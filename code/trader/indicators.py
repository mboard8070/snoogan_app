# trader/indicators.py - XGBoost version for trend prediction
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# XGBoost import with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[XGBoost] Not installed - using indicator-only mode")

# Model state
_xgb_model = None
_last_training_date = None
_feature_importance = {}
MODEL_PATH = Path("/home/mboard76/nvidia-workbench/snoogan_app/code/trader/xgb_trend_model.pkl")


def _calculate_vwap(df: pd.DataFrame) -> float:
    if df is None or df.empty: return 0.0
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vol_sum = df['volume'].sum()
    if vol_sum == 0: return typical_price.iloc[-1]
    return (typical_price * df['volume']).cumsum().iloc[-1] / vol_sum


def _ema(series: pd.Series, span: int) -> float:
    if series is None or len(series) < span: return 0.0
    return series.ewm(span=span, adjust=False).mean().iloc[-1]


def _rsi(series: pd.Series, period: int = 14) -> float:
    if series is None or len(series) <= period: return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    avg_loss = loss.iloc[-1]
    if avg_loss == 0: return 100.0
    rs = gain.iloc[-1] / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> tuple:
    if series is None or len(series) < 26: return 0.0, 0.0
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) <= period: return 0.0
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, high_close, low_close)
    return tr.rolling(window=period).mean().iloc[-1]


def _kst(series: pd.Series) -> pd.Series:
    """
    Calculate KST (Know Sure Thing) indicator.
    KST combines multiple Rate of Change indicators smoothed with SMAs.

    Formula (ThinkOrSwim default):
    - ROC1 = 10-period ROC smoothed with 10-period SMA, weight 1
    - ROC2 = 15-period ROC smoothed with 10-period SMA, weight 2
    - ROC3 = 20-period ROC smoothed with 10-period SMA, weight 3
    - ROC4 = 30-period ROC smoothed with 15-period SMA, weight 4
    - KST = (ROC1 * 1) + (ROC2 * 2) + (ROC3 * 3) + (ROC4 * 4)
    """
    if series is None or len(series) < 45:  # Need enough bars for longest lookback
        return pd.Series([0.0] * len(series) if series is not None else [0.0])

    # Rate of Change calculations
    roc1 = ((series / series.shift(10)) - 1) * 100
    roc2 = ((series / series.shift(15)) - 1) * 100
    roc3 = ((series / series.shift(20)) - 1) * 100
    roc4 = ((series / series.shift(30)) - 1) * 100

    # Smooth each ROC with SMA
    sroc1 = roc1.rolling(window=10).mean()
    sroc2 = roc2.rolling(window=10).mean()
    sroc3 = roc3.rolling(window=10).mean()
    sroc4 = roc4.rolling(window=15).mean()

    # Weighted sum
    kst = (sroc1 * 1) + (sroc2 * 2) + (sroc3 * 3) + (sroc4 * 4)

    return kst


def get_kst_momentum(series: pd.Series) -> dict:
    """
    Get KST momentum direction for entry filtering.

    Matches ThinkOrSwim logic:
    - CALL entry: KST rising for 2 consecutive bars (KST[1] > KST[2] > KST[3])
    - PUT entry: KST falling for 2 consecutive bars (KST[1] < KST[2] < KST[3])

    Returns dict with:
        - kst_current: current KST value
        - kst_rising: True if KST rising for 2 bars (bullish)
        - kst_falling: True if KST falling for 2 bars (bearish)
        - kst_direction: 'rising', 'falling', or 'choppy'
    """
    kst = _kst(series)

    if len(kst) < 4 or kst.isna().iloc[-1]:
        return {
            'kst_current': 0.0,
            'kst_rising': False,
            'kst_falling': False,
            'kst_direction': 'choppy'
        }

    # Get last 4 KST values (current and 3 bars back)
    kst_1 = kst.iloc[-1]  # Most recent (current bar)
    kst_2 = kst.iloc[-2]  # 1 bar ago
    kst_3 = kst.iloc[-3]  # 2 bars ago
    kst_4 = kst.iloc[-4]  # 3 bars ago

    # ThinkOrSwim logic: 2 consecutive bars of momentum
    # Rising: KST[2] > KST[3] AND KST[1] > KST[2]
    kst_rising = (kst_2 > kst_3) and (kst_1 > kst_2)

    # Falling: KST[2] < KST[3] AND KST[1] < KST[2]
    kst_falling = (kst_2 < kst_3) and (kst_1 < kst_2)

    if kst_rising:
        direction = 'rising'
    elif kst_falling:
        direction = 'falling'
    else:
        direction = 'choppy'

    return {
        'kst_current': round(kst_1, 4),
        'kst_rising': kst_rising,
        'kst_falling': kst_falling,
        'kst_direction': direction
    }


def _extract_features(bars: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Extract features for XGBoost from price bars.
    Returns a dict of features that can be used for prediction.
    """
    if bars is None or len(bars) < lookback:
        return None

    close = bars['close']
    high = bars['high']
    low = bars['low']
    volume = bars['volume']

    price = close.iloc[-1]

    # RSI at multiple timeframes
    rsi_14 = _rsi(close, 14)
    rsi_7 = _rsi(close, 7)

    # MACD
    macd_line, macd_signal = _macd(close)
    macd_hist = macd_line - macd_signal

    # EMAs and crossovers
    ema_9 = _ema(close, 9)
    ema_21 = _ema(close, 21)
    ema_50 = _ema(close, 50) if len(close) >= 50 else ema_21

    ema_cross_9_21 = 1 if ema_9 > ema_21 else -1
    ema_cross_21_50 = 1 if ema_21 > ema_50 else -1

    # Price relative to EMAs
    price_vs_ema9 = (price - ema_9) / price * 100 if price > 0 else 0
    price_vs_ema21 = (price - ema_21) / price * 100 if price > 0 else 0

    # VWAP
    vwap = _calculate_vwap(bars)
    price_vs_vwap = (price - vwap) / price * 100 if price > 0 else 0

    # ATR (volatility)
    atr = _atr(bars, 14)
    atr_pct = atr / price * 100 if price > 0 else 0

    # Momentum features
    returns_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
    returns_10 = (close.iloc[-1] / close.iloc[-11] - 1) * 100 if len(close) > 10 else 0
    returns_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0

    # Volume features
    vol_avg = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
    vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 1.0

    # High/Low range
    high_20 = high.rolling(20).max().iloc[-1] if len(high) >= 20 else high.max()
    low_20 = low.rolling(20).min().iloc[-1] if len(low) >= 20 else low.min()
    price_position = (price - low_20) / (high_20 - low_20) if (high_20 - low_20) > 0 else 0.5

    # Candle features (last few candles)
    body_pct = (close.iloc[-1] - bars['open'].iloc[-1]) / price * 100 if price > 0 else 0
    upper_wick = (high.iloc[-1] - max(close.iloc[-1], bars['open'].iloc[-1])) / price * 100 if price > 0 else 0
    lower_wick = (min(close.iloc[-1], bars['open'].iloc[-1]) - low.iloc[-1]) / price * 100 if price > 0 else 0

    # Time features
    hour = datetime.now().hour
    minute = datetime.now().minute
    time_of_day = hour + minute / 60  # 9.5 = 9:30am

    return {
        'rsi_14': rsi_14,
        'rsi_7': rsi_7,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_hist': macd_hist,
        'ema_cross_9_21': ema_cross_9_21,
        'ema_cross_21_50': ema_cross_21_50,
        'price_vs_ema9': price_vs_ema9,
        'price_vs_ema21': price_vs_ema21,
        'price_vs_vwap': price_vs_vwap,
        'atr_pct': atr_pct,
        'returns_5': returns_5,
        'returns_10': returns_10,
        'returns_20': returns_20,
        'vol_ratio': vol_ratio,
        'price_position': price_position,
        'body_pct': body_pct,
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
        'time_of_day': time_of_day,
    }


def _xgb_predict(bars: pd.DataFrame) -> str:
    """Use XGBoost model to predict trend direction."""
    global _xgb_model

    if not XGBOOST_AVAILABLE or _xgb_model is None:
        return "chop"

    try:
        features = _extract_features(bars)
        if features is None:
            return "chop"

        # Convert to DataFrame for prediction
        X = pd.DataFrame([features])

        # Predict probabilities for each class
        probs = _xgb_model.predict_proba(X)[0]

        # Classes: 0=bear, 1=chop, 2=bull
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]

        # Only return strong signals
        if confidence < 0.45:
            return "chop"

        if pred_class == 2:
            return "bull"
        elif pred_class == 0:
            return "bear"
        else:
            return "chop"

    except Exception as e:
        print(f"[XGBoost Predict Error] {e}")
        return "chop"


def _extract_features_vectorized(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for all rows at once using vectorized operations.
    Much faster than calling _extract_features() in a loop.
    """
    df = bars.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_price = df['open']

    # RSI (vectorized)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    gain7 = delta.clip(lower=0).rolling(7).mean()
    loss7 = (-delta.clip(upper=0)).rolling(7).mean()
    rs7 = gain7 / (loss7 + 1e-10)
    df['rsi_7'] = 100 - (100 / (1 + rs7))

    # MACD (vectorized)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # EMAs
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    df['ema_cross_9_21'] = (ema_9 > ema_21).astype(int) * 2 - 1
    df['ema_cross_21_50'] = (ema_21 > ema_50).astype(int) * 2 - 1
    df['price_vs_ema9'] = (close - ema_9) / close * 100
    df['price_vs_ema21'] = (close - ema_21) / close * 100

    # VWAP (simplified - use rolling)
    typical = (high + low + close) / 3
    vwap = (typical * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
    df['price_vs_vwap'] = (close - vwap) / close * 100

    # ATR
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr = tr.rolling(14).mean()
    df['atr_pct'] = atr / close * 100

    # Returns
    df['returns_5'] = close.pct_change(5) * 100
    df['returns_10'] = close.pct_change(10) * 100
    df['returns_20'] = close.pct_change(20) * 100

    # Volume
    vol_avg = volume.rolling(20).mean()
    df['vol_ratio'] = volume / (vol_avg + 1e-10)

    # Price position in range
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    df['price_position'] = (close - low_20) / (high_20 - low_20 + 1e-10)

    # Candle features
    df['body_pct'] = (close - open_price) / close * 100
    df['upper_wick'] = (high - np.maximum(close, open_price)) / close * 100
    df['lower_wick'] = (np.minimum(close, open_price) - low) / close * 100

    # Time features (use index if datetime)
    if hasattr(df.index, 'hour'):
        df['time_of_day'] = df.index.hour + df.index.minute / 60
    else:
        df['time_of_day'] = 12.0  # Default midday

    feature_cols = [
        'rsi_14', 'rsi_7', 'macd_line', 'macd_signal', 'macd_hist',
        'ema_cross_9_21', 'ema_cross_21_50', 'price_vs_ema9', 'price_vs_ema21',
        'price_vs_vwap', 'atr_pct', 'returns_5', 'returns_10', 'returns_20',
        'vol_ratio', 'price_position', 'body_pct', 'upper_wick', 'lower_wick',
        'time_of_day'
    ]

    return df[feature_cols]


def train_xgb_daily(bars_1m: pd.DataFrame):
    """
    Train XGBoost model daily on recent price data.
    Creates labels based on future price movement.
    """
    global _xgb_model, _last_training_date, _feature_importance

    if not XGBOOST_AVAILABLE:
        print("[XGBoost] Not available - skipping training")
        return

    today = datetime.now().date()
    if _last_training_date == today:
        return

    min_bars = 500
    if len(bars_1m) < min_bars:
        print(f"[XGBoost] Not enough bars ({len(bars_1m)} < {min_bars}) - skipping training")
        return

    print(f"[XGBoost] Training on {len(bars_1m)} bars...")
    start_time = datetime.now()

    # Vectorized feature extraction (FAST)
    print("[XGBoost] Extracting features...")
    X_all = _extract_features_vectorized(bars_1m)

    # Create labels: future price direction
    lookahead = 10
    close = bars_1m['close']
    future_returns = close.shift(-lookahead) / close - 1
    future_returns_pct = future_returns * 100

    # Labels: 0=bear, 1=chop, 2=bull
    y_all = pd.Series(1, index=bars_1m.index)  # Default chop
    y_all[future_returns_pct > 0.1] = 2   # Bull
    y_all[future_returns_pct < -0.1] = 0  # Bear

    # Drop rows with NaN (from rolling calculations and future shift)
    valid_mask = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all[valid_mask].copy()
    y = y_all[valid_mask].values

    if len(X) < 100:
        print(f"[XGBoost] Not enough training samples ({len(X)})")
        return

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(['bear', 'chop', 'bull'], [counts[unique == i][0] if i in unique else 0 for i in range(3)]))
    print(f"[XGBoost] Class distribution: {class_dist}")

    # Train XGBoost classifier
    _xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0,
        random_state=42
    )

    _xgb_model.fit(X, y)

    # Store feature importance
    importance = _xgb_model.feature_importances_
    _feature_importance = dict(zip(X.columns, importance))

    # Sort and display top features
    sorted_importance = sorted(_feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("[XGBoost] Top 5 features:")
    for feat, imp in sorted_importance[:5]:
        print(f"  {feat}: {imp:.4f}")

    # Save model
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(_xgb_model, f)
    except Exception as e:
        print(f"[XGBoost] Could not save model: {e}")

    _last_training_date = today
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"[XGBoost] Training complete in {train_time:.1f}s on {len(X)} samples")


def load_xgb_model():
    """Load saved XGBoost model if available."""
    global _xgb_model

    if not XGBOOST_AVAILABLE:
        return False

    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, 'rb') as f:
                _xgb_model = pickle.load(f)
            print("[XGBoost] Loaded saved model")
            return True
        except Exception as e:
            print(f"[XGBoost] Could not load model: {e}")
    return False


def get_feature_importance() -> dict:
    """Return feature importance from last training."""
    return _feature_importance


def get_trend_signal(bars_1m: pd.DataFrame, bars_5m: pd.DataFrame = None, bars_15m: pd.DataFrame = None) -> str:
    if bars_1m is None or bars_1m.empty:
        return "chop"

    valid_tfs = [tf for tf in [bars_1m, bars_5m, bars_15m] if tf is not None and not tf.empty]
    if not valid_tfs:
        return "chop"

    price = bars_1m['close'].iloc[-1]
    num_tfs = len(valid_tfs)

    vwap_votes = sum(price > _calculate_vwap(tf) for tf in valid_tfs)

    ema_tfs = [tf for tf in valid_tfs if len(tf) >= 21]
    ema_votes = sum(_ema(tf['close'], 9) > _ema(tf['close'], 21) for tf in ema_tfs)

    macd_tfs = [tf for tf in valid_tfs if len(tf) >= 26]
    macd_votes = sum(_macd(tf['close'])[0] > _macd(tf['close'])[1] for tf in macd_tfs)

    rsi_subset = [tf for tf in [bars_1m, bars_5m] if tf is not None and len(tf) > 14]
    rsi_votes = sum(1 if _rsi(tf['close']) > 50 else -1 if _rsi(tf['close']) < 50 else 0 for tf in rsi_subset)

    atr_1m = _atr(bars_1m) if len(bars_1m) >= 14 else 0.0

    # XGBoost prediction (replaces LSTM)
    xgb_pred = _xgb_predict(bars_1m)
    xgb_vote = 1 if xgb_pred == "bull" else -1 if xgb_pred == "bear" else 0

    # XGBoost vote weight: +1.2 for bull, -1.2 for bear, 0 for chop
    xgb_bull_weight = 1.2 if xgb_vote > 0 else 0
    xgb_bear_weight = 1.2 if xgb_vote < 0 else 0

    bull_score = (vwap_votes / num_tfs if num_tfs else 0) + \
                 (ema_votes / len(ema_tfs) if ema_tfs else 0) + \
                 1.5 * (macd_votes / len(macd_tfs) if macd_tfs else 0) + \
                 (1 if rsi_votes > 0 else 0) + \
                 xgb_bull_weight

    bear_score = ((num_tfs - vwap_votes) / num_tfs if num_tfs else 0) + \
                 ((len(ema_tfs) - ema_votes) / len(ema_tfs) if ema_tfs else 0) + \
                 1.5 * ((len(macd_tfs) - macd_votes) / len(macd_tfs) if macd_tfs else 0) + \
                 (1 if rsi_votes < 0 else 0) + \
                 xgb_bear_weight

    if bull_score >= 3.5 or (bull_score >= 3 and atr_1m < price * 0.003):
        return "bull"
    elif bear_score >= 3.5 or (bear_score >= 3 and atr_1m < price * 0.003):
        return "bear"
    else:
        return "chop"


def get_full_indicator_set(bars: pd.DataFrame) -> dict:
    """
    Get complete indicator set for learning/brain integration.
    Returns dict with all relevant indicators for pattern recognition.
    """
    if bars is None or bars.empty or len(bars) < 26:
        return {
            'rsi': 50.0,
            'macd_line': 0.0,
            'macd_signal': 0.0,
            'macd_bull': False,
            'atr': 0.0,
            'trend_strength': 0.0,
            'price': 0.0,
            'vwap': 0.0
        }

    close = bars['close']
    price = close.iloc[-1]

    # RSI
    rsi = _rsi(close, 14)

    # MACD
    macd_line, macd_signal = _macd(close)
    macd_bull = macd_line > macd_signal

    # ATR (volatility)
    atr = _atr(bars, 14)

    # VWAP
    vwap = _calculate_vwap(bars)

    # Trend strength (bull_score - bear_score from get_trend_signal logic)
    # Simplified version: use EMA crossover + RSI deviation from 50
    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema_strength = (ema9 - ema21) / price * 100 if price > 0 else 0  # Percentage
    rsi_strength = (rsi - 50) / 50  # -1 to +1 scale
    trend_strength = (ema_strength + rsi_strength) / 2

    return {
        'rsi': round(rsi, 2),
        'macd_line': round(macd_line, 6),
        'macd_signal': round(macd_signal, 6),
        'macd_bull': macd_bull,
        'atr': round(atr, 4),
        'trend_strength': round(trend_strength, 4),
        'price': round(price, 2),
        'vwap': round(vwap, 2)
    }


# Backward compatibility aliases
def train_lstm_daily(bars_1m: pd.DataFrame, **kwargs):
    """Alias for backward compatibility - calls XGBoost training."""
    train_xgb_daily(bars_1m)


# Try to load saved model on import
load_xgb_model()
