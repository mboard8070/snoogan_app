# trader/indicators.py - XGBoost version for trend prediction
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import pickle
import json
from pathlib import Path
from filelock import FileLock, Timeout

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

# Stats persistence paths
SCRIPT_DIR = Path(__file__).parent
STATS_STATE_FILE = SCRIPT_DIR / "indicator_stats_state.json"
STATS_LOCK_FILE = SCRIPT_DIR / "indicator_stats_state.json.lock"
MAX_TRADE_HISTORY = 500


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


# Volatility regime detection for flat day protection
# NOTE: Using AND logic - BOTH conditions must be met to block trading
# This avoids being overly restrictive (user reported ATR alone blocks too many trades)
MIN_ATR_PCT = 0.0012  # 0.12% - looser threshold since we require both conditions
MIN_RANGE_RATIO = 0.5  # Today's range must be >= 50% of 20-day ADR
LOW_VOL_SIZE_CAP = 10  # Max contracts in low vol
COMPRESSED_SIZE_CAP = 8  # Max contracts when range compressed
MAX_VWAP_CROSSES_PER_HOUR = 4  # More than this per hour = grinding/choppy day


def get_daily_range_ratio(bars_1m: pd.DataFrame, lookback_days: int = 20) -> float:
    """
    Calculate today's range vs average daily range (ADR).
    Returns ratio: 0.5 = today's range is 50% of average.

    Args:
        bars_1m: DataFrame with 1-minute OHLCV bars
        lookback_days: Number of days for ADR calculation

    Returns:
        Ratio of today's range to ADR (e.g., 0.4 = 40% of average)
    """
    if bars_1m is None or bars_1m.empty or len(bars_1m) < 100:
        return 1.0  # Default to normal if not enough data

    try:
        # Resample to daily bars
        daily = bars_1m.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        if len(daily) < 2:
            return 1.0

        # Calculate daily ranges
        daily['range'] = daily['high'] - daily['low']

        # Today's range (last complete or current day)
        today_range = daily['range'].iloc[-1]

        # Average daily range (excluding today)
        if len(daily) > lookback_days:
            adr = daily['range'].iloc[-(lookback_days + 1):-1].mean()
        else:
            adr = daily['range'].iloc[:-1].mean()

        if adr == 0:
            return 1.0

        return today_range / adr

    except Exception as e:
        print(f"[Volatility] Error calculating daily range ratio: {e}")
        return 1.0


def get_vwap_cross_count(bars_1m: pd.DataFrame) -> dict:
    """
    Count how many times price has crossed VWAP today.
    High cross count indicates grinding/choppy market conditions.

    Args:
        bars_1m: DataFrame with 1-minute OHLCV bars (should include today's data)

    Returns:
        dict with:
            - cross_count: Number of VWAP crosses today
            - crosses_per_hour: Normalized cross rate
            - hours_traded: Hours of trading data today
            - is_choppy: True if cross rate exceeds threshold
    """
    if bars_1m is None or bars_1m.empty or len(bars_1m) < 10:
        return {
            'cross_count': 0,
            'crosses_per_hour': 0.0,
            'hours_traded': 0.0,
            'is_choppy': False
        }

    try:
        # Filter to today's bars only
        today = bars_1m.index[-1].date()
        today_bars = bars_1m[bars_1m.index.date == today].copy()

        if len(today_bars) < 10:
            return {
                'cross_count': 0,
                'crosses_per_hour': 0.0,
                'hours_traded': 0.0,
                'is_choppy': False
            }

        # Calculate intraday VWAP (cumulative from market open)
        typical_price = (today_bars['high'] + today_bars['low'] + today_bars['close']) / 3
        cumulative_tp_vol = (typical_price * today_bars['volume']).cumsum()
        cumulative_vol = today_bars['volume'].cumsum()
        vwap_series = cumulative_tp_vol / (cumulative_vol + 1e-10)

        # Count VWAP crosses (when price crosses from above to below or vice versa)
        price = today_bars['close']
        above_vwap = price > vwap_series
        crosses = (above_vwap != above_vwap.shift(1)).sum()

        # Calculate hours traded today
        if len(today_bars) > 1:
            time_diff = (today_bars.index[-1] - today_bars.index[0]).total_seconds() / 3600
            hours_traded = max(time_diff, 0.1)  # Minimum 6 minutes
        else:
            hours_traded = 0.1

        crosses_per_hour = crosses / hours_traded if hours_traded > 0 else 0

        # Determine if choppy (too many crosses per hour)
        is_choppy = crosses_per_hour > MAX_VWAP_CROSSES_PER_HOUR

        return {
            'cross_count': int(crosses),
            'crosses_per_hour': round(crosses_per_hour, 1),
            'hours_traded': round(hours_traded, 2),
            'is_choppy': is_choppy
        }

    except Exception as e:
        print(f"[Volatility] Error calculating VWAP crosses: {e}")
        return {
            'cross_count': 0,
            'crosses_per_hour': 0.0,
            'hours_traded': 0.0,
            'is_choppy': False
        }


def get_volatility_regime(bars_1m: pd.DataFrame) -> dict:
    """
    Returns volatility metrics for entry filtering.
    Used to detect flat/range-bound days where trading should be avoided.

    Args:
        bars_1m: DataFrame with 1-minute OHLCV bars

    Returns:
        dict with:
            - atr_pct: ATR as % of price (e.g., 0.002 = 0.2%)
            - daily_range_ratio: today's range vs 20-day ADR
            - vwap_crosses: VWAP cross count data
            - is_low_vol: True if trading should be avoided
            - reason: Why is_low_vol is True (if applicable)
    """
    if bars_1m is None or bars_1m.empty:
        return {
            'atr_pct': 0.0,
            'daily_range_ratio': 1.0,
            'vwap_crosses': {'cross_count': 0, 'crosses_per_hour': 0.0, 'is_choppy': False},
            'is_low_vol': False,
            'reason': None
        }

    try:
        price = bars_1m['close'].iloc[-1]
        atr = _atr(bars_1m, 14)
        atr_pct = atr / price if price > 0 else 0.0

        daily_range_ratio = get_daily_range_ratio(bars_1m, 20)
        vwap_data = get_vwap_cross_count(bars_1m)

        # Determine if low volatility / grinding day
        # Two ways to trigger:
        # 1. Low ATR AND compressed range (original logic)
        # 2. High VWAP cross rate (grinding/choppy - price keeps reverting)
        is_low_vol = False
        reason = None

        if atr_pct < MIN_ATR_PCT and daily_range_ratio < MIN_RANGE_RATIO:
            is_low_vol = True
            reason = f"FLAT DAY: ATR={atr_pct*100:.2f}% + Range={daily_range_ratio:.0%} of ADR"
        elif vwap_data['is_choppy'] and daily_range_ratio < 0.7:
            # High VWAP crosses + compressed range = grinding day
            is_low_vol = True
            reason = f"GRINDING DAY: {vwap_data['crosses_per_hour']:.1f} VWAP crosses/hr + Range={daily_range_ratio:.0%}"

        return {
            'atr_pct': round(atr_pct, 5),
            'daily_range_ratio': round(daily_range_ratio, 3),
            'vwap_crosses': vwap_data,
            'is_low_vol': is_low_vol,
            'reason': reason
        }

    except Exception as e:
        print(f"[Volatility] Error calculating regime: {e}")
        return {
            'atr_pct': 0.0,
            'daily_range_ratio': 1.0,
            'vwap_crosses': {'cross_count': 0, 'crosses_per_hour': 0.0, 'is_choppy': False},
            'is_low_vol': False,
            'reason': None
        }


# =============================================================================
# STATISTICAL MATH FOR SPREADS (Binomial, Conditional Probability, GARCH, etc.)
# =============================================================================

# Trade history storage for binomial calculations
_trade_history = []
_regime_stats = {}  # {regime: {'wins': int, 'losses': int, 'returns': []}}


def _load_stats_state():
    """Load trade history and regime stats from disk."""
    global _trade_history, _regime_stats

    if not STATS_STATE_FILE.exists():
        return

    try:
        lock = FileLock(str(STATS_LOCK_FILE), timeout=5)
        with lock:
            with open(STATS_STATE_FILE, 'r') as f:
                state = json.load(f)
                _trade_history = state.get('trade_history', [])
                _regime_stats = state.get('regime_stats', {})
                print(f"[Stats] Loaded {len(_trade_history)} trades, {len(_regime_stats)} regimes from disk")
    except Timeout:
        print("[Stats] Could not acquire lock for loading state")
    except Exception as e:
        print(f"[Stats] Error loading state: {e}")


def _save_stats_state():
    """Persist trade history and regime stats to disk."""
    global _trade_history, _regime_stats

    try:
        # Trim history to MAX_TRADE_HISTORY
        if len(_trade_history) > MAX_TRADE_HISTORY:
            _trade_history = _trade_history[-MAX_TRADE_HISTORY:]

        # Trim returns per regime to last 100
        for regime in _regime_stats:
            if len(_regime_stats[regime].get('returns', [])) > 100:
                _regime_stats[regime]['returns'] = _regime_stats[regime]['returns'][-100:]

        state = {
            'trade_history': _trade_history,
            'regime_stats': _regime_stats,
            'last_updated': datetime.now().isoformat(),
            'version': 1
        }

        lock = FileLock(str(STATS_LOCK_FILE), timeout=5)
        with lock:
            temp_file = STATS_STATE_FILE.with_suffix('.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            temp_file.replace(STATS_STATE_FILE)

    except Timeout:
        print("[Stats] Could not acquire lock for saving state")
    except Exception as e:
        print(f"[Stats] Error saving state: {e}")


def record_trade(result: dict):
    """
    Record a trade result for statistical tracking.

    Args:
        result: dict with keys:
            - pnl: float (profit/loss in dollars)
            - win: bool (True if profitable)
            - regime: str (e.g., 'bull', 'bear', 'chop', 'low_vol')
            - setup_type: str (e.g., 'call_spread', 'put_spread')
            - entry_time: datetime
            - holding_period: int (minutes/bars held)
    """
    global _trade_history, _regime_stats

    _trade_history.append(result)

    # Update regime stats
    regime = result.get('regime', 'unknown')
    if regime not in _regime_stats:
        _regime_stats[regime] = {'wins': 0, 'losses': 0, 'returns': []}

    if result.get('win', False):
        _regime_stats[regime]['wins'] += 1
    else:
        _regime_stats[regime]['losses'] += 1

    if 'pnl' in result:
        _regime_stats[regime]['returns'].append(result['pnl'])

    # Persist to disk after each trade
    _save_stats_state()


def get_binomial_win_probability(
    regime: str = None,
    setup_type: str = None,
    min_trades: int = 20
) -> dict:
    """
    Calculate win probability using binomial framework.

    Uses binomial distribution to estimate:
    - Point estimate of win rate
    - Confidence interval (Wilson score interval - better for small samples)
    - Whether we have enough data to trust the estimate

    Args:
        regime: Filter by market regime (optional)
        setup_type: Filter by setup type (optional)
        min_trades: Minimum trades needed for reliable estimate

    Returns:
        dict with:
            - win_rate: Point estimate of P(win)
            - ci_lower: 95% CI lower bound
            - ci_upper: 95% CI upper bound
            - n_trades: Sample size
            - reliable: True if n_trades >= min_trades
            - avg_win: Average winning trade
            - avg_loss: Average losing trade
            - expected_value: (win_rate * avg_win) - (loss_rate * avg_loss)
    """
    # Filter trades
    trades = _trade_history.copy()

    if regime is not None:
        trades = [t for t in trades if t.get('regime') == regime]
    if setup_type is not None:
        trades = [t for t in trades if t.get('setup_type') == setup_type]

    n = len(trades)

    if n == 0:
        return {
            'win_rate': 0.5,  # Prior assumption
            'ci_lower': 0.0,
            'ci_upper': 1.0,
            'n_trades': 0,
            'reliable': False,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'expected_value': 0.0
        }

    # Count wins
    wins = sum(1 for t in trades if t.get('win', False))
    losses = n - wins

    # Point estimate
    p_hat = wins / n

    # Wilson score interval (better than normal approximation for small n)
    # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    z = 1.96  # 95% CI
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    ci_lower = max(0, center - spread)
    ci_upper = min(1, center + spread)

    # Calculate average win/loss
    winning_trades = [t['pnl'] for t in trades if t.get('win') and 'pnl' in t]
    losing_trades = [t['pnl'] for t in trades if not t.get('win') and 'pnl' in t]

    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0

    # Expected value per trade
    expected_value = (p_hat * avg_win) - ((1 - p_hat) * avg_loss)

    return {
        'win_rate': round(p_hat, 4),
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'n_trades': n,
        'reliable': n >= min_trades,
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'expected_value': round(expected_value, 2)
    }


def get_conditional_expected_value(current_regime: str) -> dict:
    """
    Calculate expected value conditioned on current market regime.

    Implements: E[return | regime] using historical data.
    This is Bayesian in spirit - updating our expectation based on conditions.

    Args:
        current_regime: Current market regime ('bull', 'bear', 'chop', 'low_vol')

    Returns:
        dict with:
            - expected_return: E[return | regime]
            - regime_win_rate: P(win | regime)
            - overall_win_rate: P(win) unconditional
            - regime_advantage: How much better/worse this regime is
            - sample_size: Number of trades in this regime
            - recommendation: 'favorable', 'neutral', or 'unfavorable'
    """
    # Get overall stats
    overall = get_binomial_win_probability()

    # Get regime-specific stats
    regime_stats = get_binomial_win_probability(regime=current_regime)

    # Calculate regime advantage
    if overall['win_rate'] > 0:
        regime_advantage = (regime_stats['win_rate'] - overall['win_rate']) / overall['win_rate']
    else:
        regime_advantage = 0.0

    # Determine recommendation
    if regime_stats['n_trades'] < 10:
        recommendation = 'insufficient_data'
    elif regime_stats['expected_value'] > 0 and regime_stats['win_rate'] > 0.5:
        recommendation = 'favorable'
    elif regime_stats['expected_value'] < 0 or regime_stats['win_rate'] < 0.4:
        recommendation = 'unfavorable'
    else:
        recommendation = 'neutral'

    return {
        'expected_return': regime_stats['expected_value'],
        'regime_win_rate': regime_stats['win_rate'],
        'overall_win_rate': overall['win_rate'],
        'regime_advantage': round(regime_advantage, 4),
        'sample_size': regime_stats['n_trades'],
        'recommendation': recommendation,
        'ci_lower': regime_stats['ci_lower'],
        'ci_upper': regime_stats['ci_upper']
    }


def garch_volatility(returns: pd.Series, omega: float = 0.00001, alpha: float = 0.1, beta: float = 0.85) -> dict:
    """
    Simple GARCH(1,1) volatility estimation.

    GARCH captures volatility clustering: "After a big move, expect more volatility."

    Model: σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}

    Where:
        - ω (omega): Long-term variance constant
        - α (alpha): Weight on recent squared return (reaction to news)
        - β (beta): Weight on previous variance (persistence)
        - α + β < 1 for stationarity (typically α + β ≈ 0.95)

    Args:
        returns: Series of returns (e.g., close.pct_change())
        omega: Long-term variance constant (default 0.00001)
        alpha: ARCH coefficient (default 0.1)
        beta: GARCH coefficient (default 0.85)

    Returns:
        dict with:
            - current_vol: Current GARCH volatility estimate (annualized %)
            - vol_series: Full volatility series
            - vol_forecast_1: 1-step ahead forecast
            - long_term_vol: Unconditional (long-term) volatility
            - vol_regime: 'high', 'normal', or 'low' relative to long-term
    """
    if returns is None or len(returns) < 30:
        return {
            'current_vol': 0.0,
            'vol_series': None,
            'vol_forecast_1': 0.0,
            'long_term_vol': 0.0,
            'vol_regime': 'unknown'
        }

    returns = returns.dropna()
    n = len(returns)

    # Initialize variance with sample variance
    var_series = np.zeros(n)
    var_series[0] = returns.var()

    # GARCH(1,1) recursion
    for t in range(1, n):
        var_series[t] = omega + alpha * returns.iloc[t-1]**2 + beta * var_series[t-1]

    # Current volatility (annualized, assuming 1-min bars -> ~252*390 bars/year)
    # Adjust multiplier based on your bar frequency
    current_var = var_series[-1]
    current_vol = np.sqrt(current_var) * np.sqrt(252 * 390) * 100  # Annualized %

    # 1-step forecast
    vol_forecast_var = omega + alpha * returns.iloc[-1]**2 + beta * current_var
    vol_forecast_1 = np.sqrt(vol_forecast_var) * np.sqrt(252 * 390) * 100

    # Long-term (unconditional) volatility: σ² = ω / (1 - α - β)
    if (alpha + beta) < 1:
        long_term_var = omega / (1 - alpha - beta)
        long_term_vol = np.sqrt(long_term_var) * np.sqrt(252 * 390) * 100
    else:
        long_term_vol = current_vol  # Fallback if non-stationary

    # Determine volatility regime
    vol_ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0
    if vol_ratio > 1.3:
        vol_regime = 'high'
    elif vol_ratio < 0.7:
        vol_regime = 'low'
    else:
        vol_regime = 'normal'

    return {
        'current_vol': round(current_vol, 2),
        'vol_series': pd.Series(np.sqrt(var_series) * np.sqrt(252 * 390) * 100, index=returns.index),
        'vol_forecast_1': round(vol_forecast_1, 2),
        'long_term_vol': round(long_term_vol, 2),
        'vol_regime': vol_regime
    }


def get_spread_zscore(
    spread_series: pd.Series,
    lookback: int = 20,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5
) -> dict:
    """
    Calculate z-score of spread for mean reversion signals.

    For spread trading (pairs, options spreads), we want to know:
    - How far is current spread from its mean?
    - Is it likely to revert?

    Z-score = (current - mean) / std

    Args:
        spread_series: Series of spread values (e.g., price_A - price_B, or option spread price)
        lookback: Window for mean/std calculation
        entry_threshold: Z-score magnitude for entry signal (default 2.0)
        exit_threshold: Z-score magnitude for exit signal (default 0.5)

    Returns:
        dict with:
            - zscore: Current z-score
            - mean: Rolling mean
            - std: Rolling std
            - upper_band: mean + entry_threshold * std
            - lower_band: mean - entry_threshold * std
            - signal: 'long' (z < -threshold), 'short' (z > threshold), 'exit', or 'hold'
            - half_life: Estimated mean reversion half-life in bars
    """
    if spread_series is None or len(spread_series) < lookback:
        return {
            'zscore': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'upper_band': 0.0,
            'lower_band': 0.0,
            'signal': 'hold',
            'half_life': None
        }

    # Calculate rolling statistics
    roll_mean = spread_series.rolling(lookback).mean()
    roll_std = spread_series.rolling(lookback).std()

    current_spread = spread_series.iloc[-1]
    current_mean = roll_mean.iloc[-1]
    current_std = roll_std.iloc[-1]

    if current_std == 0 or np.isnan(current_std):
        zscore = 0.0
    else:
        zscore = (current_spread - current_mean) / current_std

    # Calculate bands
    upper_band = current_mean + entry_threshold * current_std
    lower_band = current_mean - entry_threshold * current_std

    # Determine signal
    if zscore < -entry_threshold:
        signal = 'long'  # Spread is cheap, expect reversion up
    elif zscore > entry_threshold:
        signal = 'short'  # Spread is expensive, expect reversion down
    elif abs(zscore) < exit_threshold:
        signal = 'exit'  # Close to mean, exit positions
    else:
        signal = 'hold'

    # Estimate half-life using Ornstein-Uhlenbeck model
    # Half-life = -ln(2) / ln(β) where β is AR(1) coefficient
    half_life = None
    try:
        if len(spread_series) >= lookback + 10:
            spread_lag = spread_series.shift(1).dropna()
            spread_now = spread_series.iloc[1:]

            # Simple AR(1) regression: spread_t = α + β * spread_{t-1} + ε
            if len(spread_lag) > 10:
                beta = np.corrcoef(spread_lag, spread_now)[0, 1]
                if 0 < beta < 1:
                    half_life = -np.log(2) / np.log(beta)
                    half_life = round(half_life, 1)
    except:
        pass

    return {
        'zscore': round(zscore, 3),
        'mean': round(current_mean, 4),
        'std': round(current_std, 4),
        'upper_band': round(upper_band, 4),
        'lower_band': round(lower_band, 4),
        'signal': signal,
        'half_life': half_life
    }


def get_rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.05,
    annualization_factor: int = 252 * 390
) -> dict:
    """
    Calculate rolling Sharpe ratio for performance tracking.

    Sharpe = (Return - Risk_Free) / Volatility

    Measures risk-adjusted returns. Higher = better.
    - Sharpe > 1.0: Good
    - Sharpe > 2.0: Very good
    - Sharpe > 3.0: Excellent (rare, check for overfitting)

    Args:
        returns: Series of returns (e.g., strategy PnL or pct returns)
        window: Rolling window in bars
        risk_free_rate: Annual risk-free rate (default 5%)
        annualization_factor: Bars per year (default 252*390 for 1-min bars)

    Returns:
        dict with:
            - current_sharpe: Most recent Sharpe ratio (annualized)
            - sharpe_series: Full rolling Sharpe series
            - sharpe_trend: 'improving', 'stable', or 'degrading'
            - cumulative_return: Total return over window
            - volatility: Annualized volatility
            - sortino: Sortino ratio (uses downside vol only)
    """
    if returns is None or len(returns) < window:
        return {
            'current_sharpe': 0.0,
            'sharpe_series': None,
            'sharpe_trend': 'unknown',
            'cumulative_return': 0.0,
            'volatility': 0.0,
            'sortino': 0.0
        }

    returns = returns.dropna()

    # Per-bar risk-free rate
    rf_per_bar = risk_free_rate / annualization_factor

    # Rolling calculations
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()

    # Sharpe ratio (annualized)
    excess_return = roll_mean - rf_per_bar
    sharpe_series = (excess_return / roll_std) * np.sqrt(annualization_factor)
    sharpe_series = sharpe_series.replace([np.inf, -np.inf], 0)

    current_sharpe = sharpe_series.iloc[-1] if not np.isnan(sharpe_series.iloc[-1]) else 0.0

    # Cumulative return over window
    cumulative_return = (1 + returns.iloc[-window:]).prod() - 1 if len(returns) >= window else 0.0

    # Annualized volatility
    volatility = roll_std.iloc[-1] * np.sqrt(annualization_factor) * 100 if not np.isnan(roll_std.iloc[-1]) else 0.0

    # Sortino ratio (downside deviation only)
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    roll_downside_std = downside_returns.rolling(window).std()
    sortino = (excess_return.iloc[-1] / roll_downside_std.iloc[-1]) * np.sqrt(annualization_factor) if roll_downside_std.iloc[-1] > 0 else 0.0

    # Determine Sharpe trend
    if len(sharpe_series) >= window:
        recent_sharpe = sharpe_series.iloc[-10:].mean()
        older_sharpe = sharpe_series.iloc[-window:-window+10].mean()

        if recent_sharpe > older_sharpe + 0.3:
            sharpe_trend = 'improving'
        elif recent_sharpe < older_sharpe - 0.3:
            sharpe_trend = 'degrading'
        else:
            sharpe_trend = 'stable'
    else:
        sharpe_trend = 'unknown'

    return {
        'current_sharpe': round(current_sharpe, 3),
        'sharpe_series': sharpe_series,
        'sharpe_trend': sharpe_trend,
        'cumulative_return': round(cumulative_return * 100, 2),  # As percentage
        'volatility': round(volatility, 2),
        'sortino': round(sortino, 3) if not np.isnan(sortino) else 0.0
    }


def get_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> dict:
    """
    Calculate Kelly Criterion for optimal position sizing.

    Kelly% = W - (1-W)/R
    Where:
        W = Win rate
        R = Win/Loss ratio (avg_win / avg_loss)

    Kelly tells you the optimal fraction of capital to risk.
    Most traders use fractional Kelly (25-50%) for safety.

    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive number)

    Returns:
        dict with:
            - full_kelly: Optimal fraction (can be > 1 or negative)
            - half_kelly: Conservative recommendation
            - quarter_kelly: Very conservative recommendation
            - edge: Mathematical edge per trade
            - recommendation: Position sizing recommendation
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return {
            'full_kelly': 0.0,
            'half_kelly': 0.0,
            'quarter_kelly': 0.0,
            'edge': 0.0,
            'recommendation': 'no_edge'
        }

    # Win/Loss ratio
    R = avg_win / avg_loss

    # Kelly formula
    W = win_rate
    kelly = W - (1 - W) / R

    # Edge per trade
    edge = (W * avg_win) - ((1 - W) * avg_loss)

    # Fractional Kelly (safer)
    half_kelly = kelly / 2
    quarter_kelly = kelly / 4

    # Recommendation
    if kelly <= 0:
        recommendation = 'no_edge'
    elif kelly < 0.1:
        recommendation = 'small_edge'
    elif kelly < 0.25:
        recommendation = 'moderate_edge'
    else:
        recommendation = 'strong_edge'

    return {
        'full_kelly': round(max(0, kelly), 4),
        'half_kelly': round(max(0, half_kelly), 4),
        'quarter_kelly': round(max(0, quarter_kelly), 4),
        'edge': round(edge, 2),
        'recommendation': recommendation
    }


def get_spread_statistics(bars_1m: pd.DataFrame) -> dict:
    """
    Comprehensive spread statistics combining all new indicators.

    Call this to get a full statistical picture for spread trading decisions.

    Args:
        bars_1m: DataFrame with OHLCV data

    Returns:
        dict with all statistical metrics for spread trading
    """
    if bars_1m is None or bars_1m.empty or len(bars_1m) < 60:
        return {
            'garch': {'current_vol': 0.0, 'vol_regime': 'unknown'},
            'sharpe': {'current_sharpe': 0.0, 'sharpe_trend': 'unknown'},
            'binomial': {'win_rate': 0.5, 'reliable': False},
            'kelly': {'half_kelly': 0.0, 'recommendation': 'no_edge'}
        }

    returns = bars_1m['close'].pct_change().dropna()

    # GARCH volatility
    garch = garch_volatility(returns)

    # Rolling Sharpe
    sharpe = get_rolling_sharpe(returns)

    # Binomial win probability (overall)
    binomial = get_binomial_win_probability()

    # Kelly criterion based on historical trades
    kelly = get_kelly_criterion(
        binomial['win_rate'],
        binomial['avg_win'],
        binomial['avg_loss']
    )

    # Get current regime for conditional stats
    volatility_regime = get_volatility_regime(bars_1m)
    current_regime = 'low_vol' if volatility_regime['is_low_vol'] else get_trend_signal(bars_1m)
    conditional_ev = get_conditional_expected_value(current_regime)

    return {
        'garch': {
            'current_vol': garch['current_vol'],
            'vol_forecast': garch['vol_forecast_1'],
            'long_term_vol': garch['long_term_vol'],
            'vol_regime': garch['vol_regime']
        },
        'sharpe': {
            'current_sharpe': sharpe['current_sharpe'],
            'sortino': sharpe['sortino'],
            'sharpe_trend': sharpe['sharpe_trend'],
            'volatility': sharpe['volatility']
        },
        'binomial': {
            'win_rate': binomial['win_rate'],
            'ci_lower': binomial['ci_lower'],
            'ci_upper': binomial['ci_upper'],
            'n_trades': binomial['n_trades'],
            'reliable': binomial['reliable'],
            'expected_value': binomial['expected_value']
        },
        'conditional': {
            'regime': current_regime,
            'regime_win_rate': conditional_ev['regime_win_rate'],
            'regime_advantage': conditional_ev['regime_advantage'],
            'recommendation': conditional_ev['recommendation']
        },
        'kelly': {
            'half_kelly': kelly['half_kelly'],
            'quarter_kelly': kelly['quarter_kelly'],
            'edge': kelly['edge'],
            'recommendation': kelly['recommendation']
        }
    }


def clear_trade_history():
    """Clear trade history for fresh start and persist to disk."""
    global _trade_history, _regime_stats
    _trade_history = []
    _regime_stats = {}
    _save_stats_state()


def get_trade_history_summary() -> dict:
    """Get summary of recorded trades."""
    n_trades = len(_trade_history)
    if n_trades == 0:
        return {'n_trades': 0, 'regimes': {}}

    return {
        'n_trades': n_trades,
        'regimes': {k: {'wins': v['wins'], 'losses': v['losses']} for k, v in _regime_stats.items()}
    }


# Backward compatibility aliases
def train_lstm_daily(bars_1m: pd.DataFrame, **kwargs):
    """Alias for backward compatibility - calls XGBoost training."""
    train_xgb_daily(bars_1m)


# =============================================================================
# Load persisted stats on module import
# =============================================================================
_load_stats_state()


# Try to load saved model on import
load_xgb_model()
