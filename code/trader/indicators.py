# trader/indicators.py - Enhanced for Scalping with MACD, ATR, and Adjusted Thresholds
import pandas as pd
import numpy as np

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
    """Returns MACD line, signal line"""
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

def get_trend_signal(bars_1m: pd.DataFrame, bars_5m: pd.DataFrame, bars_15m: pd.DataFrame) -> str:
    """
    Enhanced for scalping: Adds MACD votes, uses ATR for context, adjusts RSI to 30/70.
    Returns "bull", "bear", or "chop" with higher weight on faster signals.
    """
    if bars_1m is None or bars_1m.empty:
        print("!! Indicators: bars_1m is None/Empty. Returning chop.")
        return "chop"
    
    valid_tfs = [tf for tf in [bars_1m, bars_5m, bars_15m] if tf is not None and not tf.empty]
    if not valid_tfs:
        return "chop"
    
    price = bars_1m['close'].iloc[-1]
    num_tfs = len(valid_tfs)
    
    # VWAP votes (price above VWAP = bull)
    vwap_votes = sum(price > _calculate_vwap(tf) for tf in valid_tfs)
    
    # EMA crossover votes (9 > 21 = bull)
    ema_tfs = [tf for tf in valid_tfs if len(tf) >= 21]
    ema_votes = sum(_ema(tf['close'], 9) > _ema(tf['close'], 21) for tf in ema_tfs)
    
    # MACD votes (MACD > signal = bull, weighted higher for scalps)
    macd_tfs = [tf for tf in valid_tfs if len(tf) >= 26]
    macd_votes = sum(_macd(tf['close'])[0] > _macd(tf['close'])[1] for tf in macd_tfs)
    
    # RSI votes (now >70 bull, <30 bear; using 1m/5m only)
    rsi_subset = [tf for tf in [bars_1m, bars_5m] if tf is not None and len(tf) > 14]
    rsi_votes = sum(1 if _rsi(tf['close']) > 70 else -1 if _rsi(tf['close']) < 30 else 0 for tf in rsi_subset)
    
    # ATR for context (high ATR = potential chop, but not a vote)
    atr_1m = _atr(bars_1m) if len(bars_1m) >= 14 else 0.0
    
    # Scoring: Weight MACD higher (x1.5) for speed; need 3+ bull points for "bull"
    bull_score = (vwap_votes / num_tfs) + (ema_votes / len(ema_tfs)) + 1.5 * (macd_votes / len(macd_tfs)) + (rsi_votes > 0)
    bear_score = ((num_tfs - vwap_votes) / num_tfs) + ((len(ema_tfs) - ema_votes) / len(ema_tfs)) + 1.5 * ((len(macd_tfs) - macd_votes) / len(macd_tfs)) + (rsi_votes < 0)
    
    if bull_score >= 3 or (bull_score >= 2.5 and atr_1m < price * 0.001):  # Low volatility favors trend
        return "bull"
    elif bear_score >= 3 or (bear_score >= 2.5 and atr_1m < price * 0.001):
        return "bear"
    else:
        return "chop"

# Example usage (test with your data):
# signal = get_trend_signal(your_1m_df, your_5m_df, your_15m_df)
# print(signal)