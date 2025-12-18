# trader/indicators.py - Hardened for NoneType and Empty Data

import pandas as pd
import numpy as np

def _calculate_vwap(df: pd.DataFrame) -> float:
    if df is None or df.empty: return 0.0
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    # Check for zero volume to avoid DivisionByZero
    vol_sum = df['volume'].sum()
    if vol_sum == 0: return typical_price.iloc[-1]
    return (typical_price * df['volume']).cumsum().iloc[-1] / vol_sum

def _ema(series: pd.Series, span: int) -> float:
    if series is None or len(series) < span: return 0.0
    return series.ewm(span=span, adjust=False).mean().iloc[-1]

def _rsi(series: pd.Series, period: int = 14) -> float:
    if series is None or len(series) <= period: return 50.0 # Neutral
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    
    # Handle division by zero in loss
    avg_loss = loss.iloc[-1]
    if avg_loss == 0: return 100.0
    
    rs = gain.iloc[-1] / avg_loss
    return 100 - (100 / (1 + rs))

def get_trend_signal(bars_1m: pd.DataFrame, bars_5m: pd.DataFrame, bars_15m: pd.DataFrame) -> str:
    """
    Returns "bull", "bear", or "chop" based on 2/3 agreement across timeframes.
    Hardened against NoneType and empty dataframes.
    """
    # 1. Validate mandatory 1m data
    if bars_1m is None or bars_1m.empty:
        print("!! Indicators: bars_1m is None/Empty. Returning chop.")
        return "chop"

    # Create a list of only valid timeframes to iterate through
    valid_tfs = [tf for tf in [bars_1m, bars_5m, bars_15m] if tf is not None and not tf.empty]
    
    # We need at least the 1m chart to proceed
    price = bars_1m['close'].iloc[-1]

    # VWAP votes
    vwap_votes = sum(price > _calculate_vwap(tf) for tf in valid_tfs)

    # EMA crossover votes
    # Only vote on TFs that have enough data for a 21-period EMA
    ema_votes = sum(_ema(tf['close'], 9) > _ema(tf['close'], 21) 
                    for tf in valid_tfs if len(tf) >= 21)

    # RSI votes (using available TFs from the subset of 1m and 5m)
    rsi_subset = [tf for tf in [bars_1m, bars_5m] if tf is not None and len(tf) > 14]
    rsi_votes = sum(_rsi(tf['close']) > 55 for tf in rsi_subset)

    # Scoring Logic
    # Note: We adjust the threshold based on how many TFs actually provided data
    num_tfs = len(valid_tfs)
    
    # If only 1m is provided, we need a simple majority of indicators
    is_bullish = (vwap_votes >= 1) and (ema_votes >= 1)
    is_bearish = (vwap_votes == 0) and (ema_votes == 0)

    # If multiple TFs are provided, use your 2/3 rule
    if num_tfs >= 2:
        bull_score = sum([vwap_votes >= 2, ema_votes >= 2, rsi_votes >= 1])
        bear_score = sum([vwap_votes <= 1, ema_votes <= 1, rsi_votes <= 0])
        
        if bull_score >= 2: return "bull"
        if bear_score >= 2: return "bear"
        return "chop"
    
    # Single TF fallback
    if is_bullish and rsi_votes >= 1: return "bull"
    if is_bearish and rsi_votes == 0: return "bear"
    
    return "chop"