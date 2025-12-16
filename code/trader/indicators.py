# trader/indicators.py - trend detection (VWAP, EMA crossover, RSI + momentum)

import pandas as pd
import numpy as np

def _calculate_vwap(df: pd.DataFrame) -> float:
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum().iloc[-1] / df['volume'].cumsum().iloc[-1]

def _ema(series: pd.Series, span: int) -> float:
    return series.ewm(span=span, adjust=False).mean().iloc[-1]

def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def get_trend_signal(bars_1m: pd.DataFrame, bars_5m: pd.DataFrame, bars_15m: pd.DataFrame) -> str:
    """
    Returns "bull", "bear", or "chop" based on 2/3 agreement across timeframes.
    """
    price = bars_1m['close'].iloc[-1]

    # VWAP votes
    vwap_votes = sum(price > _calculate_vwap(tf) for tf in [bars_1m, bars_5m, bars_15m])

    # EMA crossover votes
    ema_votes = sum(_ema(tf['close'], 9) > _ema(tf['close'], 21) for tf in [bars_1m, bars_5m, bars_15m])

    # RSI votes (using 1m and 5m only for responsiveness)
    rsi_votes = sum(_rsi(tf['close']) > 55 for tf in [bars_1m, bars_5m])

    bull_score = sum([vwap_votes >= 2, ema_votes >= 2, rsi_votes >= 1])
    bear_score = sum([vwap_votes <= 1, ema_votes <= 1, rsi_votes <= 0])

    if bull_score >= 2:
        return "bull"
    elif bear_score >= 2:
        return "bear"
    return "chop"