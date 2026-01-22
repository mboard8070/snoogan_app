# code/data/data_client.py
# ── ENVIRONMENT LOADING – MUST BE FIRST ──
from pathlib import Path
from dotenv import load_dotenv
import os

# Load variables.env from project root (same as Streamlit does)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # data → code → project root
load_dotenv(PROJECT_ROOT / "variables.env")

# Now safe to read environment variables
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "alpaca")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
MODE = os.getenv("MODE", "internal")
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "100000"))
CONTRACT_SIZE = int(os.getenv("CONTRACT_SIZE", "1"))

# --- Standard imports ---
import pandas as pd
from datetime import datetime, timedelta
import re
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import time as time_mod
import logging

# Configure logging for cache diagnostics
logger = logging.getLogger(__name__)

# =============================================================================
# MODULE-LEVEL MARKET DATA CACHE
# =============================================================================
# Shared cache across all strategies to eliminate redundant API calls
# Key format: (method_name, ticker, timeframe/expiration, start_time_rounded)
# TTL: 15 seconds for bars, 12 seconds for option chains

_market_data_cache = {}
_BARS_CACHE_TTL = 15  # seconds
_CHAIN_CACHE_TTL = 12  # seconds
_MARK_CACHE_TTL = 5   # seconds (more volatile)


def _cache_key_for_bars(ticker: str, timeframe: str, start: datetime, end: datetime) -> tuple:
    """Generate cache key for bar data, rounded to 10-second intervals"""
    # Round start/end to reduce cache misses from minor time differences
    start_rounded = start.replace(second=(start.second // 10) * 10, microsecond=0)
    end_rounded = end.replace(second=(end.second // 10) * 10, microsecond=0)
    return ('bars', ticker, timeframe, start_rounded.isoformat(), end_rounded.isoformat())


def _cache_key_for_chain(ticker: str, expiration: str) -> tuple:
    """Generate cache key for option chain data"""
    return ('chain', ticker, expiration)


def _cache_key_for_mark(ticker: str) -> tuple:
    """Generate cache key for underlying mark"""
    return ('mark', ticker)


def _get_cached(key: tuple, ttl: int):
    """Get data from cache if not expired"""
    if key in _market_data_cache:
        ts, data = _market_data_cache[key]
        if time_mod.time() - ts < ttl:
            logger.debug(f"[CACHE HIT] {key[0]}:{key[1]}")
            return data, True
    return None, False


def _set_cached(key: tuple, data):
    """Store data in cache with current timestamp"""
    if data is not None:
        _market_data_cache[key] = (time_mod.time(), data)
        logger.debug(f"[CACHE SET] {key[0]}:{key[1]}")


def clear_market_data_cache():
    """Clear the entire cache (useful for testing or forced refresh)"""
    global _market_data_cache
    _market_data_cache = {}
    logger.info("[CACHE] Cleared all market data cache")

# --- Black-Scholes Model Functions ---
def black_scholes_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(0, (S - K) if option_type == 'call' else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_implied_volatility(mid_price, S, K, T, r, option_type):
    if mid_price <= 0 or T <= 1/365:
        return None
    def price_difference(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - mid_price
    try:
        iv = brentq(price_difference, 0.01, 5.0, maxiter=100)
        return iv
    except ValueError:
        return None

def calculate_greeks(S, K, T, r, IV, option_type):
    if T <= 0 or IV is None or IV <= 0:
        return None, None, None, None
    d1 = (np.log(S / K) + (r + 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))
    d2 = d1 - IV * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * IV * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * IV / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * IV / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return delta, gamma, theta, vega

# --- DataClient Class ---
if DATA_PROVIDER == "alpaca":
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, OptionChainRequest, StockSnapshotRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed  # Required for SIP feed

    class DataClient:
        def __init__(self):
            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                raise ValueError("Alpaca keys missing in .env")
            self.stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            self.option_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            self.starting_balance = STARTING_BALANCE
            self.contract_size = CONTRACT_SIZE
            self.mode = MODE

        def get_spy_bars(self, start: datetime, end: datetime, ticker: str = "SPY", timeframe: str = "1Min") -> pd.DataFrame:
            # Check cache first
            cache_key = _cache_key_for_bars(ticker, timeframe, start, end)
            cached_data, hit = _get_cached(cache_key, _BARS_CACHE_TTL)
            if hit:
                return cached_data

            # Map string timeframe to Alpaca TimeFrame
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "30Min": TimeFrame(30, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
            }
            tf = tf_map.get(timeframe, TimeFrame.Minute)

            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf,
                start=start,
                end=end,
                adjustment="all"
            )
            bars = self.stock_client.get_stock_bars(request)
            df = bars.df
            if df.empty:
                logger.debug(f"[DATA] No bars returned for {ticker}")
                return df

            # Handle MultiIndex (symbol + timestamp)
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)  # Remove symbol level

            # Ensure proper DatetimeIndex in EST
            df.index = pd.to_datetime(df.index)
            df = df.tz_convert('America/New_York') if df.index.tz is not None else df.tz_localize('America/New_York')

            # Sort just in case
            df = df.sort_index()

            # Cache the result
            _set_cached(cache_key, df)

            return df

        def get_underlying_mark(self, symbol: str = "SPY") -> float | None:
            """
            Fetch live midpoint mark for underlying ETF using SIP consolidated feed.
            Requires Algo Trader Plus (paid) subscription.
            Returns midpoint (bid+ask)/2 if available, else latest trade price, else None.
            """
            # Check cache first (shorter TTL for real-time marks)
            cache_key = _cache_key_for_mark(symbol)
            cached_data, hit = _get_cached(cache_key, _MARK_CACHE_TTL)
            if hit:
                return cached_data

            request = StockSnapshotRequest(
                symbol_or_symbols=symbol,
                feed=DataFeed.SIP  # Critical: uses full consolidated real-time quotes
            )
            try:
                snapshot = self.stock_client.get_stock_snapshot(request)
                data = snapshot[symbol]

                # Prefer quote midpoint
                if data.latest_quote and data.latest_quote.bid_price > 0 and data.latest_quote.ask_price > 0:
                    mark = (data.latest_quote.bid_price + data.latest_quote.ask_price) / 2
                    logger.debug(f"[DATA] {symbol} SIP mark (quote): {mark:.2f}")
                    _set_cached(cache_key, mark)
                    return mark

                # Fallback to latest trade
                if data.latest_trade and data.latest_trade.price > 0:
                    mark = data.latest_trade.price
                    logger.debug(f"[DATA] {symbol} SIP mark (trade fallback): {mark:.2f}")
                    _set_cached(cache_key, mark)
                    return mark

                logger.debug(f"[DATA] {symbol} No valid quote or trade in SIP snapshot")
            except Exception as e:
                logger.warning(f"[DATA] {symbol} SIP snapshot failed: {e}")

            return None

        def get_spy_option_chain(self, expiration_date: str, ticker: str = "SPY") -> pd.DataFrame:
            # Check cache first
            cache_key = _cache_key_for_chain(ticker, expiration_date)
            cached_data, hit = _get_cached(cache_key, _CHAIN_CACHE_TTL)
            if hit:
                return cached_data

            request = OptionChainRequest(
                underlying_symbol=ticker,
                expiration_date=expiration_date
            )
            chain_dict = self.option_client.get_option_chain(request)
            if not chain_dict:
                return pd.DataFrame()

            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=5)
                # This call will also use the cache
                current_bars = self.get_spy_bars(start_time, end_time, ticker=ticker)
                S = current_bars['close'].iloc[-1] if not current_bars.empty else {"SPY": 680.0, "QQQ": 480.0, "IWM": 220.0}.get(ticker, 100.0)
            except Exception:
                S = {"SPY": 680.0, "QQQ": 480.0, "IWM": 220.0}.get(ticker, 100.0)

            r = 0.04
            now = datetime.now()
            expiration_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
            T = max(((expiration_dt - now).total_seconds() / (365.25 * 24 * 3600)), 1e-6)

            rows = []
            symbol_pattern = re.compile(rf'{ticker}\d{{6}}([CP])(\d{{8}})')
            for symbol, snap in chain_dict.items():
                m = symbol_pattern.search(symbol)
                if not m:
                    continue
                opt_type = "call" if m.group(1) == "C" else "put"
                K = float(m.group(2)) / 1000.0
                bid_price = snap.latest_quote.bid_price if snap.latest_quote else None
                ask_price = snap.latest_quote.ask_price if snap.latest_quote else None
                calc_iv = snap.implied_volatility
                calc_delta = snap.greeks.delta if snap.greeks else None
                calc_gamma = snap.greeks.gamma if snap.greeks else None
                calc_theta = snap.greeks.theta if snap.greeks else None
                calc_vega = snap.greeks.vega if snap.greeks else None

                if (calc_iv is None or calc_delta is None) and bid_price and ask_price and bid_price > 0:
                    mid_price = (bid_price + ask_price) / 2
                    calculated_iv = calculate_implied_volatility(mid_price, S, K, T, r, opt_type)
                    if calculated_iv is not None:
                        calc_iv = calculated_iv
                        calc_delta, calc_gamma, calc_theta, calc_vega = calculate_greeks(S, K, T, r, calculated_iv, opt_type)

                rows.append({
                    "symbol": symbol,
                    "strike_price": K,
                    "option_type": opt_type,
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "delta": calc_delta,
                    "gamma": calc_gamma,
                    "theta": calc_theta,
                    "vega": calc_vega,
                    "implied_volatility": calc_iv,
                })

            result = pd.DataFrame(rows)

            # Cache the result
            _set_cached(cache_key, result)

            return result

        def get_spx_option_chain(self, expiration_date: str) -> pd.DataFrame:
            return self.get_spy_option_chain(expiration_date, ticker="SPY")

elif DATA_PROVIDER == "polygon":
    class DataClient:
        def __init__(self): raise NotImplementedError("Polygon stub")
        def get_spy_bars(self, start, end, ticker="SPY", timeframe="1Min"): raise NotImplementedError
        def get_spy_option_chain(self, expiration_date, ticker="SPY"): raise NotImplementedError
        def get_underlying_mark(self, symbol="SPY"): raise NotImplementedError

# ── LAZY SINGLETON ──
_instance = None
def get_data_client():
    """Return the singleton DataClient instance, creating it only on first call."""
    global _instance
    if _instance is None:
        _instance = DataClient()
    return _instance

# Compatibility for existing code
data_client = get_data_client()