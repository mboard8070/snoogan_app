# data_client.py – Multi-ticker support (SPY, QQQ, IWM)
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta
import re
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

load_dotenv()

DATA_PROVIDER = os.getenv("DATA_PROVIDER", "alpaca")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
MODE = os.getenv("MODE", "internal")
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "100000"))
CONTRACT_SIZE = int(os.getenv("CONTRACT_SIZE", "1"))

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

if DATA_PROVIDER == "alpaca":
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, OptionChainRequest
    from alpaca.data.timeframe import TimeFrame

    class DataClient:
        def __init__(self):
            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                raise ValueError("Alpaca keys missing in .env")
            self.stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            self.option_client = OptionHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            self.starting_balance = STARTING_BALANCE
            self.contract_size = CONTRACT_SIZE
            self.mode = MODE

        def get_spy_bars(self, start: datetime, end: datetime, ticker: str = "SPY") -> pd.DataFrame:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                adjustment="all"
            )
            bars = self.stock_client.get_stock_bars(request)
            df = bars.df
            if df.empty:
                print(f"[DATA] No bars returned for {ticker}")
                return df
            
            # Handle MultiIndex (symbol + timestamp) – common even for single ticker
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)  # Remove symbol level
            
            # Ensure proper DatetimeIndex in EST
            df.index = pd.to_datetime(df.index)  # Force conversion if needed
            df = df.tz_convert('America/New_York') if df.index.tz is not None else df.index.tz_localize('America/New_York')
            
            # Sort just in case
            df = df.sort_index()
            
            return df

        def get_spy_option_chain(self, expiration_date: str, ticker: str = "SPY") -> pd.DataFrame:
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

            return pd.DataFrame(rows)

        def get_spx_option_chain(self, expiration_date: str) -> pd.DataFrame:
            return self.get_spy_option_chain(expiration_date, ticker="SPY")

elif DATA_PROVIDER == "polygon":
    class DataClient:
        def __init__(self): raise NotImplementedError("Polygon stub")
        def get_spy_bars(self, start, end, ticker="SPY"): raise NotImplementedError
        def get_spy_option_chain(self, expiration_date, ticker="SPY"): raise NotImplementedError

data_client = DataClient()