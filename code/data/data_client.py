# data_client.py – UPDATED VERSION (Includes get_spy_option_chain)

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

        def get_spy_bars(self, start: datetime, end: datetime) -> pd.DataFrame:
            request = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                adjustment="all"
            )
            bars = self.stock_client.get_stock_bars(request)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)
            if df.index.tz is not None:
                df = df.tz_localize(None)
            return df

        # FIX: Renamed to match the strategy's request for SPY
        def get_spy_option_chain(self, expiration_date: str) -> pd.DataFrame:
            """Fetches the SPY option chain and calculates Greeks."""
            request = OptionChainRequest(
                underlying_symbol="SPY",
                expiration_date=expiration_date
            )
            chain_dict = self.option_client.get_option_chain(request)
            
            # Guard against empty/None API response
            if not chain_dict:
                return pd.DataFrame()

            # Get current price for BSM calculations
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=5)
                current_bars = self.get_spy_bars(start_time, end_time)
                S = current_bars['close'].iloc[-1]
            except Exception:
                S = 680.0  # Fallback proxy

            r = 0.04
            now = datetime.now()
            expiration_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
            T = max(((expiration_dt - now).total_seconds() / (365.25 * 24 * 3600)), 1e-6)

            rows = []
            symbol_pattern = re.compile(r'SPY\d{6}([CP])(\d{8})')

            for symbol, snap in chain_dict.items():
                m = symbol_pattern.search(symbol)
                if not m: continue
                
                opt_type = "call" if m.group(1) == "C" else "put"
                K = float(m.group(2)) / 1000.0

                bid_price = snap.latest_quote.bid_price if snap.latest_quote else None
                ask_price = snap.latest_quote.ask_price if snap.latest_quote else None

                calc_iv = snap.implied_volatility
                calc_delta = snap.greeks.delta if snap.greeks else None
                calc_gamma = snap.greeks.gamma if snap.greeks else None
                calc_theta = snap.greeks.theta if snap.greeks else None
                calc_vega = snap.greeks.vega if snap.greeks else None

                # Fallback to BSM if Greeks are missing from API
                if (calc_iv is None or calc_delta is None) and bid_price and ask_price and bid_price > 0:
                    mid_price = (bid_price + ask_price) / 2
                    calculated_iv = calculate_implied_volatility(mid_price, S, K, T, r, opt_type)
                    if calculated_iv is not None:
                        calc_iv = calculated_iv
                        calc_delta, calc_gamma, calc_theta, calc_vega = calculate_greeks(S, K, T, r, calculated_iv, opt_type)

                rows.append({
                    "symbol": symbol, "strike_price": K, "option_type": opt_type,
                    "bid_price": bid_price, "ask_price": ask_price,
                    "delta": calc_delta, "gamma": calc_gamma, "theta": calc_theta, "vega": calc_vega,
                    "implied_volatility": calc_iv,
                })

            return pd.DataFrame(rows)

        # Keep legacy method name for compatibility if needed elsewhere
        def get_spx_option_chain(self, expiration_date: str) -> pd.DataFrame:
            return self.get_spy_option_chain(expiration_date)

elif DATA_PROVIDER == "polygon":
    class DataClient:
        def __init__(self): raise NotImplementedError("Polygon stub")
        def get_spy_bars(self, start, end): raise NotImplementedError
        def get_spy_option_chain(self, expiration_date): raise NotImplementedError

data_client = DataClient()