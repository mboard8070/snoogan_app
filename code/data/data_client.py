# data_client.py – FINAL VERSION (Alpaca real-time + BSM fallback for Greeks - FINAL LOGIC)

from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime, timedelta, date 
import re
import numpy as np
from scipy.stats import norm 
from scipy.optimize import brentq 

load_dotenv()

# --- Black-Scholes Model Functions ---
# Note: These functions assume continuous dividend yield (q) is negligible or incorporated 
# into the risk-free rate (r), which is standard for simple index option pricing.

def black_scholes_price(S, K, T, r, sigma, option_type):
    """Calculates the theoretical option price (used as target for IV search)."""
    if T <= 0 or sigma <= 0:
        return max(0, (S - K) if option_type == 'call' else (K - S))

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: # 'put'
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def calculate_implied_volatility(mid_price, S, K, T, r, option_type):
    """Calculates Implied Volatility using the brentq root finder."""
    if mid_price <= 0 or T <= 1/365:
        return None 
    
    def price_difference(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - mid_price

    try:
        # Search for IV between 1e-6 (near zero) and 5.0 (500% vol)
        iv = brentq(price_difference, 1e-6, 5.0, maxiter=100)
        return iv
    except ValueError:
        # Fails if the price is outside the theoretical min/max bounds (e.g., deep ITM/OTM)
        return None

def calculate_greeks(S, K, T, r, IV, option_type):
    """Calculates option Greeks (Delta, Gamma, Theta, Vega) using BSM."""
    if T <= 0 or IV is None or IV <= 0:
        return None, None, None, None

    d1 = (np.log(S / K) + (r + 0.5 * IV ** 2) * T) / (IV * np.sqrt(T))
    d2 = d1 - IV * np.sqrt(T)
    
    # Delta
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        
    # Gamma (same for Call/Put)
    gamma = norm.pdf(d1) / (S * IV * np.sqrt(T))
    
    # Vega (standard format is divided by 100)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 
    
    # Theta (daily decay)
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * IV / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * IV / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
    return delta, gamma, theta, vega

# ───── ENV VARS ─────
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "alpaca")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

MODE = os.getenv("MODE", "internal")
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "100000"))
CONTRACT_SIZE = int(os.getenv("CONTRACT_SIZE", "1"))

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

            # expose to the rest of the bot
            self.starting_balance = STARTING_BALANCE
            self.contract_size = CONTRACT_SIZE
            self.mode = MODE

        # ───── SPY BARS ─────
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
                df = df.droplevel(0)          # drop symbol level
            if df.index.tz is not None:
                df = df.tz_localize(None)
            return df

        # ───── FULL SPX OPTION CHAIN (calls + puts) ─────
        def get_spx_option_chain(self, expiration_date: str) -> pd.DataFrame:
            """
            Returns a clean DataFrame with calculated greeks if Alpaca's feed is missing them.
            """
            request = OptionChainRequest(
                underlying_symbol="SPX",
                expiration_date=expiration_date
            )
            chain_dict = self.option_client.get_option_chain(request)

            # --- BSM INPUT SETUP ---
            # 1. Underlying Price S: Approximated as SPY * 10
            try:
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=5)
                current_bars = self.get_spy_bars(start_time, end_time)
                SPY_S = current_bars['close'].iloc[-1]
                S = SPY_S * 10.0 
            except Exception as e:
                print(f"Warning: Could not get live SPY price for BSM: {e}. Defaulting to S=6810.0")
                S = 6810.0 
            
            # 2. Risk-free rate (r)
            r = 0.04 
            now = datetime.now()
            expiration_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
            
            print(f"--- BSM Calculation Basis: S (Approx. SPX Price) = {S:.2f} ---")
            # --- END BSM INPUT SETUP ---


            rows = []
            # Symbol pattern is correct.
            symbol_pattern = re.compile(r'SPX\d{6}([CP])(\d{8})')

            for symbol, snap in chain_dict.items():
                # ---- parse strike & type from symbol ----
                m = symbol_pattern.search(symbol)
                if not m:
                    continue 

                opt_type = "call" if m.group(1) == "C" else "put"
                
                # FIXED: The 8-digit padded strike is divided by 1000.0 to get the correct SPX strike (e.g., 6815.0)
                K = float(m.group(2)) / 1000.0 
                
                # ---- Extract Prices and Time (T) ----
                bid_price = snap.latest_quote.bid_price if snap.latest_quote else None
                ask_price = snap.latest_quote.ask_price if snap.latest_quote else None
                
                # Time to Expiration (T in years). Must be > 0.
                T_days = (expiration_dt - now).days
                T = max(T_days / 365.0, 1e-6)

                # ---- Greek/IV Logic: Use Alpaca first, then calculate ----
                calc_iv = snap.implied_volatility
                calc_delta = snap.greeks.delta if snap.greeks else None
                calc_gamma = snap.greeks.gamma if snap.greeks else None
                calc_theta = snap.greeks.theta if snap.greeks else None
                calc_vega = snap.greeks.vega if snap.greeks else None

                # Fallback calculation if Alpaca data is missing
                if (calc_iv is None or calc_delta is None) and bid_price is not None and ask_price is not None:
                    
                    mid_price = (bid_price + ask_price) / 2
                    
                    # REFINED LOGIC: Only proceed if mid_price is above zero. 
                    if mid_price > 0: 
                        
                        # 1. Calculate IV
                        calculated_iv = calculate_implied_volatility(mid_price, S, K, T, r, opt_type)
                        
                        if calculated_iv is not None:
                            calc_iv = calculated_iv 
                            
                            # 2. Calculate Greeks using the derived IV
                            calculated_delta, calculated_gamma, calculated_theta, calculated_vega = calculate_greeks(S, K, T, r, calculated_iv, opt_type)
                            
                            calc_delta = calculated_delta
                            calc_gamma = calculated_gamma
                            calc_theta = calculated_theta
                            calc_vega = calculated_vega


                row = {
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
                }
                rows.append(row)

            return pd.DataFrame(rows)

elif DATA_PROVIDER == "polygon":
    # Placeholder for alternative data provider
    class DataClient:
        def __init__(self):
            raise NotImplementedError("Polygon provider not wired yet")

        def get_spy_bars(self, start, end):
            raise NotImplementedError

        def get_spx_option_chain(self, expiration_date):
            raise NotImplementedError

# ───── SINGLETON ─────
data_client = DataClient()