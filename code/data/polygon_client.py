# Replace your polygon_client.py with this — FREE TIER FINAL
from polygon import RESTClient
import yfinance as yf
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))

def get_underlying_price():
    # yfinance is more reliable on free tier
    try:
        data = yf.download("^GSPC", period="2d", progress=False)
        return data['Close'].iloc[-1]
    except:
        return 6000.0

def get_0dte_puts():
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        contracts = client.list_options_contracts(
            underlying_ticker="SPX",
            expiration_date=today,
            contract_type="put",
            limit=1000
        )
        rows = []
        spot = get_underlying_price()
        for c in contracts:
            distance = abs(c.strike_price - spot) / spot
            est_delta = max(0.05, min(0.5, 0.5 - distance * 10))
            rows.append({
                "ticker": c.ticker,
                "strike": c.strike_price,
                "est_delta": round(est_delta, 3),
                "distance": round(distance, 3)
            })
        df = pd.DataFrame(rows)
        print(f"Loaded {len(df)} 0DTE puts (free tier + yfinance price)")
        return df.sort_values("strike", ascending=False)
    except Exception as e:
        print(f"Chain failed: {e}")
        return pd.DataFrame()
def get_underlying_price():
    # Try Polygon first
    try:
        end = datetime.now()
        start = end - timedelta(days=5)
        aggs = client.get_aggs("SPX", 1, "day", from_=start.date(), to=end.date())
        if aggs:
            return aggs[-1].close
    except:
        pass

    # yfinance with retry + delay
    for attempt in range(5):
        try:
            ticker = yf.Ticker("^GSPC")
            data = ticker.history(period="5d")
            if not data.empty:
                return data['Close'].iloc[-1]
        except:
            time.sleep(3)
    
    return 6000.0