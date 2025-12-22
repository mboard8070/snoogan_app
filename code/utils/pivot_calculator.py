# code/utils/pivot_calculator.py — Alpaca edition for real intraday sesh pivots
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

def get_pivots():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return "**SNOOGANS' DAILY PIVOTS — 8:30 AM EST**\n\nData unavailable — check your Alpaca keys, ya slacker.\nTrade above R1, fade below S1 — lunch money secured. 37."

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    est = ZoneInfo("America/New_York")
    today = datetime.now(est).date()
    symbols = [
        {"ticker": "SPY", "name": "SPX", "multi": 10},
        {"ticker": "QQQ", "name": "NDX", "multi": 40}
    ]
    message = "**SNOOGANS' DAILY PIVOTS — 8:30 AM EST**\n\n"
    
    for sym in symbols:
        symbol = sym["ticker"]
        name = sym["name"]
        multi = sym["multi"]
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=today - timedelta(days=10),
                limit=3
            )
            bars = client.get_stock_bars(request)
            bars_list = bars[symbol]
            if len(bars_list) < 2:
                raise Exception("Not enough data")
            
            last_bar = bars_list[-1]
            last_ts = last_bar.timestamp.astimezone(est).date()
            if last_ts == today:
                yesterday_bar = bars_list[-2]
            else:
                yesterday_bar = last_bar
            
            h = yesterday_bar.high * multi
            l = yesterday_bar.low * multi
            c = yesterday_bar.close * multi
            
            pp = (h + l + c) / 3
            r1 = 2 * pp - l
            s1 = 2 * pp - h
            r2 = pp + (h - l)
            s2 = pp - (h - l)
            r3 = h + 2 * (pp - l)
            s3 = l - 2 * (h - pp)
            
            message += f"**{name}** H:${h:.0f} L:${l:.0f} C:${c:.0f}\n"
            message += f"R3: ${r3:.0f}\n"
            message += f"R2: ${r2:.0f}\n"
            message += f"R1: ${r1:.0f}\n"
            message += f"PP: ${pp:.0f}\n"
            message += f"S1: ${s1:.0f}\n"
            message += f"S2: ${s2:.0f}\n"
            message += f"S3: ${s3:.0f}\n\n"
        except Exception as e:
            message += f"**{name}**: Data unavailable — market's sleeping in ({e})\n\n"
    
    message += "Trade above R1, fade below S1 — lunch money secured. 37."
    return message

# Test
if __name__ == "__main__":
    print(get_pivots())