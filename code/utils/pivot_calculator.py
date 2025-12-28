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

# Holiday list from strategy.py – no tradin' on these, ya bum
holidays_2025 = {
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25)
}

def is_trading_day(day):
    if day.weekday() >= 5:  # Weekends out
        return False
    return day not in holidays_2025

def get_previous_trading_day(today):
    prev = today - timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    return prev

def get_pivots():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return "**SNOOGANS' DAILY PIVOTS — 8:30 AM EST**\n\nData unavailable — check your Alpaca keys, ya slacker.\nTrade above R1, fade below S1 — lunch money secured. 37."
    
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    est = ZoneInfo("America/New_York")
    today = datetime.now(est).date()
    prev_day = get_previous_trading_day(today)
    
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
                start=prev_day - timedelta(days=5),  # Buffer for safety
                end=today,  # Up to but not including partial today
                limit=None  # Get all in range – no skimpy limits
            )
            bars = client.get_stock_bars(request)
            bars_list = bars[symbol]
            if not bars_list:
                raise Exception("No bars fetched")
            
            # Find the bar for prev_day (most recent complete)
            yesterday_bar = None
            for bar in reversed(bars_list):  # Newest first
                bar_date = bar.timestamp.astimezone(est).date()
                if bar_date == prev_day:
                    yesterday_bar = bar
                    break
            
            if yesterday_bar is None:
                raise Exception("No bar for previous trading day")
            
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