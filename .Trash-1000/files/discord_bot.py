# code/utils/discord_bot.py – Webhook-only alerts (auto greeting + pivots + trade alerts)

from dotenv import load_dotenv
import os
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()  # force load .env in this process

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not DISCORD_WEBHOOK_URL:
    print("CRITICAL: DISCORD_WEBHOOK_URL missing — alerts disabled")
    def send_webhook(message: str):
        print(f"[SIM MODE] {message}")
else:
    print("Discord webhook loaded — alerts live")
    def send_webhook(message: str):
        try:
            response = requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
            if response.status_code == 204:
                print(f"Posted: {message[:60]}...")
            else:
                print(f"Webhook error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Webhook exception: {e}")

# Alpaca client for pivots
client = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    print("Alpaca client loaded for pivots")
else:
    print("Alpaca keys missing — using fallback pivots")

def calculate_pivots(high, low, close):
    pp = (high + low + close) / 3
    return {k: round(v, 2) for k, v in {
        "PP": pp,
        "R1": 2*pp - low,
        "S1": 2*pp - high,
        "R2": pp + (high - low),
        "S2": pp - (high - low),
        "R3": pp + 2*(high - low),
        "S3": pp - 2*(high - low)
    }.items()}

def get_yesterdays_spx():
    if not client:
        return {"high": 6820.0, "low": 6780.0, "close": 6800.0}
    est = ZoneInfo("America/New_York")
    end = datetime.now(est).date() - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=20)
    for sym in ["SPX", "^GSPC"]:
        try:
            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment="all"
            )).df
            if not bars.empty:
                latest = bars.iloc[-1]
                return {"high": latest["high"], "low": latest["low"], "close": latest["close"]}
        except Exception as e:
            print(f"SPX data fail: {e}")
    return {"high": 6820.0, "low": 6780.0, "close": 6800.0}

def send_greeting():
    pivots = calculate_pivots(**get_yesterdays_spx())
    pivot_str = f"PP {pivots['PP']} | R1 {pivots['R1']} S1 {pivots['S1']} | R2 {pivots['R2']} S2 {pivots['S2']} | R3 {pivots['R3']} S3 {pivots['S3']}"
    msg = f"**MORNING GREETING**\nMorning pivots: {pivot_str}\nTiny 30-delta vertical credit spreads loading… Defined risk only.\nLunch money secured today, boss."
    send_webhook(msg)

def send_entry(is_put=True, short=6820, long=6810, credit=2.50, underlying="SPX"):
    typ = "put" if is_put else "call"
    msg = f"**ENTRY**\nSold tiny {typ} credit spread {short}/{long} for {credit:.2f} credit on {underlying}\nLunch money loading."
    send_webhook(msg)

def send_exit(is_put=True, short=6820, long=6810, credit=2.50, pnl=1.25, underlying="SPX"):
    typ = "put" if is_put else "call"
    result = "WIN" if pnl > 0 else "LOSS"
    msg = f"**EXIT – {result}**\nClosed {typ} spread {short}/{long}, original credit {credit:.2f}, P/L {pnl:+.2f}\nLunch money banked."
    send_webhook(msg)

# Auto greeting on import
send_greeting()

if __name__ == "__main__":
    print("Snoogans webhook alerts ready.")