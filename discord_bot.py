# discord_bot.py – Webhook-only alerts (SPY pivots scaled to SPX + EOD summary + once-per-day greeting & EOD)

from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Force load .env from project root
project_root = Path(__file__).parent
env_path = project_root / "variables.env"
load_dotenv(dotenv_path=env_path)

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

# Alpaca client
client = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    print("Alpaca client loaded for real pivots")
else:
    print("Alpaca keys missing — using fallback pivots")

# SPY to SPX ratio
SPX_RATIO = 10.0

# Trading session tracking
trades_today = []

est = ZoneInfo("America/New_York")
TRADING_CUTOFF = time(16, 0)
SUMMARY_TIME = time(16, 5)

# --- Once-per-day flags ---
GREETING_FLAG_FILE = project_root / ".greeting_sent_today"
EOD_FLAG_FILE = project_root / ".eod_summary_sent_today"

def has_sent_greeting_today():
    if not GREETING_FLAG_FILE.exists():
        return False
    try:
        flag_date = date.fromisoformat(GREETING_FLAG_FILE.read_text().strip())
        return flag_date == date.today()
    except:
        return False

def mark_greeting_sent():
    GREETING_FLAG_FILE.write_text(date.today().isoformat())

def has_sent_eod_today():
    if not EOD_FLAG_FILE.exists():
        return False
    try:
        flag_date = date.fromisoformat(EOD_FLAG_FILE.read_text().strip())
        return flag_date == date.today()
    except:
        return False

def mark_eod_sent():
    EOD_FLAG_FILE.write_text(date.today().isoformat())

def calculate_pivots(high, low, close, ratio=SPX_RATIO):
    pp = (high + low + close) / 3
    pivots_spy = {
        "PP": pp,
        "R1": 2 * pp - low,
        "S1": 2 * pp - high,
        "R2": pp + (high - low),
        "S2": pp - (high - low),
        "R3": pp + 2 * (high - low),
        "S3": pp - 2 * (high - low)
    }
    pivots_spx = {k: round(v * ratio, 1) for k, v in pivots_spy.items()}
    return pivots_spx

def get_yesterdays_spy():
    if not client:
        return {"high": 682.0, "low": 678.0, "close": 680.0}

    end = datetime.now(est).date() - timedelta(days=1)
    start = end - timedelta(days=30)

    try:
        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="all"
        )).df

        if not bars.empty and "high" in bars.columns:
            latest = bars.iloc[-1]
            return {
                "high": latest["high"],
                "low": latest["low"],
                "close": latest["close"]
            }
    except Exception as e:
        print(f"SPY data fail: {e}")

    return {"high": 682.0, "low": 678.0, "close": 680.0}

def send_greeting():
    if has_sent_greeting_today():
        print("Morning greeting already sent today — skipping.")
        return
    
    spy_data = get_yesterdays_spy()
    pivots = calculate_pivots(**spy_data)
    
    pivot_str = (
        f"PP {pivots['PP']} | "
        f"R1 {pivots['R1']} S1 {pivots['S1']} | "
        f"R2 {pivots['R2']} S2 {pivots['S2']} | "
        f"R3 {pivots['R3']} S3 {pivots['S3']}"
    )
    
    msg = (
        f"**MORNING GREETING**\n"
        f"Morning SPX pivots (from SPY × {SPX_RATIO}): {pivot_str}\n"
        f"Tiny 16-delta vertical credit spreads loading… Defined risk only."
    )
    send_webhook(msg)
    mark_greeting_sent()
    print("Morning greeting sent for today.")

def send_entry(is_put=True, short=6820, long=6810, credit=2.50, underlying="SPX"):
    typ = "put" if is_put else "call"
    msg = f"**ENTRY**\nSold tiny {typ} credit spread {short}/{long} for {credit:.2f} credit on {underlying}"
    send_webhook(msg)
    trades_today.append({"type": "entry"})

def send_exit(is_put=True, short=6820, long=6810, credit=2.50, pnl=1.25, underlying="SPX"):
    typ = "put" if is_put else "call"
    result = "WIN" if pnl > 0 else "LOSS"
    msg = f"**EXIT – {result}**\nClosed {typ} spread {short}/{long}, original credit {credit:.2f}, P/L {pnl:+.2f}"
    send_webhook(msg)
    trades_today.append({"type": "exit", "pnl": pnl})

def send_eod_summary():
    now = datetime.now(est)
    
    if SUMMARY_TIME <= now.time() < time(17, 0):
        if has_sent_eod_today():
            print("EOD summary already sent today — skipping.")
            return
        
        entries = len([t for t in trades_today if t.get("type") == "entry"])
        
        if entries == 0:
            msg = (
                "**END OF DAY – NO TRADES**\n"
                "Trading window closed at 4:00 PM EST.\n"
                "Market remained range-bound with insufficient IV edge or clean setup for 30-delta vertical credit spreads.\n"
                "PPO agent stayed patient—no high-conviction signals triggered.\n"
                "Capital preserved, zero drawdown today.\n"
                "Defined risk only. Ready for tomorrow."
            )
        else:
            msg = (
                f"**END OF DAY – {entries} TRADE(S) INITIATED**\n"
                "All positions managed per rules.\n"
                "Full details in Streamlit dashboard.\n"
                "Defined risk only."
            )
        
        send_webhook(msg)
        trades_today.clear()
        mark_eod_sent()
        print("EOD summary sent for today.")

# Send greeting on launch (only once per day)
send_greeting()

if __name__ == "__main__":
    print("Webhook alerts ready.")