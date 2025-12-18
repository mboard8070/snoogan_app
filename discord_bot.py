# discord_bot.py – Webhook alerts with real-time entry/exit triggers

from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configuration
project_root = Path(__file__).parent
env_path = project_root / "variables.env"
load_dotenv(dotenv_path=env_path)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

def send_webhook(message: str):
    if not DISCORD_WEBHOOK_URL:
        print(f"[LOG ONLY] {message}")
        return
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
        if response.status_code != 204:
            print(f"Webhook error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Webhook exception: {e}")

# Trading session tracking
trades_today = []
est = ZoneInfo("America/New_York")
SUMMARY_TIME = time(16, 5)

# Flags to prevent duplicate messages
GREETING_FLAG_FILE = project_root / ".greeting_sent_today"
EOD_FLAG_FILE = project_root / ".eod_summary_sent_today"

def has_sent_flag(file_path):
    if not file_path.exists(): return False
    return file_path.read_text().strip() == date.today().isoformat()

def mark_flag_sent(file_path):
    file_path.write_text(date.today().isoformat())

def send_greeting():
    if has_sent_flag(GREETING_FLAG_FILE): return
    msg = "🚀 **SNOOGANS ONLINE**\n30-Delta Vertical Credit Spread Strategy engaged. Monitoring SPY for trend alignment."
    send_webhook(msg)
    mark_flag_sent(GREETING_FLAG_FILE)

def send_entry(is_put=True, short=0, long=0, credit=0.0, underlying="SPY"):
    typ = "PUT 🟢" if is_put else "CALL 🔴"
    msg = (f"📥 **ENTRY EXECUTED**\n"
           f"**Type:** {underlying} {typ} Spread\n"
           f"**Strikes:** {short}/{long}\n"
           f"**Net Credit:** ${credit:.2f}\n"
           f"**Status:** Monitoring for exit criteria.")
    send_webhook(msg)
    trades_today.append({"type": "entry"})

def send_exit(is_put=True, short=0, long=0, credit=0.0, pnl=0.0, underlying="SPY"):
    typ = "PUT" if is_put else "CALL"
    status = "✅ WIN" if pnl > 0 else "❌ LOSS"
    msg = (f"📤 **EXIT EXECUTED - {status}**\n"
           f"**Type:** {underlying} {typ} Spread\n"
           f"**Strikes:** {short}/{long}\n"
           f"**Realized P/L:** ${pnl:+.2f}")
    send_webhook(msg)
    trades_today.append({"type": "exit", "pnl": pnl})

def send_eod_summary():
    now = datetime.now(est)
    if SUMMARY_TIME <= now.time() < time(17, 0):
        if has_sent_flag(EOD_FLAG_FILE): return
        
        entries = len([t for t in trades_today if t.get("type") == "entry"])
        if entries == 0:
            msg = "**END OF DAY**\nNo trades triggered today. Market conditions did not meet 30-delta strategy requirements."
        else:
            msg = f"**END OF DAY**\nTrading complete. Total trades initiated: {entries}. See Dashboard for full PnL."
        
        send_webhook(msg)
        mark_flag_sent(EOD_FLAG_FILE)

# Run greeting on import
send_greeting()