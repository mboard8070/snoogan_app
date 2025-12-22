# discord_bot.py – Now with built-in scheduling, no external cron needed
from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from datetime import datetime, time, date
from zoneinfo import ZoneInfo
import time as sleep_time  # To avoid conflict with datetime.time
from code.utils.pivot_calculator import get_pivots

# Configuration
project_root = Path(__file__).parent
env_path = project_root / "variables.env"
load_dotenv(dotenv_path=env_path)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

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

# Flags to prevent duplicates
GREETING_FLAG_FILE = project_root / ".greeting_sent_today"
PIVOTS_FLAG_FILE = project_root / ".pivots_sent_today"
EOD_FLAG_FILE = project_root / ".eod_summary_sent_today"

def has_sent_flag(file_path):
    if not file_path.exists(): return False
    today_str = date.today().isoformat()
    return file_path.read_text().strip() == today_str

def mark_flag_sent(file_path):
    file_path.write_text(date.today().isoformat())

def reset_flags_if_new_day(file_paths):
    today_str = date.today().isoformat()
    for fp in file_paths:
        if fp.exists() and fp.read_text().strip() != today_str:
            fp.unlink()  # Delete old flag on new day

# Hardcoded 2025 US stock market full closures (NYSE/Nasdaq)
HOLIDAYS_2025 = {
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
}

def is_trading_day(today_date: date) -> bool:
    if today_date.weekday() >= 5:  # Saturday or Sunday
        return False
    if today_date in HOLIDAYS_2025:
        return False
    return True

def send_greeting():
    if has_sent_flag(GREETING_FLAG_FILE): return
    msg = "🚀 **SNOOGANS ONLINE**\n30-Delta Vertical Credit Spread Strategy engaged. Monitoring SPX / NDX for theta decay. Tiny positions only – lunch money secured."
    send_webhook(msg)
    mark_flag_sent(GREETING_FLAG_FILE)

def send_pivots():
    if has_sent_flag(PIVOTS_FLAG_FILE): return
    pivots_msg = get_pivots()
    send_webhook(pivots_msg)
    mark_flag_sent(PIVOTS_FLAG_FILE)

def send_entry(is_put=True, short=0, long=0, credit=0.0, underlying="SPX"):
    typ = "PUT 🟢" if is_put else "CALL 🔴"
    msg = (f"📥 **ENTRY EXECUTED – {underlying}**\n"
           f"**Type:** {underlying} {typ} Spread\n"
           f"**Strikes:** {short}/{long}\n"
           f"**Net Credit:** ${credit:.2f}\n"
           f"**Status:** Monitoring for 50% target or 2x stop.")
    send_webhook(msg)
    trades_today.append({"type": "entry"})

def send_exit(is_put=True, short=0, long=0, credit=0.0, pnl=0.0, underlying="SPX"):
    typ = "PUT" if is_put else "CALL"
    status = "✅ WIN" if pnl > 0 else "❌ LOSS"
    msg = (f"📤 **EXIT EXECUTED - {status} – {underlying}**\n"
           f"**Type:** {underlying} {typ} Spread\n"
           f"**Strikes:** {short}/{long}\n"
           f"**Realized P/L:** ${pnl:+.2f}")
    send_webhook(msg)
    trades_today.append({"type": "exit", "pnl": pnl})

def send_eod_summary():
    if has_sent_flag(EOD_FLAG_FILE): return
    entries = len([t for t in trades_today if t.get("type") == "entry"])
    if entries == 0:
        msg = "**END OF DAY**\nNo spreads triggered today. Conditions weren't spicy enough – staying zen."
    else:
        msg = f"**END OF DAY**\nTrading wrapped. Initiated {entries} tiny spread(s). Dashboard got the full deets."
    send_webhook(msg)
    mark_flag_sent(EOD_FLAG_FILE)

# Main scheduling loop
def main_loop():
    print("Snoogans bot loop started – chillin' in Discord forever. Snoochie boochies.")
    while True:
        now = datetime.now(est)
        today_date = now.date()
        current_time = now.time()

        # Reset flags if it's a new day
        reset_flags_if_new_day([GREETING_FLAG_FILE, PIVOTS_FLAG_FILE, EOD_FLAG_FILE])
        trades_today.clear()  # Reset daily trades list

        if is_trading_day(today_date):
            # Morning messages around 8:30 AM EST
            if time(8, 25) <= current_time < time(9, 0):
                send_greeting()
                send_pivots()

            # EOD summary after 4:05 PM
            if time(16, 5) <= current_time < time(17, 0):
                send_eod_summary()

        # Sleep 60 seconds – gentle on the CPU, checks every minute
        sleep_time.sleep(60)

if __name__ == "__main__":
    main_loop()