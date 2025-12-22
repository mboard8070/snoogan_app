# code/discord_notifier.py – Clean Discord blasts for greets, pivots, entries/exits, EOD
from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from datetime import datetime, time, date
from zoneinfo import ZoneInfo
from code.utils.pivot_calculator import get_pivots  # Your Alpaca pivot magic

project_root = Path(__file__).parent.parent
load_dotenv(project_root / "variables.env")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
est = ZoneInfo("America/New_York")

# Daily flag files – one-and-done per day
GREETING_FLAG = project_root / ".greeting_sent_today"
PIVOTS_FLAG = project_root / ".pivots_sent_today"
EOD_FLAG = project_root / ".eod_summary_sent_today"

def _has_sent(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    return file_path.read_text().strip() == date.today().isoformat()

def _mark_sent(file_path: Path):
    file_path.write_text(date.today().isoformat())

def send_webhook(msg: str):
    if not DISCORD_WEBHOOK_URL:
        print(f"[DISCORD LOG ONLY] {msg}")
        return
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        if response.status_code != 204:
            print(f"Discord webhook error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Discord send exception: {e}")

def send_greeting_if_needed():
    now = datetime.now(est)
    if _has_sent(GREETING_FLAG):
        return
    if time(8, 25) <= now.time() < time(9, 30) and now.weekday() < 5:  # Rough trading day check
        msg = "🚀 **SNOOGANS ONLINE**\n30-Delta Vertical Credit Spreads locked and loaded on SPX/NDX. Tiny positions, defined risk only – theta decay about to pay for lunch. Snoochie boochies."
        send_webhook(msg)
        _mark_sent(GREETING_FLAG)

def send_pivots_if_needed():
    now = datetime.now(est)
    if _has_sent(PIVOTS_FLAG):
        return
    if time(8, 25) <= now.time() < time(9, 30) and now.weekday() < 5:
        pivots_msg = get_pivots()
        send_webhook(pivots_msg)
        _mark_sent(PIVOTS_FLAG)

def send_entry(is_put: bool = True, short: int = 0, long: int = 0, credit: float = 0.0, underlying: str = "SPX"):
    typ = "PUT 🟢" if is_put else "CALL 🔴"
    msg = (f"📥 **ENTRY – {underlying} {typ} 30-DELTA SPREAD**\n"
           f"Strikes: {short}/{long} | Credit: ${credit:.2f}\n"
           f"Monitoring 50% profit (then 10% trail), 2.1x stop, or 4pm cutoff. Stay zen.")
    send_webhook(msg)

def send_exit(is_put: bool = True, short: int = 0, long: int = 0, credit: float = 0.0, pnl: float = 0.0, underlying: str = "SPX"):
    status = "✅ WIN" if pnl > 0 else "❌ LOSS"
    trail_note = " (trail hit)" if pnl > 0.5 * credit else ""
    msg = (f"📤 **EXIT {status}{trail_note} – {underlying}**\n"
           f"Strikes: {short}/{long} | P/L: ${pnl:+.2f}\n"
           f"Another tiny 30-delta bag secured. Theta doin' work.")
    send_webhook(msg)

def send_eod_if_needed(trade_count_today: int = 0):
    now = datetime.now(est)
    if _has_sent(EOD_FLAG):
        return
    if time(16, 5) <= now.time() < time(17, 0):
        if trade_count_today == 0:
            msg = "**END OF DAY**\nNo 30-delta setups today – market wasn't spicy enough. Staying chill, no trauma."
        else:
            msg = f"**END OF DAY**\n{trade_count_today} tiny spread(s) executed. Dashboard got the full PnL deets."
        send_webhook(msg)
        _mark_sent(EOD_FLAG)