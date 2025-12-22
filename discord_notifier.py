# code/discord_notifier.py
import os
from dotenv import load_dotenv
from pathlib import Path
import requests
from datetime import datetime, time, date
from zoneinfo import ZoneInfo
from code.utils.pivot_calculator import get_pivots
import streamlit as st

project_root = Path(__file__).parent.parent
load_dotenv(project_root / "variables.env")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
est = ZoneInfo("America/New_York")

GREETING_FLAG = project_root / ".greeting_sent_today"
PIVOTS_FLAG = project_root / ".pivots_sent_today"
EOD_FLAG = project_root / ".eod_summary_sent_today"

def send_webhook(msg: str):
    if not DISCORD_WEBHOOK_URL:
        print(f"[DISCORD LOG ONLY] {msg}")
        return
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        if response.status_code != 204:
            print(f"Discord webhook failed {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Discord send exception: {e}")

def _has_sent(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    try:
        return file_path.read_text().strip() == date.today().isoformat()
    except:
        return False

def _mark_sent(file_path: Path):
    try:
        file_path.write_text(date.today().isoformat())
    except:
        pass

def set_live_mode():
    if 'discord_live_mode' not in st.session_state:
        st.session_state.discord_live_mode = False
    st.session_state.discord_live_mode = True
    print("[DISCORD] App live mode activated.")

def set_strategy_ready():
    st.session_state.discord_strategy_ready = True
    print("[DISCORD] Strategy fully ready — real trade alerts now enabled.")

def is_ready() -> bool:
    live = st.session_state.get('discord_live_mode', False)
    ready = st.session_state.get('discord_strategy_ready', False)
    return live and ready

def send_greeting_if_needed():
    now = datetime.now(est)
    if _has_sent(GREETING_FLAG):
        return
    if now.weekday() >= 5:
        return
    if time(8, 0) <= now.time() <= time(9, 0):
        msg = ("**SNOOGANS ONLINE**\n"
               "Strategy: Vertical credit spreads on SPY/QQQ/IWM (0DTE/1DTE).\n"
               "Risk management: 50% profit target (then 10% trail), 1x stop loss, 2:30 PM EST cutoff.\n"
               "Defined risk only. Session started.")
        send_webhook(msg)
        _mark_sent(GREETING_FLAG)

def force_send_greeting():
    msg = ("**SNOOGANS ONLINE**\n"
           "Strategy: Vertical credit spreads on SPY/QQQ/IWM (0DTE/1DTE).\n"
           "Risk management: 50% profit target (then 10% trail), 1x stop loss, 2:30 PM EST cutoff.\n"
           "Defined risk only. Session started.")
    send_webhook(msg)
    print("[DISCORD] Forced greeting sent.")

def send_pivots_if_needed():
    now = datetime.now(est)
    if _has_sent(PIVOTS_FLAG):
        return
    if time(8, 0) <= now.time() <= time(9, 0) and now.weekday() < 5:
        pivots_msg = get_pivots()
        if pivots_msg:
            send_webhook(pivots_msg)
            _mark_sent(PIVOTS_FLAG)

def force_send_pivots():
    pivots_msg = get_pivots()
    if pivots_msg:
        send_webhook(pivots_msg)
        print("[DISCORD] Forced pivots sent.")

def send_entry(is_put: bool = True, short: float = 0.0, long: float = 0.0, credit: float = 0.0, underlying: str = "SPY"):
    if not is_ready():
        print(f"[BLOCKED PREMATURE ENTRY] Not ready yet: {underlying} {short:.2f}/{long:.2f}")
        return
    if short == 0.0 or long == 0.0 or credit <= 0:
        print(f"[BLOCKED INVALID ENTRY] {underlying} {short:.2f}/{long:.2f} @ {credit}")
        return
    typ = "PUT" if is_put else "CALL"
    direction = "🟢" if is_put else "🔴"
    msg = (f"**ENTRY – {underlying} {typ} CREDIT SPREAD** {direction}\n"
           f"Strikes: {short:.2f}/{long:.2f} | Credit received: ${credit:.2f}\n"
           f"Management: 50% profit target (then 10% trail), 1x stop loss, or 2:30 PM EST cutoff.")
    send_webhook(msg)

def send_exit(is_put: bool = True, short: float = 0.0, long: float = 0.0, credit: float = 0.0, pnl: float = 0.0, underlying: str = "SPY"):
    if not is_ready():
        print(f"[BLOCKED PREMATURE EXIT] Not ready yet: {underlying} {short:.2f}/{long:.2f}")
        return
    if short == 0.0 or long == 0.0:
        print(f"[BLOCKED INVALID EXIT] {underlying} {short:.2f}/{long:.2f}")
        return
    status = "WIN ✅" if pnl > 0 else "LOSS ❌"
    reason = " (trail hit)" if (pnl >= 0.5 * credit and pnl < credit) else ""
    msg = (f"**EXIT {status}{reason} – {underlying}**\n"
           f"Strikes: {short:.2f}/{long:.2f} | P/L: ${pnl:+.2f}")
    send_webhook(msg)

def send_eod_if_needed(trade_count_today: int = 0):
    if not is_ready():
        return
    now = datetime.now(est)
    if _has_sent(EOD_FLAG):
        return
    if time(14, 30) <= now.time() < time(17, 0):
        if trade_count_today == 0:
            msg = "**END OF DAY**\nNo qualifying setups today. Capital preserved."
        else:
            msg = f"**END OF DAY**\n{trade_count_today} trade(s) executed today. Full details on dashboard."
        send_webhook(msg)
        _mark_sent(EOD_FLAG)