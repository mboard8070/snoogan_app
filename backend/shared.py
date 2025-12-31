# backend/shared.py
from pathlib import Path
import json
import os

# Correct project root (backend is at project root level)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
TRADES_FILE = KNOWLEDGE_DIR / "trades.json"

def load_trades():
    if not TRADES_FILE.exists():
        return []
    try:
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, PermissionError):
        return []

def get_trade_count():
    return len(load_trades())

def get_starting_balance():
    return float(os.getenv("STARTING_BALANCE", "100000"))