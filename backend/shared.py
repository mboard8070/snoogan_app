# backend/shared.py
from pathlib import Path
import json
import os
import glob

# Correct project root (backend is at project root level)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
TRADES_FILE = KNOWLEDGE_DIR / "trades.json"

def load_trades():
    """Load trades from trades.json and all trades_batch_*.json files."""
    all_trades = []

    # Load from trades.json if exists
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE, "r") as f:
                trades = json.load(f)
                if isinstance(trades, list):
                    all_trades.extend(trades)
        except (json.JSONDecodeError, PermissionError):
            pass

    # Load from batch files
    batch_pattern = str(KNOWLEDGE_DIR / "trades_batch_*.json")
    for batch_file in glob.glob(batch_pattern):
        try:
            with open(batch_file, "r") as f:
                trades = json.load(f)
                if isinstance(trades, list):
                    all_trades.extend(trades)
        except (json.JSONDecodeError, PermissionError):
            pass

    return all_trades

def get_trade_count():
    return len(load_trades())

def get_starting_balance():
    return float(os.getenv("STARTING_BALANCE", "100000"))
