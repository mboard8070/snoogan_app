# backend/shared.py
from pathlib import Path
import json
import os
import glob

# Correct project root (backend is at project root level)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Trades are saved by strategies in code/data/knowledge/
KNOWLEDGE_DIR = PROJECT_ROOT / "code" / "data" / "knowledge"
TRADES_FILE = KNOWLEDGE_DIR / "trades.json"
# Archived batches are stored in data/knowledge/
ARCHIVE_DIR = PROJECT_ROOT / "data" / "knowledge"
# Shared equity file (single source of truth)
SHARED_EQUITY_FILE = PROJECT_ROOT / "shared_equity.json"
BATCH_SIZE = 30

def load_trades():
    """Load trades from trades.json and all trades_batch_*.json files."""
    all_trades = []

    # Load from trades.json if exists (current batch)
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE, "r") as f:
                trades = json.load(f)
                if isinstance(trades, list):
                    all_trades.extend(trades)
        except (json.JSONDecodeError, PermissionError):
            pass

    # Load from batch files (archived batches in ARCHIVE_DIR)
    batch_pattern = str(ARCHIVE_DIR / "trades_batch_*.json")
    for batch_file in glob.glob(batch_pattern):
        try:
            with open(batch_file, "r") as f:
                trades = json.load(f)
                if isinstance(trades, list):
                    all_trades.extend(trades)
        except (json.JSONDecodeError, PermissionError):
            pass

    return all_trades

def get_current_batch_count():
    """Get the count of trades in the current batch (trades.json only)."""
    if not TRADES_FILE.exists():
        return 0
    try:
        with open(TRADES_FILE, "r") as f:
            trades = json.load(f)
            if isinstance(trades, list):
                return len(trades)
    except (json.JSONDecodeError, PermissionError):
        pass
    return 0

def get_total_learned():
    """Get the total number of trades that have been learned (archived batches only)."""
    total = 0
    batch_pattern = str(ARCHIVE_DIR / "trades_batch_*.json")
    for batch_file in glob.glob(batch_pattern):
        try:
            with open(batch_file, "r") as f:
                trades = json.load(f)
                if isinstance(trades, list):
                    total += len(trades)
        except (json.JSONDecodeError, PermissionError):
            pass
    return total

def get_trade_count():
    """Get current batch count (for progress bar compatibility)."""
    return get_current_batch_count()

def get_starting_balance():
    return float(os.getenv("STARTING_BALANCE", "100000"))

def get_shared_equity():
    """Get the shared equity value from the JSON file (single source of truth)."""
    if SHARED_EQUITY_FILE.exists():
        try:
            with open(SHARED_EQUITY_FILE, 'r') as f:
                data = json.load(f)
                return data.get('equity', get_starting_balance())
        except (json.JSONDecodeError, KeyError):
            pass
    return get_starting_balance()
