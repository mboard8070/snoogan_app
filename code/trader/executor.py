# trader/executor.py - paper execution (real Alpaca live switch later)

import os
from datetime import datetime

class Executor:
    def __init__(self):
        self.mode = os.getenv("MODE", "internal")  # internal or paper or live

    def place_credit_spread(self, direction: str, short_strike: float, long_strike: float, credit: float) -> dict:
        position = {
            "direction": direction,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "credit_received": credit,
            "entry_time": datetime.now(),
            "contracts": int(os.getenv("CONTRACT_SIZE", "1"))
        }
        print(f"[PAPER] {direction.upper()} credit spread entered: {short_strike}/{long_strike} @ {credit:.2f}")
        return position

    def close_position(self, position: dict, debit: float) -> float:
        pnl = (position["credit_received"] - debit) * 100 * position["contracts"]
        reason = "profit" if pnl > 0 else "stop" if pnl < 0 else "cutoff"
        print(f"[PAPER] Closed {reason}: PnL ${pnl:.2f}")
        return pnl