# trader/position_manager.py - position monitoring

from datetime import time

CUTOFF_TIME = time(14, 0)  # 2 PM EST

def should_close(position: dict, current_debit: float, daily_pnl: float, starting_balance: float) -> tuple[bool, str]:
    pnl_pct = (position["credit_received"] - current_debit) / position["credit_received"]

    if current_debit <= 0.05 or pnl_pct >= 0.50:
        return True, "profit_target"
    if pnl_pct <= -1.5:  # 1.5x stop
        return True, "stop_loss"
    if datetime.now().time() >= CUTOFF_TIME:
        return True, "time_cutoff"
    if daily_pnl <= -0.03 * starting_balance:
        return True, "daily_nanny"
    return False, ""