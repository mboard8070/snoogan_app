# code/trader/position_manager.py - position monitoring
from datetime import time, datetime, timedelta, timezone

# --- CONFIGURATION ---
CUTOFF_TIME = time(14, 0)  # 2 PM EST/EDT

def get_ny_time():
    """Returns current time in New York (EST/EDT)"""
    # Simple UTC offset for New York (UTC-5). 
    # Use pytz.timezone("America/New_York") for more precision with DST.
    utc_now = datetime.now(timezone.utc)
    return utc_now - timedelta(hours=5)

def should_close(position: dict, current_debit: float, daily_pnl: float, starting_balance: float) -> tuple[bool, str]:
    """
    Standalone function for logic consistency with Jupyter-style imports.
    Evaluates vertical spread exit criteria.
    """
    if not position:
        return False, ""
    
    # 1. Logic Setup: Calculate PnL percentage based on credit received
    credit_received = position.get("credit_received", 0)
    if credit_received <= 0:
        return False, ""
        
    # PnL % = (Credit Collected - Cost to Buy Back) / Credit Collected
    pnl_pct = (credit_received - current_debit) / credit_received
    
    # 2. Profit Target (The 50% Rule or the Nickel Rule)
    # Exit if we made 50% profit OR if the spread is worth only $0.05
    if current_debit <= 0.05 or pnl_pct >= 0.50:
        return True, "profit_target"
    
    # 3. Stop Loss (1.5x Rule)
    # If pnl_pct is -1.5, it means the debit has grown to 2.5x the credit collected
    if pnl_pct <= -1.5:
        return True, "stop_loss"
    
    # 4. Time Cutoff (2 PM EST)
    ny_now = get_ny_time()
    if ny_now.time() >= CUTOFF_TIME:
        return True, "time_cutoff"
    
    # 5. Daily Nanny (3% account protection)
    if daily_pnl <= -0.03 * starting_balance:
        return True, "daily_nanny"
    
    return False, ""

class PositionManager:
    """
    Manages the state of the current active trade.
    Prints output that will be captured by the Streamlit Verbose Log.
    """
    def __init__(self):
        self.position = None

    def open_position(self, details: dict):
        """Sets active position and prints debug info for the UI"""
        self.position = details
        print(f"[{datetime.now().strftime('%H:%M:%S')}] POSITION OPENED")
        print(f"  Symbol: {details.get('symbol', 'N/A')}")
        print(f"  Credit: ${details.get('credit_received', 0):.2f}")
        print(f"  Delta: {details.get('delta', '30')}")

    def close_position(self, realized_pnl: float, reason: str):
        """Clears active position and logs exit reason"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] POSITION CLOSED")
        print(f"  Reason: {reason.upper()}")
        print(f"  PnL: ${realized_pnl:+.2f}")
        self.position = None

    def get_status(self):
        """Returns True if a position is currently tracked"""
        return self.position is not None