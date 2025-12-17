# code/trader/strategy.py - main trading logic (Verbosity Updated)

from datetime import time, datetime, timedelta
import os
from data_client import data_client
from indicators import get_trend_signal
from position_manager import should_close

ENTRY_START = time(9, 50)
ENTRY_END = time(10, 30)

class TradingStrategy:
    def __init__(self):
        from executor import Executor
        self.executor = Executor()
        self.position = None
        self.daily_pnl = 0.0
        self.balance = float(os.getenv("STARTING_BALANCE", "100000"))

    def in_entry_window(self) -> bool:
        now = datetime.now().time()
        return ENTRY_START <= now <= ENTRY_END

    def run_cycle(self):
        now = datetime.now()
        timestamp = now.strftime('%H:%M:%S')
        print(f"[{timestamp}] === Starting trading cycle ===")

        # Fetching Data
        bars_1m = data_client.get_spy_bars(now - timedelta(hours=1), now)
        chain = data_client.get_spx_option_chain(now.strftime("%Y-%m-%d"))

        # ENTRY LOGIC
        if self.in_entry_window() and self.position is None:
            print(f"[{timestamp}] Entry window open. Checking indicators...")
            trend = get_trend_signal(bars_1m, None, None)
            
            # Map trend for clearer logging
            status_map = {"bull": "BULLISH 🟢", "bear": "BEARISH 🔴", "chop": "CHOPPY 🟡"}
            current_status = status_map.get(trend, trend.upper())
            print(f"[{timestamp}] Market Vote: {current_status}")

            if trend == "chop":
                print(f"[{timestamp}] Result: FAILED (Reason: Market is choppy/sideways)")
            elif chain.empty:
                print(f"[{timestamp}] Result: FAILED (Reason: Option chain data unavailable)")
            else:
                print(f"[{timestamp}] Result: PASSED. Proceeding to strike selection.")
                direction = "put" if trend == "bull" else "call"
                target_chain = chain[chain['option_type'] == direction].copy()
                
                if 'delta' in target_chain.columns and target_chain['delta'].notnull().any():
                    target_chain['abs_delta'] = target_chain['delta'].abs()
                    selected = target_chain.iloc[(target_chain['abs_delta'] - 0.30).abs().argsort()[:1]]
                else:
                    target_chain['credit'] = (target_chain['bid_price'] + target_chain['ask_price']) / 2
                    selected = target_chain.nlargest(1, 'credit')

                if not selected.empty:
                    short_strike = selected['strike_price'].values[0]
                    long_strike = short_strike - 5 if direction == "put" else short_strike + 5
                    credit = selected['credit'].values[0] if 'credit' in selected else (selected['bid_price'].values[0] + selected['ask_price'].values[0]) / 2

                    print(f"[{timestamp}] EXECUTION: Selling {direction.upper()} Vertical Spread")
                    print(f"   Strikes: {short_strike}/{long_strike} | Net Credit: ${credit:.2f}")
                    
                    self.position = self.executor.place_credit_spread(direction, short_strike, long_strike, credit)
                else:
                    print(f"[{timestamp}] Result: FAILED (Reason: No strikes found near 30-delta)")

        # MONITORING & EXIT LOGIC
        if self.position:
            print(f"[{timestamp}] Monitoring active {self.position['short_strike']} position...")
            current_chain = data_client.get_spx_option_chain(now.strftime("%Y-%m-%d"))
            short_row = current_chain[current_chain['strike_price'] == self.position['short_strike']]
            long_row = current_chain[current_chain['strike_price'] == self.position['long_strike']]
            
            if not short_row.empty and not long_row.empty:
                debit = (short_row['bid_price'].values[0] + long_row['ask_price'].values[0]) / 2 - (long_row['bid_price'].values[0] + short_row['ask_price'].values[0]) / 2
                print(f"[{timestamp}] Current Debit to close: ${debit:.2f}")
                
                close, reason = should_close(self.position, debit, self.daily_pnl, self.balance)
                
                if close:
                    print(f"[{timestamp}] EXIT SIGNAL: {reason.upper()} triggered.")
                    pnl = self.executor.close_position(self.position, debit)
                    self.daily_pnl += pnl
                    self.balance += pnl
                    print(f"[{timestamp}] POSITION CLOSED. Realized PnL: ${pnl:+.2f}")
                    self.position = None
                else:
                    print(f"[{timestamp}] Status: Holding (Criteria for '{reason}' not met)")

        elif not self.in_entry_window():
            # If after 10:30 and no position, just a quiet heartbeat
            if now.minute % 5 == 0: # Only print this every 5 mins to keep log clean
                 print(f"[{timestamp}] Entry window closed. No active positions.")

        print(f"[{timestamp}] === Cycle complete ===")