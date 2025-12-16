# trader/strategy.py - main trading logic

from datetime import time, datetime, timedelta
from data_client import data_client
from executor import Executor
from indicators import get_trend_signal
from position_manager import should_close

ENTRY_START = time(9, 50)
ENTRY_END = time(10, 30)

class TradingStrategy:
    def __init__(self):
        self.executor = Executor()
        self.position = None
        self.daily_pnl = 0.0
        self.balance = float(os.getenv("STARTING_BALANCE", "100000"))

    def in_entry_window(self) -> bool:
        now = datetime.now().time()
        return ENTRY_START <= now <= ENTRY_END

    def run_cycle(self):
        now = datetime.now()

        # Pull fresh data
        bars_1m = data_client.get_spy_bars(now - timedelta(hours=1), now)
        chain = data_client.get_spx_option_chain(now.strftime("%Y-%m-%d"))

        # Entry
        if self.in_entry_window() and self.position is None:
            trend = get_trend_signal(bars_1m, None, None)  # extend with 5m/15m later
            if trend != "chop" and not chain.empty:
                direction = "put" if trend == "bull" else "call"
                target_chain = chain[chain['option_type'] == direction]
                if 'delta' in target_chain.columns and target_chain['delta'].notnull().any():
                    target_chain = target_chain.copy()
                    target_chain['abs_delta'] = target_chain['delta'].abs()
                    selected = target_chain.iloc[(target_chain['abs_delta'] - 0.30).abs().argsort()[:1]]
                else:
                    # Fallback to highest credit if no delta
                    target_chain['credit'] = (target_chain['bid_price'] + target_chain['ask_price']) / 2
                    selected = target_chain.nlargest(1, 'credit')

                if not selected.empty:
                    short_strike = selected['strike_price'].values[0]
                    long_strike = short_strike - 5 if direction == "put" else short_strike + 5
                    credit = selected['credit'].values[0] if 'credit' in selected else (selected['bid_price'].values[0] + selected['ask_price'].values[0]) / 2
                    self.position = self.executor.place_credit_spread(direction, short_strike, long_strike, credit)

        # Management
        if self.position:
            current_chain = data_client.get_spx_option_chain(now.strftime("%Y-%m-%d"))
            short_row = current_chain[current_chain['strike_price'] == self.position['short_strike']]
            long_row = current_chain[current_chain['strike_price'] == self.position['long_strike']]
            if not short_row.empty and not long_row.empty:
                debit = (short_row['bid_price'].values[0] + long_row['ask_price'].values[0]) / 2 - (long_row['bid_price'].values[0] + short_row['ask_price'].values[0]) / 2  # approximate
                close, reason = should_close(self.position, debit, self.daily_pnl, self.balance)
                if close:
                    pnl = self.executor.close_position(self.position, debit)
                    self.daily_pnl += pnl
                    self.balance += pnl
                    self.position = None