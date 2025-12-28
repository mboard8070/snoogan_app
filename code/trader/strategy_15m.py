# code/trader/strategy_15m.py – 15-minute strategy with full persistence (positions, equity_history, daily_pnl)
# Below is the same code with detailed comments explaining what EACH METHOD does

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from code.data.data_client import data_client
from indicators import get_trend_signal
from discord_notifier import send_entry, send_exit, set_strategy_ready

est = ZoneInfo("America/New_York")
project_root = Path(__file__).parent.parent
STATE_FILE = project_root / "strategy_15m_state.json"  # Separate from 1m to avoid conflicts

class TradingStrategy15m:
    def __init__(self):
        """
        Constructor: Initializes the strategy instance.
        - Loads persisted state (positions, equity curve, daily PnL, closed trades) from disk.
        - Sets the daily loss limit to -3% of current equity.
        - Tracks whether the "strategy ready" Discord message has been sent.
        """
        self._load_state()
        self.daily_loss_limit = -0.03 * self.equity_history[-1]
        self._strategy_ready_sent = False
        self.prefix = "[15m] "
        self.trades_today = []  # For EOD trade count (not currently used elsewhere)

    def _load_state(self):
        """
        Private method: Loads persistent state from strategy_15m_state.json.
        - If file exists: restores open positions, equity history, daily PnL, last PnL date, and closed trades.
        - If file missing: starts fresh with default starting balance from env var or $100k.
        - Resets daily PnL to zero if we're on a new trading day.
        """
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                self.positions = state.get('positions', {})
                self.equity_history = state.get('equity_history', [float(os.getenv("STARTING_BALANCE", 100000))])
                self.daily_pnl = state.get('daily_pnl', 0.0)
                self.last_pnl_date = state.get('last_pnl_date', date.today().isoformat())
                self.closed_trades = state.get('closed_trades', [])
        else:
            self.positions = {}
            self.equity_history = [float(os.getenv("STARTING_BALANCE", 100000))]
            self.daily_pnl = 0.0
            self.last_pnl_date = date.today().isoformat()
            self.closed_trades = []
        # Reset daily PnL on new trading day
        if self.last_pnl_date != date.today().isoformat():
            self.daily_pnl = 0.0
            self.last_pnl_date = date.today().isoformat()

    def _save_state(self):
        """
        Private method: Persists current state to strategy_15m_state.json.
        - Saves open positions, full equity history, daily PnL, last PnL date, and recent closed trades.
        - Called at the end of every cycle to ensure crash recovery.
        """
        state = {
            'positions': self.positions,
            'equity_history': self.equity_history,
            'daily_pnl': self.daily_pnl,
            'last_pnl_date': self.last_pnl_date,
            'closed_trades': self.closed_trades
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)

    def _log_closed_trades_batch(self):
        """
        Private method: Batches closed trades for long-term storage.
        - When 30+ closed trades accumulate in memory, writes them to a timestamped JSON file in /data/knowledge.
        - Clears the in-memory list afterward to prevent unbounded growth.
        - Used for later analysis / RL training data.
        """
        if len(self.closed_trades) >= 30:
            timestamp = datetime.now(est).strftime("%Y%m%d_%H%M%S")
            rag_folder = Path("/data/knowledge")  # Fixed path
            rag_folder.mkdir(parents=True, exist_ok=True)
            batch_file = rag_folder / f"trades_batch_15m_{timestamp}.json"
            with open(batch_file, 'w') as f:
                json.dump(self.closed_trades, f, indent=4)
            print(f"{self.prefix} Logged 30 closed trades to {batch_file.name}")
            self.closed_trades = []

    def is_trading_day(self):
        """
        Checks if today is a valid trading day.
        - Excludes weekends and hardcoded 2025 US market holidays.
        - Returns True only on regular trading days.
        """
        today = date.today()
        if today.weekday() >= 5:
            return False
        holidays_2025 = {
            date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
            date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
            date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
            date(2025, 12, 25)
        }
        return today not in holidays_2025

    def is_scanning_active(self):
        """
        Determines if the bot should be actively scanning (8:00–16:00 EST).
        - Returns False outside market hours or on non-trading days.
        """
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        return time(8, 0) <= now <= time(16, 0)

    def can_enter_trades(self):
        """
        Defines the entry window: 9:40–14:30 EST.
        - Prevents new entries too early (pre-open volatility) or too late (near close).
        - Still allows management of existing positions outside this window.
        """
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        return time(9, 40) <= now < time(14, 30)

    def check_daily_loss_nanny(self):
        """
        Enforces the 3% daily loss limit.
        - If daily PnL hits or exceeds the limit, prints nanny message and blocks further trading today.
        - Returns True if still okay to trade.
        """
        if self.daily_pnl <= self.daily_loss_limit:
            print(f"{self.prefix}[NANNY] 3% daily loss hit – no more trades today.")
            return False
        return True

    def run_cycle(self):
        """
        Main loop method called every 15 minutes.
        - Prints cycle header with time, daily PnL, and open positions.
        - Checks scanning window and daily loss nanny.
        - Sends "strategy ready" Discord message once per day.
        - Loops through SPY, QQQ, IWM:
            • Fetches minute bars and resamples to 15m.
            • Computes trend on 15m bars.
            • Attempts new entry if no position and conditions met.
            • Otherwise just monitors.
        - Manages all open positions.
        - Saves state and logs closed trade batches.
        """
        now = datetime.now(est)
        tickers = ["SPY", "QQQ", "IWM"]
        print(f"{self.prefix}=============================================")
        print(f"{self.prefix} CYCLE: {now.strftime('%H:%M:%S')}")
        print(f"{self.prefix} DAILY PnL: ${self.daily_pnl:+.2f} | LIMIT: -${-self.daily_loss_limit:.2f}")
        print(f"{self.prefix} OPEN POSITIONS: {len(self.positions)}")
        print(f"{self.prefix}---------------------------------------------")
        if not self.is_scanning_active():
            print(f"{self.prefix} Outside 8AM-4PM window – scanning paused.")
            print(f"{self.prefix}=============================================")
            return
        if not self.check_daily_loss_nanny():
            print(f"{self.prefix}=============================================")
            return
        if not self._strategy_ready_sent:
            set_strategy_ready()
            self._strategy_ready_sent = True
        for ticker in tickers:
            minute_bars = data_client.get_spy_bars(
                now - timedelta(days=7), now, ticker=ticker
            )
            if minute_bars is None or minute_bars.empty or len(minute_bars) < 20:
                print(f"{self.prefix} [DATA SKIP] {ticker} - insufficient minute bars")
                continue
            bars_15m = minute_bars.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            if bars_15m.empty or len(bars_15m) < 20:
                print(f"{self.prefix} [DATA SKIP] {ticker} - not enough 15m bars")
                continue
            trend = get_trend_signal(bars_15m, None, None)
            price = bars_15m['close'].iloc[-1]
            chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker)
            if chain is None or chain.empty:
                print(f"{self.prefix} [DATA SKIP] {ticker} - empty option chain")
                continue
            print(f"{self.prefix} [{ticker}] Price: {price:.2f} | Trend: {trend.upper()} (15m)")
            if ticker not in self.positions:
                if trend != "chop" and self.can_enter_trades():
                    self._attempt_entry(ticker, trend, chain, price)
            else:
                print(f"{self.prefix} [MONITOR] Managing {ticker} position")
        self.manage_positions()
        self._save_state()
        self._log_closed_trades_batch()
        print(f"{self.prefix}=============================================")

    def _get_spread_price(self, short_contract, long_contract):
        """
        Private method: Calculates the executable credit for a vertical spread.
        - Uses midpoints: (short bid + short ask)/2 - (long bid + long ask)/2.
        - Returns None if either leg is illiquid (zero bid/ask) or net credit < $0.20.
        - Rounded to two decimals.
        """
        short_bid = short_contract['bid']
        short_ask = short_contract['ask']
        long_bid = long_contract['bid']
        long_ask = long_contract['ask']
        if short_bid == 0 or long_ask == 0:
            return None
        credit = short_bid - long_ask
        if credit < 0.20:
            return None
        return round((short_bid + short_ask)/2 - (long_bid + long_ask)/2, 2)

    def _attempt_entry(self, ticker, trend, chain, price):
        """
        Core entry logic for a new vertical credit spread.
        - Determines put vs call based on trend ("up" → put spread).
        - Filters option chain to correct type.
        - Finds short leg candidates within ±0.05 of target delta (-0.30 put / +0.30 call).
        - For each short candidate, finds long leg candidates within ±0.05 of -0.15/+0.15 delta, on the correct side.
        - Calculates credit for every valid pair.
        - Selects the pair with the HIGHEST credit.
        - Enters 1-contract position if credit ≥ $0.20, saves details, sends Discord entry message.
        """
        is_put = trend == "up"
        type_col = None
        for col in ['type', 'contract_type', 'option_type', 'right', 'call_put']:
            if col in chain.columns:
                type_col = col
                break
        if type_col is None:
            print(f"{self.prefix} [{ticker}] No recognized call/put column in chain – skipping entry")
            return
        target_type = 'put' if is_put else 'call'
        opts = chain[chain[type_col].str.lower() == target_type].copy()
        opts['delta'] = pd.to_numeric(opts['delta'], errors='coerce')
        opts = opts.dropna(subset=['delta'])
        if opts.empty:
            print(f"{self.prefix} [{ticker}] No valid deltas in chain – skipping entry")
            return
        short_delta_target = -0.30 if is_put else 0.30
        long_delta_target = -0.15 if is_put else 0.15
        delta_tolerance = 0.15
        short_candidates = opts[abs(opts['delta'] - short_delta_target) <= delta_tolerance]
        if short_candidates.empty:
            print(f"{self.prefix} [{ticker}] No short legs near {short_delta_target:.2f} delta – skipping entry")
            return
        candidates = []
        for _, short_row in short_candidates.iterrows():
            strike_condition = (opts['strike'] < short_row['strike']) if is_put else (opts['strike'] > short_row['strike'])
            long_candidates = opts[strike_condition & (abs(opts['delta'] - long_delta_target) <= delta_tolerance)]
            for _, long_row in long_candidates.iterrows():
                credit = self._get_spread_price(short_row.to_dict(), long_row.to_dict())
                if credit is not None:
                    candidates.append((credit, short_row, long_row))
        if not candidates:
            print(f"{self.prefix} [{ticker}] No valid delta spreads with credit >= $0.20 – skipping entry")
            return
        best_credit, best_short, best_long = max(candidates, key=lambda x: x[0])
        short_strike = float(best_short['strike'])
        long_strike = float(best_long['strike'])
        contracts = 1
        self.positions[ticker] = {
            "is_put": is_put,
            "short": short_strike,
            "long": long_strike,
            "credit": best_credit,
            "contracts": contracts,
            "best_value": best_credit,
            "trail_active": False,
            "trail_level": None,
            "entry_time": datetime.now(est).isoformat()
        }
        self.trades_today.append("entry")
        send_entry(
            is_put=is_put,
            short=short_strike,
            long=long_strike,
            credit=best_credit * contracts * 100,
            underlying=ticker
        )
        print(f"{self.prefix} [ENTRY] {ticker} {'PUT' if is_put else 'CALL'} {short_strike:.1f}/{long_strike:.1f} | "
              f"Entry: {best_credit:.2f}")

    def _get_current_mark(self, ticker, pos):
        """
        Private method: Fetches current midpoint mark value of an open spread.
        - Pulls fresh 0DTE chain.
        - Looks up short and long legs by strike and type.
        - Returns the same midpoint credit calculation as entry.
        - Used for profit tracking and exit decisions.
        """
        now = datetime.now(est)
        chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker)
        if chain is None or chain.empty:
            return None
        type_col = None
        for col in ['type', 'contract_type', 'option_type', 'right', 'call_put']:
            if col in chain.columns:
                type_col = col
                break
        if type_col is None:
            return None
        target_type = 'put' if pos["is_put"] else 'call'
        short_row = chain[(chain['strike'] == pos["short"]) & (chain[type_col].str.lower() == target_type)]
        long_row = chain[(chain['strike'] == pos["long"]) & (chain[type_col].str.lower() == target_type)]
        if short_row.empty or long_row.empty:
            return None
        short_contract = short_row.iloc[0].to_dict()
        long_contract = long_row.iloc[0].to_dict()
        return self._get_spread_price(short_contract, long_contract)

    def manage_positions(self):
        """
        Monitors and exits all open positions every cycle.
        - Updates best (lowest) mark seen.
        - Exit conditions:
            • Spread value ≥ entry credit (loss = full credit)
            • After 14:30 EST (forced close at current mark)
            • 50% profit taken → activates 10% trailing stop on the best mark
            • Trailing stop hit
        - Calls exit_position() when any condition triggers.
        """
        now = datetime.now(est)
        for ticker, pos in list(self.positions.items()):
            current_value = self._get_current_mark(ticker, pos)
            if current_value is None:
                print(f"{self.prefix} [{ticker}] Unable to fetch live mark – skipping management this cycle")
                continue
            print(f"{self.prefix} [MONITOR] {ticker} {'PUT' if pos['is_put'] else 'CALL'} {pos['short']:.1f}/{pos['long']:.1f} | "
                  f"Entry: {pos['credit']:.2f} | Current: {current_value:.2f}")
            if current_value < pos["best_value"]:
                pos["best_value"] = current_value
            realized = None
            credit = pos["credit"]
            if current_value >= credit:  # Full loss
                realized = -credit * pos["contracts"] * 100
            elif now.time() >= time(14, 30):  # Forced close
                realized = (credit - current_value) * pos["contracts"] * 100
            elif (credit - current_value) >= 0.5 * credit:  # 50% profit
                if not pos["trail_active"]:
                    pos["trail_active"] = True
                    pos["trail_level"] = pos["best_value"] * 1.10
                    print(f"{self.prefix} [{ticker}] 50% profit hit – 10% trailing stop activated @ {pos['trail_level']:.2f}")
                if current_value >= pos["trail_level"]:
                    realized = (pos["trail_level"] - current_value) * pos["contracts"] * 100
            if realized is not None:
                self.exit_position(ticker, pos, realized, current_value)

    def exit_position(self, ticker, pos, realized_pnl, final_mark):
        """
        Finalizes a trade exit.
        - Updates daily PnL and equity curve.
        - Sends Discord exit message.
        - Logs detailed closed trade dictionary for later analysis.
        - Removes position from open dict.
        """
        self.daily_pnl += realized_pnl
        self.equity_history.append(self.equity_history[-1] + realized_pnl)
        send_exit(
            is_put=pos["is_put"],
            short=pos["short"],
            long=pos["long"],
            credit=pos["credit"],
            pnl=realized_pnl,
            underlying=ticker
        )
        print(f"{self.prefix} EXITED {ticker} spread – P/L ${realized_pnl:+.2f} (mark ${final_mark:.2f})")
        closed_trade = {
            "ticker": ticker,
            "is_put": pos["is_put"],
            "short": pos["short"],
            "long": pos["long"],
            "credit": pos["credit"],
            "pnl": realized_pnl,
            "entry_time": pos.get("entry_time", "unknown"),
            "exit_time": datetime.now(est).isoformat()
        }
        self.closed_trades.append(closed_trade)
        del self.positions[ticker]
        self.trades_today.append("exit")