# scalp_strategy.py
# Standalone scalping bot – buys single ATM 0DTE options
# Run separately from the vertical credit spread strategy

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from code.data.data_client import data_client
from indicators import get_trend_signal
# Updated imports for scalp-specific Discord messages
from discord_notifier import send_scalp_entry, send_scalp_exit, set_strategy_ready

est = ZoneInfo("America/New_York")
project_root = Path(__file__).parent.parent
STATE_FILE = project_root / "scalp_strategy_state.json"   # separate state file

class ScalpStrategy:
    def __init__(self):
        self._load_state()
        self.daily_loss_limit = -0.03 * self.equity_history[-1]
        self._strategy_ready_sent = False
        self.trades_today = []

    def _load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                self.positions = state.get('positions', {})                    # {ticker: pos_dict}
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

        if self.last_pnl_date != date.today().isoformat():
            self.daily_pnl = 0.0
            self.last_pnl_date = date.today().isoformat()

    def _save_state(self):
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
        if len(self.closed_trades) >= 30:
            timestamp = datetime.now(est).strftime("%Y%m%d_%H%M%S")
            rag_folder = Path("/data/knowledge")
            rag_folder.mkdir(parents=True, exist_ok=True)
            batch_file = rag_folder / f"scalp_trades_batch_{timestamp}.json"
            with open(batch_file, 'w') as f:
                json.dump(self.closed_trades, f, indent=4)
            print(f"Logged 30 scalp trades to {batch_file.name}")
            self.closed_trades = []

    def is_trading_day(self):
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
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        return time(8, 0) <= now <= time(16, 0)

    def can_enter_trades(self):
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        return time(9, 40) <= now < time(14, 30)

    def check_daily_loss_nanny(self):
        if self.daily_pnl <= self.daily_loss_limit:
            print("[NANNY] 3% daily loss hit – scalp bot shutting down for today.")
            return False
        return True

    def run_cycle(self):
        now = datetime.now(est)
        tickers = ["SPY", "QQQ", "IWM"]

        print("=============================================")
        print(f" SCALP CYCLE: {now.strftime('%H:%M:%S')}")
        print(f" DAILY PnL: ${self.daily_pnl:+.2f} | LIMIT: -${-self.daily_loss_limit:.2f}")
        print(f" OPEN POSITIONS: {len(self.positions)}")
        print("---------------------------------------------")

        if not self.is_scanning_active():
            print(" Outside 8AM-4PM window – scanning paused.")
            return
        if not self.check_daily_loss_nanny():
            return
        if not self._strategy_ready_sent:
            set_strategy_ready()
            self._strategy_ready_sent = True

        for ticker in tickers:
            bars = data_client.get_spy_bars(now - timedelta(days=7), now, ticker=ticker)
            if bars is None or bars.empty or len(bars) < 20:
                print(f" [DATA SKIP] {ticker} - insufficient bars")
                continue
            trend = get_trend_signal(bars, None, None)
            price = bars['close'].iloc[-1]
            chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker)
            if chain is None or chain.empty:
                print(f" [DATA SKIP] {ticker} - empty option chain")
                continue

            print(f" [{ticker}] Price: {price:.2f} | Trend: {trend.upper()}")

            if ticker not in self.positions:
                if trend != "chop" and self.can_enter_trades():
                    self._attempt_entry(ticker, trend, chain, price)
            else:
                print(f" [MONITOR] Managing {ticker} scalp position")

        self.manage_positions()
        self._save_state()
        self._log_closed_trades_batch()
        print("=============================================")

    def _get_option_mark(self, contract):
        bid = contract['bid']
        ask = contract['ask']
        if bid == 0 or ask == 0:
            return None
        return round((bid + ask) / 2, 2)

    def _attempt_entry(self, ticker, trend, chain, price):
        is_call = trend == "up"
        type_col = next((c for c in ['type', 'contract_type', 'option_type', 'right', 'call_put'] if c in chain.columns), None)
        if type_col is None:
            print(f" [{ticker}] No call/put column found – skipping")
            return

        target_type = 'call' if is_call else 'put'
        opts = chain[chain[type_col].str.lower() == target_type].copy()
        if opts.empty:
            return

        # ATM = closest strike
        closest_idx = abs(opts['strike'] - price).argmin()
        row = opts.iloc[closest_idx]
        contract = row.to_dict()
        debit = self._get_option_mark(contract)
        if debit is None or debit < 0.20:
            print(f" [{ticker}] Invalid or thin ATM debit ({debit}) – skipping")
            return

        strike = float(row['strike'])
        contracts = 1

        self.positions[ticker] = {
            "is_call": is_call,
            "strike": strike,
            "debit": debit,
            "contracts": contracts,
            "best_mark": debit,          # highest mark seen (for trailing stop)
            "trail_active": False,
            "trail_level": None,
            "entry_time": datetime.now(est).isoformat()
        }

        # SCALP-SPECIFIC DISCORD ENTRY
        send_scalp_entry(
            is_call=is_call,
            strike=strike,
            debit=debit * contracts * 100,
            underlying=ticker
        )
        print(f" [SCALP ENTRY] {ticker} ATM {'CALL' if is_call else 'PUT'} {strike:.1f} @ {debit:.2f}")

    def _get_current_mark(self, ticker, pos):
        now = datetime.now(est)
        chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker)
        if chain is None or chain.empty:
            return None
        type_col = next((c for c in ['type', 'contract_type', 'option_type', 'right', 'call_put'] if c in chain.columns), None)
        if type_col is None:
            return None
        target_type = 'call' if pos["is_call"] else 'put'
        row = chain[(chain['strike'] == pos["strike"]) & (chain[type_col].str.lower() == target_type)]
        if row.empty:
            return None
        return self._get_option_mark(row.iloc[0].to_dict())

    def manage_positions(self):
        now = datetime.now(est)
        for ticker, pos in list(self.positions.items()):
            # Fresh trend for chop check
            bars = data_client.get_spy_bars(now - timedelta(days=7), now, ticker=ticker)
            if bars is None or bars.empty or len(bars) < 20:
                continue
            trend = get_trend_signal(bars, None, None)

            mark = self._get_current_mark(ticker, pos)
            if mark is None:
                print(f" [{ticker}] No live mark – skipping this cycle")
                continue

            print(f" [SCALP MONITOR] {ticker} {'CALL' if pos['is_call'] else 'PUT'} {pos['strike']:.1f} | "
                  f"Debit: {pos['debit']:.2f} → Mark: {mark:.2f}")

            if mark > pos["best_mark"]:
                pos["best_mark"] = mark

            realized = None
            debit = pos["debit"]

            if mark <= 0.00:                                     # 1× debit stop loss
                realized = -debit * pos["contracts"] * 100
            elif trend == "chop":                                # exit on chop
                realized = (mark - debit) * pos["contracts"] * 100
            elif now.time() >= time(14, 30):                      # forced close
                realized = (mark - debit) * pos["contracts"] * 100
            elif (mark - debit) >= 0.5 * debit:                   # 50% profit → trail
                if not pos["trail_active"]:
                    pos["trail_active"] = True
                    pos["trail_level"] = pos["best_mark"] * 0.90
                    print(f" [{ticker}] 50% profit → 10% trail activated @ {pos['trail_level']:.2f}")
                if mark <= pos["trail_level"]:
                    realized = (mark - debit) * pos["contracts"] * 100

            if realized is not None:
                self.exit_position(ticker, pos, realized, mark)

    def exit_position(self, ticker, pos, realized_pnl, final_mark):
        self.daily_pnl += realized_pnl
        self.equity_history.append(self.equity_history[-1] + realized_pnl)

        # SCALP-SPECIFIC DISCORD EXIT
        send_scalp_exit(
            is_call=pos["is_call"],
            strike=pos["strike"],
            debit=pos["debit"] * pos["contracts"] * 100,
            pnl=realized_pnl,
            underlying=ticker
        )
        print(f" [SCALP EXIT] {ticker} → P/L ${realized_pnl:+.2f} (mark {final_mark:.2f})")

        closed = {
            "ticker": ticker,
            "is_call": pos["is_call"],
            "strike": pos["strike"],
            "debit": pos["debit"],
            "pnl": realized_pnl,
            "entry_time": pos.get("entry_time"),
            "exit_time": datetime.now(est).isoformat()
        }
        self.closed_trades.append(closed)
        del self.positions[ticker]


if __name__ == "__main__":
    bot = ScalpStrategy()
    # Example loop – replace with your scheduler / timer
    import time
    while True:
        bot.run_cycle()
        time.sleep(60)   # run every minute