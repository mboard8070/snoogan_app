# code/trader/strategy_15m.py – 15-minute vertical credit spread strategy (Alpaca Data)
# Updated: Fixed $5 width spreads, lowered min credit to $0.15, widened delta to 20-45
# Added: End-of-day forced closeout at/after 4:00 PM EST to prevent overnight open positions
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
STATE_FILE = project_root / "strategy_15m_state.json"

class TradingStrategy15m:
    def __init__(self):
        self._load_state()
        self.daily_loss_limit = -0.03 * self.equity_history[-1]
        self._strategy_ready_sent = False
        self.prefix = "[15m] "
        self.trades_today = []

    def _load_state(self):
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
            batch_file = rag_folder / f"trades_batch_15m_{timestamp}.json"
            with open(batch_file, 'w') as f:
                json.dump(self.closed_trades, f, indent=4)
            print(f"{self.prefix} Logged 30 closed trades to {batch_file.name}")
            self.closed_trades = []

    def is_trading_day(self):
        today = date.today()
        if today.weekday() >= 5:  # Saturday or Sunday
            return False
        # NYSE/NASDAQ full closures 2026-2028
        holidays_2026 = {
            date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
            date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
            date(2026, 11, 26), date(2026, 12, 25)
        }
        holidays_2027 = {
            date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
            date(2027, 5, 31), date(2027, 6, 19), date(2027, 7, 5), date(2027, 9, 6),
            date(2027, 11, 25), date(2027, 12, 27)
        }
        holidays_2028 = {
            date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14), date(2028, 5, 29),
            date(2028, 6, 19), date(2028, 7, 4), date(2028, 9, 4), date(2028, 11, 23),
            date(2028, 12, 25)
        }
        if today.year == 2026:
            holidays = holidays_2026
        elif today.year == 2027:
            holidays = holidays_2027
        elif today.year == 2028:
            holidays = holidays_2028
        else:
            print(f"{self.prefix}[WARNING] No holidays defined for {today.year} – assuming trading day")
            holidays = set()
        return today not in holidays

    def is_scanning_active(self):
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        return time(8, 0) <= now <= time(16, 0)

    def can_enter_trades(self):
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        return time(9, 35) <= now < time(14, 30)

    def check_daily_loss_nanny(self):
        if self.daily_pnl <= self.daily_loss_limit:
            print(f"{self.prefix}[NANNY] 3% daily loss hit – no more trades today.")
            return False
        return True

    def run_cycle(self):
        now = datetime.now(est)
        current_time = now.time()
        tickers = ["SPY", "QQQ", "IWM"]
        print(f"{self.prefix}=============================================")
        print(f"{self.prefix} CYCLE: {now.strftime('%H:%M:%S')}")
        print(f"{self.prefix} DAILY PnL: ${self.daily_pnl:+.2f} | LIMIT: -${-self.daily_loss_limit:.2f}")
        print(f"{self.prefix} OPEN POSITIONS: {len(self.positions)}")
        print(f"{self.prefix}---------------------------------------------")

        # Force close all positions at/after 4:00 PM EST on trading days
        if len(self.positions) > 0 and self.is_trading_day() and current_time >= time(16, 0):
            print(f"{self.prefix} [EOD FORCE CLOSE] Market closed – forcing exit on all open positions")
            for ticker, pos in list(self.positions.items()):
                current_value = self._get_current_mark(ticker, pos)
                if current_value is None:
                    print(f"{self.prefix} [{ticker}] No final mark available – closing at $0.00 (full loss)")
                    current_value = 0.0
                realized = (pos["credit"] - current_value) * pos["contracts"] * 100
                self.exit_position(ticker, pos, realized, current_value)
            self._save_state()
            self._log_closed_trades_batch()
            print(f"{self.prefix}=============================================")
            return

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
                    try:
                        self._attempt_entry(ticker, trend, chain, price, now.strftime("%Y-%m-%d"))
                    except Exception as e:
                        print(f"{self.prefix} Unexpected entry error: {e}")
            else:
                print(f"{self.prefix} [MONITOR] Managing {ticker} position")

        self.manage_positions()
        self._save_state()
        self._log_closed_trades_batch()
        print(f"{self.prefix}=============================================")

    def _get_spread_price(self, short_contract, long_contract):
        bid_col = next((col for col in ['bid', 'bid_price', 'b'] if col in short_contract), None)
        ask_col = next((col for col in ['ask', 'ask_price', 'a'] if col in short_contract), None)
        if bid_col is None or ask_col is None:
            return None
        short_bid = short_contract.get(bid_col, 0)
        short_ask = short_contract.get(ask_col, 0)
        long_bid = long_contract.get(bid_col, 0)
        long_ask = long_contract.get(ask_col, 0)
        if short_bid == 0 or long_ask == 0:
            return None
        credit = short_bid - long_ask
        if credit < 0.15:
            return None
        return round((short_bid + short_ask)/2 - (long_bid + long_ask)/2, 2)

    def _attempt_entry(self, ticker, trend, chain, price, expiration_date):
        is_put = trend == "bull"  # bull -> put credit spreads
        type_col = next((c for c in ['type', 'contract_type', 'option_type', 'right', 'call_put', 't'] if c in chain.columns), None)
        if type_col is None:
            print(f"{self.prefix} [{ticker}] No recognized call/put column – skipping entry")
            return
        target_type = 'put' if is_put else 'call'
        opts = chain[chain[type_col].str.lower() == target_type].copy()
        if opts.empty:
            print(f"{self.prefix} [{ticker}] No {target_type}s in chain – skipping entry")
            return
        strike_col = next((c for c in ['strike', 'strike_price', 'k', 'S'] if c in opts.columns), None)
        if strike_col is None:
            print(f"{self.prefix} [{ticker}] No strike column found – skipping entry")
            return
        opts = opts.sort_values(strike_col)
        delta_col = next((c for c in ['delta', 'd'] if c in opts.columns), None)
        short_candidates = pd.DataFrame()
        if delta_col is not None:
            if is_put:
                short_candidates = opts[(opts[delta_col] >= -0.45) & (opts[delta_col] <= -0.20)]  # 20-45 delta
            else:
                short_candidates = opts[(opts[delta_col] >= 0.20) & (opts[delta_col] <= 0.45)]
            if not short_candidates.empty:
                print(f"{self.prefix} [{ticker}] Found {len(short_candidates)} short strikes in 20-45 delta range")
            else:
                print(f"{self.prefix} [{ticker}] No short strikes in 20-45 delta range – falling back to OTM pct")
        if short_candidates.empty:
            short_otm_pct = 0.007  # ~0.7% OTM
            if is_put:
                short_strike_target = price * (1 - short_otm_pct)
                short_candidates = opts[opts[strike_col] <= short_strike_target]
            else:
                short_strike_target = price * (1 + short_otm_pct)
                short_candidates = opts[opts[strike_col] >= short_strike_target]
            print(f"{self.prefix} [{ticker}] Fallback: {len(short_candidates)} short candidates at ~0.7% OTM")
        if short_candidates.empty:
            print(f"{self.prefix} [{ticker}] No suitable short strikes found – skipping entry")
            return

        candidates = []
        for _, short_row in short_candidates.iterrows():
            short_strike = short_row[strike_col]
            target_long_strike = short_strike - 5 if is_put else short_strike + 5
            long_opts = opts[opts[strike_col] == target_long_strike]
            if long_opts.empty:
                continue
            for _, long_row in long_opts.iterrows():
                credit = self._get_spread_price(short_row.to_dict(), long_row.to_dict())
                if credit is not None:
                    candidates.append((credit, short_row, long_row))

        print(f"{self.prefix} [{ticker}] Found {len(candidates)} viable $5-wide spreads >= $0.15 credit")
        if not candidates:
            print(f"{self.prefix} [{ticker}] No valid $5-wide spreads with credit >= $0.15 – skipping entry")
            return

        best_credit, best_short, best_long = max(candidates, key=lambda x: x[0])
        short_strike = float(best_short[strike_col])
        long_strike = float(best_long[strike_col])
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
            "entry_time": datetime.now(est).isoformat(),
            "expiration_date": expiration_date
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
        expiration = pos.get("expiration_date")
        if not expiration:
            print(f"{self.prefix} [{ticker}] No expiration stored in position – falling back to today")
            expiration = datetime.now(est).strftime("%Y-%m-%d")
        chain = data_client.get_spy_option_chain(expiration, ticker=ticker)
        if chain is None or chain.empty:
            print(f"{self.prefix} [{ticker}] Empty chain for expiration {expiration} – skipping management")
            return None
        type_col = next((c for c in ['type', 'contract_type', 'option_type', 'right', 'call_put', 't'] if c in chain.columns), None)
        if type_col is None:
            return None
        strike_col = next((c for c in ['strike', 'strike_price', 'k', 'S'] if c in chain.columns), None)
        if strike_col is None:
            return None
        target_type = 'put' if pos["is_put"] else 'call'
        short_row = chain[(chain[strike_col] == pos["short"]) & (chain[type_col].str.lower() == target_type)]
        long_row = chain[(chain[strike_col] == pos["long"]) & (chain[type_col].str.lower() == target_type)]
        if short_row.empty or long_row.empty:
            return None
        short_contract = short_row.iloc[0].to_dict()
        long_contract = long_row.iloc[0].to_dict()
        return self._get_spread_price(short_contract, long_contract)

    def manage_positions(self):
        now = datetime.now(est)
        for ticker, pos in list(self.positions.items()):
            current_value = self._get_current_mark(ticker, pos)
            if current_value is None:
                print(f"{self.prefix} [{ticker}] Unable to fetch live mark – skipping management this cycle")
                continue
            print(f"{self.prefix} [MONITOR] {ticker} {'PUT' if pos['is_put'] else 'CALL'} {pos['short']:.1f}/{pos['long']:.1f} | "
                  f"Entry: {pos['credit']:.2f} | Current: {current_value:.2f}")
            if current_value < pos["best_value"]:
                old_trail = pos.get("trail_level")
                pos["best_value"] = current_value
                if pos["trail_active"]:
                    pos["trail_level"] = pos["best_value"] * 1.10
                    print(f"{self.prefix} [{ticker}] Trailing stop moved up to {pos['trail_level']:.2f} (from {old_trail:.2f if old_trail else 'N/A'})")
            realized = None
            credit = pos["credit"]
            if current_value >= 2.0 * credit:
                realized = (credit - current_value) * pos["contracts"] * 100
            elif now.time() >= time(14, 30):
                realized = (credit - current_value) * pos["contracts"] * 100
            elif (credit - current_value) >= 0.5 * credit:
                if not pos["trail_active"]:
                    pos["trail_active"] = True
                    pos["trail_level"] = pos["best_value"] * 1.10
                    print(f"{self.prefix} [{ticker}] 50% profit hit – 10% trailing stop activated @ {pos['trail_level']:.2f}")
                if current_value >= pos["trail_level"]:
                    realized = (credit - current_value) * pos["contracts"] * 100
            if realized is not None:
                self.exit_position(ticker, pos, realized, current_value)

    def exit_position(self, ticker, pos, realized_pnl, final_mark):
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