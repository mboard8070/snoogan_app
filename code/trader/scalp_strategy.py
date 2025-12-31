# code/trader/scalp_strategy.py - Removed Polygon/VIX; hard 40% profit target
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from code.data.data_client import data_client
from indicators import get_trend_signal, _macd
from discord_notifier import send_scalp_entry, send_scalp_exit, set_strategy_ready, send_webhook

est = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / "scalp_strategy_state.json"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
TRADES_FILE = KNOWLEDGE_DIR / "trades.json"

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
        self._validate_and_clean_positions()

    def _validate_and_clean_positions(self):
        required_keys = {"is_call", "strike_price", "debit", "contracts", "best_mark", "entry_time"}
        keys_to_remove = []
        for ticker, pos in self.positions.items():
            missing = required_keys - pos.keys()
            if missing:
                print(f"[WARNING] Invalid position for {ticker} - missing keys: {missing}. Removing.")
                keys_to_remove.append(ticker)
            if "chop_count" not in pos:
                pos["chop_count"] = 0
            if "trail_active" not in pos:
                pos["trail_active"] = False
                pos["trail_level"] = None
            if isinstance(pos.get("entry_time"), datetime):
                pos["entry_time"] = pos["entry_time"].isoformat()
        for ticker in keys_to_remove:
            del self.positions[ticker]

    def _save_state(self):
        state = {
            'positions': self.positions,
            'equity_history': self.equity_history,
            'daily_pnl': self.daily_pnl,
            'last_pnl_date': self.last_pnl_date,
            'closed_trades': self.closed_trades
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4, default=str)

    def _save_individual_trade(self, closed_trade):
        if TRADES_FILE.exists():
            with open(TRADES_FILE, 'r') as f:
                try:
                    all_trades = json.load(f)
                except json.JSONDecodeError:
                    all_trades = []
        else:
            all_trades = []
        all_trades.append(closed_trade)
        with open(TRADES_FILE, 'w') as f:
            json.dump(all_trades, f, indent=4, default=str)

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
        early_closes_2025 = {date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24)}
        if today.year != 2025:
            print(f"[WARNING] No holidays defined for {today.year}")
            return False
        self.is_early_close = today in early_closes_2025
        return today not in holidays_2025

    def is_scanning_active(self):
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        end_time = time(13, 0) if getattr(self, 'is_early_close', False) else time(16, 0)
        return time(8, 0) <= now <= end_time

    def can_enter_trades(self):
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        end_time = time(12, 30) if getattr(self, 'is_early_close', False) else time(15, 0)
        return time(9, 30) <= now < end_time

    def check_daily_loss_nanny(self):
        unrealized = sum(((self._get_current_mark(t, p) or p['debit']) - p['debit']) * p['contracts'] * 100
                         for t, p in self.positions.items())
        if self.daily_pnl + unrealized <= self.daily_loss_limit:
            print("[NANNY] 3% daily loss (incl unrealized) hit – shutdown today.")
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
            print(" Outside window – paused.")
            return
        if not self.check_daily_loss_nanny():
            return
        if not self._strategy_ready_sent:
            set_strategy_ready()
            self._strategy_ready_sent = True
        for ticker in tickers:
            bars = data_client.get_spy_bars(now - timedelta(days=7), now, ticker=ticker)
            if bars is None or bars.empty or len(bars) < 26:
                print(f" [DATA SKIP] {ticker} - insufficient bars")
                continue
            trend = get_trend_signal(bars, None, None)
            macd_line, macd_signal = _macd(bars['close'])
            is_momentum_bull = macd_line > macd_signal
            price = bars['close'].iloc[-1]
            chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker)
            if chain is None or chain.empty:
                continue
            print(f" [{ticker}] Price: {price:.2f} | Trend: {trend.upper()} | MACD Bull: {is_momentum_bull}")
            if ticker not in self.positions:
                if (trend == "bull" or (trend != "chop" and is_momentum_bull)) and self.can_enter_trades():
                    self._attempt_entry(ticker, trend, chain, price)
            else:
                print(f" [MONITOR] Managing {ticker} scalp position")
        self.manage_positions()
        self._save_state()
        print("=============================================")

    def _get_option_mark(self, row):
        bid = row.get('bid', 0) or row.get('b', 0) or row.get('bid_price', 0) or 0
        ask = row.get('ask', 0) or row.get('a', 0) or row.get('ask_price', 0) or 0
        if bid == 0 or ask == 0:
            return None
        return round((bid + ask) / 2, 2)

    def _attempt_entry(self, ticker, trend, chain, price):
        is_call = trend == "bull"
        type_col = next((c for c in ['type', 'contract_type', 'option_type', 'right', 'call_put', 't'] if c in chain.columns), None)
        if type_col is None:
            return
        target_type = 'call' if is_call else 'put'
        opts = chain[chain[type_col].str.lower() == target_type].copy()
        if opts.empty:
            return
        strike_col = 'strike_price' if 'strike_price' in opts.columns else 'strike'
        closest_idx = abs(opts[strike_col] - price).argmin()
        row = opts.iloc[closest_idx]
        debit = self._get_option_mark(row)
        if debit is None or debit < 0.20:
            return
        strike_price = float(row[strike_col])
        contracts = 1
        self.positions[ticker] = {
            "is_call": is_call,
            "strike_price": strike_price,
            "debit": debit,
            "contracts": contracts,
            "best_mark": debit,
            "trail_active": False,
            "trail_level": None,
            "entry_time": datetime.now(est).isoformat(),
            "chop_count": 0
        }
        send_scalp_entry(is_call=is_call, strike=strike_price, debit=debit * contracts * 100, underlying=ticker)
        print(f" [SCALP ENTRY] {ticker} ATM {'CALL' if is_call else 'PUT'} {strike_price:.1f} @ {debit:.2f}")

    def _get_current_mark(self, ticker, pos):
        now = datetime.now(est)
        chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker)
        if chain is None or chain.empty:
            return None
        type_col = next((c for c in ['type', 'contract_type', 'option_type', 'right', 'call_put', 't'] if c in chain.columns), None)
        if type_col is None:
            return None
        target_type = 'call' if pos["is_call"] else 'put'
        strike_col = 'strike_price' if 'strike_price' in chain.columns else 'strike'
        row = chain[(chain[strike_col] == pos["strike_price"]) & (chain[type_col].str.lower() == target_type)]
        if row.empty:
            return None
        return self._get_option_mark(row.iloc[0])

    def manage_positions(self):
        now = datetime.now(est)
        for ticker, pos in list(self.positions.items()):
            bars = data_client.get_spy_bars(now - timedelta(days=7), now, ticker=ticker)
            if bars is None or bars.empty or len(bars) < 20:
                continue
            trend = get_trend_signal(bars, None, None)
            mark = self._get_current_mark(ticker, pos)
            if mark is None:
                print(f" [{ticker}] No live mark – skipping cycle")
                continue
            try:
                entry_dt = datetime.fromisoformat(pos["entry_time"])
            except:
                entry_dt = now
            print(f" [SCALP MONITOR] {ticker} {'CALL' if pos['is_call'] else 'PUT'} {pos['strike_price']:.1f} | Debit: {pos['debit']:.2f} → Mark: {mark:.2f}")
            if mark > pos["best_mark"]:
                pos["best_mark"] = mark
                if pos["trail_active"]:
                    pos["trail_level"] = pos["best_mark"] * 0.90
                    print(f" [{ticker}] Trail moved to {pos['trail_level']:.2f}")
            realized = None
            debit = pos["debit"]
            profit_pct = (mark - debit) / debit if debit > 0 else 0

            # HARD 40% PROFIT TARGET
            if profit_pct >= 0.40:
                realized = (mark - debit) * pos["contracts"] * 100
                print(f" [{ticker}] Hard 40% profit target hit – exiting")

            elif mark <= 0.00:
                realized = -debit * pos["contracts"] * 100

            elif profit_pct <= -0.30:
                realized = (mark - debit) * pos["contracts"] * 100
                print(f" [{ticker}] 30% loss stop hit")

            elif (pos["is_call"] and trend == "bear") or (not pos["is_call"] and trend == "bull"):
                realized = (mark - debit) * pos["contracts"] * 100
                print(f" [{ticker}] Reversal exit")

            elif trend == "chop":
                pos["chop_count"] += 1
                if pos["chop_count"] >= 1:
                    realized = (mark - debit) * pos["contracts"] * 100
            else:
                pos["chop_count"] = 0

            if realized is None and (now - entry_dt) > timedelta(minutes=10):
                realized = (mark - debit) * pos["contracts"] * 100
                print(f" [{ticker}] 10-min max hold exit")

            if realized is None and now.time() >= (time(12, 45) if getattr(self, 'is_early_close', False) else time(15, 30)):
                realized = (mark - debit) * pos["contracts"] * 100

            # Trail activation at 20% (only if <40%)
            if realized is None and profit_pct >= 0.20:
                if not pos["trail_active"]:
                    pos["trail_active"] = True
                    pos["trail_level"] = pos["best_mark"] * 0.90
                    print(f" [{ticker}] 20% profit → 10% trail activated @ {pos['trail_level']:.2f}")
                    send_webhook(f"**TRAIL ACTIVATED – {ticker}** 20% profit. 10% trail active @ ${pos['trail_level']:.2f}")
                if mark <= pos["trail_level"]:
                    realized = (mark - debit) * pos["contracts"] * 100

            if realized is not None:
                self.exit_position(ticker, pos, realized, mark)

    def exit_position(self, ticker, pos, realized_pnl, final_mark):
        self.daily_pnl += realized_pnl
        self.equity_history.append(self.equity_history[-1] + realized_pnl)
        try:
            from strategy import TradingStrategy
            main = TradingStrategy()
            main.daily_pnl += realized_pnl
            main.equity_history.append(main.equity_history[-1] + realized_pnl)
            main._save_state()
        except Exception as e:
            print(f"[UI SYNC ERROR] {e}")
        send_scalp_exit(is_call=pos["is_call"], strike=pos["strike_price"], debit=pos["debit"] * pos["contracts"] * 100,
                        pnl=realized_pnl, underlying=ticker)
        print(f" [SCALP EXIT] {ticker} → P/L ${realized_pnl:+.2f} (mark {final_mark:.2f})")
        closed = {
            "ticker": ticker,
            "is_call": pos["is_call"],
            "strike_price": pos["strike_price"],
            "debit": pos["debit"],
            "pnl": realized_pnl,
            "entry_time": pos["entry_time"],
            "exit_time": datetime.now(est).isoformat()
        }
        self.closed_trades.append(closed)
        del self.positions[ticker]
        self._save_individual_trade(closed)

if __name__ == "__main__":
    bot = ScalpStrategy()
    import time
    while True:
        bot.run_cycle()
        time.sleep(60)