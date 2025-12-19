import os, json, pytz, numpy as np, pandas as pd
from datetime import time, datetime, timedelta
from pathlib import Path
from scipy.stats import norm
from code.data.data_client import data_client
from indicators import get_trend_signal
from discord_bot import send_entry, send_exit

EST = pytz.timezone('US/Eastern')
STATE_FILE = Path(".current_position.json")
COMPLETED_TRADES_FILE = Path("completed_trades.json")
KNOWLEDGE_DIR = Path("data/knowledge")
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)  # Make sure folder exists

def get_approx_delta(S, K, T, r=0.04, sigma=0.20, option_type="call"):
    if T <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1

class TradingStrategy:
    def __init__(self):
        from executor import Executor
        self.executor = Executor()
        self.daily_pnl = 0.0
        self.starting_balance = float(os.getenv("STARTING_BALANCE", "100000"))
        self.position = self._load_state()
        self.completed_trades = self._load_completed_trades()

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    d = json.load(f)
                    if 'strike' in d: d['short_strike'] = d.pop('strike')
                    if 'trailing_active' not in d:
                        d['trailing_active'] = False
                    if 'best_credit' not in d:
                        d['best_credit'] = None
                    return d
            except:
                return None
        return None

    def _load_completed_trades(self):
        if COMPLETED_TRADES_FILE.exists():
            try:
                with open(COMPLETED_TRADES_FILE, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_completed_trades(self):
        with open(COMPLETED_TRADES_FILE, 'w') as f:
            json.dump(self.completed_trades, f, indent=2)

    def _batch_to_knowledge(self):
        batch_size = 30
        total = len(self.completed_trades)
        if total < batch_size:
            return

        # Find next batch number
        existing_batches = [p for p in KNOWLEDGE_DIR.iterdir() if p.name.startswith("trades_batch_") and p.suffix == ".json"]
        batch_num = max([int(p.stem.split("_")[-1]) for p in existing_batches] + [0]) + 1
        batch_file = KNOWLEDGE_DIR / f"trades_batch_{batch_num:03d}.json"

        # Take last 30 (or whatever multiple)
        start_idx = (total // batch_size) * batch_size - batch_size
        batch_trades = self.completed_trades[start_idx:]

        with open(batch_file, 'w') as f:
            json.dump(batch_trades, f, indent=2)

        print(f" [KNOWLEDGE] Dropped {len(batch_trades)} trades into {batch_file.name} – RAG brain food secured")
        # Optional: clear old ones if you want rolling batches only
        # self.completed_trades = self.completed_trades[-batch_size:]  # keep last batch in master
        # self._save_completed_trades()

    def _update_state(self):
        if self.position:
            with open(STATE_FILE, 'w') as f:
                json.dump(self.position, f)
        elif STATE_FILE.exists():
            os.remove(STATE_FILE)

    def run_cycle(self):
        now = datetime.now(EST)
        bars = data_client.get_spy_bars(now - timedelta(hours=1), now)
        chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"))
        if bars is None or bars.empty or chain is None:
            return

        self.last_spy_price = bars['close'].iloc[-1]
        trend = get_trend_signal(bars, None, None)

        print(f"=============================================")
        print(f" CYCLE: {now.strftime('%H:%M:%S')} | SPY: {self.last_spy_price:.4f}")
        print(f" DAILY PnL: ${self.daily_pnl:+.2f} | LIMIT: -${(self.starting_balance * 0.03):.2f}")
        print(f"---------------------------------------------")

        if self.position is None:
            print(f" [BASH] SIGNAL: {trend.upper()}")
            if trend != "chop":
                self._attempt_entry(trend, chain)
        else:
            self._monitor_position(chain)

        print(f"=============================================")

    def _attempt_entry(self, trend, chain):
        if self.position is not None or STATE_FILE.exists():
            return

        direction = "put" if trend == "bull" else "call"
        target_chain = chain[chain['option_type'] == direction].copy()
        target_chain['delta'] = pd.to_numeric(target_chain['delta'], errors='coerce')

        for idx, row in target_chain.iterrows():
            if pd.isna(row['delta']):
                target_chain.at[idx, 'delta'] = get_approx_delta(
                    self.last_spy_price, row['strike_price'], 0.002, option_type=direction
                )

        target_chain['abs_delta'] = target_chain['delta'].abs()
        sel = target_chain.iloc[(target_chain['abs_delta'] - 0.30).abs().argsort()[:1]]

        if not sel.empty:
            s = sel['strike_price'].values[0]
            c = round((sel['bid_price'].values[0] + sel['ask_price'].values[0]) / 2, 2)
            long_strike = s + 1 if direction == "put" else s - 1

            if c >= 0.20:
                self.position = {
                    'side': direction,
                    'short_strike': s,
                    'long_strike': long_strike,
                    'entry_credit': c,
                    'trailing_active': False,
                    'best_credit': None,
                    'entry_time': datetime.now(EST).isoformat()
                }
                self._update_state()

                self.executor.place_credit_spread(direction, s, long_strike, c)

                send_entry(
                    is_put=(direction == "put"),
                    short=s,
                    long=long_strike,
                    credit=c
                )

    def _monitor_position(self, chain):
        p = self.position
        row = chain[
            (chain['option_type'] == p['side']) &
            (chain['strike_price'] == p['short_strike'])
        ]
        if row.empty:
            return

        cur = (row['bid_price'].values[0] + row['ask_price'].values[0]) / 2
        entry = p['entry_credit']
        profit_50pct_level = entry * 0.50

        status = "TRAILING ACTIVE" if p['trailing_active'] else "50% PROFIT TARGET"
        print(f" [MONITOR] {p['side'].upper()} {p['short_strike']} | Entry: {entry:.2f} | Current: {cur:.2f} | Mode: {status}")

        if cur >= entry * 1.50:
            self._close_and_clear("STOP LOSS", cur)
            return

        if not p['trailing_active']:
            if cur <= profit_50pct_level:
                print(f" [PROFIT] 50% profit hit @ {cur:.2f} — hard take-profit DISABLED")
                print(f" [TRAIL] Switching to 10% trailing stop. New best: {cur:.2f} | Stop level: {cur * 1.10:.2f}")
                p['trailing_active'] = True
                p['best_credit'] = cur
                self._update_state()
            return

        if cur < p['best_credit']:
            old_stop = p['best_credit'] * 1.10
            p['best_credit'] = cur
            new_stop = cur * 1.10
            self._update_state()
            print(f" [TRAIL] Theta still decaying – new low: {cur:.2f} | Trailing stop ratcheted to {new_stop:.2f} (was {old_stop:.2f})")

        trail_stop_price = p['best_credit'] * 1.10
        if cur >= trail_stop_price:
            print(f" [TRAIL STOP HIT] Spread widened to {cur:.2f} ≥ {trail_stop_price:.2f} — locking in profits")
            self._close_and_clear("TRAILING STOP (10%)", cur)

    def _close_and_clear(self, reason, debit):
        p = self.position
        pnl = round((p['entry_credit'] - debit) * 100, 2)
        self.daily_pnl += pnl

        trade_record = {
            "entry_time": p.get('entry_time', datetime.now(EST).isoformat()),
            "exit_time": datetime.now(EST).isoformat(),
            "side": p['side'],
            "short_strike": p['short_strike'],
            "long_strike": p['long_strike'],
            "entry_credit": p['entry_credit'],
            "exit_debit": debit,
            "pnl": pnl,
            "reason": reason,
            "trailing_used": p.get('trailing_active', False)
        }

        self.completed_trades.append(trade_record)
        self._save_completed_trades()
        print(f" [DIARY] Trade logged | {reason} | PnL: ${pnl:+.2f} | Total trades: {len(self.completed_trades)}")

        # Check for batch drop
        if len(self.completed_trades) % 30 == 0:
            self._batch_to_knowledge()

        send_exit(
            is_put=(p['side'] == "put"),
            short=p['short_strike'],
            long=p['long_strike'],
            credit=p['entry_credit'],
            pnl=pnl
        )

        print(f" [EXIT] {reason} | PnL: ${pnl:+.2f}")
        self.position = None
        self._update_state()