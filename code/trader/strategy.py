import os, json, pytz, numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import norm
from code.data.data_client import data_client
from indicators import get_trend_signal
from discord_bot import send_entry, send_exit

EST = pytz.timezone('US/Eastern')

STATE_FILE = Path(".current_position.json")
COMPLETED_TRADES_FILE = Path("completed_trades.json")
DAILY_PNL_FILE = Path("daily_pnl.json")
EQUITY_HISTORY_FILE = Path("equity_history.json")
KNOWLEDGE_DIR = Path("data/knowledge")
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def get_approx_delta(S, K, T, r=0.04, sigma=0.20, option_type="call"):
    if T <= 0:
        return 0.5 if option_type == "call" else -0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(-d1)


class TradingStrategy:
    def __init__(self):
        from executor import Executor
        self.executor = Executor()
        self.starting_balance = float(os.getenv("STARTING_BALANCE", "100000"))
        self.daily_pnl = self._load_daily_pnl()
        self.equity_history = self._load_equity_history()
        self.positions = self._load_state()
        self.completed_trades = self._load_completed_trades()

    # Persistence helpers (unchanged from final version)
    def _load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _update_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump(self.positions, f)

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
        if total < batch_size or total % batch_size != 0:
            return
        existing_batches = [p for p in KNOWLEDGE_DIR.iterdir() if p.name.startswith("trades_batch_") and p.suffix == ".json"]
        batch_num = max([int(p.stem.split("_")[-1]) for p in existing_batches] + [0]) + 1
        batch_file = KNOWLEDGE_DIR / f"trades_batch_{batch_num:03d}.json"
        start_idx = total - batch_size
        batch_trades = self.completed_trades[start_idx:]
        with open(batch_file, 'w') as f:
            json.dump(batch_trades, f, indent=2)
        print(f"[KNOWLEDGE] Dropped {len(batch_trades)} trades into {batch_file.name}")

    def _load_daily_pnl(self):
        today = datetime.now(EST).date().isoformat()
        if DAILY_PNL_FILE.exists():
            try:
                with open(DAILY_PNL_FILE, 'r') as f:
                    data = json.load(f)
                    if data.get("date") == today:
                        return data.get("pnl", 0.0)
            except:
                pass
        return 0.0

    def _save_daily_pnl(self):
        today = datetime.now(EST).date().isoformat()
        with open(DAILY_PNL_FILE, 'w') as f:
            json.dump({"date": today, "pnl": self.daily_pnl}, f, indent=2)

    def _load_equity_history(self):
        if EQUITY_HISTORY_FILE.exists():
            try:
                with open(EQUITY_HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return [self.starting_balance]

    def _save_equity_history(self):
        with open(EQUITY_HISTORY_FILE, 'w') as f:
            json.dump(self.equity_history, f, indent=2)

    def run_cycle(self):
        now = datetime.now(EST)
        tickers = ["SPY", "QQQ", "IWM"]
        print(f"=============================================")
        print(f" CYCLE: {now.strftime('%H:%M:%S')}")
        print(f" DAILY PnL (persistent): ${self.daily_pnl:+.2f} | LIMIT: -${(self.starting_balance * 0.03):.2f}")
        print(f" OPEN POSITIONS: {len(self.positions)}")
        print(f"---------------------------------------------")

        for ticker in tickers:
            bars = data_client.get_spy_bars(now - timedelta(hours=1), now, ticker=ticker)
            if bars is None or bars.empty:
                print(f" [DATA SKIP] {ticker} – no bars data")
                continue

            trend = get_trend_signal(bars, None, None)
            price = bars['close'].iloc[-1]

            chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker)
            if chain is None or chain.empty:
                print(f" [DATA SKIP] {ticker} – option chain empty")
                continue

            if ticker not in self.positions:
                if trend != "chop":
                    print(f" [SIGNAL] {trend.upper()} on {ticker} @ {price:.2f}")
                    self._attempt_entry(trend, chain, underlying=ticker, price=price)
            else:
                self._monitor_position(ticker, chain, price)

        print(f"=============================================")

    def _attempt_entry(self, trend, chain, underlying, price):
        print(f" [ENTRY ATTEMPT] {underlying} {trend.upper()} – scanning {len(chain)} options")
        direction = "put" if trend == "bull" else "call"
        target_chain = chain[chain['option_type'] == direction].copy()
        target_chain['delta'] = pd.to_numeric(target_chain['delta'], errors='coerce')

        for idx, row in target_chain.iterrows():
            if pd.isna(row['delta']):
                target_chain.at[idx, 'delta'] = get_approx_delta(
                    price, row['strike_price'], 0.002, option_type=direction
                )

        target_chain['abs_delta'] = target_chain['delta'].abs()

        TARGET_DELTA = 0.30
        sel = target_chain.iloc[(target_chain['abs_delta'] - TARGET_DELTA).abs().argsort()[:1]]

        if sel.empty:
            print(f" [ENTRY SKIP] {underlying} – no ~30 delta strike found")
            return

        s = sel['strike_price'].values[0]
        c = round((sel['bid_price'].values[0] + sel['ask_price'].values[0]) / 2, 2)
        long_strike = s + 1 if direction == "put" else s - 1

        if c < 0.20:
            print(f" [ENTRY SKIP] {underlying} – credit {c:.2f} < $0.20")
            return

        print(f" [ENTRY] {underlying} {direction.upper()} {s}/{long_strike} @ {c:.2f} credit – just sold another tiny spread, lunch money secured, snoochie boochies")

        pos = {
            'underlying': underlying,
            'side': direction,
            'short_strike': s,
            'long_strike': long_strike,
            'entry_credit': c,
            'trailing_active': False,
            'best_credit': None,
            'entry_time': datetime.now(EST).isoformat()
        }
        self.positions[underlying] = pos
        self._update_state()
        self.executor.place_credit_spread(direction, s, long_strike, c)
        send_entry(
            is_put=(direction == "put"),
            short=s,
            long=long_strike,
            credit=c,
            underlying=underlying
        )

    def _monitor_position(self, ticker, chain, price):
        p = self.positions[ticker]
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

        print(f" [MONITOR] {ticker} {p['side'].upper()} {p['short_strike']} | Entry: {entry:.2f} | Current: {cur:.2f} | Mode: {status}")

        if cur >= entry * 2.0:
            self._close_and_clear(ticker, "STOP LOSS", cur)
            return

        if not p['trailing_active']:
            if cur <= profit_50pct_level:
                print(f" [PROFIT] 50% captured on {ticker} — flipping to trailing mode")
                p['trailing_active'] = True
                p['best_credit'] = cur
                self._update_state()
            return

        if cur < p['best_credit']:
            p['best_credit'] = cur
            self._update_state()
            print(f" [TRAIL] {ticker} ratcheted – new stop @ {cur * 1.10:.2f}")

        if cur >= p['best_credit'] * 1.10:
            print(f" [TRAIL STOP] {ticker} triggered — locking in gains")
            self._close_and_clear(ticker, "TRAILING STOP (10%)", cur)

    def _close_and_clear(self, ticker, reason, debit):
        p = self.positions[ticker]
        pnl = round((p['entry_credit'] - debit) * 100, 2)
        self.daily_pnl += pnl
        self._save_daily_pnl()

        current_equity = self.starting_balance + self.daily_pnl
        if not self.equity_history or self.equity_history[-1] != current_equity:
            self.equity_history.append(current_equity)
            self._save_equity_history()

        trade_record = {
            "underlying": p['underlying'],
            "entry_time": p.get('entry_time'),
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

        print(f" [DIARY] {ticker} {reason} | PnL: ${pnl:+.2f}")

        if len(self.completed_trades) % 30 == 0:
            self._batch_to_knowledge()

        send_exit(
            is_put=(p['side'] == "put"),
            short=p['short_strike'],
            long=p['long_strike'],
            credit=p['entry_credit'],
            pnl=pnl,
            underlying=p['underlying']
        )
        print(f" [EXIT] {ticker} {reason} | PnL: ${pnl:+.2f}")

        del self.positions[ticker]
        self._update_state()