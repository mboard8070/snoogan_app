import os, json, pytz, numpy as np, pandas as pd, requests
from datetime import time, datetime, timedelta
from pathlib import Path
from scipy.stats import norm
from data_client import data_client
from indicators import get_trend_signal

EST = pytz.timezone('US/Eastern')
STATE_FILE = Path(".current_position.json")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")

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

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    d = json.load(f)
                    if 'strike' in d: d['short_strike'] = d.pop('strike')
                    return d
            except: return None
        return None

    def _update_state(self):
        if self.position:
            with open(STATE_FILE, 'w') as f: json.dump(self.position, f)

    def send_alert(self, msg):
        if not DISCORD_WEBHOOK:
            print(" [BASH] ALERT ERROR: No Discord Webhook found.")
            return False
        try:
            print(f" [BASH] Sending Discord Alert...")
            r = requests.post(DISCORD_WEBHOOK, json={"content": f"🚀 **Snoogans:** {msg}"}, timeout=8)
            return r.status_code in [200, 204]
        except: return False

    def run_cycle(self):
        now = datetime.now(EST)
        bars = data_client.get_spy_bars(now - timedelta(hours=1), now)
        chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"))
        if bars is None or bars.empty or chain is None: return

        self.last_spy_price = bars['close'].iloc[-1]
        trend = get_trend_signal(bars, None, None)

        print(f"=============================================")
        print(f" CYCLE: {now.strftime('%H:%M:%S')} | SPY: {self.last_spy_price:.4f}")
        print(f" DAILY PnL: ${self.daily_pnl:+.2f} | LIMIT: -${(self.starting_balance * 0.03):.2f}")
        print(f"---------------------------------------------")

        # ALERT RETRY LOGIC
        if self.position:
            if not self.position.get('alert_sent'):
                if self.send_alert(f"ENTRY: {self.position['side'].upper()} {self.position['short_strike']}"):
                    self.position['alert_sent'] = True; self._update_state()
            
            if self.position.get('pending_exit_msg'):
                if self.send_alert(self.position['pending_exit_msg']):
                    if STATE_FILE.exists(): os.remove(STATE_FILE)
                    self.position = None; return

        # RESTORED SIGNAL OUTPUT
        if self.position is None:
            print(f" [BASH] SIGNAL: {trend.upper()}")
            if trend != "chop": 
                self._attempt_entry(trend, chain)
        else:
            self._monitor_position(chain)
            
        print(f"=============================================")

    def _attempt_entry(self, trend, chain):
        if STATE_FILE.exists(): return
        direction = "put" if trend == "bull" else "call"
        target_chain = chain[chain['option_type'] == direction].copy()
        target_chain['delta'] = pd.to_numeric(target_chain['delta'], errors='coerce')
        
        for idx, row in target_chain.iterrows():
            if pd.isna(row['delta']):
                target_chain.at[idx, 'delta'] = get_approx_delta(self.last_spy_price, row['strike_price'], 0.002, option_type=direction)

        target_chain['abs_delta'] = target_chain['delta'].abs()
        sel = target_chain.iloc[(target_chain['abs_delta'] - 0.30).abs().argsort()[:1]]
        
        if not sel.empty:
            s, c = sel['strike_price'].values[0], round((sel['bid_price'].values[0] + sel['ask_price'].values[0]) / 2, 2)
            if c >= 0.20:
                self.position = {'side': direction, 'short_strike': s, 'entry_credit': c, 'alert_sent': False}
                self._update_state()
                self.executor.place_credit_spread(direction, s, s+1, c)

    def _monitor_position(self, chain):
        p = self.position
        row = chain[(chain['option_type'] == p['side']) & (chain['strike_price'] == p['short_strike'])]
        if row.empty: return
        cur = (row['bid_price'].values[0] + row['ask_price'].values[0]) / 2
        print(f" [MONITOR] {p['side'].upper()} {p['short_strike']} | Entry: {p['entry_credit']} | Current: {cur:.2f}")

        if cur <= (p['entry_credit'] * 0.50): self._close_and_clear("TAKE PROFIT", cur)
        elif cur >= (p['entry_credit'] * 1.50): self._close_and_clear("STOP LOSS", cur)

    def _close_and_clear(self, reason, debit):
        pnl = round((self.position['entry_credit'] - debit) * 100, 2)
        self.daily_pnl += pnl
        self.position['pending_exit_msg'] = f"EXIT: {reason} | PnL: ${pnl:+.2f}"
        self._update_state()