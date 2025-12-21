import os, json, pytz, numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import norm
from code.data.data_client import data_client
from indicators import get_trend_signal
from discord_bot import send_entry, send_exit

EST = pytz.timezone('US/Eastern')
from strategy import TradingStrategy

class TradingStrategy15m(TradingStrategy):
    def __init__(self):
        super().__init__()
        self.prefix = "[15m] "

    def run_cycle(self):
        now = datetime.now(EST)
        tickers = ["SPY", "QQQ", "IWM"]
        print(f"{self.prefix}=============================================")
        print(f"{self.prefix} CYCLE: {now.strftime('%H:%M:%S')}")
        print(f"{self.prefix} DAILY PnL (persistent): ${self.daily_pnl:+.2f} | LIMIT: -${(self.starting_balance * 0.03):.2f}")
        print(f"{self.prefix} OPEN POSITIONS: {len(self.positions)}")
        print(f"{self.prefix}---------------------------------------------")

        for ticker in tickers:
            minute_bars = data_client.get_spy_bars(
                now - timedelta(days=7),
                now,
                ticker=ticker
            )
            if minute_bars is None or minute_bars.empty:
                print(f"{self.prefix} [DATA SKIP] {ticker} - no minute bars")
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
                print(f"{self.prefix} [DATA SKIP] {ticker} - option chain empty")
                continue

            if ticker not in self.positions:
                if trend != "chop":
                    print(f"{self.prefix} [SIGNAL] {trend.upper()} on {ticker} @ {price:.2f} (15m chart)")
                    self._attempt_entry(trend, chain, underlying=ticker, price=price)
            else:
                self._monitor_position(ticker, chain, price)

        print(f"{self.prefix}=============================================")
