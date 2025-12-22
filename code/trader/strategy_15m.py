import os, json, pytz, numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import norm
from code.data.data_client import data_client
from indicators import get_trend_signal
from discord_notifier import send_entry, send_exit
from strategy import TradingStrategy  # Inherits holidays, nanny, etc.

EST = pytz.timezone('US/Eastern')

class TradingStrategy15m(TradingStrategy):
    def __init__(self):
        super().__init__()
        self.prefix = "[15m] "

    def run_cycle(self):
        now = datetime.now(EST)
        tickers = ["SPY", "QQQ", "IWM"]
        print(f"{self.prefix}=============================================")
        print(f"{self.prefix} CYCLE: {now.strftime('%H:%M:%S')}")
        print(f"{self.prefix} DAILY PnL (persistent): ${self.daily_pnl:+.2f} | LIMIT: -${(-self.daily_loss_limit):.2f}")
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

    def _attempt_entry(self, trend, chain, underlying, price):
        # Your existing entry logic here – when you successfully enter:
        # Determine real underlying name and multiplier
        underlying_real = "SPX" if underlying == "SPY" else ("NDX" if underlying == "QQQ" else "RUT")
        multiplier = 10 if underlying == "SPY" else (40 if underlying == "QQQ" else 5)

        is_put = trend == "down"  # puts on downtrend, calls on uptrend
        short_strike = 5500  # placeholder – pull from your actual delta targeting
        long_strike = short_strike - 50  # placeholder
        credit = 1.80  # placeholder – your real credit received
        contracts = 1  # tiny sizing

        # After successful order fill:
        self.positions[underlying] = {
            "is_put": is_put,
            "short": short_strike * multiplier,
            "long": long_strike * multiplier,
            "credit": credit,
            "entry_time": datetime.now(EST),
            "pnl": 0.0
        }
        self.trades_today.append("entry")

        send_entry(
            is_put=is_put,
            short=short_strike * multiplier,
            long=long_strike * multiplier,
            credit=credit * contracts * 100,
            underlying=underlying_real
        )
        print(f"{self.prefix} Entered tiny {underlying_real} {'PUT' if is_put else 'CALL'} spread – lunch money secured.")

    def _monitor_position(self, ticker, chain, price):
        pos = self.positions[ticker]
        # Your existing monitoring logic – calculate current spread value/mark
        current_value = 0.90  # placeholder – real bid on the spread

        pnl_per = pos["credit"] - current_value
        realized = None

        # 50% profit target
        if pnl_per >= 0.5 * pos["credit"]:
            realized = pnl_per * 100
        # 2.1x stop loss
        elif current_value >= 2.1 * pos["credit"]:
            realized = -1.1 * pos["credit"] * 100
        # 4:00 PM cutoff
        elif datetime.now(EST).time() >= datetime.strptime("16:00", "%H:%M").time():
            realized = pnl_per * 100

        if realized is not None:
            self.daily_pnl += realized
            self.equity_history.append(self.equity_history[-1] + realized)

            underlying_real = "SPX" if ticker == "SPY" else ("NDX" if ticker == "QQQ" else "RUT")
            send_exit(
                is_put=pos["is_put"],
                short=pos["short"],
                long=pos["long"],
                credit=pos["credit"],
                pnl=realized,
                underlying=underlying_real
            )
            print(f"{self.prefix} Exited {underlying_real} spread – P/L ${realized:+.2f}. Theta doin' work.")
            del self.positions[ticker]
            self.trades_today.append("exit")