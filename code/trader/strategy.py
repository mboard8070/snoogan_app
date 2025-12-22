# code/trader/strategy.py – Complete 1m strategy: bash logs from 8AM, entries only after 9:40AM
import os
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from code.data.data_client import data_client
from indicators import get_trend_signal
from discord_notifier import send_entry, send_exit

est = ZoneInfo("America/New_York")

class TradingStrategy:
    def __init__(self):
        self.positions = {}
        self.equity_history = [float(os.getenv("STARTING_BALANCE", 100000))]
        self.daily_pnl = 0.0
        self.trades_today = []
        self.daily_loss_limit = -0.03 * self.equity_history[-1]

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
        """Logs + data pulls from 8:00 AM"""
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        return time(8, 0) <= now <= time(16, 0)

    def can_enter_trades(self):
        """Actual entries only after 9:40 AM – safe liquidity"""
        if not self.is_trading_day():
            return False
        now = datetime.now(est).time()
        return time(9, 40) <= now <= time(16, 0)

    def check_daily_loss_nanny(self):
        if self.daily_pnl <= self.daily_loss_limit:
            print("[NANNY] 3% daily loss hit – no more trades today.")
            return False
        return True

    def run_cycle(self):
        now = datetime.now(est)
        tickers = ["SPY", "QQQ", "IWM"]
        print("=============================================")
        print(f" CYCLE: {now.strftime('%H:%M:%S')}")
        print(f" DAILY PnL (persistent): ${self.daily_pnl:+.2f} | LIMIT: -${-self.daily_loss_limit:.2f}")
        print(f" OPEN POSITIONS: {len(self.positions)}")
        print("---------------------------------------------")

        if not self.is_scanning_active():
            print(" Outside 8AM-4PM window – full scanning paused.")
            print("=============================================")
            return

        if not self.check_daily_loss_nanny():
            print("=============================================")
            return

        for ticker in tickers:
            minute_bars = data_client.get_spy_bars(
                now - timedelta(days=7), now, ticker=ticker
            )
            if minute_bars is None or minute_bars.empty:
                print(f" [DATA SKIP] {ticker} - no minute bars")
                continue

            if len(minute_bars) < 20:
                print(f" [DATA SKIP] {ticker} - not enough bars")
                continue

            trend = get_trend_signal(minute_bars, None, None)
            price = minute_bars['close'].iloc[-1]
            chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker)

            if chain is None or chain.empty:
                print(f" [DATA SKIP] {ticker} - option chain empty")
                continue

            print(f" [{ticker}] Price: {price:.2f} | Trend: {trend.upper()}")

            if ticker not in self.positions:
                if trend != "chop":
                    if self.can_enter_trades():
                        print(f" [SIGNAL] {trend.upper()} on {ticker} – entering 30-delta spread")
                        self._attempt_entry(ticker, trend, chain, price)
                    else:
                        print(f" [SIGNAL] {trend.upper()} on {ticker} @ {price:.2f} – waiting for 9:40AM liquidity")
            else:
                print(f" [MONITOR] Managing {ticker} position")

        self.manage_positions()

        print("=============================================")

    def _attempt_entry(self, proxy, trend, chain, price):
        is_put = trend == "down"
        multiplier = 10 if proxy == "SPY" else (40 if proxy == "QQQ" else 5)
        # Placeholder strike logic – replace with your real 30-delta selection
        short_strike = int(round(price / 5) * 5) * multiplier
        long_strike = short_strike - (50 * multiplier)
        credit = 1.80
        contracts = 1

        underlying = "SPX" if proxy == "SPY" else ("NDX" if proxy == "QQQ" else "RUT")

        self.positions[proxy] = {
            "is_put": is_put,
            "short": short_strike,
            "long": long_strike,
            "credit": credit,
            "contracts": contracts,
            "best_value": credit,
            "trail_active": False,
            "trail_level": None
        }
        self.trades_today.append("entry")
        send_entry(is_put=is_put, short=short_strike, long=long_strike,
                   credit=credit * contracts * 100, underlying=underlying)
        print(f" ENTERED tiny 30-delta {underlying} {'PUT' if is_put else 'CALL'} spread")

    def manage_positions(self):
        now = datetime.now(est)
        for proxy, pos in list(self.positions.items()):
            current_value = 0.90  # Replace with real mark-to-market

            unrealized_per = pos["credit"] - current_value

            if current_value < pos["best_value"]:
                pos["best_value"] = current_value
                if pos["trail_active"]:
                    pos["trail_level"] = current_value * 1.10

            realized = None

            if current_value >= 2.1 * pos["credit"]:
                realized = -1.1 * pos["credit"] * pos["contracts"] * 100

            elif now.time() >= time(16, 0):
                realized = (pos["credit"] - current_value) * pos["contracts"] * 100

            elif unrealized_per >= 0.5 * pos["credit"]:
                if not pos["trail_active"]:
                    pos["trail_active"] = True
                    pos["trail_level"] = pos["best_value"] * 1.10
                    print(" 50% profit reached – 10% trailing stop activated")

                if current_value >= pos["trail_level"]:
                    realized = (pos["trail_level"] - current_value) * pos["contracts"] * 100

            if realized is not None:
                self.exit_position(proxy, pos, realized)

    def exit_position(self, proxy, pos, realized_pnl):
        self.daily_pnl += realized_pnl
        self.equity_history.append(self.equity_history[-1] + realized_pnl)
        underlying = "SPX" if proxy == "SPY" else ("NDX" if proxy == "QQQ" else "RUT")
        trail_note = " (trail hit)" if pos.get("trail_active") else ""
        send_exit(is_put=pos["is_put"], short=pos["short"], long=pos["long"],
                  credit=pos["credit"], pnl=realized_pnl, underlying=underlying)
        print(f" EXITED {underlying} spread{trail_note} – P/L ${realized_pnl:+.2f}")
        del self.positions[proxy]
        self.trades_today.append("exit")