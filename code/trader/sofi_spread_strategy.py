from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, time, date
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from code.data.data_client import data_client
from code.rag.stock_options_learner import StockOptionsLearner
from code.trader.indicators import get_full_indicator_set, get_trend_signal
from code.trader.sofi_xgb import predict_latest

logger = logging.getLogger(__name__)
EST = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_FILE = PROJECT_ROOT / "code" / "trader" / "sofi_spread_strategy_state.json"
CHART_CONTEXT_FILE = PROJECT_ROOT / "code" / "trader" / "sofi_chart_context.json"
NEWS_CONTEXT_FILE = PROJECT_ROOT / "data" / "news" / "latest_market_context.json"
TRADE_LOG_FILE = PROJECT_ROOT / "data" / "logs" / "sofi_spread_trades.jsonl"


class SofiSpreadStrategy:
    """SOFI-only, $1-wide vertical credit spread strategy for small-account shadow trading."""

    ticker = "SOFI"
    spread_width = 1.0
    max_contracts = 1
    max_risk_dollars = 85.0
    estimated_opening_fees = 2.26
    min_credit = 0.12
    max_leg_spread_pct = 0.20
    max_leg_spread_abs = 0.02
    min_open_interest = 100
    min_option_volume = 10
    target_dte = 21
    bias = "bullish"
    bearish_xgb_min_confidence = 0.45
    chop_veto_confidence = 0.65

    def __init__(self):
        self.learner = StockOptionsLearner()
        self.positions = {}
        self.closed_trades = []
        self.daily_pnl = 0.0
        self.last_pnl_date = date.today().isoformat()
        self.current_trends = {}
        self._load_state()

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            self._save_state()
            return
        state = json.loads(STATE_FILE.read_text())
        self.positions = state.get("positions", {})
        self.closed_trades = state.get("closed_trades", [])
        self.daily_pnl = state.get("daily_pnl", 0.0)
        self.last_pnl_date = state.get("last_pnl_date", date.today().isoformat())
        self.current_trends = state.get("current_trends", {})
        if self.last_pnl_date != date.today().isoformat():
            self.daily_pnl = 0.0
            self.last_pnl_date = date.today().isoformat()
            self._save_state()

    def _save_state(self) -> None:
        STATE_FILE.write_text(json.dumps({
            "positions": self.positions,
            "closed_trades": self.closed_trades,
            "daily_pnl": self.daily_pnl,
            "last_pnl_date": self.last_pnl_date,
            "current_trends": self.current_trends,
        }, indent=2, sort_keys=True))

    def can_enter_trades(self) -> bool:
        now = datetime.now(EST)
        if now.weekday() >= 5:
            return False
        return time(9, 45) <= now.time() < time(14, 30)

    def scan(self, force: bool = False) -> dict:
        now = datetime.now(EST)
        if self.ticker in self.positions:
            return {"action": "monitor", "position": self.positions[self.ticker]}
        shadow_write = not force
        if not force and not self.can_enter_trades():
            return {"action": "skip", "reason": "outside entry window"}

        bars = data_client.get_spy_bars(now - timedelta(hours=8), now, ticker=self.ticker, timeframe="1Min")
        if bars.empty or len(bars) < 60:
            return {"action": "skip", "reason": "insufficient bars", "bars": len(bars)}

        indicators = get_full_indicator_set(bars)
        trend = get_trend_signal(bars)
        xgb_prediction = predict_latest(bars)
        self.current_trends[self.ticker] = trend

        setup = self._choose_setup(bars, trend, indicators, xgb_prediction)
        if not setup["allow"]:
            return {
                "action": "skip",
                "reason": setup["reason"],
                "trend": trend,
                "setup": setup,
                "xgb": xgb_prediction.__dict__ if xgb_prediction else None,
                "rsi": indicators.get("rsi"),
            }
        is_put = setup["is_put"]

        expiration = self._next_expiration(now)
        chain = data_client.get_spy_option_chain(expiration, ticker=self.ticker)
        if chain.empty:
            return {"action": "skip", "reason": "empty option chain", "expiration": expiration}

        candidates = self._find_candidates(chain, is_put)
        if not candidates:
            return {"action": "skip", "reason": "no valid spread candidates", "trend": trend, "expiration": expiration}

        candidate = max(candidates, key=lambda item: (item["credit"], -item["max_risk"]))
        state_key = self.learner.state_key(
            ticker=self.ticker,
            spread_type="put_credit" if is_put else "call_credit",
            trend=trend,
            rsi=float(indicators.get("rsi", 50.0)),
            delta=candidate.get("short_delta"),
            credit=candidate["credit"],
            width=self.spread_width,
            spread_pct=candidate["spread_pct"],
        )
        decision = self.learner.decide(state_key)
        result = {**candidate, "action": "candidate", "trend": trend, "setup": setup, "xgb": xgb_prediction.__dict__ if xgb_prediction else None, "rsi": indicators.get("rsi"), "state_key": state_key, "decision": decision.__dict__}
        if not decision.should_enter:
            result["action"] = "skip"
            result["reason"] = decision.reason
            return result

        position = {
            "is_put": is_put,
            "short": candidate["short"],
            "long": candidate["long"],
            "credit": candidate["credit"],
            "contracts": 1,
            "max_risk": candidate["max_risk"],
            "expiration_date": expiration,
            "entry_time": now.isoformat(),
            "state_key": state_key,
            "trend": trend,
            "setup": setup,
            "xgb": xgb_prediction.__dict__ if xgb_prediction else None,
            "rsi": indicators.get("rsi"),
            "shadow": True,
            "news_context": setup.get("news_context"),
        }
        result["action"] = "shadow_entry" if shadow_write else "candidate"
        result["position"] = position
        if shadow_write:
            self.positions[self.ticker] = position
            self._log_trade_event({
                "event": "entry",
                "ticker": self.ticker,
                "timestamp": now.isoformat(),
                "is_put": is_put,
                "short": candidate["short"],
                "long": candidate["long"],
                "credit": candidate["credit"],
                "gross_credit_dollars": candidate.get("gross_credit_dollars"),
                "estimated_opening_fees": candidate.get("estimated_opening_fees"),
                "net_opening_credit": candidate.get("net_opening_credit"),
                "max_risk": candidate["max_risk"],
                "contracts": 1,
                "expiration_date": expiration,
                "state_key": state_key,
                "xgb": xgb_prediction.__dict__ if xgb_prediction else None,
                "setup": setup,
            })
            self._save_state()
        return result


    def _chart_context(self) -> dict:
        if not CHART_CONTEXT_FILE.exists():
            return {}
        try:
            return json.loads(CHART_CONTEXT_FILE.read_text())
        except Exception:
            return {}

    def _news_context(self) -> dict:
        if not NEWS_CONTEXT_FILE.exists():
            return {}
        try:
            context = json.loads(NEWS_CONTEXT_FILE.read_text())
        except Exception:
            return {}
        generated_at = context.get("generated_at")
        if not generated_at:
            return context
        try:
            age = datetime.now(EST) - datetime.fromisoformat(generated_at)
        except ValueError:
            return context
        if age > timedelta(minutes=30):
            context["stale"] = True
        return context

    def _choose_setup(self, bars: pd.DataFrame, trend: str, indicators: dict, xgb_prediction) -> dict:
        chart_context = self._chart_context()
        news_context = self._news_context()
        close = pd.to_numeric(bars["close"], errors="coerce").dropna()
        if len(close) < 60:
            return {"allow": False, "reason": "insufficient bars for setup", "is_put": True}

        price = float(close.iloc[-1])
        ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        recent_low = float(close.tail(78).min())
        reclaimed_ema9 = price >= ema9
        above_ema21 = price >= ema21
        above_recent_low = price >= recent_low * 1.02
        rsi = float(indicators.get("rsi", 50.0) or 50.0)

        xgb_trend = xgb_prediction.trend if xgb_prediction else None
        xgb_conf = xgb_prediction.confidence if xgb_prediction else 0.0
        xgb_strong_bear = xgb_prediction and xgb_trend == "bear" and xgb_conf >= self.bearish_xgb_min_confidence
        xgb_strong_chop = xgb_prediction and xgb_trend == "chop" and xgb_conf >= self.chop_veto_confidence

        cup_base_constructive = (above_recent_low and (reclaimed_ema9 or above_ema21) and rsi < 68)
        news_symbol_sentiment = news_context.get("symbol_sentiment", {})
        sofi_news_score = news_symbol_sentiment.get(self.ticker, 0) + news_context.get("macro_sentiment", 0)
        high_negative_news = (
            not news_context.get("stale")
            and news_context.get("highest_impact") == "high"
            and sofi_news_score < 0
        )

        if self.bias == "bullish":
            if high_negative_news:
                return {
                    "allow": False,
                    "reason": "high-impact negative news risk blocks bullish SOFI entry",
                    "is_put": True,
                    "price": price,
                    "ema9": ema9,
                    "ema21": ema21,
                    "ema50": ema50,
                    "rsi": rsi,
                    "chart_context": chart_context,
                    "news_context": news_context,
                }
            if xgb_strong_bear and not above_ema21:
                return {
                    "allow": False,
                    "reason": "bullish bias blocked by strong bearish XGBoost and weak EMA posture",
                    "is_put": True,
                    "price": price,
                    "ema9": ema9,
                    "ema21": ema21,
                    "ema50": ema50,
                    "rsi": rsi,
                    "chart_context": chart_context,
                    "news_context": news_context,
                }
            if trend in {"bull", "chop"} or cup_base_constructive:
                return {
                    "allow": True,
                    "reason": "bullish/base setup prefers put credit spread",
                    "is_put": True,
                    "price": price,
                    "ema9": ema9,
                    "ema21": ema21,
                    "ema50": ema50,
                    "rsi": rsi,
                    "cup_base_constructive": cup_base_constructive,
                    "chart_context": chart_context,
                    "news_context": news_context,
                }
            if trend == "bear":
                return {
                    "allow": False,
                    "reason": "bearish short-term trend, but bullish SOFI bias avoids call credit spread",
                    "is_put": True,
                    "price": price,
                    "ema9": ema9,
                    "ema21": ema21,
                    "ema50": ema50,
                    "rsi": rsi,
                    "chart_context": chart_context,
                    "news_context": news_context,
                }

        if xgb_strong_chop:
            return {"allow": False, "reason": "SOFI XGBoost strongly predicts chop", "is_put": trend == "bull"}
        if trend == "bull":
            return {"allow": True, "reason": "bull trend", "is_put": True}
        if trend == "bear" and xgb_prediction and xgb_trend == "bear" and xgb_conf >= self.bearish_xgb_min_confidence and price < ema21:
            return {"allow": True, "reason": "confirmed bearish breakdown", "is_put": False}
        return {"allow": False, "reason": "no confirmed setup", "is_put": True}

    def _find_candidates(self, chain: pd.DataFrame, is_put: bool) -> list[dict]:
        target_type = "put" if is_put else "call"
        opts = chain[chain["option_type"].str.lower() == target_type].copy()
        opts = opts.dropna(subset=["strike_price", "bid_price", "ask_price"])
        opts = opts.sort_values("strike_price")
        if opts.empty:
            return []

        delta = opts["delta"].abs() if "delta" in opts else pd.Series(index=opts.index, dtype=float)
        opts = opts[(delta >= 0.20) & (delta <= 0.45)] if not delta.empty else opts
        candidates = []
        for _, short in opts.iterrows():
            short_strike = float(short["strike_price"])
            long_strike = short_strike - self.spread_width if is_put else short_strike + self.spread_width
            long_rows = chain[(chain["option_type"].str.lower() == target_type) & ((chain["strike_price"] - long_strike).abs() < 0.001)]
            if long_rows.empty:
                continue
            for _, long in long_rows.iterrows():
                candidate = self._candidate(short, long, is_put)
                if candidate:
                    candidates.append(candidate)
        return candidates

    def _candidate(self, short: pd.Series, long: pd.Series, is_put: bool) -> Optional[dict]:
        short_bid = float(short.get("bid_price") or 0)
        short_ask = float(short.get("ask_price") or 0)
        long_bid = float(long.get("bid_price") or 0)
        long_ask = float(long.get("ask_price") or 0)
        if min(short_bid, short_ask, long_bid, long_ask) <= 0:
            return None
        short_mid = (short_bid + short_ask) / 2
        long_mid = (long_bid + long_ask) / 2
        short_spread_abs = short_ask - short_bid
        long_spread_abs = long_ask - long_bid
        short_spread_pct = short_spread_abs / short_mid if short_mid else 1
        long_spread_pct = long_spread_abs / long_mid if long_mid else 1
        spread_pct = max(short_spread_pct, long_spread_pct)
        short_liquid = short_spread_pct <= self.max_leg_spread_pct or short_spread_abs <= self.max_leg_spread_abs
        long_liquid = long_spread_pct <= self.max_leg_spread_pct or long_spread_abs <= self.max_leg_spread_abs
        if not (short_liquid and long_liquid):
            return None
        if float(short.get("open_interest") or 0) < self.min_open_interest:
            return None
        if float(short.get("volume") or 0) < self.min_option_volume:
            return None

        credit = round(short_mid - long_mid, 2)
        gross_credit = round(credit * 100, 2)
        net_opening_credit = round(gross_credit - self.estimated_opening_fees, 2)
        max_risk = round((self.spread_width - credit) * 100 + self.estimated_opening_fees, 2)
        if credit < self.min_credit or max_risk > self.max_risk_dollars or max_risk <= 0:
            return None
        return {
            "ticker": self.ticker,
            "type": "put_credit" if is_put else "call_credit",
            "short": float(short["strike_price"]),
            "long": float(long["strike_price"]),
            "credit": credit,
            "gross_credit_dollars": gross_credit,
            "estimated_opening_fees": self.estimated_opening_fees,
            "net_opening_credit": net_opening_credit,
            "max_risk": max_risk,
            "short_delta": float(short.get("delta")) if pd.notna(short.get("delta")) else None,
            "spread_pct": round(spread_pct, 4),
            "contracts": 1,
        }

    def _next_expiration(self, now: datetime) -> str:
        expirations = self._available_expirations()
        if not expirations:
            exp = now.date() + timedelta(days=self.target_dte)
            while exp.weekday() >= 5:
                exp += timedelta(days=1)
            return exp.isoformat()

        target = now.date() + timedelta(days=self.target_dte)
        future_expirations = [exp for exp in expirations if exp >= now.date()]
        if not future_expirations:
            return expirations[-1].isoformat()
        return min(future_expirations, key=lambda exp: abs((exp - target).days)).isoformat()

    def _available_expirations(self) -> list[date]:
        chain = data_client._request_json(f"https://api.tastytrade.com/option-chains/{self.ticker}/nested")
        dates = []
        for item in chain.get("data", {}).get("items", []):
            for expiration in item.get("expirations", []):
                expiration_date = expiration.get("expiration-date")
                if expiration_date:
                    dates.append(date.fromisoformat(expiration_date))
        return sorted(set(dates))


    def monitor_positions(self) -> list[dict]:
        events = []
        now = datetime.now(EST)
        for ticker, pos in list(self.positions.items()):
            current_value = self._get_current_spread_mark(pos)
            if current_value is None:
                continue
            credit = float(pos["credit"])
            contracts = int(pos.get("contracts", 1))
            pnl = round((credit - current_value) * contracts * 100 - self.estimated_opening_fees, 2)
            pos["current_value"] = current_value
            pos["unrealized_pnl_after_fees"] = pnl
            exit_reason = None
            if current_value <= max(0.05, credit * 0.45):
                exit_reason = "profit_target"
            elif current_value >= credit * 1.65:
                exit_reason = "stop_loss"
            elif now.time() >= time(15, 45):
                exit_reason = "time_exit"
            if exit_reason:
                events.append(self.exit_position(ticker, pos, current_value, pnl, exit_reason))
        self._save_state()
        return events

    def _get_current_spread_mark(self, pos: dict) -> Optional[float]:
        chain = data_client.get_spy_option_chain(pos["expiration_date"], ticker=self.ticker)
        if chain.empty:
            return None
        target_type = "put" if pos["is_put"] else "call"
        short = chain[(chain["option_type"].str.lower() == target_type) & ((chain["strike_price"] - pos["short"]).abs() < 0.001)]
        long = chain[(chain["option_type"].str.lower() == target_type) & ((chain["strike_price"] - pos["long"]).abs() < 0.001)]
        if short.empty or long.empty:
            return None
        short_mid = (float(short.iloc[0]["bid_price"] or 0) + float(short.iloc[0]["ask_price"] or 0)) / 2
        long_mid = (float(long.iloc[0]["bid_price"] or 0) + float(long.iloc[0]["ask_price"] or 0)) / 2
        if short_mid <= 0 or long_mid <= 0:
            return None
        return round(short_mid - long_mid, 2)

    def exit_position(self, ticker: str, pos: dict, final_mark: float, pnl_after_fees: float, reason: str) -> dict:
        event = {
            "event": "exit",
            "ticker": ticker,
            "timestamp": datetime.now(EST).isoformat(),
            "reason": reason,
            "is_put": pos["is_put"],
            "short": pos["short"],
            "long": pos["long"],
            "credit": pos["credit"],
            "final_mark": final_mark,
            "contracts": pos.get("contracts", 1),
            "estimated_opening_fees": self.estimated_opening_fees,
            "pnl_after_fees": pnl_after_fees,
            "expiration_date": pos.get("expiration_date"),
            "state_key": pos.get("state_key"),
        }
        self.daily_pnl += pnl_after_fees
        self.closed_trades.append(event)
        if pos.get("state_key"):
            self.learner.record_trade_outcome(pos["state_key"], pnl_after_fees, float(pos.get("max_risk", 1)))
        self.positions.pop(ticker, None)
        self._log_trade_event(event)
        self._save_state()
        return event

    def _log_trade_event(self, event: dict) -> None:
        TRADE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRADE_LOG_FILE.open("a") as f:
            f.write(json.dumps(event, sort_keys=True, default=str) + "\n")

    def get_status(self) -> dict:
        return {
            "ticker": self.ticker,
            "open_positions": len(self.positions),
            "positions": self.positions,
            "daily_pnl": self.daily_pnl,
            "current_trends": self.current_trends,
            "learner_stats": self.learner.stats,
        }
