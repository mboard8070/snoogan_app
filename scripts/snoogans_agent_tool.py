from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "trader"))

from code.data.data_client import data_client
from code.trader.indicators import get_full_indicator_set, get_trend_signal
from code.rag.snoogans_brain import SnoogansBrain


EST = ZoneInfo("America/New_York")


def build_snapshot(symbol: str, lookback_hours: int) -> dict:
    now = datetime.now(EST)
    bars = data_client.get_spy_bars(now - timedelta(hours=lookback_hours), now, ticker=symbol, timeframe="1Min")
    if bars.empty:
        return {"symbol": symbol, "error": "no bars returned"}

    indicators = get_full_indicator_set(bars)
    trend = get_trend_signal(bars)
    mark = data_client.get_underlying_mark(symbol)
    return {
        "symbol": symbol,
        "mark": mark,
        "trend": trend,
        "bars": len(bars),
        "last_bar": bars.index[-1].isoformat(),
        "price": indicators.get("price"),
        "rsi": indicators.get("rsi"),
        "macd_line": indicators.get("macd_line"),
        "macd_signal": indicators.get("macd_signal"),
        "atr": indicators.get("atr"),
        "trend_strength": indicators.get("trend_strength"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Snoogans live Tastytrade snapshot tool")
    parser.add_argument("symbols", nargs="*", default=["SOFI", "LUMN", "LUNR"])
    parser.add_argument("--lookback-hours", type=int, default=8)
    parser.add_argument("--ask-brain", action="store_true", help="Ask the local Snoogans model to comment on the snapshot")
    args = parser.parse_args()

    snapshots = [build_snapshot(symbol.upper(), args.lookback_hours) for symbol in args.symbols]
    for item in snapshots:
        if "error" in item:
            print(f"{item['symbol']}: {item['error']}")
            continue
        print(
            f"{item['symbol']}: mark={item['mark']} trend={item['trend']} "
            f"bars={item['bars']} rsi={item['rsi']} atr={item['atr']} "
            f"last_bar={item['last_bar']}"
        )

    if args.ask_brain:
        prompt = "Review this live Tastytrade market snapshot. Do not recommend live orders. Identify watchlist quality, risk, and what would invalidate a setup.\n\n"
        prompt += "\n".join(str(item) for item in snapshots)
        brain = SnoogansBrain()
        print("\nSnoogans brain:\n")
        print(brain.ask(prompt))


if __name__ == "__main__":
    main()
