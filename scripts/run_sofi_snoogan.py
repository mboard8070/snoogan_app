from __future__ import annotations

import json
import sys
import time as time_mod
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "trader"))

from code.trader.sofi_spread_strategy import SofiSpreadStrategy
from code.utils.telegram_notifier import send_telegram_message
from code.sentiment.news_x_scanner import scan_market_news

EST = ZoneInfo("America/New_York")
RUN_LOG = PROJECT_ROOT / "data" / "logs" / "sofi_snoogan_runner.log"


def log(msg: str) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(EST).isoformat()} {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a") as f:
        f.write(line + "\n")


def entry_message(result: dict) -> str:
    pos = result["position"]
    typ = "PUT" if pos["is_put"] else "CALL"
    return (
        f"Snoogans SOFI shadow ENTRY\n"
        f"{typ} credit spread {pos['short']:.2f}/{pos['long']:.2f} exp {pos['expiration_date']}\n"
        f"Credit: ${pos['credit'] * 100:.2f} gross | est fees: ${result.get('estimated_opening_fees', 0):.2f}\n"
        f"Premium after fees: ${result.get('net_opening_credit', 0):.2f}\n"
        f"Fee-aware max risk: ${pos['max_risk']:.2f}\n"
        f"Trend: {pos.get('trend')} | XGB: {(pos.get('xgb') or {}).get('trend')} {(pos.get('xgb') or {}).get('confidence', 0):.0%}"
    )


def exit_message(event: dict) -> str:
    typ = "PUT" if event["is_put"] else "CALL"
    return (
        f"Snoogans SOFI shadow EXIT\n"
        f"{typ} credit spread {event['short']:.2f}/{event['long']:.2f} exp {event.get('expiration_date')}\n"
        f"Reason: {event['reason']}\n"
        f"Entry credit: ${event['credit'] * 100:.2f} gross | final mark: ${event['final_mark'] * 100:.2f}\n"
        f"P/L after est fees: ${event['pnl_after_fees']:+.2f}"
    )


def news_message(context: dict) -> str:
    items = context.get("high_impact_items", [])[:3]
    lines = [
        "Snoogans market news risk scan",
        f"Impact: {context.get('highest_impact')} | net sentiment: {context.get('net_sentiment'):+}",
    ]
    for item in items:
        symbols = ",".join(item.get("symbols") or ["market"])
        lines.append(f"{symbols}: {item.get('title')}")
    return "\n".join(lines)


def main() -> None:
    strategy = SofiSpreadStrategy()
    send_telegram_message("Snoogans SOFI runner online. Shadow mode only. Watching for 3-week $1-wide SOFI spreads.")
    log("runner started")
    sent_entry_for_state = set()
    last_news_scan = None
    sent_news_ids = set()

    while True:
        now = datetime.now(EST)
        if now.weekday() >= 5 or now.time() >= time(16, 5):
            log("runner finished")
            send_telegram_message(f"Snoogans SOFI runner finished. Daily shadow P/L: ${strategy.daily_pnl:+.2f}")
            break

        try:
            if last_news_scan is None or (now - last_news_scan).total_seconds() >= 300:
                news_context = scan_market_news()
                high_ids = {item["id"] for item in news_context.get("high_impact_items", [])}
                new_high_ids = high_ids - sent_news_ids
                if new_high_ids and news_context.get("highest_impact") == "high":
                    send_telegram_message(news_message(news_context))
                    sent_news_ids.update(high_ids)
                log("news " + json.dumps({
                    "highest_impact": news_context.get("highest_impact"),
                    "net_sentiment": news_context.get("net_sentiment"),
                    "item_count": news_context.get("item_count"),
                    "x_enabled": news_context.get("x_enabled"),
                    "grok_build_x_enabled": news_context.get("grok_build_x_enabled"),
                }, sort_keys=True, default=str))
                last_news_scan = now

            for event in strategy.monitor_positions():
                send_telegram_message(exit_message(event))
                log("exit " + json.dumps(event, sort_keys=True, default=str))

            result = strategy.scan(force=False)
            if result.get("action") == "shadow_entry":
                state_key = result.get("state_key")
                if state_key not in sent_entry_for_state:
                    send_telegram_message(entry_message(result))
                    log("entry " + json.dumps(result, sort_keys=True, default=str))
                    sent_entry_for_state.add(state_key)
            else:
                log("scan " + json.dumps({"action": result.get("action"), "reason": result.get("reason")}, sort_keys=True, default=str))
        except Exception as exc:
            log(f"error {type(exc).__name__}: {exc}")

        time_mod.sleep(60)


if __name__ == "__main__":
    main()
