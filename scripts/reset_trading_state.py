from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / "variables.env")
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "100000"))

STATE_FILES = [
    PROJECT_ROOT / "code" / "strategy_state.json",
    PROJECT_ROOT / "strategy_15m_state.json",
    PROJECT_ROOT / "code" / "strategy_15m_state.json",
    PROJECT_ROOT / "code" / "trader" / "scalp_strategy_state.json",
]

POSITION_FILES = [
    PROJECT_ROOT / "code" / "positions.json",
    PROJECT_ROOT / "code" / "positions_15m.json",
]


def fresh_state() -> dict:
    return {
        "positions": {},
        "equity_history": [STARTING_BALANCE],
        "daily_pnl": 0.0,
        "last_pnl_date": date.today().isoformat(),
        "closed_trades": [],
        "current_trends": {},
        "trades_today": [],
    }


def main() -> None:
    for path in STATE_FILES:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fresh_state(), indent=4))
        print(f"reset {path.relative_to(PROJECT_ROOT)}")

    for path in POSITION_FILES:
        path.write_text("{}")
        print(f"reset {path.relative_to(PROJECT_ROOT)}")

    (PROJECT_ROOT / "shared_equity.json").write_text(
        json.dumps({"equity": STARTING_BALANCE, "last_updated": date.today().isoformat()}, indent=2)
    )
    (PROJECT_ROOT / "shared_equity_history.json").write_text(json.dumps([STARTING_BALANCE]))
    print("reset shared equity")


if __name__ == "__main__":
    main()
