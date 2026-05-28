from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "trader"))

from code.trader.sofi_spread_strategy import SofiSpreadStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="SOFI spread shadow scanner")
    parser.add_argument("--force", action="store_true", help="Scan outside entry hours")
    args = parser.parse_args()
    strategy = SofiSpreadStrategy()
    result = strategy.scan(force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
