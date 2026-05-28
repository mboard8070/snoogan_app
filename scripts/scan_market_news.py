from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.sentiment.news_x_scanner import scan_market_news


if __name__ == "__main__":
    print(json.dumps(scan_market_news(), indent=2, sort_keys=True))
