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
from code.trader.sofi_xgb import train_model

CACHE_DIR = PROJECT_ROOT / "data" / "market_cache"
CACHE_PATH = CACHE_DIR / "SOFI_5Min.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SOFI-specific XGBoost directional model")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--horizon-bars", type=int, default=18, help="18 five-minute bars = 90 minutes")
    parser.add_argument("--threshold-pct", type=float, default=0.35)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists() and not args.refresh:
        df = __import__("pandas").read_parquet(CACHE_PATH)
    else:
        now = datetime.now(ZoneInfo("America/New_York"))
        df = data_client.get_spy_bars(now - timedelta(days=args.days), now, ticker="SOFI", timeframe="5Min")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = __import__("pandas").to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df.to_parquet(CACHE_PATH)
    metadata = train_model(df, horizon_bars=args.horizon_bars, threshold_pct=args.threshold_pct)
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
