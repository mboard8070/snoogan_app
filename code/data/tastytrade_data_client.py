from __future__ import annotations

import asyncio
import json
import math
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import websockets
from dotenv import load_dotenv


API_BASE = "https://api.tastytrade.com"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TASTY_ENV_FALLBACK = Path("/home/mboard76/tastytrade-live-data/.env")


def _load_tasty_env() -> None:
    load_dotenv(PROJECT_ROOT / "variables.env")
    if TASTY_ENV_FALLBACK.exists():
        load_dotenv(TASTY_ENV_FALLBACK, override=False)


def _to_float(value: Any) -> float | None:
    if value is None or value == "" or value == "NaN":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def _timestamp(value: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("America/New_York")
    return ts.tz_convert("America/New_York")


class TastytradeDataClient:
    """Read-only tastytrade market data adapter for Snoogans strategies."""

    def __init__(self):
        _load_tasty_env()
        self.client_secret = os.getenv("TASTYTRADE_CLIENT_SECRET", "").strip()
        self.refresh_token = os.getenv("TASTYTRADE_REFRESH_TOKEN", "").strip()
        self.scopes = os.getenv("TASTYTRADE_SCOPES", "read trade openid").replace(",", " ")
        self.starting_balance = float(os.getenv("STARTING_BALANCE", "100000"))
        self.contract_size = int(os.getenv("CONTRACT_SIZE", "1"))
        self.mode = os.getenv("MODE", "internal")
        self._access_token: str | None = None
        self._quote_token: dict | None = None

        if not self.client_secret or not self.refresh_token:
            raise ValueError("Tastytrade credentials missing. Set TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN.")

    def _request_json(self, url: str, *, data: bytes | None = None, method: str = "GET") -> dict:
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if not url.endswith("/oauth/token"):
            headers["Authorization"] = f"Bearer {self._get_access_token()}"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.loads(response.read().decode())

    def _get_access_token(self) -> str:
        if self._access_token:
            return self._access_token

        form = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "scope": self.scopes,
            }
        ).encode()
        payload = self._request_json(f"{API_BASE}/oauth/token", data=form, method="POST")
        self._access_token = payload["access_token"]
        return self._access_token

    def _get_quote_token(self) -> dict:
        if self._quote_token:
            return self._quote_token
        self._quote_token = self._request_json(f"{API_BASE}/api-quote-tokens")["data"]
        return self._quote_token

    def get_underlying_mark(self, symbol: str = "SPY") -> float | None:
        query = urllib.parse.urlencode({"equity": symbol})
        payload = self._request_json(f"{API_BASE}/market-data/by-type?{query}")
        items = payload.get("data", {}).get("items", [])
        if not items:
            return None
        item = items[0]
        for key in ("mark", "mid", "last", "last-mkt", "close"):
            value = _to_float(item.get(key))
            if value and value > 0:
                return value
        return None

    def get_spy_bars(
        self,
        start: datetime,
        end: datetime,
        ticker: str = "SPY",
        timeframe: str = "1Min",
    ) -> pd.DataFrame:
        interval = self._candle_interval(timeframe)
        rows = asyncio.run(self._fetch_candles(f"{ticker}{{={interval}}}", start, end))
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("America/New_York")
        df = df.set_index("timestamp").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]
        start_ts = _timestamp(start)
        end_ts = _timestamp(end)
        return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    def get_spy_option_chain(self, expiration_date: str, ticker: str = "SPY") -> pd.DataFrame:
        chain = self._request_json(f"{API_BASE}/option-chains/{ticker}/nested")
        items = chain.get("data", {}).get("items", [])
        if not items:
            return pd.DataFrame()

        strikes: list[dict] = []
        for chain_item in items:
            for expiration in chain_item.get("expirations", []):
                if expiration.get("expiration-date") == expiration_date:
                    strikes.extend(expiration.get("strikes", []))

        if not strikes:
            return pd.DataFrame()

        option_symbols = []
        metadata: dict[str, dict] = {}
        for strike in strikes:
            strike_price = _to_float(strike.get("strike-price"))
            for option_type in ("call", "put"):
                symbol = strike.get(option_type)
                if not symbol:
                    continue
                option_symbols.append(symbol)
                metadata[symbol] = {"strike_price": strike_price, "option_type": option_type}

        quote_map = self._get_equity_option_quotes(option_symbols)
        rows = []
        for symbol in option_symbols:
            quote = quote_map.get(symbol, {})
            meta = metadata[symbol]
            rows.append(
                {
                    "symbol": symbol,
                    "strike_price": meta["strike_price"],
                    "option_type": meta["option_type"],
                    "bid_price": _to_float(quote.get("bid")),
                    "ask_price": _to_float(quote.get("ask")),
                    "delta": _to_float(quote.get("delta")),
                    "gamma": _to_float(quote.get("gamma")),
                    "theta": _to_float(quote.get("theta")),
                    "vega": _to_float(quote.get("vega")),
                    "implied_volatility": _to_float(quote.get("volatility")),
                    "open_interest": _to_float(quote.get("open-interest")),
                    "volume": _to_float(quote.get("volume")),
                }
            )
        return pd.DataFrame(rows)

    def get_spx_option_chain(self, expiration_date: str) -> pd.DataFrame:
        return self.get_spy_option_chain(expiration_date, ticker="SPY")

    def _get_equity_option_quotes(self, option_symbols: list[str]) -> dict[str, dict]:
        quotes: dict[str, dict] = {}
        for idx in range(0, len(option_symbols), 80):
            chunk = option_symbols[idx : idx + 80]
            query = urllib.parse.urlencode({"equity-option": chunk}, doseq=True)
            payload = self._request_json(f"{API_BASE}/market-data/by-type?{query}")
            for item in payload.get("data", {}).get("items", []):
                quotes[item["symbol"]] = item
        return quotes

    def _candle_interval(self, timeframe: str) -> str:
        mapping = {
            "1Min": "m",
            "5Min": "5m",
            "15Min": "15m",
            "30Min": "30m",
            "1Hour": "h",
            "1Day": "d",
        }
        return mapping.get(timeframe, "m")

    async def _fetch_candles(self, candle_symbol: str, start: datetime, end: datetime) -> list[dict]:
        quote_token = self._get_quote_token()
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        rows: dict[int, dict] = {}

        async with websockets.connect(quote_token["dxlink-url"], ping_interval=None, open_timeout=20) as websocket:
            async def send(payload: dict) -> None:
                await websocket.send(json.dumps(payload))

            async def recv(timeout: int = 10) -> dict:
                return json.loads(await asyncio.wait_for(websocket.recv(), timeout=timeout))

            await send(
                {
                    "type": "SETUP",
                    "channel": 0,
                    "version": "1.0-DXF-JS/0.3.0",
                    "keepaliveTimeout": 60,
                    "acceptKeepaliveTimeout": 60,
                }
            )
            await recv()
            await send({"type": "AUTH", "channel": 0, "token": quote_token["token"]})
            while True:
                message = await recv()
                if message.get("state") == "AUTHORIZED":
                    break

            await send({"type": "CHANNEL_REQUEST", "channel": 11, "service": "FEED", "parameters": {"contract": "AUTO"}})
            while True:
                message = await recv()
                if message.get("type") == "CHANNEL_OPENED":
                    break

            await send(
                {
                    "type": "FEED_SETUP",
                    "channel": 11,
                    "acceptAggregationPeriod": 1,
                    "acceptDataFormat": "FULL",
                    "acceptEventFields": {
                        "Candle": ["eventSymbol", "time", "open", "high", "low", "close", "volume"]
                    },
                }
            )
            await send(
                {
                    "type": "FEED_SUBSCRIPTION",
                    "channel": 11,
                    "reset": True,
                    "add": [{"type": "Candle", "symbol": candle_symbol, "fromTime": start_ms}],
                }
            )

            idle_count = 0
            while idle_count < 3:
                try:
                    message = await recv(timeout=3)
                except asyncio.TimeoutError:
                    idle_count += 1
                    continue
                if message.get("type") != "FEED_DATA":
                    continue
                for row in message.get("data", []):
                    ts = int(row.get("time", 0))
                    if start_ms <= ts <= end_ms:
                        rows[ts] = row
                if rows and max(rows) >= end_ms:
                    break

        return list(rows.values())
