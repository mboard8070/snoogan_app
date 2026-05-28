from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "news"
LATEST_CONTEXT_FILE = OUTPUT_DIR / "latest_market_context.json"
EVENT_LOG_FILE = OUTPUT_DIR / "market_news_events.jsonl"

SYMBOLS = ["SOFI", "LUMN", "LUNR", "SPY", "QQQ"]
NEWS_QUERIES = [
    "SOFI stock OR SoFi Technologies",
    "LUMN stock OR Lumen Technologies",
    "LUNR stock OR Intuitive Machines",
    "Federal Reserve rates stock market",
    "market moving earnings guidance stocks",
]

NEGATIVE_TERMS = {
    "downgrade", "lawsuit", "sec", "investigation", "misses", "missed",
    "loss", "losses", "fraud", "bankruptcy", "default", "dilution",
    "offering", "selloff", "plunge", "cuts guidance", "weak guidance",
    "short report", "recall", "halted", "bearish",
}
POSITIVE_TERMS = {
    "upgrade", "beats", "beat", "raises guidance", "record revenue",
    "profit", "profitable", "partnership", "approval", "contract",
    "buyback", "bullish", "breakout", "surge", "launch",
}
HIGH_IMPACT_TERMS = {
    "earnings", "guidance", "fed", "federal reserve", "rates", "cpi",
    "inflation", "sec", "lawsuit", "investigation", "offering",
    "downgrade", "upgrade", "contract", "approval", "short report",
}
MACRO_TERMS = {"fed", "federal reserve", "rates", "cpi", "inflation", "bond market", "treasury", "tariff"}


@dataclass
class NewsItem:
    source: str
    title: str
    url: str
    published: str | None
    summary: str = ""

    @property
    def uid(self) -> str:
        key = f"{self.source}|{self.title}|{self.url}".encode()
        return hashlib.sha256(key).hexdigest()[:16]


def _fresh_enough(published: str | None, max_age: timedelta = timedelta(days=3)) -> bool:
    if not published:
        return True
    try:
        parsed = parsedate_to_datetime(published)
    except (TypeError, ValueError):
        return True
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return datetime.now(parsed.tzinfo) - parsed <= max_age


def _load_env() -> None:
    env_path = PROJECT_ROOT / "variables.env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _text_score(text: str) -> tuple[int, list[str]]:
    lower = text.lower()
    hits: list[str] = []
    score = 0
    for term in POSITIVE_TERMS:
        if term in lower:
            score += 1
            hits.append(f"+{term}")
    for term in NEGATIVE_TERMS:
        if term in lower:
            score -= 1
            hits.append(f"-{term}")
    return score, sorted(hits)


def _impact(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in HIGH_IMPACT_TERMS):
        return "high"
    if re.search(r"\b(SOFI|LUMN|LUNR|SPY|QQQ)\b", text, re.IGNORECASE):
        return "medium"
    return "low"


def _symbols(text: str) -> list[str]:
    found = []
    for symbol in SYMBOLS:
        if re.search(rf"\b{re.escape(symbol)}\b", text, re.IGNORECASE):
            found.append(symbol)
    if re.search(r"\bsofi\b|sofi technologies|sofiusd", text, re.IGNORECASE) and "SOFI" not in found:
        found.append("SOFI")
    if re.search(r"lumen technologies|\blumen\b", text, re.IGNORECASE) and "LUMN" not in found:
        found.append("LUMN")
    if re.search(r"intuitive machines", text, re.IGNORECASE) and "LUNR" not in found:
        found.append("LUNR")
    return found


def _is_macro(text: str) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lower) for term in MACRO_TERMS)


def fetch_google_news() -> list[NewsItem]:
    items: list[NewsItem] = []
    headers = {"User-Agent": "snoogans-news-scanner/1.0"}
    for query in NEWS_QUERIES:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        try:
            response = requests.get(url, headers=headers, timeout=12)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception:
            continue
        for item in root.findall(".//item")[:8]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            published = (item.findtext("pubDate") or "").strip() or None
            if title and link:
                items.append(NewsItem("google_news", title, link, published))
        time.sleep(0.2)
    return items


def fetch_yahoo_rss() -> list[NewsItem]:
    items: list[NewsItem] = []
    headers = {"User-Agent": "snoogans-news-scanner/1.0"}
    for symbol in ["SOFI", "LUMN", "LUNR"]:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
        try:
            response = requests.get(url, headers=headers, timeout=12)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception:
            continue
        for item in root.findall(".//item")[:10]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            published = (item.findtext("pubDate") or "").strip() or None
            summary = (item.findtext("description") or "").strip()
            if title and link:
                items.append(NewsItem("yahoo_finance", title, link, published, summary))
    return items


def fetch_grok_build_x() -> list[NewsItem]:
    if os.getenv("GROK_BUILD_X_SEARCH", "1").lower() not in {"1", "true", "yes", "on"}:
        return []
    prompt = (
        "Use X search only if available. Do not use regular web search. "
        "Search recent public X posts from the last 24 hours for market-moving posts about "
        "SOFI, LUMN, LUNR, SPY, and QQQ. Return only compact JSON with an items array. "
        "Each item must have title, url, published, symbols, sentiment_score, and impact. "
        "Use sentiment_score -2 to 2 and impact low, medium, or high."
    )
    try:
        completed = subprocess.run(
            [
                "grok", "-p", prompt,
                "--output-format", "plain",
                "--no-alt-screen",
                "--max-turns", "30",
                "--disable-web-search",
            ],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=45,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    body = completed.stdout.strip()
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return []
    items = []
    for row in payload.get("items", [])[:20]:
        title = str(row.get("title") or "").strip()
        url = str(row.get("url") or "").strip()
        if title:
            items.append(NewsItem("grok_x_search", title, url or "grok://x-search", row.get("published")))
    return items


def fetch_x_recent() -> list[NewsItem]:
    bearer = os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        return []
    query = '(SOFI OR "$SOFI" OR "SoFi" OR LUMN OR "$LUMN" OR LUNR OR "$LUNR") lang:en -is:retweet'
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": "25",
        "tweet.fields": "created_at,public_metrics,author_id",
    }
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {bearer}"}, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    items = []
    for tweet in payload.get("data", []):
        text = tweet.get("text", "").strip()
        tweet_id = tweet.get("id")
        if text and tweet_id:
            items.append(NewsItem("x", text, f"https://x.com/i/web/status/{tweet_id}", tweet.get("created_at")))
    return items


def analyze_items(items: Iterable[NewsItem]) -> dict:
    deduped: dict[str, NewsItem] = {}
    for item in items:
        deduped.setdefault(item.uid, item)

    analyzed = []
    net_sentiment = 0
    symbol_sentiment = {symbol: 0 for symbol in SYMBOLS}
    macro_sentiment = 0
    highest = "low"
    rank = {"none": -1, "low": 0, "medium": 1, "high": 2}
    for item in deduped.values():
        if not _fresh_enough(item.published):
            continue
        text = f"{item.title} {item.summary}"
        score, hits = _text_score(text)
        impact = _impact(text)
        symbols = _symbols(text)
        macro = _is_macro(text)
        net_sentiment += score
        for symbol in symbols:
            symbol_sentiment[symbol] += score
        if macro:
            macro_sentiment += score
        if rank[impact] > rank[highest]:
            highest = impact
        analyzed.append({
            "id": item.uid,
            "source": item.source,
            "title": item.title,
            "url": item.url,
            "published": item.published,
            "symbols": symbols,
            "macro": macro,
            "sentiment_score": score,
            "impact": impact,
            "signals": hits,
        })

    analyzed.sort(key=lambda row: (rank[row["impact"]], abs(row["sentiment_score"])), reverse=True)
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "symbols": SYMBOLS,
        "item_count": len(analyzed),
        "highest_impact": highest if analyzed else "none",
        "net_sentiment": net_sentiment,
        "symbol_sentiment": symbol_sentiment,
        "macro_sentiment": macro_sentiment,
        "items": analyzed[:30],
        "high_impact_items": [row for row in analyzed if row["impact"] == "high" and (row["symbols"] or row["macro"] )][:10],
        "x_enabled": bool(os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")),
        "grok_build_x_enabled": os.getenv("GROK_BUILD_X_SEARCH", "1").lower() in {"1", "true", "yes", "on"},
    }


def save_context(context: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_CONTEXT_FILE.write_text(json.dumps(context, indent=2, sort_keys=True))
    with EVENT_LOG_FILE.open("a") as f:
        f.write(json.dumps(context, sort_keys=True) + "\n")


def scan_market_news() -> dict:
    _load_env()
    items = []
    items.extend(fetch_google_news())
    items.extend(fetch_yahoo_rss())
    items.extend(fetch_x_recent())
    items.extend(fetch_grok_build_x())
    context = analyze_items(items)
    save_context(context)
    return context
