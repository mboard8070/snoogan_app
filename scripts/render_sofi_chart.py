from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "data" / "market_cache" / "SOFI_5Min.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "market_cache" / "SOFI_5Min_chart.html"


def sx(i: int, n: int, left: int, width: int) -> float:
    return left + (i / max(n - 1, 1)) * width


def sy(v: float, lo: float, hi: float, top: int, height: int) -> float:
    return top + (hi - v) / max(hi - lo, 1e-9) * height


def main() -> None:
    df = pd.read_parquet(CACHE_PATH).sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    recent = df.tail(950).copy()
    recent["ema9"] = recent["close"].ewm(span=9, adjust=False).mean()
    recent["ema21"] = recent["close"].ewm(span=21, adjust=False).mean()
    recent["ema50"] = recent["close"].ewm(span=50, adjust=False).mean()

    W, H = 1400, 820
    left, top, chart_w, chart_h = 70, 60, 1260, 560
    vol_top, vol_h = 650, 120
    lo = float(recent[["low", "ema9", "ema21", "ema50"]].min().min())
    hi = float(recent[["high", "ema9", "ema21", "ema50"]].max().max())
    pad = (hi - lo) * 0.06
    lo -= pad
    hi += pad
    max_vol = float(recent["volume"].max())
    n = len(recent)
    candle_w = max(1.0, chart_w / n * 0.55)

    html_head = """<!doctype html><html><head><meta charset=\"utf-8\"><title>SOFI Chart</title>
<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827} svg{background:white;border:1px solid #d1d5db} .muted{color:#64748b}</style></head><body>
<h2>SOFI 5-Minute Tastytrade Candles</h2>
<p class=\"muted\">%s to %s | Close %.2f | EMA9 %.2f | EMA21 %.2f | EMA50 %.2f</p>
<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">""" % (recent.index.min(), recent.index.max(), recent.close.iloc[-1], recent.ema9.iloc[-1], recent.ema21.iloc[-1], recent.ema50.iloc[-1], W, H, W, H)
    parts = [html_head]

    for j in range(6):
        y = top + j * chart_h / 5
        price = hi - j * (hi - lo) / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="12" y="{y+4:.1f}" font-size="12" fill="#475569">{price:.2f}</text>')

    for i, row in enumerate(recent.itertuples()):
        x = sx(i, n, left, chart_w)
        vh = (float(row.volume) / max_vol) * vol_h if max_vol else 0
        color = '#22c55e' if row.close >= row.open else '#ef4444'
        parts.append(f'<rect x="{x-candle_w/2:.2f}" y="{vol_top+vol_h-vh:.2f}" width="{candle_w:.2f}" height="{vh:.2f}" fill="{color}" opacity="0.35"/>')

    for i, row in enumerate(recent.itertuples()):
        x = sx(i, n, left, chart_w)
        y_high = sy(float(row.high), lo, hi, top, chart_h)
        y_low = sy(float(row.low), lo, hi, top, chart_h)
        y_open = sy(float(row.open), lo, hi, top, chart_h)
        y_close = sy(float(row.close), lo, hi, top, chart_h)
        color = '#16a34a' if row.close >= row.open else '#dc2626'
        body_y = min(y_open, y_close)
        body_h = max(abs(y_close-y_open), 1.0)
        parts.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="{color}" stroke-width="1"/>')
        parts.append(f'<rect x="{x-candle_w/2:.2f}" y="{body_y:.2f}" width="{candle_w:.2f}" height="{body_h:.2f}" fill="{color}"/>')

    def path_for(col: str) -> str:
        pts = [f'{sx(i,n,left,chart_w):.2f},{sy(float(v),lo,hi,top,chart_h):.2f}' for i, v in enumerate(recent[col])]
        return 'M ' + ' L '.join(pts)

    for col, color in [('ema9','#2563eb'),('ema21','#f97316'),('ema50','#7c3aed')]:
        parts.append(f'<path d="{path_for(col)}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append(f'<text x="{left}" y="35" font-size="13" fill="#2563eb">EMA9</text>')
    parts.append(f'<text x="{left+55}" y="35" font-size="13" fill="#f97316">EMA21</text>')
    parts.append(f'<text x="{left+120}" y="35" font-size="13" fill="#7c3aed">EMA50</text>')
    parts.append('</svg></body></html>')
    OUT_PATH.write_text('\n'.join(parts))
    print(OUT_PATH)
    print('rows', len(recent), 'first', recent.index.min(), 'last', recent.index.max())
    print('close', float(recent.close.iloc[-1]), 'ema9', float(recent.ema9.iloc[-1]), 'ema21', float(recent.ema21.iloc[-1]), 'ema50', float(recent.ema50.iloc[-1]))


if __name__ == "__main__":
    main()
