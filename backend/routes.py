# backend/routes.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
from .shared import get_trade_count, get_starting_balance
from code.trader.strategy import TradingStrategy
from code.trader.strategy_15m import TradingStrategy15m
from code.trader.scalp_strategy import ScalpStrategy
from code.rag.snoogans_brain import SnoogansBrain
from code.rag.adaptive_learner import AdaptiveLearner
from code.data.data_client import get_data_client

router = APIRouter()

# Cache learner instance
_learner: AdaptiveLearner | None = None

def get_learner() -> AdaptiveLearner:
    """Get or create the adaptive learner singleton."""
    global _learner
    if _learner is None:
        _learner = AdaptiveLearner()
    return _learner

# Cache brain instance
brain = SnoogansBrain(chunk_size=400, chunk_overlap=150)

@router.get("/state")
async def get_dashboard_state():
    # Instantiate fresh strategies to read latest state from disk/files
    strat_1m = TradingStrategy()
    strat_15m = TradingStrategy15m()
    strat_scalp = ScalpStrategy()

    # Get full status from each strategy
    status_1m = strat_1m.get_status()
    status_15m = strat_15m.get_status()
    status_scalp = strat_scalp.get_status()

    # Get equity from history (actual tracked equity) or fall back to starting balance
    equity_history = strat_1m.equity_history
    if equity_history and len(equity_history) > 0:
        equity = equity_history[-1]
    else:
        equity = get_starting_balance()

    daily_pnl = strat_1m.daily_pnl

    return {
        "equity": round(equity, 2),
        "daily_pnl": round(daily_pnl, 2),
        "trade_count": get_trade_count(),
        "market_open": datetime.now().time().replace(tzinfo=None) >= datetime.strptime("09:30", "%H:%M").time() and
                       datetime.now().time().replace(tzinfo=None) <= datetime.strptime("16:00", "%H:%M").time(),
        "positions_1m": strat_1m.positions,
        "positions_15m": strat_15m.positions,
        "positions_scalp": strat_scalp.positions,
        "equity_history": equity_history[-100:] if equity_history else [],
        # Full strategy status for Trading Logs
        "status_1m": {
            "daily_pnl": status_1m.get("daily_pnl", 0),
            "daily_loss_limit": status_1m.get("daily_loss_limit", 0),
            "trades_today": status_1m.get("trades_today", 0),
            "can_trade": status_1m.get("can_trade", False),
            "current_trends": status_1m.get("current_trends", {}),
        },
        "status_15m": {
            "daily_pnl": status_15m.get("daily_pnl", 0),
            "daily_loss_limit": status_15m.get("daily_loss_limit", 0),
            "trades_today": status_15m.get("trades_today", 0),
            "can_trade": status_15m.get("can_trade", False),
            "current_trends": status_15m.get("current_trends", {}),
        },
        "status_scalp": {
            "daily_pnl": status_scalp.get("daily_pnl", 0),
            "daily_loss_limit": status_scalp.get("daily_loss_limit", 0),
            "trades_today": status_scalp.get("trades_today", 0),
            "can_trade": status_scalp.get("can_trade", False),
            "current_trends": status_scalp.get("current_trends", {}),
        },
    }

@router.post("/ask")
async def ask_snoogans(question: dict):
    reply = brain.ask(question["question"])
    return {"reply": reply}

@router.get("/learner")
async def get_learner_data():
    """Get RL learner stats, best/worst states, and recent decisions."""
    learner = get_learner()
    stats = learner.get_stats()
    decisions = learner.get_decision_summary()
    best_states = learner.get_best_states(5)
    worst_states = learner.get_worst_states(5)
    recent_decisions = learner.get_recent_decisions(10)

    return {
        "stats": stats,
        "decisions": decisions,
        "best_states": best_states,
        "worst_states": worst_states,
        "recent_decisions": recent_decisions,
    }

# WebSocket for real-time logs
@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state = await get_dashboard_state()
            await websocket.send_json({"type": "state", "data": state})
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass

# WebSocket for real-time candle data
candle_subscribers: Dict[str, List[WebSocket]] = {}

@router.websocket("/ws/candles")
async def websocket_candles(websocket: WebSocket):
    """WebSocket for real-time candlestick data from Alpaca."""
    await websocket.accept()
    subscribed_tickers: List[str] = []

    try:
        from datetime import timedelta
        from concurrent.futures import ThreadPoolExecutor
        import functools

        executor = ThreadPoolExecutor(max_workers=3)
        loop = asyncio.get_event_loop()

        def fetch_bars_sync(ticker: str, days: int = 5):
            """Fetch bars synchronously (runs in thread pool)."""
            try:
                data_client = get_data_client()
                end = datetime.now()
                start = end - timedelta(days=days)
                bars = data_client.get_spy_bars(start, end, ticker=ticker)
                return bars
            except Exception as e:
                print(f"Error in fetch_bars_sync for {ticker}: {e}")
                return None

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
                if message.get("type") == "subscribe":
                    ticker = message.get("ticker", "").upper()
                    if ticker and ticker not in subscribed_tickers:
                        subscribed_tickers.append(ticker)
                        try:
                            bars = await loop.run_in_executor(
                                executor, functools.partial(fetch_bars_sync, ticker, 5)
                            )
                            if bars is not None and len(bars) > 0:
                                candles = []
                                for idx, row in bars.tail(100).iterrows():
                                    candles.append({
                                        "time": int(idx.timestamp()),
                                        "open": float(row["open"]),
                                        "high": float(row["high"]),
                                        "low": float(row["low"]),
                                        "close": float(row["close"]),
                                        "volume": int(row.get("volume", 0)),
                                    })
                                await websocket.send_json({
                                    "type": "history",
                                    "ticker": ticker,
                                    "candles": candles,
                                })
                                print(f"Sent {len(candles)} candles for {ticker}")
                            else:
                                print(f"No bars returned for {ticker}")
                        except Exception as e:
                            print(f"Error fetching bars for {ticker}: {e}")
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Candle WebSocket error: {e}")
