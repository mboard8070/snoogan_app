# backend/routes.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
import json
from datetime import datetime
from .shared import get_trade_count, get_starting_balance
# load_trades no longer needed directly
from code.trader.strategy import TradingStrategy
from code.trader.strategy_15m import TradingStrategy15m
from code.trader.scalp_strategy import ScalpStrategy
from code.rag.snoogans_brain import SnoogansBrain

router = APIRouter()

# Cache brain instance
brain = SnoogansBrain(chunk_size=400, chunk_overlap=150)

@router.get("/state")
async def get_dashboard_state():
    # Instantiate fresh strategies to read latest state from disk/files
    strat_1m = TradingStrategy()
    strat_15m = TradingStrategy15m()
    strat_scalp = ScalpStrategy()
    
    balance = get_starting_balance()
    pnl_1m = strat_1m.daily_pnl
    equity = balance + pnl_1m  # adjust if you track total PnL differently
    
    return {
        "equity": round(equity, 2),
        "daily_pnl": round(pnl_1m, 2),
        "trade_count": get_trade_count(),
        "market_open": datetime.now().time().replace(tzinfo=None) >= datetime.strptime("09:30", "%H:%M").time() and 
                       datetime.now().time().replace(tzinfo=None) <= datetime.strptime("16:00", "%H:%M").time(),
        "positions_1m": strat_1m.positions,
        "positions_15m": strat_15m.positions,
        "positions_scalp": strat_scalp.positions,
        "equity_history": strat_1m.equity_history[-100:],  # last 100 points
    }

@router.post("/ask")
async def ask_snoogans(question: dict):
    reply = brain.ask(question["question"])  # Smart router handles temporal, RAG, and general
    return {"reply": reply}

# Optional: WebSocket for real-time logs (highly recommended)
@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # You can extend this to capture live output from run_cycle if you move loops here
            # For now, just push state every few seconds
            state = await get_dashboard_state()
            await websocket.send_json({"type": "state", "data": state})
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass