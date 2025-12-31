import streamlit as st
import pandas as pd
import numpy as np
import io, json, os, sys
from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime, time as dt_time
from dotenv import load_dotenv
# --- 1. BOOTSTRAP & PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
for p in [PROJECT_ROOT, CODE_DIR, CODE_DIR / "trader", CODE_DIR / "rag"]:
    if str(p.resolve()) not in sys.path:
        sys.path.insert(0, str(p.resolve()))
ENV_PATH = PROJECT_ROOT / "variables.env"
load_dotenv(ENV_PATH)
from strategy import TradingStrategy
from strategy_15m import TradingStrategy15m
from scalp_strategy import ScalpStrategy
from snoogans_brain import SnoogansBrain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# --- DISCORD NOTIFIER IMPORTS ---
from discord_notifier import (
    send_greeting_if_needed,
    send_pivots_if_needed,
    send_eod_if_needed,
    set_live_mode,
    force_send_greeting,
    force_send_pivots
)
# --- 2. UI CONFIG & CSS ---
st.set_page_config(page_title="Snoogans", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .block-container { padding-top: 3.5rem; padding-bottom: 0rem; max-width: 98%; }
    div[data-testid="stVerticalBlock"] { gap: 0.4rem; }
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)
# --- 3. BRAIN & ROUTER LOGIC ---
@st.cache_resource
def load_brain():
    return SnoogansBrain(chunk_size=400, chunk_overlap=150)
RAG_BRAIN = load_brain()
ROUTER_LLM = OllamaLLM(model="snoogans:latest", temperature=0.0, base_url="http://127.0.0.1:11434")
router_prompt = PromptTemplate.from_template("Classify as 'MANIFESTO' or 'GENERAL'. Question: {question}\nAnswer:")
router_chain = router_prompt | ROUTER_LLM | StrOutputParser()
def get_snoogans_rant(question: str) -> str:
    try:
        classification = router_chain.invoke({"question": question}).strip().upper()
    except:
        classification = "GENERAL"
    if "MANIFESTO" in classification:
        st.toast("Query classified as **MANIFESTO**. Using RAG context.")
        return RAG_BRAIN.ask_rag(question)
    return RAG_BRAIN.ask_general(question)
# --- 4. DATA PERSISTENCE & INSTANCES ---
if 'strategy_1m' not in st.session_state:
    st.session_state.strategy_1m = TradingStrategy()
if 'strategy_15m' not in st.session_state:
    st.session_state.strategy_15m = TradingStrategy15m()
if 'strategy_scalp' not in st.session_state:
    st.session_state.strategy_scalp = ScalpStrategy()

BATCH_SIZE = 30

def refresh_dashboard_state():
    # Reload main strategy state from disk (syncs scalp PnL/equity)
    st.session_state.strategy_1m = TradingStrategy()
    
    # Count individual trades from project-relative trades.json
    KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
    TRADES_FILE = KNOWLEDGE_DIR / "trades.json"
    trade_count = 0
    if TRADES_FILE.exists():
        with open(TRADES_FILE, 'r') as f:
            try:
                all_trades = json.load(f)
                trade_count = len(all_trades)
            except json.JSONDecodeError:
                trade_count = 0
    return trade_count

# === LIVE HEADER FRAGMENT (auto-updates every 10 seconds) ===
header_placeholder = st.empty()
@st.fragment(run_every=10)
def update_header():
    trade_count = refresh_dashboard_state()
    bal = float(os.getenv("STARTING_BALANCE", 100000))
    pnl = st.session_state.strategy_1m.daily_pnl
    current_eq = bal + pnl

    with header_placeholder.container():
        m1, m2, m3, m4 = st.columns([2, 2, 4, 2])
        m1.metric("Equity", f"${current_eq:,.2f}")
        m2.metric("Daily PnL", f"${pnl:+.2f}")
        with m3:
            st.caption(f"Brain Learning Progress → {trade_count} trades saved")
            progress_value = trade_count / 300.0  # full bar at 300 trades
            st.progress(min(progress_value, 1.0))
            if trade_count > 0 and trade_count % BATCH_SIZE == 0:
                st.success("🍃 Fresh batch of 30 trades completed – brain gettin' smarter")
            else:
                next_batch = BATCH_SIZE - (trade_count % BATCH_SIZE)
                st.caption(f"Next batch in {next_batch} trades")
        m4.metric("Market", "🟢 OPEN" if dt_time(9,30) <= datetime.now().time() <= dt_time(16,0) else "🔴 CLOSED")
        st.divider()

update_header()

st.markdown("### Discord Manual Fire")
col1, col2 = st.columns(2)
with col1:
    if st.button("FIRE GREETING", key="manual_greeting", use_container_width=True):
        print("[BUTTON] Manual greeting fired")
        force_send_greeting()
with col2:
    if st.button("FIRE PIVOTS", key="manual_pivots", use_container_width=True):
        print("[BUTTON] Manual pivots fired")
        force_send_pivots()

# --- 5. DASHBOARD GRID ---
col_left, col_right = st.columns([1, 1], gap="medium")
with col_left:
    st.subheader("Performance & Position")
    st.line_chart(
        pd.Series(st.session_state.strategy_1m.equity_history, name="Equity"),
        height=230
    )
    with st.container(border=True):
        if st.session_state.strategy_1m.positions:
            st.write("**Current Positions (1m):**")
            pos_list = []
            for ticker, p in st.session_state.strategy_1m.positions.items():
                row = p.copy()
                row['ticker'] = ticker
                pos_list.append(row)
            st.table(pd.DataFrame(pos_list))
        else:
            st.info("Scanning SPY / QQQ / IWM for .30 Delta setups...")
        if st.session_state.strategy_15m.positions:
            st.write("**Current Positions (15m):**")
            pos_list_15m = []
            for ticker, p in st.session_state.strategy_15m.positions.items():
                row = p.copy()
                row['ticker'] = ticker
                pos_list_15m.append(row)
            st.table(pd.DataFrame(pos_list_15m))
        if st.session_state.strategy_scalp.positions:
            st.write("**Current Positions (Scalp):**")
            scalp_pos_list = []
            for ticker, p in st.session_state.strategy_scalp.positions.items():
                is_call = "CALL" if p.get("is_call", False) else "PUT"
                strike = p.get("strike_price", 0.0)
                debit = p.get("debit", 0.0)
                contracts = p.get("contracts", 1)
                best_mark = p.get("best_mark", debit)
                trail_active = "Yes" if p.get("trail_active", False) else "No"
                trail_level = p.get("trail_level", 0.0)
                entry_time = p.get("entry_time", "unknown")
                unrealized = (best_mark - debit) * contracts * 100
                scalp_pos_list.append({
                    "Ticker": ticker,
                    "Type": is_call,
                    "Strike": f"{strike:.1f}",
                    "Debit": f"${debit:.2f}",
                    "Contracts": contracts,
                    "Best Mark": f"${best_mark:.2f}",
                    "Unrealized": unrealized,
                    "Trail Active": trail_active,
                    "Trail Level": f"${trail_level:.2f}" if trail_active == "Yes" else "-",
                    "Entry Time": entry_time.split(".")[0]
                })
            df_scalp = pd.DataFrame(scalp_pos_list)
            st.dataframe(
                df_scalp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Unrealized": st.column_config.NumberColumn(
                        "Unrealized P/L",
                        format="$%.2f",
                        help="Current unrealized profit/loss on this position"
                    ),
                    "Debit": st.column_config.NumberColumn(format="$%.2f"),
                    "Best Mark": st.column_config.NumberColumn(format="$%.2f"),
                    "Trail Level": st.column_config.NumberColumn(format="$%.2f"),
                }
            )
        else:
            st.info("No active scalp positions — waiting for trend signal.")
with col_right:
    t1, t2 = st.tabs(["💬 Snoogans Chat", "📜 Trading Logs"])
    with t1:
        chat_box = st.container(height=360)
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "Router active. Standing by."}]
 
        with chat_box:
            for msg in st.session_state.chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
 
        if prompt := st.chat_input("Ask Snoogans..."):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                reply = get_snoogans_rant(prompt)
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})
            st.rerun()
    with t2:
        st.write(f"Last Heartbeat: {datetime.now().strftime('%H:%M:%S')}")
        st.subheader("🖥️ 1-Minute Snoogans Log")
        log_display_1m = st.empty()
        @st.fragment(run_every=12)
        def sync_1m_cycle():
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                try:
                    st.session_state.strategy_1m.run_cycle()
                except Exception as e:
                    print(f"[1m] Cycle Error: {e}")
        
            log_display_1m.code(output_buffer.getvalue() or "Scanning...", language="bash", wrap_lines=True)
        
            send_greeting_if_needed()
            send_pivots_if_needed()
            today_trades = len(getattr(st.session_state.strategy_1m, "trades_today", []))
            send_eod_if_needed(today_trades)
            refresh_dashboard_state()
        sync_1m_cycle()
        st.subheader("🖥️ 15-Minute Snoogans Log")
        log_display_15m = st.empty()
        @st.fragment(run_every=900)
        def sync_15m_cycle():
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                try:
                    st.session_state.strategy_15m.run_cycle()
                except Exception as e:
                    print(f"[15m] Cycle Error: {e}")
        
            log_display_15m.code(output_buffer.getvalue() or "Scanning...", language="bash", wrap_lines=True)
        
            send_greeting_if_needed()
            send_pivots_if_needed()
            today_trades = len(getattr(st.session_state.strategy_15m, "trades_today", []))
            send_eod_if_needed(today_trades)
            refresh_dashboard_state()
        sync_15m_cycle()
        st.subheader("🖥️ Scalp Snoogans Log")
        log_display_scalp = st.empty()
        @st.fragment(run_every=60)
        def sync_scalp_cycle():
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                try:
                    st.session_state.strategy_scalp.run_cycle()
                except Exception as e:
                    print(f"[Scalp] Cycle Error: {e}")
        
            log_display_scalp.code(output_buffer.getvalue() or "Scanning...", language="bash", wrap_lines=True)
            refresh_dashboard_state()
        sync_scalp_cycle()
# === FINAL STEP: ENABLE LIVE DISCORD ALERTS ===
set_live_mode()