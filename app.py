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
from snoogans_brain import SnoogansBrain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
def load_brain(): return SnoogansBrain(chunk_size=400, chunk_overlap=150)
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

# --- 4. DATA PERSISTENCE & HEADER ---
if 'strategy' not in st.session_state:
    st.session_state.strategy = TradingStrategy()

# Load completed trades count for brain progress
COMPLETED_TRADES_FILE = PROJECT_ROOT / "completed_trades.json"
BATCH_SIZE = 30
trade_count = 0
if COMPLETED_TRADES_FILE.exists():
    try:
        with open(COMPLETED_TRADES_FILE, 'r') as f:
            completed = json.load(f)
            trade_count = len(completed)
    except:
        trade_count = 0

progress = min(trade_count / BATCH_SIZE, 1.0)
trades_till_batch = BATCH_SIZE - (trade_count % BATCH_SIZE)

# Equity
bal = float(os.getenv("STARTING_BALANCE", 100000))
pnl = st.session_state.strategy.daily_pnl
current_eq = bal + pnl

# Header metrics
m1, m2, m3, m4 = st.columns([2, 2, 4, 2])
m1.metric("Equity", f"${current_eq:,.2f}")
m2.metric("Daily PnL", f"${pnl:+.2f}")
with m3:
    st.caption(f"Brain Learning Progress → {trade_count}/{trade_count // BATCH_SIZE * BATCH_SIZE + BATCH_SIZE} (Next batch in {trades_till_batch} trades)")
    st.progress(progress)
    if trades_till_batch == BATCH_SIZE:
        st.success("🍃 Fresh knowledge batch dropped – RAG brain gettin' fatter")
m4.metric("Market", "🟢 OPEN" if dt_time(9,30) <= datetime.now().time() <= dt_time(16,0) else "🔴 CLOSED")

st.divider()

# --- 5. DASHBOARD GRID ---
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("Performance & Position")
    st.line_chart(
        pd.Series(st.session_state.strategy.equity_history, name="Equity"),
        height=230
    )
   
    with st.container(border=True):
        if st.session_state.strategy.positions:
            st.write("**Current Positions:**")
            pos_list = []
            for ticker, p in st.session_state.strategy.positions.items():
                row = p.copy()
                row['ticker'] = ticker
                pos_list.append(row)
            st.table(pd.DataFrame(pos_list))
        else:
            st.info("Scanning SPY / QQQ / IWM for .30 Delta setups...")

with col_right:
    t1, t2 = st.tabs(["💬 Snoogans Chat", "📜 Trading Logs"])
   
    with t1:
        chat_box = st.container(height=360)
        if "chat" not in st.session_state:
            st.session_state.chat = [{"role": "assistant", "content": "Router active. Standing by."}]
       
        with chat_box:
            for msg in st.session_state.chat:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
       
        if prompt := st.chat_input("Ask Snoogans..."):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                reply = get_snoogans_rant(prompt)
                st.markdown(reply)
                st.session_state.chat.append({"role": "assistant", "content": reply})
            st.rerun()

    with t2:
        st.write(f"Last Heartbeat: {datetime.now().strftime('%H:%M:%S')}")
        log_display = st.empty()

# --- 6. BACKGROUND ENGINE ---
@st.fragment(run_every=10)
def sync_trading_cycle():
    output_buffer = io.StringIO()
   
    with redirect_stdout(output_buffer):
        try:
            st.session_state.strategy.run_cycle()
        except Exception as e:
            print(f"Cycle Error: {e}")
    
    log_display.code(output_buffer.getvalue() or "Scanning...", language="bash", wrap_lines=True)

sync_trading_cycle()