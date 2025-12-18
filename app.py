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
    if str(p.resolve()) not in sys.path: sys.path.insert(0, str(p.resolve()))

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

# --- 3. BRAIN & ROUTER LOGIC (RESTORED) ---
@st.cache_resource
def load_brain(): return SnoogansBrain(chunk_size=400, chunk_overlap=150)

RAG_BRAIN = load_brain()
ROUTER_LLM = OllamaLLM(model="snoogans:latest", temperature=0.0, base_url="http://127.0.0.1:11434")
router_prompt = PromptTemplate.from_template("Classify as 'MANIFESTO' or 'GENERAL'. Question: {question}\nAnswer:")
router_chain = router_prompt | ROUTER_LLM | StrOutputParser()

def get_snoogans_rant(question: str) -> str:
    try: classification = router_chain.invoke({"question": question}).strip().upper()
    except: classification = "GENERAL"
    if "MANIFESTO" in classification:
        st.toast("Query classified as **MANIFESTO**. Using RAG context.")
        return RAG_BRAIN.ask_rag(question)
    return RAG_BRAIN.ask_general(question)

# --- 4. DATA PERSISTENCE & HEADER ---
if 'strategy' not in st.session_state:
    st.session_state.strategy = TradingStrategy()

# Restore Equity History for Charting
if 'equity_history' not in st.session_state:
    st.session_state.equity_history = [float(os.getenv("STARTING_BALANCE", 100000))]

bal = float(os.getenv("STARTING_BALANCE", 100000))
pnl = st.session_state.strategy.daily_pnl
current_eq = bal + pnl

# Update history if pnl changed
if st.session_state.equity_history[-1] != current_eq:
    st.session_state.equity_history.append(current_eq)

m1, m2, m3, m4 = st.columns([2, 2, 4, 2])
m1.metric("Equity", f"${current_eq:,.2f}")
m2.metric("Daily PnL", f"${pnl:+.2f}")
with m3:
    # Simulated progress - replace with actual trade count if available
    st.caption("Brain Learning Progress (Batch of 30)")
    st.progress(0.1) 
m4.metric("Market", "🟢 OPEN" if dt_time(9,30) <= datetime.now().time() <= dt_time(16,0) else "🔴 CLOSED")

st.divider()

# --- 5. DASHBOARD GRID (RESTORED) ---
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("Performance & Position")
    # Restore the Equity Graph
    st.line_chart(pd.DataFrame({'Equity': st.session_state.equity_history}), height=230)
    
    with st.container(border=True):
        pos = st.session_state.strategy.position
        if pos:
            st.write("**Current Position:**")
            st.table(pd.DataFrame([pos]))
        else:
            st.info("Scanning for .30 Delta Setup...")

with col_right:
    t1, t2 = st.tabs(["💬 Snoogans Chat", "📜 Trading Logs"])
    
    with t1:
        # Restore Chat Interface
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
    initial_pos = json.dumps(st.session_state.strategy.position)
    
    with redirect_stdout(output_buffer):
        try:
            st.session_state.strategy.run_cycle()
        except Exception as e:
            print(f"Cycle Error: {e}")
    
    # Force UI update if trade state changed
    if initial_pos != json.dumps(st.session_state.strategy.position):
        st.rerun()

    log_display.code(output_buffer.getvalue() or "Scanning...", language="bash", wrap_lines=True)

sync_trading_cycle()