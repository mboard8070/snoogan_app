import streamlit as st
import pandas as pd
import numpy as np
import threading
import subprocess
import os
import sys
import io
import time  # <--- This is for time.sleep()
from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime, time as dt_time, timedelta # <--- Renamed to dt_time
from dotenv import load_dotenv

# --- 1. ROBUST PATH & ENV SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE_NAME = "variables.env"
ENV_PATH = PROJECT_ROOT / ENV_FILE_NAME
load_dotenv(ENV_PATH)

CODE_DIR = PROJECT_ROOT / "code"
paths_to_add = [PROJECT_ROOT, CODE_DIR, CODE_DIR / "trader", CODE_DIR / "data", CODE_DIR / "rag"]
for p in paths_to_add:
    if str(p.resolve()) not in sys.path:
        sys.path.insert(0, str(p.resolve()))

# --- 2. IMPORTS ---
try:
    from strategy import TradingStrategy
    from discord_bot import send_eod_summary
    from snoogans_brain import SnoogansBrain
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    st.error(f"Critical Import Failure: {e}")
    st.stop()

# --- 3. THE "GREAT" ROUTER & RAG LOGIC ---
@st.cache_resource
def load_brain():
    return SnoogansBrain(chunk_size=400, chunk_overlap=150)

RAG_BRAIN = load_brain()
ROUTER_LLM = OllamaLLM(
    model="snoogans:latest",
    temperature=0.0,
    base_url="http://127.0.0.1:11434"
)

router_prompt = PromptTemplate.from_template(
    """Classify as 'MANIFESTO' if question is about trading rules (delta, entry/exit, stop, sizing, etc.). Else 'GENERAL'.
Respond STRICTLY with one word: 'MANIFESTO' or 'GENERAL'.
Question: {question}
Answer:"""
)
router_chain = router_prompt | ROUTER_LLM | StrOutputParser()

def get_snoogans_rant(question: str) -> str:
    try:
        classification = router_chain.invoke({"question": question}).strip().upper()
    except:
        classification = "GENERAL"
  
    if "MANIFESTO" in classification:
        st.info("Query classified as **MANIFESTO**. Using RAG for strict manifesto quote.")
        return RAG_BRAIN.ask_rag(question)
  
    st.info("Query classified as **GENERAL**. Using General LLM for conversational response.")
    return RAG_BRAIN.ask_general(question)

# --- 4. BACKGROUND THREADS ---
if 'bot_started' not in st.session_state:
    def launch_services():
        discord_script = PROJECT_ROOT / "discord_bot.py"
        if discord_script.exists():
            subprocess.Popen(["python3", str(discord_script)], cwd=str(PROJECT_ROOT))
        
        while True:
            try:
                send_eod_summary() 
            except Exception as e:
                print(f"EOD Error: {e}")
            time.sleep(60) # <--- Now works because 'time' is the module

    threading.Thread(target=launch_services, daemon=True).start()
    st.session_state.strategy = TradingStrategy()
    st.session_state.bot_started = True

# --- 5. UI CONFIG & SIDEBAR ---
st.set_page_config(page_title="Snoogans", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.header("Admin Controls")
    force_run = st.button("⚡ Force Trading Cycle", use_container_width=True)
    if st.button("Clear Chat Cache"):
        st.session_state.chat = [{"role": "assistant", "content": "Cache cleared."}]
        st.rerun()

c1, c2, c3, c4, c5 = st.columns([2.2, 1.2, 1.2, 1.2, 2])
with c1:
    balance = getattr(st.session_state, 'equity', 100000.0)
    st.markdown(f"<div style='font-size:1.5rem; font-weight:bold'>${balance:,.0f}</div>", unsafe_allow_html=True)
with c5:
    st.caption(f"System Time: {datetime.now().strftime('%H:%M:%S EST')}")

st.markdown("---")

# --- 6. MAIN DASHBOARD ---
left, mid, right = st.columns([3, 2.2, 2], gap="medium")

with left:
    st.subheader("Equity Curve")
    st.line_chart(pd.DataFrame({'Equity': [st.session_state.get('equity', 100000)] * 15}), height=250)
    
    st.subheader("Trade Log")
    if hasattr(st.session_state.strategy, 'trades') and st.session_state.strategy.trades:
        st.dataframe(pd.DataFrame(st.session_state.strategy.trades[-5:]), use_container_width=True)
    else:
        st.caption("Waiting for execution...")

with mid:
    st.subheader("Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Win Rate", "0.0%")
        st.metric("Profit Factor", "0.0")
    with col2:
        st.metric("Avg Loss", "N/A")
        st.metric("2 PM Cutoff", "Active")

with right:
    st.subheader("Open Positions")
    if st.session_state.strategy.position:
        st.dataframe(pd.DataFrame([st.session_state.strategy.position]), use_container_width=True)
    else:
        st.caption("No open positions")

st.markdown("---")

# --- 7. CHAT & HEARTBEAT LOGS ---
bot_left, bot_right = st.columns([1, 1], gap="large")

with bot_left:
    st.subheader("Chat with Snoogans")
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "Snoogans online. Target .30 delta. Lunch money mode."}]

    chat_container = st.container(height=450)
    with chat_container:
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

# --- 8. HEARTBEAT FRAGMENT ---
with bot_right:
    st.subheader("Live Trading Cycle Logs")
    log_status = st.empty()
    log_display = st.empty()

@st.fragment(run_every=10)
def sync_trading_cycle(manual_trigger=False):
    try:
        sh, sm = int(os.getenv("ENTRY_START_HOUR", 9)), int(os.getenv("ENTRY_START_MINUTE", 40))
        eh, em = int(os.getenv("ENTRY_END_HOUR", 14)), int(os.getenv("ENTRY_END_MINUTE", 30))
        # Use 'dt_time' here because we renamed the import
        win_start, win_end = dt_time(sh, sm), dt_time(eh, em)
    except:
        win_start, win_end = dt_time(9, 40), dt_time(14, 30)

    now_t = datetime.now().time()
    is_in_win = win_start <= now_t <= win_end
    has_pos = st.session_state.strategy.position is not None

    if not is_in_win and not has_pos and not manual_trigger:
        log_status.warning(f"⏸️ Heartbeat Paused ({win_start.strftime('%H:%M')} - {win_end.strftime('%H:%M')})")
        log_display.info("Outside entry window. No active positions.")
        return

    log_status.caption(f"Last Heartbeat: {datetime.now().strftime('%H:%M:%S')}")
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        try:
            st.session_state.strategy.run_cycle()
        except Exception as e:
            print(f"Cycle Error: {e}")
    
    log_display.code(output_buffer.getvalue() if output_buffer.getvalue() else "Scanning markets...", language="bash")

if force_run:
    sync_trading_cycle(manual_trigger=True)
else:
    sync_trading_cycle()