# /home/mboard76/nvidia-workbench/snoogan_app/code/dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import threading
import subprocess
import os
import sys
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv()

# --- 🎯 RAG INTEGRATION & ROUTING SETUP ---

# 1. Path setup to find snoogans_brain.py (Robust Pathing)
current_dir = Path(__file__).parent 
module_dir = current_dir / ".." / "rag" 

module_dir_str = str(module_dir.resolve())
if module_dir_str not in sys.path:
    sys.path.append(module_dir_str)

# 2. Import the RAG class
try:
    from snoogans_brain import SnoogansBrain
except Exception as e:
    st.error(f"FATAL RAG ERROR: Could not import SnoogansBrain. Check path and file name. {e}")
    st.stop()

# 3. Initialize the RAG brain globally
@st.cache_resource
def load_snoogans_brain():
    return SnoogansBrain(chunk_size=400, chunk_overlap=150) 

RAG_BRAIN = load_snoogans_brain()

# 4. Initialize the fast Router LLM
# FIX: Swapping the Router LLM to use the more robust core model (snoogans:latest) 
# which should be better at following strict instructions (temperature=0.0).
ROUTER_MODEL = "snoogans:latest" 
ROUTER_LLM = OllamaLLM(model=ROUTER_MODEL, temperature=0.0) 

# 5. Router Chain Definition (UPDATED PROMPT for accuracy)
router_prompt = PromptTemplate.from_template(
    """You are a query classifier for a trading bot. Your only task is to determine if the user's question relates to the *specific trading rules* documented in the "Snoogans Manifesto."

Topics that are always MANIFESTO: **delta, credit, entry/exit time, stop loss, sizing, cutoff, account loss limit, contract stack, trade stack, and overall strategy.**
Any other question (greetings, jokes, market commentary, general advice, etc.) is GENERAL.

Respond STRICTLY with one word: 'MANIFESTO' or 'GENERAL'.

Question: {question}
Classification:"""
)
router_chain = router_prompt | ROUTER_LLM | StrOutputParser()

# === RAG Router Function ===
def get_snoogans_rant(details: str) -> str:
    """Classifies the query and sends it to the appropriate chain (RAG or General)."""
    
    # Classify the query
    try:
        classification = router_chain.invoke({"question": details}).strip().upper()
    except Exception as e:
        print(f"Router failed: {e}. Defaulting to General Chat.")
        classification = "GENERAL" 
    
    # Show status for debugging the routing logic
    if classification == "MANIFESTO":
        st.info("Query classified as **MANIFESTO**. Using RAG for strict manifesto quote.")
        return RAG_BRAIN.ask_rag(details) 
    else:
        st.info("Query classified as **GENERAL**. Using General LLM for conversational response.")
        return RAG_BRAIN.ask_general(details) 


# === Auto-launch Discord bot (THREADING FIX) ===
# FIX: Using the correct, writable project root for the subprocess call
project_root = "/home/mboard76/nvidia-workbench/snoogan_app" 
def start_discord_bot():
    discord_script = "code/utils/discord_bot.py"
    full_path = os.path.join(project_root, discord_script)
    
    if os.path.exists(full_path):
        # Only run the subprocess, no st. commands
        subprocess.Popen(["python3", full_path])
        print(f"INFO: Snoogans Discord bot launched from {full_path}.")
    else:
        print(f"WARNING: discord_bot.py not found at {full_path}.")

# Fire bot in background on dashboard load (Use session state to run only once)
if 'bot_started' not in st.session_state:
    threading.Thread(target=start_discord_bot, daemon=True).start()
    st.session_state['bot_started'] = True


# --- START STREAMLIT APP ---
st.set_page_config(page_title="Snoogans", layout="wide", initial_sidebar_state="collapsed")

# Session state 
if "equity" not in st.session_state:
    st.session_state.equity = 142350
    st.session_state.trades = []
    st.session_state.opens = []
    st.session_state.chat = [{"role": "assistant", "content": "Snoogans online. Tiny spreads. Lunch money loading…"}]

# Display Bot Status
st.sidebar.markdown("---")
if st.session_state['bot_started']:
    st.sidebar.caption("🤖 Discord Bot: Active")
else:
    st.sidebar.warning("🤖 Discord Bot: Launch Error (Check terminal log)")

# === Top bar (Includes fix for layout stability) ===
c1, c2, c3, c4, c5 = st.columns([2.2, 1.2, 1.2, 1.2, 2])
with c1:
    st.markdown(
        f"<div style='font-size:1.4rem; font-weight:bold'>${st.session_state.equity:,.0f}</div>"
        "<div style='font-size:0.9rem; color:#00ff00'>+$3,840 (+2.8%)</div>",
        unsafe_allow_html=True
    )
with c2:
    st.markdown("<div style='font-size:0.9rem'>Today</div><div style='font-size:1.3rem'>+1.42%</div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div style='font-size:0.9rem'>Month</div><div style='font-size:1.3rem'>+9.81%</div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div style='font-size:0.9rem'>Max DD</div><div style='font-size:1.3rem'>-4.1%</div>", unsafe_allow_html=True)
with c5:
    st.caption(f"{datetime.now().strftime('%H:%M:%S')} EST")
st.markdown("---")

# === Upper row (Layout check continues) ===
upper_left, upper_mid, upper_right = st.columns([3, 2.2, 2], gap="medium")

with upper_left:
    st.subheader("Equity Curve")
    dates = pd.date_range("2025-01-01", periods=90)
    eq = 140000 + np.cumsum(np.random.randn(90)*700 + 500)
    st.line_chart(eq, height=260, use_container_width=True)
    
    st.subheader("Trade Log")
    recent = st.session_state.trades[-10:] if st.session_state.trades else []
    if recent:
        st.dataframe(pd.DataFrame(recent, columns=["Time","Message"]), hide_index=True, use_container_width=True, height=360)
    else:
        st.caption("Waiting for first spread…")

with upper_mid:
    st.subheader("Stats")
    grid = st.columns(3)
    stats = [
        ("Win Rate", "78.4%"), ("Profit Factor", "2.81"), ("Avg Win", "+61%"),
        ("Avg Loss", "-22%"), ("Sortino", "4.12"), ("Calmar", "6.8"),
        ("Contracts", "3"), ("Daily Loss", "-0.00%"), ("4pm Cutoff", "02:34")
    ]
    for i, (label, value) in enumerate(stats):
        with grid[i % 3]:
            st.markdown(f"**{label}**")
            st.markdown(f"<h3 style='font-size:1.6rem; margin:-12px 0 16px 0'>{value}</h3>", unsafe_allow_html=True)

with upper_right:
    st.subheader("Open Positions")
    if not st.session_state.opens:
        st.caption("No opens – chillin'")
    else:
        st.dataframe(pd.DataFrame(st.session_state.opens), use_container_width=True)

# === Bottom full-width chat ===
st.markdown("---")
chat_col, _ = st.columns([5, 1])
with chat_col:
    st.subheader("Chat with Snoogans")
    chat_container = st.container(height=640)
    with chat_container:
        for msg in st.session_state.chat[-20:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# === Chat input ===
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Snoogans thinkin'..."):
            # Calls the routing function!
            reply = get_snoogans_rant(prompt) 
        st.markdown(reply)
    
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()

# Auto-refresh
import time
time.sleep(7)
st.rerun()