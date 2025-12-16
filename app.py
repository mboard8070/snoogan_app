# app.py - Root Streamlit dashboard + bot control (real backtest stats + clean equity curve + safe refresh)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import threading
import subprocess
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()

# --- Path setup for RAG ---
current_dir = Path(__file__).parent
rag_dir = current_dir / "code" / "rag"
if str(rag_dir.resolve()) not in sys.path:
    sys.path.append(str(rag_dir.resolve()))

try:
    from snoogans_brain import SnoogansBrain
except Exception as e:
    st.error(f"RAG import failed: {e}")
    st.stop()

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

# --- Discord bot launch ---
project_root = Path(__file__).parent
discord_script = project_root / "code" / "utils" / "discord_bot.py"

def start_discord_bot():
    if discord_script.exists():
        subprocess.Popen(["python3", str(discord_script)])
        print(f"INFO: Snoogans Discord bot launched.")
    else:
        print(f"WARNING: discord_bot.py not found.")

if 'bot_started' not in st.session_state:
    threading.Thread(target=start_discord_bot, daemon=True).start()
    st.session_state.bot_started = True

# --- UI ---
st.set_page_config(page_title="Snoogans", layout="wide", initial_sidebar_state="collapsed")

# Session state init
if "equity" not in st.session_state:
    st.session_state.equity = float(os.getenv("STARTING_BALANCE", "100000"))
    st.session_state.trades = []
    st.session_state.opens = []
    st.session_state.chat = [{"role": "assistant", "content": "Snoogans online. Tiny spreads loading…"}]

# Sidebar bot status
st.sidebar.markdown("---")
if st.session_state.get('bot_started', False):
    st.sidebar.caption("🤖 Discord Bot: Active")
else:
    st.sidebar.warning("🤖 Discord Bot: Launch Error (Check terminal log)")

# Top bar
c1, c2, c3, c4, c5 = st.columns([2.2, 1.2, 1.2, 1.2, 2])
with c1:
    st.markdown(f"<div style='font-size:1.4rem; font-weight:bold'>${st.session_state.equity:,.0f}</div>", unsafe_allow_html=True)
with c5:
    st.caption(datetime.now().strftime("%H:%M:%S EST"))

st.markdown("---")

# Main layout
left, mid, right = st.columns([3, 2.2, 2], gap="medium")

with left:
    st.subheader("Equity Curve")
    
    starting_equity = float(os.getenv("STARTING_BALANCE", "100000"))
    
    if st.session_state.trades:
        pnl_list = [t.get('pnl', 0) for t in st.session_state.trades]
        cumulative = np.cumsum([0] + pnl_list)
        eq = starting_equity + cumulative
        dates = pd.date_range(datetime.now() - timedelta(days=len(eq)-1), periods=len(eq))
        chart_df = pd.DataFrame({'Equity': eq}, index=dates)
        st.line_chart(chart_df, height=260)
    else:
        dates = pd.date_range(datetime.now() - timedelta(days=14), periods=15)
        eq = [starting_equity] * 15
        chart_df = pd.DataFrame({'Equity': eq}, index=dates)
        st.line_chart(chart_df, height=260)
        st.caption("No trades yet — curve starts flat at starting balance")

    st.subheader("Trade Log")
    if st.session_state.trades:
        st.dataframe(pd.DataFrame(st.session_state.trades[-10:]), hide_index=True, use_container_width=True, height=360)
    else:
        st.caption("Waiting for first trade…")

# Real backtest stats (live when trades exist)
with mid:
    st.subheader("Stats")
    
    trades = st.session_state.trades

    if trades:
        pnl_list = [t.get('pnl', 0) for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]

        total_trades = len(pnl_list)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')

        downside = [p for p in pnl_list if p < 0]
        downside_dev = np.std(downside) if downside else 0.0
        sortino = np.mean(pnl_list) / downside_dev if downside_dev > 0 else float('inf')

        returns = np.array(pnl_list) / starting_equity
        cumulative = np.cumsum(returns)
        drawdowns = np.maximum.accumulate(cumulative) - cumulative
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0
        calmar = (cumulative[-1] / len(pnl_list) * 252) / max_dd if max_dd > 0 else float('inf')
    else:
        win_rate = avg_win = avg_loss = profit_factor = sortino = calmar = 0.0

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Win Rate (All Time)", value=f"{win_rate:.1f}%")
        st.metric(label="Profit Factor", value=f"{profit_factor:.1f}" if profit_factor != float('inf') else "∞")
        st.metric(label="Avg Win", value=f"{avg_win:+.1f}")
        st.metric(label="Sortino Ratio", value=f"{sortino:.2f}" if sortino != float('inf') else "∞")
    
    with col2:
        st.metric(label="Avg Loss", value=f"{avg_loss:+.1f}" if avg_loss else "N/A")
        st.metric(label="Calmar Ratio", value=f"{calmar:.2f}" if calmar != float('inf') else "∞")
        st.metric(label="Contracts", value=str(int(os.getenv("CONTRACT_SIZE", "1"))))
        st.metric(label="Daily Loss", value="0.00%")
        st.metric(label="2 PM Cutoff", value="Active")

with right:
    st.subheader("Open Positions")
    if st.session_state.opens:
        st.dataframe(pd.DataFrame(st.session_state.opens), use_container_width=True)
    else:
        st.caption("No opens")

# Chat
st.markdown("---")
st.subheader("Chat with Snoogans")
chat_container = st.container(height=600)
with chat_container:
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask Snoogans…"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Snoogans thinkin'..."):
            reply = get_snoogans_rant(prompt)
        st.markdown(reply)
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()

# Safe auto-refresh fallback (works on all versions)
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if (datetime.now() - st.session_state.last_refresh).seconds > 15:
    st.session_state.last_refresh = datetime.now()
    st.rerun()