# app.py – Snoogans Cockpit v3 (dashboard + Discord bot auto-launch fixed)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import threading
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Snoogans", layout="wide", initial_sidebar_state="collapsed")

# === LLM Setup ===
llm = OllamaLLM(model="supertrader:latest", temperature=0.7)

rationale_prompt = PromptTemplate.from_template(
    "Explain this like a Red Bank degenerate: {details}"
)
rationale_chain = rationale_prompt | llm

def get_snoogans_rant(details: str) -> str:
    return rationale_chain.invoke({"details": details})

# === Auto-launch Discord bot (fixed path + python3) ===
def start_discord_bot():
    discord_script = "../utils/discord_bot.py"  # exact relative path from dashboard folder
    full_path = os.path.join(os.path.dirname(__file__), discord_script)
    
    if os.path.exists(full_path):
        # Use python3 for Ubuntu zen, inherit dashboard venv
        subprocess.Popen(["python3", full_path])
        st.success("Snoogans Discord bot launched – @mention ready for Jersey rants")
    else:
        st.warning(f"discord_bot.py not found at {full_path} – alerts/mentions offline")

# Fire bot in background on dashboard load
threading.Thread(target=start_discord_bot, daemon=True).start()

# Session state
if "equity" not in st.session_state:
    st.session_state.equity = 142350
    st.session_state.trades = []
    st.session_state.opens = []
    st.session_state.chat = [{"role": "assistant", "content": "Snoogans online. Tiny spreads. Lunch money loading…"}]

# === Top bar ===
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

# === Upper row ===
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
            reply = get_snoogans_rant(prompt)
        st.markdown(reply)
    
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()

# Auto-refresh
import time
time.sleep(7)
st.rerun()