# app.py – Ultra-compact Snoogans cockpit v2
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Snoogans", layout="wide", initial_sidebar_state="collapsed")

# --- Session state ---
if "equity" not in st.session_state:
    st.session_state.equity = 142350
    st.session_state.trades = []
    st.session_state.opens = []
    st.session_state.chat = [{"role": "assistant", "content": "Snoogans online. Tiny spreads only. Lunch money loading…"}]

# === Tiny top bar – 50% smaller numbers ===
c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.2, 1.2, 2])
with c1:
    st.markdown(f"**Account**  <span style='font-size:1.4rem'>${st.session_state.equity:,.0f}</span> <small>+$3,840</small>", unsafe_allow_html=True)
with c2:
    st.markdown("<small>Today</small><br><span style='font-size:1.2rem'>+1.42%</span>", unsafe_allow_html=True)
with c3:
    st.markdown("<small>Month</small><br><span style='font-size:1.2rem'>+9.81%</span>", unsafe_allow_html=True)
with c4:
    st.markdown("<small>Max DD</small><br><span style='font-size:1.2rem'>-4.1%</span>", unsafe_allow_html=True)
with c5:
    st.caption(f"{datetime.now().strftime('%H:%M:%S')} EST")

st.markdown("---")

# === Main layout – three tight columns ===
left, mid, right = st.columns([3, 2.2, 2], gap="medium")

with left:
    st.subheader("Equity Curve")
    dates = pd.date_range("2025-01-01", periods=90)
    eq = 140000 + np.cumsum(np.random.randn(90)*700 + 500)
    st.line_chart(eq, height=260, use_container_width=True)

    st.subheader("Open Positions")
    if st.session_state.opens:
        st.dataframe(pd.DataFrame(st.session_state.opens), hide_index=True, use_container_width=True, height=150)
    else:
        st.caption("No opens – chillin'")

with mid:
    st.subheader("Trade Log")
    recent = st.session_state.trades[-10:] if st.session_state.trades else []
    if recent:
        st.dataframe(pd.DataFrame(recent, columns=["Time","Message"]), hide_index=True, use_container_width=True, height=360)
    else:
        st.caption("Waiting for first spread…")

with right:
    st.subheader("Stats")
    grid = st.columns(3)
    stats = [
        ("Win Rate", "78.4%"), ("Profit Factor", "2.81"), ("Avg Win", "+61%"),
        ("Avg Loss", "-22%"), ("Sortino", "4.12"), ("Calmar", "6.8"),
        ("Contracts", "3"), ("Daily Loss", "-0.0%"), ("4pm Cutoff", "02:34")
    ]
    for i, (label, value) in enumerate(stats):
        with grid[i % 3]:
            st.metric(label, value)

    st.markdown("---")
    st.subheader("Chat with Snoogans")

    # Chat messages directly above input
    chat_container = st.container(height=320)
    with chat_container:
        for msg in st.session_state.chat[-12:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# Input pinned at absolute bottom
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.chat.append({"role": "user", "content": prompt})
    reply = "Just dumped another 16-delta put spread while you were typing. +$310. Lunch money secured, ya filthy animal 🤙"
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()

# Auto-refresh
import time
time.sleep(7)
st.rerun()