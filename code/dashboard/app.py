import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Snoogans", layout="wide", initial_sidebar_state="collapsed")

# Fake live data (replace later with real bot)
if "equity" not in st.session_state:
    st.session_state.equity = 142350
    st.session_state.trades = []
    st.session_state.opens = []
    st.session_state.chat_history = [{"role": "assistant", "content": "Snoogans online. Tiny spreads only. Lunch money loading…"}]

# === TOP BAR – pinned, tiny, always visible ===
c1, c2, c3, c4, c5, c6 = st.columns([1.8,1,1,1,1,2])
with c1: st.metric("Account", f"${st.session_state.equity:,.0f}", "+$3,840")
with c2: st.metric("Today", "+1.42%", "+$2,020")
with c3: st.metric("Month", "+9.81%", "+$12,740")
with c4: st.metric("DD", "-4.1%")
with c5: st.metric("Open", len(st.session_state.opens))
with c6: st.caption(f"Updated {datetime.now().strftime('%H:%M%S')} EST")

st.markdown("---")

# === MAIN ROW – everything else squeezed ===
left, mid, right = st.columns([3.2, 2, 1.6], gap="medium")

with left:
    st.subheader("Equity Curve")
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    eq = np.cumsum(np.random.randn(90)*800 + 600) + 140000
    st.line_chart(eq, height=280, use_container_width=True)

    st.subheader("Open Positions")
    if st.session_state.opens:
        st.dataframe(pd.DataFrame(st.session_state.opens), hide_index=True, use_container_width=True, height=180)
    else:
        st.info("No opens — chillin’")

with mid:
    st.subheader("Trade Log")
    if st.session_state.trades:
        log_df = pd.DataFrame(st.session_state.trades[-12:], columns=["Time","Message"])
        st.dataframe(log_df, hide_index=True, use_container_width=True, height=380)
    else:
        st.info("Waiting for first spread…")

with right:
    st.subheader("Stats")
    st.metric("Win Rate", "78.4%")
    st.metric("Profit Factor", "2.81")
    st.metric("Avg Win", "+61%")
    st.metric("Avg Loss", "-22%")
    st.metric("Sortino", "4.12")
    st.markdown("---")
    
    st.subheader("Chat with Snoogans")
    chat = st.container(height=380)
    for msg in st.session_state.chat_history[-10:]:
        with chat.chat_message(msg["role"]):
            st.write(msg["content"])

# === Chat input pinned at very bottom ===
if prompt := st.chat_input("Talk to me — why that trade, next setup, size, whatever…"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    # instant Jersey reply
    reply = "Got it boss. Just dumped another 16-delta put spread while you were typing. Lunch money secured 🤙"
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()

# Auto-refresh every 8 seconds
import time
time.sleep(8)
st.rerun()