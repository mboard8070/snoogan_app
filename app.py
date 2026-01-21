"""
Iron Spark Trading Dashboard
A Streamlit-based trading dashboard with multiple strategy timeframes,
RAG-powered chat, and Discord notifications.
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, time as dt_time, date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AppConfig:
    """Application configuration constants."""
    PAGE_TITLE: str = "Iron Spark"
    LAYOUT: str = "wide"
    SIDEBAR_STATE: str = "collapsed"

    BATCH_SIZE: int = 30
    MAX_TRADES_FOR_PROGRESS: int = 300

    MARKET_OPEN: dt_time = dt_time(9, 30)
    MARKET_CLOSE: dt_time = dt_time(16, 0)

    HEADER_REFRESH_SECONDS: int = 10
    STRATEGY_1M_REFRESH_SECONDS: int = 12
    STRATEGY_15M_REFRESH_SECONDS: int = 12  # Frequent refresh for P&L tracking (uses 15m bars for indicators)
    STRATEGY_15M_MONITOR_SECONDS: int = 60  # Position monitoring between full cycles
    STRATEGY_SCALP_REFRESH_SECONDS: int = 60
    LEARNER_REFRESH_SECONDS: int = 60

    CHAT_BOX_HEIGHT: int = 280
    EQUITY_CHART_HEIGHT: int = 160

    RAG_CHUNK_SIZE: int = 400
    RAG_CHUNK_OVERLAP: int = 150

    OLLAMA_MODEL: str = "snoogans:latest"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"

    # Column layout ratios
    HEADER_COLUMNS: tuple = (2, 2, 4, 2)  # equity, pnl, progress, market
    MAIN_COLUMNS: tuple = (1, 1)  # left, right


CONFIG = AppConfig()


# =============================================================================
# PATH SETUP & ENVIRONMENT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
KNOWLEDGE_DIR = CODE_DIR / "data" / "knowledge"  # Trades are saved here by strategies
TRADES_FILE = KNOWLEDGE_DIR / "trades.json"
ARCHIVE_DIR = PROJECT_ROOT / "data" / "knowledge"  # Archived batches go here
ENV_PATH = PROJECT_ROOT / "variables.env"
SHARED_EQUITY_FILE = PROJECT_ROOT / "shared_equity.json"
SHARED_EQUITY_HISTORY_FILE = PROJECT_ROOT / "shared_equity_history.json"

# Add required paths to sys.path
_paths_to_add = [
    PROJECT_ROOT,
    CODE_DIR,
    CODE_DIR / "trader",
    CODE_DIR / "rag",
]
for path in _paths_to_add:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

load_dotenv(ENV_PATH)


# =============================================================================
# IMPORTS (after path setup)
# =============================================================================

from strategy import TradingStrategy
from strategy_15m import TradingStrategy15m
from scalp_strategy import ScalpStrategy
from snoogans_brain import SnoogansBrain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from discord_notifier import (
    send_greeting_if_needed,
    send_eod_if_needed,
    set_live_mode,
)

# Import adaptive learner for monitoring
try:
    from adaptive_learner import get_learner
    LEARNER_AVAILABLE = True
except ImportError:
    LEARNER_AVAILABLE = False

# Import indicator stats functions
try:
    from indicators import (
        get_trade_history_summary,
        get_binomial_win_probability,
        get_conditional_expected_value,
        get_kelly_criterion,
    )
    INDICATOR_STATS_AVAILABLE = True
except ImportError:
    INDICATOR_STATS_AVAILABLE = False

# Path to indicator stats state file
INDICATOR_STATS_FILE = PROJECT_ROOT / "code" / "trader" / "indicator_stats_state.json"


# =============================================================================
# SHARED EQUITY HELPERS
# =============================================================================

def get_shared_equity() -> float:
    """Read the shared equity value from the JSON file."""
    if SHARED_EQUITY_FILE.exists():
        try:
            with open(SHARED_EQUITY_FILE, 'r') as f:
                data = json.load(f)
                return data.get('equity', float(os.getenv("STARTING_BALANCE", "100000")))
        except (json.JSONDecodeError, KeyError):
            pass
    return float(os.getenv("STARTING_BALANCE", "100000"))


def update_shared_equity(new_equity: float) -> None:
    """Update the shared equity value in the JSON file."""
    data = {
        'equity': new_equity,
        'last_updated': datetime.now().isoformat()
    }
    with open(SHARED_EQUITY_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def get_shared_equity_history() -> list:
    """Read the shared equity history from the JSON file."""
    if SHARED_EQUITY_HISTORY_FILE.exists():
        try:
            with open(SHARED_EQUITY_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                if isinstance(history, list):
                    return history
        except (json.JSONDecodeError, KeyError):
            pass
    # Return starting balance as initial history if no history exists
    return [float(os.getenv("STARTING_BALANCE", "100000"))]


# =============================================================================
# STREAMLIT PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title=CONFIG.PAGE_TITLE,
    layout=CONFIG.LAYOUT,
    initial_sidebar_state=CONFIG.SIDEBAR_STATE,
)

st.markdown(
    """
    <style>
    /* Main container - tighter padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }

    /* Reduce gaps between elements */
    div[data-testid="stVerticalBlock"] { gap: 0.3rem; }

    /* Compact metrics */
    .stMetric {
        background-color: #1e1e1e;
        padding: 8px;
        border-radius: 5px;
    }
    [data-testid="stMetricValue"] { font-size: 1.2rem; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem; }

    /* Compact dataframes */
    .stDataFrame { font-size: 0.85rem; }
    [data-testid="stDataFrame"] > div { max-height: 200px; overflow-y: auto; }

    /* Tabs - compact */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 6px 12px; font-size: 0.85rem; }

    /* Code blocks - scrollable with max height */
    .stCodeBlock { max-height: 250px; overflow-y: auto; }
    pre { font-size: 0.75rem !important; }

    /* Subheaders - smaller */
    h3 { font-size: 1rem !important; margin-bottom: 0.3rem !important; }
    h2 { font-size: 1.1rem !important; margin-bottom: 0.3rem !important; }

    /* Charts - responsive height */
    [data-testid="stVegaLiteChart"] { max-height: 180px; }

    /* Expanders - compact */
    .streamlit-expanderHeader { font-size: 0.85rem; padding: 0.4rem; }

    /* Progress bar - thinner */
    .stProgress > div > div { height: 8px; }

    /* Responsive columns - stack on smaller screens */
    @media (max-width: 1200px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }

    /* Hide hamburger menu for more space */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Scrollable containers */
    .scrollable-container {
        max-height: 300px;
        overflow-y: auto;
        overflow-x: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# RAG BRAIN & ROUTER SETUP
# =============================================================================

@st.cache_resource
def load_brain() -> SnoogansBrain:
    """Load and cache the RAG brain instance."""
    return SnoogansBrain(
        chunk_size=CONFIG.RAG_CHUNK_SIZE,
        chunk_overlap=CONFIG.RAG_CHUNK_OVERLAP,
    )


@st.cache_resource
def get_router_chain():
    """Create and cache the router chain for query classification."""
    llm = OllamaLLM(
        model=CONFIG.OLLAMA_MODEL,
        temperature=0.0,
        base_url=CONFIG.OLLAMA_BASE_URL,
    )
    prompt = PromptTemplate.from_template(
        "Classify as 'MANIFESTO' or 'GENERAL'. Question: {question}\nAnswer:"
    )
    return prompt | llm | StrOutputParser()


RAG_BRAIN = load_brain()
ROUTER_CHAIN = get_router_chain()


def get_snoogans_response(question: str) -> str:
    """
    Route a question through the brain's smart router.

    Handles:
        - Temporal queries (today, yesterday, this week, etc.)
        - Trade/strategy questions (uses RAG)
        - General chat

    Args:
        question: The user's question to process.

    Returns:
        The response from the appropriate handler.
    """
    return RAG_BRAIN.ask(question)


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state() -> None:
    """Initialize all session state variables."""
    if "strategy_1m" not in st.session_state:
        st.session_state.strategy_1m = TradingStrategy()
    
    if "strategy_15m" not in st.session_state:
        st.session_state.strategy_15m = TradingStrategy15m()
    
    if "strategy_scalp" not in st.session_state:
        st.session_state.strategy_scalp = ScalpStrategy()
    
    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "assistant", "content": "Router active. Standing by."}
        ]


def get_trade_count() -> int:
    """
    Get the total number of trades from the trades file.

    Returns:
        The count of trades, or 0 if file doesn't exist or is invalid.
    """
    if not TRADES_FILE.exists():
        return 0
    try:
        return len(json.loads(TRADES_FILE.read_text()))
    except (json.JSONDecodeError, IOError):
        return 0


def archive_trades_if_batch_complete() -> bool:
    """
    Archive trades when a batch of 30 is complete.

    Archives exactly one batch of 30, keeping remaining trades active.

    Returns:
        True if a batch was archived, False otherwise.
    """
    if not TRADES_FILE.exists():
        return False

    try:
        trades = json.loads(TRADES_FILE.read_text())
        if len(trades) < CONFIG.BATCH_SIZE:
            return False

        # Ensure archive directory exists
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

        # Take exactly one batch of 30
        batch = trades[: CONFIG.BATCH_SIZE]
        remaining = trades[CONFIG.BATCH_SIZE :]

        # Count existing batches for numbering
        existing_batches = len(list(ARCHIVE_DIR.glob("trades_batch_*.json")))
        batch_num = existing_batches + 1

        # Create archive file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = ARCHIVE_DIR / f"trades_batch_{batch_num:03d}_{timestamp}.json"

        # Archive the batch
        archive_file.write_text(json.dumps(batch, indent=2, default=str))

        # Keep remaining trades active
        TRADES_FILE.write_text(json.dumps(remaining, indent=2, default=str))

        print(f"[BRAIN] Archived batch {batch_num} ({CONFIG.BATCH_SIZE} trades) -> {archive_file.name}")
        return True

    except (json.JSONDecodeError, IOError) as e:
        print(f"[BRAIN] Archive error: {e}")
        return False


def refresh_dashboard_state() -> int:
    """
    Refresh the dashboard state for display purposes.

    Also checks if a batch of trades is ready to be archived.

    Returns:
        The current trade count (after any archiving).
    """
    # NOTE: Do NOT call _load_state() here - it overwrites in-memory positions
    # and causes duplicate Discord messages. Strategies manage their own state.
    # The strategies load state once in __init__ and save after each trade.

    # Archive trades if batch is complete
    archive_trades_if_batch_complete()

    return get_trade_count()


def get_starting_balance() -> float:
    """Get the starting balance from environment variables."""
    return float(os.getenv("STARTING_BALANCE", 100000))


def is_market_open() -> bool:
    """Check if the market is currently open."""
    current_time = datetime.now().time()
    return CONFIG.MARKET_OPEN <= current_time <= CONFIG.MARKET_CLOSE


def format_strategy_status(strategy, strategy_type: str = "spread") -> str:
    """
    Format strategy status for log display.

    Args:
        strategy: The strategy object to format status for.
        strategy_type: Either "spread" (1m/15m) or "scalp" for formatting.

    Returns:
        Formatted status string with position monitoring, PnL, and size info.
    """
    try:
        status = strategy.get_status()
        lines = []

        # Calculate trades_today from closed_trades with today's exit_time
        today_str = date.today().isoformat()
        closed_trades = getattr(strategy, 'closed_trades', [])
        trades_today_count = sum(
            1 for t in closed_trades
            if t.get('exit_time', '').startswith(today_str)
        )

        # Header with PnL summary
        lines.append(f"{'='*50}")
        lines.append(f"Daily P&L: ${status['daily_pnl']:+.2f} | "
                    f"Loss Limit: ${status['daily_loss_limit']:.2f}")
        lines.append(f"Equity: ${status['equity']:,.2f} | "
                    f"Trades Today: {trades_today_count}")
        lines.append(f"Can Trade: {'Yes' if status['can_trade'] else 'No'}")
        lines.append(f"{'='*50}")

        # Current trend indicators
        trends = status.get('current_trends', {})
        if trends:
            trend_parts = []
            for ticker in ["SPY", "QQQ", "IWM"]:
                trend = trends.get(ticker, "---")
                if trend == "bull":
                    trend_parts.append(f"{ticker}: BULL")
                elif trend == "bear":
                    trend_parts.append(f"{ticker}: BEAR")
                elif trend == "chop":
                    trend_parts.append(f"{ticker}: CHOP")
                else:
                    trend_parts.append(f"{ticker}: ---")
            lines.append("TRENDS: " + " | ".join(trend_parts))
            lines.append("-" * 50)

        # Position details
        if status['open_positions'] == 0:
            lines.append("No open positions")
        else:
            lines.append(f"OPEN POSITIONS ({status['open_positions']}):")
            lines.append("-" * 40)

            for ticker, pos in status['positions'].items():
                if strategy_type == "scalp":
                    # Scalp strategy (long options)
                    direction = "CALL" if pos.get("is_call") else "PUT"
                    strike = pos.get("strike_price", 0)
                    debit = pos.get("debit", 0)
                    contracts = pos.get("contracts", 1)
                    current_mark = pos.get("current_mark", pos.get("best_mark", debit))
                    best_mark = pos.get("best_mark", debit)
                    trail_active = pos.get("trail_active", False)
                    trail_level = pos.get("trail_level")
                    entry_time = pos.get("entry_time", "unknown")

                    # Calculate unrealized P&L
                    unrealized = (current_mark - debit) * contracts * 100
                    pnl_pct = ((current_mark - debit) / debit * 100) if debit > 0 else 0
                    best_unrealized = (best_mark - debit) * contracts * 100

                    lines.append(f"  {ticker}: LONG {direction} @ {strike:.1f}")
                    lines.append(f"    Debit: ${debit:.2f} x {contracts} = ${debit * contracts * 100:.2f}")
                    lines.append(f"    Current: ${current_mark:.2f} | Unrealized: ${unrealized:+.2f} ({pnl_pct:+.1f}%)")
                    lines.append(f"    Best: ${best_mark:.2f} | Best P&L: ${best_unrealized:+.2f}")
                    if trail_active:
                        lines.append(f"    Trail Active @ ${trail_level:.2f}")
                    lines.append(f"    Entry: {str(entry_time).split('T')[0] if 'T' in str(entry_time) else entry_time}")
                else:
                    # Credit spread strategies (1m/15m)
                    direction = "PUT" if pos.get("is_put") else "CALL"
                    short = pos.get("short", 0)
                    long = pos.get("long", 0)
                    credit = pos.get("credit", 0)
                    contracts = pos.get("contracts", 1)
                    current_value = pos.get("current_value", pos.get("best_value", credit))
                    best_value = pos.get("best_value", credit)
                    trail_active = pos.get("trail_active", False)
                    trail_level = pos.get("trail_level")
                    entry_time = pos.get("entry_time", "unknown")

                    # For credit spreads, lower value = more profit
                    unrealized = (credit - current_value) * contracts * 100
                    best_unrealized = (credit - best_value) * contracts * 100
                    max_profit = credit * contracts * 100

                    lines.append(f"  {ticker}: {direction} SPREAD {short:.1f}/{long:.1f}")
                    lines.append(f"    Credit: ${credit:.2f} x {contracts} = ${credit * contracts * 100:.2f}")
                    lines.append(f"    Current: ${current_value:.2f} | Unrealized: ${unrealized:+.2f}")
                    lines.append(f"    Best: ${best_value:.2f} | Best P&L: ${best_unrealized:+.2f}")
                    lines.append(f"    Max Profit: ${max_profit:.2f}")
                    if trail_active:
                        lines.append(f"    Trail Active @ ${trail_level:.2f}")
                    lines.append(f"    Entry: {str(entry_time).split('T')[0] if 'T' in str(entry_time) else entry_time}")

                lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error formatting status: {e}"


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header(placeholder: st.delta_generator.DeltaGenerator) -> None:
    """Render the dashboard header with metrics and progress."""
    trade_count = refresh_dashboard_state()

    # Get shared equity (single source of truth)
    current_equity = get_shared_equity()

    # Combined daily P&L across all strategies
    pnl = st.session_state.strategy_1m.daily_pnl + st.session_state.strategy_15m.daily_pnl + st.session_state.strategy_scalp.daily_pnl

    # Count archived batches
    archived_batches = 0
    if ARCHIVE_DIR.exists():
        archived_batches = len(list(ARCHIVE_DIR.glob("trades_batch_*.json")))
    total_learned = archived_batches * CONFIG.BATCH_SIZE

    with placeholder.container():
        col_equity, col_pnl, col_progress, col_market = st.columns(
            list(CONFIG.HEADER_COLUMNS)
        )

        col_equity.metric("Equity", f"${current_equity:,.2f}")
        col_pnl.metric("Daily PnL", f"${pnl:+.2f}")

        with col_progress:
            st.caption(
                f"Brain Learning → {trade_count}/{CONFIG.BATCH_SIZE} "
                f"(Total learned: {total_learned})"
            )
            progress_value = trade_count / CONFIG.BATCH_SIZE
            st.progress(progress_value)

            trades_to_batch = CONFIG.BATCH_SIZE - trade_count
            if trades_to_batch == CONFIG.BATCH_SIZE:
                st.caption("Starting new batch...")
            else:
                st.caption(f"{trades_to_batch} trades until next brain update")

        market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
        col_market.metric("Market", market_status)

        st.divider()


def render_positions(strategy, label: str, empty_message: str | None = None) -> None:
    """
    Render positions for a given strategy (1m/15m credit spreads).

    Args:
        strategy: The strategy object containing positions.
        label: Display label for the positions section.
        empty_message: Optional message to show when no positions exist.
    """
    if not strategy.positions:
        if empty_message:
            st.info(empty_message)
        return

    st.write(f"**Positions ({label}):**")

    positions_data = []
    for ticker, pos in strategy.positions.items():
        direction = "PUT" if pos.get("is_put") else "CALL"
        short = pos.get("short", 0)
        long = pos.get("long", 0)
        credit = pos.get("credit", 0)
        contracts = pos.get("contracts", 1)
        best_value = pos.get("best_value", credit)
        trail_active = pos.get("trail_active", False)

        # For credit spreads: profit = credit - current value
        unrealized = (credit - best_value) * contracts * 100

        positions_data.append({
            "Ticker": ticker,
            "Type": direction,
            "Strikes": f"{short:.0f}/{long:.0f}",
            "Credit": credit,
            "P/L": unrealized,
            "Trail": "✓" if trail_active else "",
        })

    st.dataframe(
        pd.DataFrame(positions_data),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Credit": st.column_config.NumberColumn(format="$%.2f"),
            "P/L": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


def render_positions_scalp() -> None:
    """Render scalp strategy positions."""
    if not st.session_state.strategy_scalp.positions:
        st.info("No active scalp positions — waiting for trend signal.")
        return

    st.write("**Positions (Scalp):**")

    positions_data = []
    for ticker, pos in st.session_state.strategy_scalp.positions.items():
        direction = "CALL" if pos.get("is_call", False) else "PUT"
        strike = pos.get("strike_price", 0.0)
        debit = pos.get("debit", 0.0)
        contracts = pos.get("contracts", 1)
        best_mark = pos.get("best_mark", debit)
        trail_active = pos.get("trail_active", False)

        # For long options: profit = current - debit
        unrealized = (best_mark - debit) * contracts * 100
        pnl_pct = ((best_mark - debit) / debit * 100) if debit > 0 else 0

        positions_data.append({
            "Ticker": ticker,
            "Type": direction,
            "Strike": f"{strike:.0f}",
            "Debit": debit,
            "P/L": unrealized,
            "P/L %": pnl_pct,
            "Trail": "✓" if trail_active else "",
        })

    st.dataframe(
        pd.DataFrame(positions_data),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Debit": st.column_config.NumberColumn(format="$%.2f"),
            "P/L": st.column_config.NumberColumn(format="$%.2f"),
            "P/L %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )


def render_chat_interface(container: Any) -> None:
    """Render the chat interface."""
    chat_box = container.container(height=CONFIG.CHAT_BOX_HEIGHT)
    
    with chat_box:
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask Iron Spark..."):
        st.session_state.chat.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            reply = get_snoogans_response(prompt)
            st.markdown(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})
        
        st.rerun()


# =============================================================================
# MAIN DASHBOARD LAYOUT
# =============================================================================

def render_left_column() -> None:
    """Render the left column with performance and positions."""
    st.subheader("Performance & Position")

    st.line_chart(
        pd.Series(get_shared_equity_history(), name="Equity"),
        height=CONFIG.EQUITY_CHART_HEIGHT,
    )

    # Position display with auto-refresh
    position_placeholder = st.empty()

    @st.fragment(run_every=CONFIG.HEADER_REFRESH_SECONDS)
    def update_positions():
        with position_placeholder.container(border=True):
            render_positions(
                st.session_state.strategy_1m,
                "1m",
                "Scanning SPY / QQQ / IWM for .30 Delta setups...",
            )
            render_positions(st.session_state.strategy_15m, "15m")
            render_positions_scalp()

    update_positions()


def render_indicator_stats() -> None:
    """Render the indicator statistics tab."""
    if not INDICATOR_STATS_AVAILABLE:
        st.warning("Indicator stats module not available")
        return

    # Stats display with auto-refresh
    stats_placeholder = st.empty()

    @st.fragment(run_every=30)
    def update_stats():
        with stats_placeholder.container():
            # Load stats from file
            regime_stats = {}
            trade_history = []
            last_updated = None

            if INDICATOR_STATS_FILE.exists():
                try:
                    with open(INDICATOR_STATS_FILE, 'r') as f:
                        state = json.load(f)
                        trade_history = state.get("trade_history", [])
                        regime_stats = state.get("regime_stats", {})
                        last_updated = state.get("last_updated")
                except Exception as e:
                    st.error(f"Error loading stats: {e}")
                    return

            # Get summary
            try:
                summary = get_trade_history_summary()
            except Exception as e:
                summary = {"total_trades": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0, "total_pnl": 0}

            # Summary metrics header
            st.subheader("📊 Binomial Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Trades", summary.get("total_trades", 0))
            col2.metric("Win Rate", f"{summary.get('win_rate', 0):.1f}%")
            col3.metric("Avg Win", f"${summary.get('avg_win', 0):.2f}")
            col4.metric("Avg Loss", f"${summary.get('avg_loss', 0):.2f}")
            col5.metric("Total P&L", f"${summary.get('total_pnl', 0):+.2f}")

            st.divider()

            # Regime breakdown
            st.subheader("Win Rate by Regime")
            regime_cols = st.columns(3)

            for i, regime in enumerate(["bull", "bear", "chop"]):
                with regime_cols[i]:
                    st.markdown(f"**{regime.upper()}**")

                    # Raw stats
                    rs = regime_stats.get(regime, {})
                    wins = rs.get("wins", 0)
                    losses = rs.get("losses", 0)
                    total = wins + losses

                    if total > 0:
                        raw_win_rate = wins / total * 100
                        st.write(f"Record: {wins}W / {losses}L")
                        st.write(f"Win Rate: {raw_win_rate:.1f}%")

                        # Binomial confidence interval
                        try:
                            binomial = get_binomial_win_probability(regime=regime)
                            ci = binomial.get("confidence_interval", [0, 1])
                            ev = binomial.get("expected_value", 0)
                            reliable = binomial.get("reliable", False)

                            st.write(f"95% CI: [{ci[0]*100:.0f}% - {ci[1]*100:.0f}%]")
                            st.write(f"Expected Value: ${ev:.2f}")

                            if reliable:
                                st.success("Statistically Reliable")
                            else:
                                st.warning(f"Need {20 - total} more trades")
                        except Exception:
                            pass

                        # Conditional EV
                        try:
                            cond_ev = get_conditional_expected_value(regime)
                            rec = cond_ev.get("recommendation", "neutral")
                            if rec == "favorable":
                                st.success(f"Favorable")
                            elif rec == "unfavorable":
                                st.error(f"Unfavorable")
                            else:
                                st.info(f"Neutral")
                        except Exception:
                            pass
                    else:
                        st.caption("No data yet")

            st.divider()

            # Kelly Criterion
            if summary.get("total_trades", 0) >= 5:
                try:
                    win_rate = summary.get("win_rate", 50) / 100
                    avg_win = summary.get("avg_win", 100)
                    avg_loss = abs(summary.get("avg_loss", -100))
                    if avg_loss > 0:
                        kelly = get_kelly_criterion(win_rate, avg_win, avg_loss)

                        st.subheader("Kelly Criterion")
                        kelly_cols = st.columns(4)
                        kelly_cols[0].metric("Full Kelly", f"{kelly.get('kelly_fraction', 0)*100:.1f}%")
                        kelly_cols[1].metric("Half Kelly", f"{kelly.get('half_kelly', 0)*100:.1f}%")
                        kelly_cols[2].metric("Max Bet", f"{kelly.get('max_bet_percent', 0):.1f}%")
                        kelly_cols[3].metric("Edge", kelly.get("recommendation", "unknown").replace("_", " ").upper())

                        st.divider()
                except Exception:
                    pass

            # Recent trades
            if trade_history:
                st.subheader("Recent Trades")
                recent_trades = trade_history[-10:][::-1]  # Last 10, reversed
                trades_data = []
                for t in recent_trades:
                    trades_data.append({
                        "Time": str(t.get("entry_time", ""))[:16],
                        "Ticker": t.get("ticker", ""),
                        "Setup": t.get("setup_type", ""),
                        "Regime": t.get("regime", "").upper(),
                        "P&L": t.get("pnl", 0),
                        "Result": "WIN" if t.get("win") else "LOSS"
                    })
                st.dataframe(
                    pd.DataFrame(trades_data),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "P&L": st.column_config.NumberColumn(format="$%.2f"),
                    }
                )

            if last_updated:
                st.caption(f"Last updated: {last_updated}")

    update_stats()


def render_right_column() -> None:
    """Render the right column with chat and trading logs."""
    tab_chat, tab_logs, tab_learner, tab_stats = st.tabs(["💬 Iron Spark Chat", "📜 Trading Logs", "🧠 RL Learner", "📊 Binomial Stats"])

    with tab_chat:
        render_chat_interface(tab_chat)

    with tab_logs:
        render_trading_logs()

    with tab_learner:
        render_learner_monitor()

    with tab_stats:
        render_indicator_stats()


def render_learner_monitor() -> None:
    """Render the RL Learner monitoring tab."""
    if not LEARNER_AVAILABLE:
        st.warning("Adaptive Learner not available")
        return

    # Debug: show state file info
    state_file = Path("/home/mboard76/nvidia-workbench/snoogan_app/code/rag/adaptive_learner_state.json")
    if state_file.exists():
        import os
        mtime = datetime.fromtimestamp(os.path.getmtime(state_file))
        st.caption(f"DEBUG: State file exists, last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.caption("DEBUG: State file does NOT exist")

    # Learner display with auto-refresh
    learner_placeholder = st.empty()

    @st.fragment(run_every=CONFIG.LEARNER_REFRESH_SECONDS)
    def update_learner():
        with learner_placeholder.container():
            learner = get_learner()
            stats = learner.get_stats()
            decisions = learner.get_decision_summary()

            # Compact header metrics in a single row
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("States", stats['total_states'])
            col2.metric("Trades", stats['total_trades'])
            col3.metric("Win %", f"{stats['win_rate']:.0f}%")
            col4.metric("P&L", f"${stats['total_pnl']:+.0f}")
            col5.metric("Epsilon", f"{stats['epsilon']:.0%}")
            good_skips = stats.get('counterfactual_good_skips', 0)
            bad_skips = stats.get('counterfactual_bad_skips', 0)
            col6.metric("Skips", f"{good_skips}/{good_skips+bad_skips}")

            # Two columns with expanders
            left, right = st.columns(2)

            with left:
                with st.expander("Best States to Trade", expanded=True):
                    best_states = learner.get_best_states(5)
                    if best_states:
                        best_data = []
                        for s in best_states:
                            parts = s['state'].split('|')
                            best_data.append({
                                "Tkr": parts[0],
                                "Trend": parts[1],
                                "RSI": parts[2],
                                "Q": f"{s['q_enter']:+.3f}"
                            })
                        st.dataframe(pd.DataFrame(best_data), use_container_width=True, hide_index=True, height=150)
                    else:
                        st.caption("No states learned yet")

            with right:
                with st.expander("States to Avoid", expanded=True):
                    worst_states = learner.get_worst_states(5)
                    if worst_states:
                        worst_data = []
                        for s in worst_states:
                            parts = s['state'].split('|')
                            worst_data.append({
                                "Tkr": parts[0],
                                "Trend": parts[1],
                                "RSI": parts[2],
                                "Q": f"{s['q_enter']:+.3f}"
                            })
                        st.dataframe(pd.DataFrame(worst_data), use_container_width=True, hide_index=True, height=150)

            # Recent decisions in expander
            with st.expander("Recent Decisions", expanded=False):
                # Debug logging
                st.caption(f"DEBUG: decision_history length = {len(learner.decision_history)}")
                st.caption(f"DEBUG: decisions summary = {decisions}")
                recent = learner.get_recent_decisions(10)
                st.caption(f"DEBUG: recent_decisions count = {len(recent)}")

                if decisions['total'] > 0:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Enters", decisions['enters'], f"{decisions['enter_rate']:.0f}%")
                    col2.metric("Skips", decisions['skips'], f"{100-decisions['enter_rate']:.0f}%")
                    col3.metric("Explores", decisions['explorations'], f"{decisions['exploration_rate']:.0f}%")

                    if recent:
                        decision_data = []
                        for d in recent:
                            decision_data.append({
                                "Time": d.get('timestamp', '')[-8:],
                                "State": d.get('state_key', '')[:25],
                                "Action": d.get('action', '').upper(),
                                "Exp": "Y" if d.get('exploration') else ""
                            })
                        st.dataframe(pd.DataFrame(decision_data), use_container_width=True, hide_index=True, height=200)
                    else:
                        st.caption("No recent decisions returned")
                else:
                    st.caption("No decisions yet (decisions['total'] == 0)")

    update_learner()


def render_trading_logs() -> None:
    """Render the trading logs section with auto-updating fragments."""
    st.write(f"Last Heartbeat: {datetime.now().strftime('%H:%M:%S')}")
    
    # 1-Minute Strategy Log
    st.subheader("🖥️ 1-Minute Iron Spark Log")
    log_display_1m = st.empty()
    
    @st.fragment(run_every=CONFIG.STRATEGY_1M_REFRESH_SECONDS)
    def sync_1m_cycle():
        output_buffer = io.StringIO()
        error_msg = None
        with redirect_stdout(output_buffer):
            try:
                st.session_state.strategy_1m.run_cycle()
            except Exception as e:
                error_msg = f"[1m] Cycle Error: {e}"
                print(error_msg)

        # Combine cycle output with position monitoring
        cycle_output = output_buffer.getvalue() or "Scanning..."
        status_output = format_strategy_status(
            st.session_state.strategy_1m, strategy_type="spread"
        )
        combined_output = f"{cycle_output}\n\n{status_output}"

        if error_msg:
            st.error(error_msg)
        log_display_1m.code(combined_output, language="bash", wrap_lines=True)

        send_greeting_if_needed()
        # Calculate TOTAL trades_today and daily_pnl from ALL strategies for EOD notification
        today_str = date.today().isoformat()
        closed_trades_1m = getattr(st.session_state.strategy_1m, 'closed_trades', [])
        closed_trades_15m = getattr(st.session_state.strategy_15m, 'closed_trades', [])
        closed_trades_scalp = getattr(st.session_state.strategy_scalp, 'closed_trades', [])
        total_trades_today = (
            sum(1 for t in closed_trades_1m if t.get('exit_time', '').startswith(today_str)) +
            sum(1 for t in closed_trades_15m if t.get('exit_time', '').startswith(today_str)) +
            sum(1 for t in closed_trades_scalp if t.get('exit_time', '').startswith(today_str))
        )
        total_daily_pnl = (
            st.session_state.strategy_1m.daily_pnl +
            st.session_state.strategy_15m.daily_pnl +
            st.session_state.strategy_scalp.daily_pnl
        )
        send_eod_if_needed(total_trades_today, total_daily_pnl)
        refresh_dashboard_state()

    sync_1m_cycle()

    # 15-Minute Strategy Log
    st.subheader("🖥️ 15-Minute Iron Spark Log")
    log_display_15m = st.empty()

    @st.fragment(run_every=CONFIG.STRATEGY_15M_REFRESH_SECONDS)
    def sync_15m_cycle():
        output_buffer = io.StringIO()
        error_msg = None
        with redirect_stdout(output_buffer):
            try:
                st.session_state.strategy_15m.run_cycle()
            except Exception as e:
                error_msg = f"[15m] Cycle Error: {e}"
                print(error_msg)

        # Combine cycle output with position monitoring
        cycle_output = output_buffer.getvalue() or "Scanning..."
        status_output = format_strategy_status(
            st.session_state.strategy_15m, strategy_type="spread"
        )
        combined_output = f"{cycle_output}\n\n{status_output}"

        if error_msg:
            st.error(error_msg)
        log_display_15m.code(combined_output, language="bash", wrap_lines=True)

        send_greeting_if_needed()
        # EOD notification is sent from 1m strategy section with combined totals
        refresh_dashboard_state()

    sync_15m_cycle()

    # 15m Position Monitoring (runs frequently between full cycles)
    @st.fragment(run_every=CONFIG.STRATEGY_15M_MONITOR_SECONDS)
    def monitor_15m_positions():
        """Lightweight position monitoring for timely exits"""
        try:
            st.session_state.strategy_15m.monitor_only()
        except Exception as e:
            pass  # Silent - full cycle will log errors

    monitor_15m_positions()

    # Scalp Strategy Log
    st.subheader("🖥️ Scalp Iron Spark Log")
    log_display_scalp = st.empty()

    @st.fragment(run_every=CONFIG.STRATEGY_SCALP_REFRESH_SECONDS)
    def sync_scalp_cycle():
        output_buffer = io.StringIO()
        error_msg = None
        with redirect_stdout(output_buffer):
            try:
                st.session_state.strategy_scalp.run_cycle()
            except Exception as e:
                error_msg = f"[Scalp] Cycle Error: {e}"
                print(error_msg)

        # Combine cycle output with position monitoring
        cycle_output = output_buffer.getvalue() or "Scanning..."
        status_output = format_strategy_status(
            st.session_state.strategy_scalp, strategy_type="scalp"
        )
        combined_output = f"{cycle_output}\n\n{status_output}"

        if error_msg:
            st.error(error_msg)
        log_display_scalp.code(combined_output, language="bash", wrap_lines=True)

        send_greeting_if_needed()
        # EOD notification is sent from 1m strategy section with combined totals
        refresh_dashboard_state()

    sync_scalp_cycle()


def main() -> None:
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Header with auto-refresh
    header_placeholder = st.empty()
    
    @st.fragment(run_every=CONFIG.HEADER_REFRESH_SECONDS)
    def update_header():
        render_header(header_placeholder)
    
    update_header()

    # Main dashboard grid
    col_left, col_right = st.columns(list(CONFIG.MAIN_COLUMNS), gap="medium")
    
    with col_left:
        render_left_column()
    
    with col_right:
        render_right_column()
    
    # Enable live Discord alerts
    set_live_mode()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
else:
    # When imported by Streamlit, run main directly
    main()
