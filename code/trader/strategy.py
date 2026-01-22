# code/trader/strategy.py – 1-minute vertical credit spread strategy (Alpaca Data)
# REFACTORED: Fixed dashboard issues with proper logging, error handling, and state management
# Updated: $5 width spreads, min credit $0.40, delta 25-40, faster exits (25% PT, 50% SL)
# Added: End-of-day forced closeout at/after 4:00 PM EST

import os
import json
import pandas as pd
import logging
import time as time_mod
from pathlib import Path
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from filelock import FileLock, Timeout
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple, List

# Import your existing modules
from code.data.data_client import data_client
from indicators import (
    get_trend_signal, train_lstm_daily, get_full_indicator_set, get_volatility_regime, record_trade,
    get_binomial_win_probability, get_conditional_expected_value, get_kelly_criterion, get_macd_crossover
)
from discord_notifier import send_entry, send_exit, set_strategy_ready

# Import brain for trade decisions (optional - fails gracefully)
try:
    from code.rag.snoogans_brain import SnoogansBrain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False

# Import adaptive learner for RL-based entry decisions
try:
    from code.rag.adaptive_learner import get_learner
    LEARNER_AVAILABLE = True
except ImportError:
    LEARNER_AVAILABLE = False

# Configure logging - use INFO for production, DEBUG for troubleshooting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/strategy_1m.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
EST = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).parent.parent
STATE_FILE = PROJECT_ROOT / "strategy_state.json"
LOCK_FILE = PROJECT_ROOT / "strategy_state.json.lock"
SHARED_EQUITY_FILE = PROJECT_ROOT.parent / "shared_equity.json"
SHARED_EQUITY_HISTORY_FILE = PROJECT_ROOT.parent / "shared_equity_history.json"


def update_shared_equity(realized_pnl: float) -> None:
    """Update the shared equity file with realized PnL and append to history."""
    try:
        current_equity = 100000.0  # Default starting balance
        if SHARED_EQUITY_FILE.exists():
            with open(SHARED_EQUITY_FILE, 'r') as f:
                data = json.load(f)
                current_equity = data.get('equity', 100000.0)

        new_equity = current_equity + realized_pnl
        with open(SHARED_EQUITY_FILE, 'w') as f:
            json.dump({
                'equity': new_equity,
                'last_updated': datetime.now(EST).isoformat()
            }, f, indent=2)

        # Append to shared equity history
        try:
            history = []
            if SHARED_EQUITY_HISTORY_FILE.exists():
                with open(SHARED_EQUITY_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            history.append(new_equity)
            # Keep last 500 entries
            history = history[-500:]
            with open(SHARED_EQUITY_HISTORY_FILE, 'w') as f:
                json.dump(history, f)
        except Exception as he:
            logger.error(f"Failed to update equity history: {he}")

        logger.debug(f"Updated shared equity: ${current_equity:.2f} + ${realized_pnl:.2f} = ${new_equity:.2f}")
    except Exception as e:
        logger.error(f"Failed to update shared equity: {e}")

# NYSE holidays for 2025 and 2026
NYSE_HOLIDAYS_2025 = {
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25)
}

NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25)
}


class TradingStrategy:
    """1-minute timeframe vertical credit spread strategy with XGBoost trend prediction"""

    def __init__(self, brain=None):
        self.prefix = "[1m]"
        self._strategy_ready_sent = False
        self.trades_today = []
        self.last_lstm_train_date = None
        self.current_trends = {}  # Track current trend for each ticker
        self._cache = {}  # Simple in-memory cache: key -> (timestamp, data)

        # Load state with error handling
        try:
            self._load_state()
            self.daily_loss_limit = -0.03 * self.equity_history[-1]
            logger.info(f"{self.prefix} Strategy initialized. Equity: ${self.equity_history[-1]:,.2f}")
        except Exception as e:
            logger.error(f"{self.prefix} Failed to initialize: {e}", exc_info=True)
            raise

        # Use shared brain if provided, otherwise create own instance
        self.brain = brain
        if self.brain is None and BRAIN_AVAILABLE:
            try:
                self.brain = SnoogansBrain()
                logger.info(f"{self.prefix} Brain loaded - will use historical patterns for trade decisions")
            except Exception as e:
                logger.warning(f"{self.prefix} Brain not available: {e}")
                self.brain = None
        elif self.brain is not None:
            logger.info(f"{self.prefix} Using shared brain instance")

        # Initialize adaptive learner for RL-based entry decisions
        self.learner = None
        if LEARNER_AVAILABLE:
            try:
                self.learner = get_learner()
                stats = self.learner.get_stats()
                logger.info(f"{self.prefix} Adaptive learner loaded - {stats['total_states']} learned states")
            except Exception as e:
                logger.warning(f"{self.prefix} Adaptive learner not available: {e}")

    @contextmanager
    def _state_lock(self, timeout: int = 5):
        """Context manager for file locking with timeout"""
        lock = FileLock(str(LOCK_FILE), timeout=timeout)
        try:
            with lock:
                yield
        except Timeout:
            logger.error(f"{self.prefix} Failed to acquire state file lock within {timeout}s")
            raise
        except Exception as e:
            logger.error(f"{self.prefix} Lock error: {e}", exc_info=True)
            raise

    def _cached_fetch(self, key: Tuple, fetch_func, ttl: int = 10):
        """Simple cache with TTL to reduce API calls within a cycle"""
        now = time_mod.time()

        if key in self._cache:
            ts, data = self._cache[key]
            if now - ts < ttl:
                logger.debug(f"{self.prefix} Cache hit for {key}")
                return data

        try:
            data = fetch_func()
            if data is not None:
                self._cache[key] = (now, data)
            return data
        except Exception as e:
            logger.error(f"{self.prefix} Error in cached fetch for {key}: {e}")
            return None

    def _load_state(self):
        """Load strategy state from JSON file with file locking"""
        default_balance = float(os.getenv("STARTING_BALANCE", 100000))
        
        if not STATE_FILE.exists():
            logger.info(f"{self.prefix} No state file found, initializing new state")
            self.positions = {}
            self.equity_history = [default_balance]
            self.daily_pnl = 0.0
            self.last_pnl_date = date.today().isoformat()
            self.closed_trades = []
            self._save_state()
            return

        try:
            with self._state_lock():
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.positions = state.get('positions', {})
                    self.equity_history = state.get('equity_history', [default_balance])
                    self.daily_pnl = state.get('daily_pnl', 0.0)
                    self.last_pnl_date = state.get('last_pnl_date', date.today().isoformat())
                    self.closed_trades = state.get('closed_trades', [])
                    self.current_trends = state.get('current_trends', {})
                    self.trades_today = state.get('trades_today', [])
                    # Load last training date to avoid re-training on restart
                    last_train = state.get('last_lstm_train_date')
                    if last_train:
                        self.last_lstm_train_date = date.fromisoformat(last_train)

            # Reset daily P&L and trades_today if new day
            if self.last_pnl_date != date.today().isoformat():
                logger.info(f"{self.prefix} New trading day - resetting daily P&L and trades_today")
                self.daily_pnl = 0.0
                self.trades_today = []
                self.last_pnl_date = date.today().isoformat()
                self._save_state()
                
        except json.JSONDecodeError as e:
            logger.error(f"{self.prefix} Corrupted state file: {e}. Creating backup and reinitializing.")
            if STATE_FILE.exists():
                backup = STATE_FILE.with_suffix('.json.backup')
                STATE_FILE.rename(backup)
            self.positions = {}
            self.equity_history = [default_balance]
            self.daily_pnl = 0.0
            self.last_pnl_date = date.today().isoformat()
            self.closed_trades = []
        except Exception as e:
            logger.error(f"{self.prefix} Error loading state: {e}", exc_info=True)
            raise

    def _save_state(self):
        """Save strategy state to JSON file with file locking"""
        state = {
            'positions': self.positions,
            'equity_history': self.equity_history,
            'daily_pnl': self.daily_pnl,
            'last_pnl_date': self.last_pnl_date,
            'closed_trades': self.closed_trades,
            'current_trends': self.current_trends,
            'trades_today': self.trades_today,
            'last_lstm_train_date': self.last_lstm_train_date.isoformat() if self.last_lstm_train_date else None,
        }

        def json_serializer(obj):
            """Handle numpy types and other non-serializable objects"""
            import numpy as np
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        try:
            with self._state_lock():
                # Write to temporary file first, then atomic rename
                temp_file = STATE_FILE.with_suffix('.json.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state, f, indent=4, default=json_serializer)
                temp_file.replace(STATE_FILE)

        except Exception as e:
            logger.error(f"{self.prefix} Failed to save state: {e}", exc_info=True)

    def _log_closed_trades_batch(self):
        """Archive closed trades to knowledge base when batch reaches 30"""
        if len(self.closed_trades) < 30:
            return
            
        try:
            timestamp = datetime.now(EST).strftime("%Y%m%d_%H%M%S")
            rag_folder = Path(__file__).parent.parent.parent / "data" / "knowledge"
            rag_folder.mkdir(parents=True, exist_ok=True)
            batch_file = rag_folder / f"trades_batch_1m_{timestamp}.json"
            
            with open(batch_file, 'w') as f:
                json.dump(self.closed_trades, f, indent=4)
            
            logger.info(f"{self.prefix} Archived 30 closed trades to {batch_file.name}")
            self.closed_trades = []
            
        except Exception as e:
            logger.error(f"{self.prefix} Failed to log trade batch: {e}", exc_info=True)

    def is_trading_day(self) -> bool:
        """Check if today is a valid trading day"""
        today = date.today()
        
        # Weekend check
        if today.weekday() >= 5:
            return False
        
        # Holiday check
        if today.year == 2025:
            holidays = NYSE_HOLIDAYS_2025
        elif today.year == 2026:
            holidays = NYSE_HOLIDAYS_2026
        else:
            logger.warning(f"{self.prefix} No holidays defined for {today.year} - assuming trading day")
            holidays = set()
        
        if today in holidays:
            logger.info(f"{self.prefix} Market holiday: {today}")
            return False
            
        return True

    def is_scanning_active(self) -> bool:
        """Check if we should be scanning for opportunities (8am-4pm ET)"""
        if not self.is_trading_day():
            return False
            
        now = datetime.now(EST).time()
        return time(8, 0) <= now <= time(16, 0)

    def can_enter_trades(self) -> bool:
        """Check if we can enter new trades (9:45am-2:30pm ET)"""
        if not self.is_trading_day():
            return False

        now = datetime.now(EST).time()
        return time(9, 45) <= now < time(14, 30)

    def check_daily_loss_nanny(self) -> bool:
        """Check if daily loss limit has been hit"""
        if self.daily_pnl <= self.daily_loss_limit:
            logger.warning(f"{self.prefix} Daily loss limit hit: ${self.daily_pnl:.2f} <= ${self.daily_loss_limit:.2f}")
            return False
        return True

    def is_conservative_mode(self) -> bool:
        """Check if we should be more selective (daily PnL >= $1000)"""
        return self.daily_pnl >= 1000.0

    def _calculate_position_size(self, confidence: float = 50.0, win_rate: float = 50.0, vol_regime: dict = None) -> int:
        """
        Scale position size 5-20 contracts based on:
        - Confidence score (0-100) - from brain or indicators
        - Historical win rate for this pattern (0-100)
        - Daily P&L (reduce size if losing, cap if winning big)
        - Volatility regime (reduce size in low vol conditions)
        """
        base = 5  # Minimum contracts

        # Confidence scaling: 0-100 → 0-10 additional contracts
        confidence_bonus = int(confidence / 10)

        # Win rate scaling: 35-70% → 0-5 additional contracts
        wr_bonus = max(0, int((win_rate - 35) / 7))

        # Daily P&L cap adjustment
        if self.daily_pnl < -500:
            size_cap = 5  # Minimum only when losing badly
        elif self.daily_pnl < 0:
            size_cap = 10  # Reduced when in the red
        elif self.daily_pnl > 1000:
            size_cap = 15  # Conservative when up big (protect gains)
        else:
            size_cap = 20  # Full range available

        # Volatility-based reduction (even if not blocking, reduce size in marginal conditions)
        if vol_regime:
            atr_pct = vol_regime.get("atr_pct", 0.003)
            range_ratio = vol_regime.get("daily_range_ratio", 1.0)

            # Reduce size when volatility is borderline (not blocking, but be cautious)
            if atr_pct < 0.002:  # ATR < 0.2%
                size_cap = min(size_cap, 10)
                logger.debug(f"{self.prefix} Vol reduction: ATR {atr_pct*100:.2f}% → cap {size_cap}")
            if range_ratio < 0.6:  # Range < 60% of ADR
                size_cap = min(size_cap, 8)
                logger.debug(f"{self.prefix} Vol reduction: Range {range_ratio:.0%} → cap {size_cap}")

        calculated = min(base + confidence_bonus + wr_bonus, size_cap)
        logger.debug(f"{self.prefix} Position size: {calculated} contracts "
                    f"(conf={confidence:.0f}, wr={win_rate:.0f}%, cap={size_cap})")
        return calculated

    def _train_model_if_needed(self, now: datetime):
        """Train XGBoost model once per trading day (non-blocking)"""
        import threading

        current_date = now.date()

        # Check if already trained today (persisted in state file)
        if hasattr(self, '_training_in_progress') and self._training_in_progress:
            logger.debug(f"{self.prefix} XGBoost training already in progress - skipping")
            return

        if self.last_lstm_train_date == current_date:
            return  # Already trained today

        if not self.is_trading_day():
            return

        # Run training in background thread to avoid blocking UI
        def _background_train():
            try:
                self._training_in_progress = True
                logger.info(f"{self.prefix} Starting XGBoost training in background...")

                historical_bars = data_client.get_spy_bars(
                    now - timedelta(days=60), now, ticker="SPY"
                )

                if historical_bars is not None and not historical_bars.empty:
                    logger.info(f"{self.prefix} Training XGBoost with {len(historical_bars)} bars")
                    train_lstm_daily(historical_bars)
                    self.last_lstm_train_date = current_date
                    self._save_state()  # Persist training date
                    logger.info(f"{self.prefix} XGBoost training complete")
                else:
                    logger.warning(f"{self.prefix} No bars received - skipping model training")
            except Exception as e:
                logger.error(f"{self.prefix} Model training failed: {e}", exc_info=True)
            finally:
                self._training_in_progress = False

        # Start background thread
        self._training_in_progress = True
        thread = threading.Thread(target=_background_train, daemon=True)
        thread.start()
        logger.info(f"{self.prefix} XGBoost training started in background thread")

    def run_cycle(self):
        """Main strategy cycle - called periodically"""
        now = datetime.now(EST)
        current_time = now.time()
        tickers = ["SPY", "QQQ", "IWM"]

        # Log cycle start at DEBUG level to reduce noise
        logger.debug(f"{self.prefix} Cycle start: {now.strftime('%H:%M:%S')}")
        mode = "CONSERVATIVE" if self.is_conservative_mode() else "NORMAL"
        logger.info(f"{self.prefix} Daily P&L: ${self.daily_pnl:+.2f} | Open: {len(self.positions)} | Mode: {mode}")

        # XGBoost training (once per day)
        try:
            self._train_model_if_needed(now)
        except Exception as e:
            logger.error(f"{self.prefix} Error in model training check: {e}", exc_info=True)

        # End-of-day forced closeout
        if len(self.positions) > 0 and self.is_trading_day() and current_time >= time(16, 0):
            logger.warning(f"{self.prefix} EOD force close - market closed")
            self._force_close_all_positions()
            return

        # Check if we should be active
        if not self.is_scanning_active():
            logger.debug(f"{self.prefix} Outside scanning hours")
            return

        if not self.check_daily_loss_nanny():
            return

        # Send strategy ready notification once
        if not self._strategy_ready_sent:
            try:
                set_strategy_ready()
                self._strategy_ready_sent = True
            except Exception as e:
                logger.error(f"{self.prefix} Failed to send ready notification: {e}")

        # Process each ticker
        for ticker in tickers:
            try:
                self._process_ticker(ticker, now)
            except Exception as e:
                logger.error(f"{self.prefix} Error processing {ticker}: {e}", exc_info=True)
                continue

        # Manage existing positions
        try:
            self.manage_positions()
        except Exception as e:
            logger.error(f"{self.prefix} Error managing positions: {e}", exc_info=True)

        # Update counterfactual tracking for RL learner
        if self.learner:
            try:
                expired = self.learner.expire_old_counterfactuals(max_age_minutes=10)
                for cf in expired:
                    logger.info(f"{self.prefix} Counterfactual resolved: {cf['ticker']} | "
                               f"Would have P&L: ${cf['would_have_pnl']:+.2f} | {cf['outcome']}")
            except Exception as e:
                logger.debug(f"{self.prefix} Counterfactual update error: {e}")

        # Save state and archive trades
        try:
            self._save_state()
            self._log_closed_trades_batch()
        except Exception as e:
            logger.error(f"{self.prefix} Error saving state: {e}", exc_info=True)

    def _force_close_all_positions(self):
        """Force close all positions at end of day"""
        for ticker, pos in list(self.positions.items()):
            try:
                current_value = self._get_current_mark(ticker, pos)
                if current_value is None:
                    logger.warning(f"{self.prefix} [{ticker}] No final mark - closing at $0.00")
                    current_value = 0.0

                realized = (pos["credit"] - current_value) * pos["contracts"] * 100
                self.exit_position(ticker, pos, realized, current_value, "eod_force_close")

            except Exception as e:
                logger.error(f"{self.prefix} Error force closing {ticker}: {e}", exc_info=True)

        self._save_state()
        self._log_closed_trades_batch()

    def _process_ticker(self, ticker: str, now: datetime):
        """Process a single ticker for entry or monitoring"""
        
        # Fetch minute bars
        minute_bars = self._fetch_minute_bars(ticker, now)
        if minute_bars is None:
            return

        # Get trend signal
        try:
            trend = get_trend_signal(minute_bars, None, None)
            self.current_trends[ticker] = trend  # Store for dashboard display
            price = minute_bars['close'].iloc[-1]
        except Exception as e:
            logger.error(f"{self.prefix} [{ticker}] Error calculating trend: {e}")
            return

        # Fetch option chain
        chain = self._fetch_option_chain(ticker, now.strftime("%Y-%m-%d"))
        if chain is None:
            return

        # Check volatility regime before entry
        vol_regime = get_volatility_regime(minute_bars)
        logger.debug(f"{self.prefix} [{ticker}] Price: ${price:.2f} | Trend: {trend} | "
                    f"ATR: {vol_regime['atr_pct']*100:.2f}% | Range: {vol_regime['daily_range_ratio']:.0%}")

        # Entry logic
        if ticker not in self.positions:
            # Volatility gate: log but don't skip (was too restrictive)
            if vol_regime.get("is_low_vol", False):
                logger.info(f"{self.prefix} [{ticker}] LOW VOL detected but proceeding: {vol_regime['reason']}")

            if trend == "chop":
                logger.info(f"{self.prefix} [{ticker}] CHOP trend - no entry attempted")
            elif not self.can_enter_trades():
                logger.info(f"{self.prefix} [{ticker}] Cannot enter trades (max positions or trade limit)")
            else:
                # Higher timeframe filter: use 15m trend as directional bias
                htf_trend = self._get_higher_timeframe_bias(ticker, now)
                if htf_trend:
                    # Only enter if 1m trend aligns with 15m trend
                    if htf_trend == "chop":
                        logger.info(f"{self.prefix} [{ticker}] HTF CHOP - skipping (15m={htf_trend}, 1m={trend})")
                        return
                    elif htf_trend != trend:
                        logger.info(f"{self.prefix} [{ticker}] HTF CONFLICT - skipping (15m={htf_trend}, 1m={trend})")
                        return
                    else:
                        logger.info(f"{self.prefix} [{ticker}] HTF ALIGNED (15m={htf_trend}, 1m={trend})")

                try:
                    self._attempt_entry(ticker, trend, chain, price, minute_bars, vol_regime)
                except Exception as e:
                    logger.error(f"{self.prefix} [{ticker}] Entry error: {e}", exc_info=True)
        else:
            logger.debug(f"{self.prefix} [{ticker}] Monitoring existing position")

    def _fetch_minute_bars(self, ticker: str, now: datetime) -> Optional[pd.DataFrame]:
        """Fetch minute bars with caching"""
        minute_bars = self._cached_fetch(
            ('bars', ticker),
            lambda: data_client.get_spy_bars(now - timedelta(days=7), now, ticker=ticker),
            ttl=10
        )

        if minute_bars is None or minute_bars.empty or len(minute_bars) < 20:
            logger.debug(f"{self.prefix} [{ticker}] Insufficient minute bars")
            return None

        return minute_bars

    def _fetch_15m_bars(self, ticker: str, now: datetime) -> Optional[pd.DataFrame]:
        """Fetch 15-minute bars for higher timeframe bias with caching"""
        bars_15m = self._cached_fetch(
            ('bars_15m', ticker),
            lambda: data_client.get_spy_bars(
                now - timedelta(days=7),
                now,
                ticker=ticker,
                timeframe="15Min"
            ),
            ttl=60  # Cache 15m bars for 60 seconds
        )

        if bars_15m is None or bars_15m.empty or len(bars_15m) < 20:
            logger.debug(f"{self.prefix} [{ticker}] Insufficient 15m bars")
            return None

        return bars_15m

    def _get_higher_timeframe_bias(self, ticker: str, now: datetime) -> Optional[str]:
        """Get 15m trend as higher timeframe directional bias"""
        bars_15m = self._fetch_15m_bars(ticker, now)
        if bars_15m is None:
            return None

        try:
            htf_trend = get_trend_signal(bars_15m, None, None)
            return htf_trend
        except Exception as e:
            logger.debug(f"{self.prefix} [{ticker}] Error getting HTF trend: {e}")
            return None

    def _fetch_option_chain(self, ticker: str, expiration: str) -> Optional[pd.DataFrame]:
        """Fetch option chain with caching"""
        chain = self._cached_fetch(
            ('chain', ticker),
            lambda: data_client.get_spy_option_chain(expiration, ticker=ticker),
            ttl=8
        )

        if chain is None or chain.empty:
            logger.debug(f"{self.prefix} [{ticker}] Empty option chain")
            return None

        return chain

    def _get_spread_price(self, short_contract: Dict, long_contract: Dict, for_entry: bool = False) -> Optional[float]:
        """Calculate net credit for a vertical spread"""
        try:
            # Find bid/ask columns (multiple possible names)
            bid_col = next((col for col in ['bid', 'bid_price', 'b']
                          if col in short_contract), None)
            ask_col = next((col for col in ['ask', 'ask_price', 'a']
                          if col in short_contract), None)

            if bid_col is None or ask_col is None:
                return None

            short_bid = short_contract.get(bid_col, 0)
            short_ask = short_contract.get(ask_col, 0)
            long_bid = long_contract.get(bid_col, 0)
            long_ask = long_contract.get(ask_col, 0)

            # Check for valid prices
            if short_bid == 0 or long_ask == 0:
                return None

            # Calculate credit
            credit = short_bid - long_ask

            # Minimum credit filter - only apply during entry
            if for_entry and credit < 0.40:
                return None

            # Return mid-market spread price
            return round((short_bid + short_ask)/2 - (long_bid + long_ask)/2, 2)

        except Exception as e:
            logger.error(f"{self.prefix} Error calculating spread price: {e}")
            return None

    def _get_adaptive_exits(self, ticker: str, is_put: bool, market_context: dict) -> dict:
        """
        Get adaptive exit parameters from brain based on historical patterns.
        Returns default values if brain unavailable or insufficient data.
        """
        # Default exit parameters for credit spreads
        defaults = {
            'profit_target_pct': 50.0,   # Take profit at 50% of max credit
            'stop_loss_pct': 50.0,       # Stop at 1.5x credit (50% loss)
            'trail_activation_pct': 50.0,  # Activate trailing at 50%
            'confidence': 0,
            'source': 'default'
        }

        if not self.brain:
            return defaults

        try:
            # Determine trade type
            trade_type = 'PUT_SPREAD' if is_put else 'CALL_SPREAD'

            # Get optimal exits from brain
            optimal = self.brain.get_optimal_exits(
                ticker=ticker,
                trend=market_context.get('trend'),
                trade_type=trade_type
            )

            # Only use brain values if confidence is high enough
            if optimal.get('confidence', 0) >= 50 and optimal.get('sample_size', 0) >= 10:
                return {
                    'profit_target_pct': min(optimal.get('profit_target_pct', 50.0), 80.0),
                    'stop_loss_pct': min(optimal.get('stop_loss_pct', 100.0), 150.0),
                    'trail_activation_pct': max(30.0, min(optimal.get('profit_target_pct', 50.0) * 0.6, 50.0)),
                    'confidence': optimal.get('confidence', 0),
                    'source': 'brain',
                    'analysis': optimal.get('analysis', '')
                }

            logger.debug(f"{self.prefix} [{ticker}] Brain exit data insufficient "
                        f"(conf={optimal.get('confidence', 0)}, n={optimal.get('sample_size', 0)}) - using defaults")

        except Exception as e:
            logger.debug(f"{self.prefix} [{ticker}] Brain exit query failed: {e}")

        return defaults

    def _attempt_entry(self, ticker: str, trend: str, chain: pd.DataFrame, price: float, bars: pd.DataFrame = None, vol_regime: dict = None):
        """Attempt to enter a new credit spread position"""

        conservative = self.is_conservative_mode()
        is_put = trend == "bull"  # Bull trend -> put credit spread
        now = datetime.now(EST)

        # Get full indicator set for learning
        indicators = get_full_indicator_set(bars) if bars is not None else {}

        # Build enhanced market context for learning
        market_context = {
            "trend": trend,
            "price": round(price, 2),
            "hour": now.hour,
            "day_of_week": now.strftime("%A"),
            "conservative_mode": conservative,
            # Enhanced indicators for learning
            "rsi": indicators.get("rsi", 50.0),
            "macd_line": indicators.get("macd_line", 0.0),
            "macd_signal": indicators.get("macd_signal", 0.0),
            "macd_bull": indicators.get("macd_bull", False),
            "atr": indicators.get("atr", 0.0),
            "trend_strength": indicators.get("trend_strength", 0.0),
        }

        # ============ MACD ALIGNMENT FILTER ============
        # REQUIRE MACD alignment with trade direction
        # Put spread (bullish): MACD histogram must be positive
        # Call spread (bearish): MACD histogram must be negative
        macd_cross = get_macd_crossover(bars, lookback=3)

        if is_put:  # Put credit spread = bullish position, need bullish MACD
            if macd_cross['crossing_down']:
                logger.info(f"{self.prefix} [{ticker}] SKIP (MACD): Bullish entry but MACD crossing DOWN "
                           f"({macd_cross['bars_since_cross']} bars ago)")
                return
            if macd_cross['macd_momentum'] <= 0:
                logger.info(f"{self.prefix} [{ticker}] SKIP (MACD): Bullish entry but MACD histogram negative "
                           f"(hist={macd_cross['macd_momentum']:.3f}) - need bullish momentum")
                return
        else:  # Call credit spread = bearish position, need bearish MACD
            if macd_cross['crossing_up']:
                logger.info(f"{self.prefix} [{ticker}] SKIP (MACD): Bearish entry but MACD crossing UP "
                           f"({macd_cross['bars_since_cross']} bars ago)")
                return
            if macd_cross['macd_momentum'] >= 0:
                logger.info(f"{self.prefix} [{ticker}] SKIP (MACD): Bearish entry but MACD histogram positive "
                           f"(hist={macd_cross['macd_momentum']:.3f}) - need bearish momentum")
                return

        logger.info(f"{self.prefix} [{ticker}] MACD ALIGNED: hist={macd_cross['macd_momentum']:.3f}")

        # ============ AI DECISION LAYER ============
        # Default confidence/win_rate if brain unavailable
        brain_confidence = 50.0
        brain_win_rate = 50.0
        learner_state_key = None

        # Query brain for trade recommendation if available
        if self.brain:
            try:
                if bars is not None and len(bars) >= 30:
                    recommendation = self.brain.get_live_recommendation(
                        ticker=ticker,
                        bars=bars,
                        is_call=not is_put  # For spreads: put_spread = bullish bias
                    )

                    if not recommendation['should_enter']:
                        reason = recommendation.get('final_reason', recommendation.get('reason', 'Unknown'))
                        warnings = recommendation.get('indicator_warnings', [])
                        hist = recommendation.get('historical_analysis', {})
                        hist_win_rate = hist.get('avg_win_rate', 50.0)

                        # RE-ENABLED: Trust brain warnings if win rate is below 50%
                        # Brain sees patterns we don't - if it says skip AND history is bad, SKIP
                        if hist_win_rate < 50.0:
                            logger.info(f"{self.prefix} [{ticker}] SKIP (BRAIN): {reason} "
                                       f"(historical win rate {hist_win_rate:.0f}% < 50%)")
                            if warnings:
                                logger.info(f"{self.prefix} [{ticker}] Warnings: {', '.join(warnings)}")
                            return
                        else:
                            logger.info(f"{self.prefix} [{ticker}] Brain suggests skip but win rate OK ({hist_win_rate:.0f}%): {reason}")

                    hist = recommendation.get('historical_analysis', {})
                    brain_confidence = hist.get('confidence', 50.0)
                    brain_win_rate = hist.get('avg_win_rate', 50.0)

                    # Conservative mode: require higher confidence and win rate
                    if conservative:
                        min_confidence = 60
                        min_win_rate = 50
                        if brain_confidence < min_confidence or brain_win_rate < min_win_rate:
                            logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): "
                                       f"Brain confidence {brain_confidence:.0f}% < {min_confidence}% or "
                                       f"win rate {brain_win_rate:.0f}% < {min_win_rate}%")
                            return

                    logger.info(f"{self.prefix} [{ticker}] Brain says ENTER: "
                              f"Win rate: {brain_win_rate:.0f}% | Confidence: {brain_confidence:.0f}%")

            except Exception as e:
                if conservative:
                    logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): Brain unavailable - {e}")
                    return
                logger.warning(f"{self.prefix} Brain query failed: {e} - proceeding with trade")

        # RL learner decision
        if self.learner and market_context:
            try:
                learner_state_key = self.learner.get_state_key(
                    ticker=ticker,
                    trend=trend,
                    rsi=market_context.get('rsi', 50.0),
                    hour=now.hour,
                    confidence=brain_confidence
                )

                rl_decision = self.learner.should_enter(learner_state_key, brain_win_rate)

                if not rl_decision['should_enter']:
                    q_vals = rl_decision.get('q_values', {})
                    # Only skip if RL has strong conviction (q_skip > q_enter + 0.5)
                    if q_vals.get('skip', 0) > q_vals.get('enter', 0) + 0.5:
                        logger.info(f"{self.prefix} [{ticker}] RL Learner says SKIP (strong signal): {rl_decision['reason']}")
                        return
                    else:
                        logger.info(f"{self.prefix} [{ticker}] RL suggests skip but weak signal - proceeding")
                else:
                    if rl_decision.get('exploration'):
                        logger.info(f"{self.prefix} [{ticker}] RL Learner: EXPLORATION entry")
                    else:
                        logger.info(f"{self.prefix} [{ticker}] RL Learner says ENTER: {rl_decision['reason']}")

            except Exception as e:
                logger.debug(f"{self.prefix} [{ticker}] RL learner error: {e}")

        # Get adaptive exit parameters from brain
        exit_params = self._get_adaptive_exits(ticker, is_put, market_context)
        if exit_params.get('source') == 'brain':
            logger.info(f"{self.prefix} [{ticker}] Using brain exits: "
                       f"PT={exit_params['profit_target_pct']:.0f}%, "
                       f"SL={exit_params['stop_loss_pct']:.0f}%")

        # ============ ADAPTIVE REGIME ADJUSTMENT ============
        # Instead of skipping unfavorable regimes, ADAPT: trade smaller, tighter stops
        setup_type = 'put_spread' if is_put else 'call_spread'
        binomial_stats = get_binomial_win_probability(regime=trend, setup_type=setup_type)
        conditional = get_conditional_expected_value(trend)

        # Default: normal trading
        regime_size_mult = 1.0  # Position size multiplier
        regime_stop_mult = 1.0  # Stop loss multiplier (lower = tighter)

        # Adapt based on regime performance
        if binomial_stats['reliable'] and binomial_stats['n_trades'] >= 10:
            win_rate = binomial_stats['win_rate']
            ev = binomial_stats['expected_value']

            if ev < -50 or win_rate < 0.35:
                # Very unfavorable: trade at 50% size, 75% stop (tighter)
                regime_size_mult = 0.5
                regime_stop_mult = 0.75
                logger.info(f"{self.prefix} [{ticker}] ADAPT (UNFAVORABLE): {trend} regime "
                           f"(win={win_rate:.0%}, EV=${ev:.0f}) → 50% size, tighter stops")
            elif ev < 0 or win_rate < 0.45:
                # Marginal: trade at 75% size
                regime_size_mult = 0.75
                regime_stop_mult = 0.85
                logger.info(f"{self.prefix} [{ticker}] ADAPT (MARGINAL): {trend} regime "
                           f"(win={win_rate:.0%}, EV=${ev:.0f}) → 75% size")
            elif ev > 50 and win_rate > 0.55:
                # Favorable: can trade larger
                regime_size_mult = 1.25
                logger.info(f"{self.prefix} [{ticker}] ADAPT (FAVORABLE): {trend} regime "
                           f"(win={win_rate:.0%}, EV=${ev:.0f}) → 125% size")

        # Log stats
        if binomial_stats['n_trades'] > 0:
            logger.debug(f"{self.prefix} [{ticker}] Stats: win_rate={binomial_stats['win_rate']:.1%}, "
                        f"EV=${binomial_stats['expected_value']:.2f}, n={binomial_stats['n_trades']}")

        # Validate chain columns
        type_col = next((c for c in ['type', 'contract_type', 'option_type', 
                                      'right', 'call_put', 't'] 
                        if c in chain.columns), None)
        if type_col is None:
            logger.warning(f"{self.prefix} [{ticker}] No option type column found")
            return

        strike_col = next((c for c in ['strike', 'strike_price', 'k', 'S'] 
                          if c in chain.columns), None)
        if strike_col is None:
            logger.warning(f"{self.prefix} [{ticker}] No strike column found")
            return

        # Filter for correct option type
        target_type = 'put' if is_put else 'call'
        opts = chain[chain[type_col].str.lower() == target_type].copy()
        
        if opts.empty:
            logger.info(f"{self.prefix} [{ticker}] SKIP: No {target_type}s in chain")
            return

        opts = opts.sort_values(strike_col)

        # Find short strike candidates (20-45 delta or ~0.7% OTM fallback)
        short_candidates = self._find_short_candidates(opts, is_put, price, strike_col)
        
        if short_candidates.empty:
            logger.info(f"{self.prefix} [{ticker}] SKIP: No suitable short strikes (delta 20-45 or OTM)")
            return

        # Find valid $5-wide spreads
        candidates = self._find_spread_candidates(
            short_candidates, opts, is_put, strike_col
        )

        if not candidates:
            logger.info(f"{self.prefix} [{ticker}] SKIP: No valid $5-wide spreads >= $0.40")
            return

        # Select best credit spread
        best_credit, best_short, best_long = max(candidates, key=lambda x: x[0])

        # Conservative mode: require higher minimum credit
        if conservative:
            min_credit = 0.25  # Higher minimum in conservative mode
            if best_credit < min_credit:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): "
                           f"Credit ${best_credit:.2f} < ${min_credit:.2f} minimum")
                return
        short_strike = float(best_short[strike_col])
        long_strike = float(best_long[strike_col])

        # Calculate adaptive position size using brain confidence/win_rate
        contracts = self._calculate_position_size(
            confidence=brain_confidence,
            win_rate=brain_win_rate,
            vol_regime=vol_regime
        )

        # Kelly criterion adjustment if we have reliable stats
        if binomial_stats['reliable']:
            kelly = get_kelly_criterion(
                binomial_stats['win_rate'],
                binomial_stats['avg_win'],
                binomial_stats['avg_loss']
            )

            if kelly['recommendation'] in ['moderate_edge', 'strong_edge']:
                # Scale up position using half-Kelly fraction (max 50% increase)
                kelly_multiplier = min(1.5, 1 + kelly['half_kelly'])
                contracts = int(contracts * kelly_multiplier)
                logger.info(f"{self.prefix} [{ticker}] Kelly boost: {kelly['half_kelly']:.1%} -> {contracts} contracts")
            elif kelly['recommendation'] == 'no_edge':
                # Reduce position if no statistical edge
                contracts = max(5, contracts // 2)
                logger.info(f"{self.prefix} [{ticker}] Kelly reduction: no edge -> {contracts} contracts")

        # Apply regime-based size adjustment (adapt, don't skip)
        if regime_size_mult != 1.0:
            contracts = max(1, int(contracts * regime_size_mult))
            logger.info(f"{self.prefix} [{ticker}] Regime adjustment: {regime_size_mult:.0%} -> {contracts} contracts")

        # Adjust stop loss for unfavorable regimes (tighter stops)
        adjusted_stop_pct = exit_params.get('stop_loss_pct', 100.0) * regime_stop_mult

        # Store position with market context for learning
        self.positions[ticker] = {
            "is_put": is_put,
            "short": short_strike,
            "long": long_strike,
            "credit": best_credit,
            "contracts": contracts,
            "best_value": best_credit,
            "worst_value": best_credit,  # Track worst for max drawdown
            "trail_active": False,
            "trail_level": None,
            "entry_time": datetime.now(EST).isoformat(),
            "market_context": market_context,
            # AI decision tracking
            "learner_state_key": learner_state_key,
            "profit_target_pct": exit_params.get('profit_target_pct', 50.0),
            "stop_loss_pct": adjusted_stop_pct,
            "regime_adjustment": regime_size_mult,  # Track for learning
        }

        self.trades_today.append("entry")

        # Send notification
        try:
            send_entry(
                is_put=is_put,
                short=short_strike,
                long=long_strike,
                credit=best_credit * contracts * 100,
                underlying=ticker
            )
        except Exception as e:
            logger.error(f"{self.prefix} Failed to send entry notification: {e}")

        logger.info(f"{self.prefix} ENTRY: {ticker} {'PUT' if is_put else 'CALL'} "
                   f"{short_strike:.1f}/{long_strike:.1f} @ ${best_credit:.2f} x{contracts}")

        # Save state immediately after entry to prevent duplicate messages on restart
        self._save_state()

    def _find_short_candidates(self, opts: pd.DataFrame, is_put: bool, 
                              price: float, strike_col: str) -> pd.DataFrame:
        """Find short strike candidates using delta or OTM percentage"""
        
        delta_col = next((c for c in ['delta', 'd'] if c in opts.columns), None)
        short_candidates = pd.DataFrame()

        # Try delta-based selection first
        if delta_col is not None:
            if is_put:
                short_candidates = opts[
                    (opts[delta_col] >= -0.40) & (opts[delta_col] <= -0.25)
                ]
            else:
                short_candidates = opts[
                    (opts[delta_col] >= 0.25) & (opts[delta_col] <= 0.40)
                ]

            if not short_candidates.empty:
                logger.debug(f"{self.prefix} Found {len(short_candidates)} "
                           "strikes in 20-45 delta range")
                return short_candidates

        # Fallback to OTM percentage
        logger.debug(f"{self.prefix} Using OTM percentage fallback")
        short_otm_pct = 0.007  # ~0.7% OTM

        if is_put:
            short_strike_target = price * (1 - short_otm_pct)
            short_candidates = opts[opts[strike_col] <= short_strike_target]
        else:
            short_strike_target = price * (1 + short_otm_pct)
            short_candidates = opts[opts[strike_col] >= short_strike_target]

        return short_candidates

    def _find_spread_candidates(self, short_candidates: pd.DataFrame,
                               opts: pd.DataFrame, is_put: bool,
                               strike_col: str) -> List[Tuple[float, pd.Series, pd.Series]]:
        """Find valid $5-wide spread combinations"""
        candidates = []

        for _, short_row in short_candidates.iterrows():
            short_strike = short_row[strike_col]
            target_long_strike = short_strike - 5 if is_put else short_strike + 5

            long_opts = opts[opts[strike_col] == target_long_strike]
            if long_opts.empty:
                continue

            for _, long_row in long_opts.iterrows():
                credit = self._get_spread_price(
                    short_row.to_dict(),
                    long_row.to_dict(),
                    for_entry=True
                )

                if credit is not None:
                    candidates.append((credit, short_row, long_row))

        logger.debug(f"{self.prefix} Found {len(candidates)} viable spreads")
        return candidates

    def _get_current_mark(self, ticker: str, pos: Dict) -> Optional[float]:
        """Get current market price for an open spread position"""
        try:
            now = datetime.now(EST)
            chain = data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker)

            if chain is None or chain.empty:
                logger.debug(f"{self.prefix} [{ticker}] Empty chain")
                return None

            type_col = next((c for c in ['type', 'contract_type', 'option_type', 
                                        'right', 'call_put', 't'] 
                            if c in chain.columns), None)
            strike_col = next((c for c in ['strike', 'strike_price', 'k', 'S'] 
                             if c in chain.columns), None)

            if type_col is None or strike_col is None:
                return None

            target_type = 'put' if pos["is_put"] else 'call'
            
            short_row = chain[
                (chain[strike_col] == pos["short"]) & 
                (chain[type_col].str.lower() == target_type)
            ]
            long_row = chain[
                (chain[strike_col] == pos["long"]) & 
                (chain[type_col].str.lower() == target_type)
            ]

            if short_row.empty or long_row.empty:
                return None

            return self._get_spread_price(
                short_row.iloc[0].to_dict(),
                long_row.iloc[0].to_dict()
            )

        except Exception as e:
            logger.error(f"{self.prefix} [{ticker}] Error fetching spread mark: {e}")
            return None

    def manage_positions(self):
        """Manage all open positions - check stops, profit targets, trailing"""
        now = datetime.now(EST)

        for ticker, pos in list(self.positions.items()):
            try:
                self._manage_single_position(ticker, pos, now)
            except Exception as e:
                logger.error(f"{self.prefix} Error managing {ticker}: {e}", exc_info=True)

    def _manage_single_position(self, ticker: str, pos: Dict, now: datetime):
        """Manage a single position with trend monitoring for chop-exit"""

        # Fetch fresh bars for trend calculation (like 15m strategy does)
        bars = self._fetch_minute_bars(ticker, now)
        trend = None
        if bars is not None and len(bars) >= 20:
            try:
                trend = get_trend_signal(bars, None, None)
                self.current_trends[ticker] = trend
            except Exception as e:
                logger.debug(f"{self.prefix} [{ticker}] Error calculating trend: {e}")

        current_value = self._get_current_mark(ticker, pos)

        if current_value is None:
            logger.info(f"{self.prefix} [{ticker}] No spread mark - skipping cycle")
            return

        # Store current value for dashboard display
        pos["current_value"] = current_value

        credit = pos["credit"]
        unrealized = (credit - current_value) * pos["contracts"] * 100

        trend_str = trend if trend else "N/A"
        logger.info(f"{self.prefix} [{ticker}] {'PUT' if pos['is_put'] else 'CALL'} "
                   f"{pos['short']:.1f}/{pos['long']:.1f} | "
                   f"Entry: ${credit:.2f} | Current: ${current_value:.2f} | "
                   f"Unrealized: ${unrealized:+.2f} | Trend: {trend_str}")

        # Update best value and trailing stop
        if current_value < pos["best_value"]:
            old_trail = pos.get("trail_level")
            pos["best_value"] = current_value

            if pos["trail_active"]:
                pos["trail_level"] = pos["best_value"] * 1.10
                logger.info(f"{self.prefix} [{ticker}] Trailing stop updated: "
                          f"${pos['trail_level']:.2f} (from ${old_trail:.2f if old_trail else 'N/A'})")

        # Update worst value for drawdown tracking
        if current_value > pos.get("worst_value", current_value):
            pos["worst_value"] = current_value

        # Check exit conditions
        realized = None
        exit_reason = None

        # 1. Stop loss: 1.5x credit (50% loss) - cut losses faster!
        if current_value >= 1.5 * credit:
            realized = (credit - current_value) * pos["contracts"] * 100
            exit_reason = "stop_loss"
            logger.warning(f"{self.prefix} [{ticker}] Stop loss hit (1.5x credit)")

        # 2. Chop exit: trend became choppy
        elif trend == "chop":
            realized = (credit - current_value) * pos["contracts"] * 100
            exit_reason = "trend_to_chop"
            logger.info(f"{self.prefix} [{ticker}] Chop exit - trend became choppy")

        # 3. Trend reversal exit (PUT spread in bear trend, CALL spread in bull trend)
        elif trend is not None and pos["is_put"] and trend == "bear":
            realized = (credit - current_value) * pos["contracts"] * 100
            exit_reason = f"trend_reversal → {trend}"
            logger.info(f"{self.prefix} [{ticker}] Trend reversal exit (PUT spread, trend={trend})")
        elif trend is not None and not pos["is_put"] and trend == "bull":
            realized = (credit - current_value) * pos["contracts"] * 100
            exit_reason = f"trend_reversal → {trend}"
            logger.info(f"{self.prefix} [{ticker}] Trend reversal exit (CALL spread, trend={trend})")

        # 4. Time-based exit after 2:30 PM
        elif now.time() >= time(14, 30):
            realized = (credit - current_value) * pos["contracts"] * 100
            exit_reason = "time_exit"
            logger.info(f"{self.prefix} [{ticker}] Time-based exit (after 2:30 PM)")

        # 5. Activate trailing stop at 10% profit - let winners run
        elif (credit - current_value) >= 0.10 * credit:
            if not pos["trail_active"]:
                pos["trail_active"] = True
                pos["trail_level"] = pos["best_value"] * 1.10
                profit_pct = (credit - current_value) / credit * 100
                logger.info(f"{self.prefix} [{ticker}] {profit_pct:.0f}% profit - trailing stop "
                          f"activated @ ${pos['trail_level']:.2f}")

        # 6. Check trailing stop - exit when price reverts 10% from best
        if pos["trail_active"] and current_value >= pos["trail_level"]:
            realized = (credit - current_value) * pos["contracts"] * 100
            exit_reason = "trailing_stop"
            logger.info(f"{self.prefix} [{ticker}] Trailing stop hit @ ${pos['trail_level']:.2f}")

        # Exit if any condition triggered
        if realized is not None:
            self.exit_position(ticker, pos, realized, current_value, exit_reason)

    def exit_position(self, ticker: str, pos: Dict, realized_pnl: float, final_mark: float,
                      exit_reason: str = "unknown"):
        """Close a position and update state with learning metrics"""
        try:
            # Update P&L and equity
            self.daily_pnl += realized_pnl
            self.equity_history.append(self.equity_history[-1] + realized_pnl)

            # Update shared equity (single source of truth)
            update_shared_equity(realized_pnl)

            # Calculate learning metrics
            credit = pos["credit"]
            best_value = pos.get("best_value", credit)
            worst_value = pos.get("worst_value", credit)
            entry_time_str = pos.get("entry_time", "unknown")

            # Max profit seen (credit - best_value) / credit * 100
            max_profit_pct = ((credit - best_value) / credit * 100) if credit > 0 else 0.0

            # Max drawdown seen (worst_value - credit) / credit * 100
            max_drawdown_pct = ((worst_value - credit) / credit * 100) if credit > 0 else 0.0

            # Hold duration in minutes
            hold_duration_minutes = 0.0
            if entry_time_str != "unknown":
                try:
                    entry_dt = datetime.fromisoformat(entry_time_str)
                    hold_duration_minutes = (datetime.now(EST) - entry_dt).total_seconds() / 60.0
                except:
                    pass

            # Send exit notification
            try:
                send_exit(
                    is_put=pos["is_put"],
                    short=pos["short"],
                    long=pos["long"],
                    credit=pos["credit"],
                    pnl=realized_pnl,
                    underlying=ticker
                )
            except Exception as e:
                logger.error(f"{self.prefix} Failed to send exit notification: {e}")

            # Enhanced exit logging with indicators
            market_ctx = pos.get("market_context", {})
            logger.info(f"{self.prefix} EXIT: {ticker} {'PUT' if pos['is_put'] else 'CALL'} "
                       f"{pos['short']:.1f}/{pos['long']:.1f} | "
                       f"P&L: ${realized_pnl:+.2f} | Reason: {exit_reason} | "
                       f"Entry: ${credit:.2f} → Exit: ${final_mark:.2f} | "
                       f"RSI: {market_ctx.get('rsi', 'N/A')} | "
                       f"Trend: {market_ctx.get('trend', 'N/A')}")

            # Archive closed trade with market context and learning metrics
            closed_trade = {
                "ticker": ticker,
                "is_put": pos["is_put"],
                "short": pos["short"],
                "long": pos["long"],
                "credit": pos["credit"],
                "pnl": realized_pnl,
                "entry_time": entry_time_str,
                "exit_time": datetime.now(EST).isoformat(),
                "market_context": market_ctx,
                # Learning metrics
                "exit_reason": exit_reason,
                "max_profit_pct": round(max_profit_pct, 2),
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "hold_duration_minutes": round(hold_duration_minutes, 1),
            }
            self.closed_trades.append(closed_trade)

            # Record trade for statistical tracking (binomial, conditional EV, etc.)
            try:
                record_trade({
                    'pnl': realized_pnl,
                    'win': realized_pnl > 0,
                    'regime': market_ctx.get('trend', 'unknown'),
                    'setup_type': 'put_spread' if pos['is_put'] else 'call_spread',
                    'entry_time': entry_time_str,
                    'holding_period': hold_duration_minutes,
                    'exit_reason': exit_reason,
                    'ticker': ticker,
                })
            except Exception as e:
                logger.debug(f"{self.prefix} Failed to record trade stats: {e}")

            # Update RL learner with trade outcome
            learner_state_key = pos.get("learner_state_key")
            if self.learner and learner_state_key:
                try:
                    max_risk = credit * pos.get("contracts", 10) * 100
                    self.learner.record_trade_outcome(
                        state_key=learner_state_key,
                        pnl=realized_pnl,
                        max_risk=max_risk
                    )
                    logger.debug(f"{self.prefix} [{ticker}] Updated RL learner with P&L=${realized_pnl:+.2f}")
                except Exception as e:
                    logger.debug(f"{self.prefix} [{ticker}] RL learner update failed: {e}")

            # Remove position
            del self.positions[ticker]
            self.trades_today.append("exit")

        except Exception as e:
            logger.error(f"{self.prefix} Error exiting position: {e}", exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status for dashboard display"""
        # Calculate trades_today from closed_trades with today's exit_time
        today_str = date.today().isoformat()
        trades_today_count = sum(
            1 for t in self.closed_trades
            if t.get('exit_time', '').startswith(today_str)
        )
        return {
            "equity": self.equity_history[-1] if self.equity_history else 0,
            "daily_pnl": self.daily_pnl,
            "daily_loss_limit": self.daily_loss_limit,
            "open_positions": len(self.positions),
            "positions": self.positions,
            "trades_today": trades_today_count,
            "is_trading_day": self.is_trading_day(),
            "can_trade": self.can_enter_trades() and self.check_daily_loss_nanny(),
            "last_lstm_train": self.last_lstm_train_date.isoformat() if self.last_lstm_train_date else None,
            "current_trends": self.current_trends,
        }