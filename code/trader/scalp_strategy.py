# code/trader/scalp_strategy.py - ATM directional option buys (long calls on bull, long puts on bear)
# REFACTORED: Fixed dashboard issues with proper logging, error handling, and state management
# Targets closest-to-ATM strike; min debit $0.20; hard 40% profit, tight stops

import os
import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
import time as time_mod
from filelock import FileLock, Timeout
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple

# Import your existing modules
from code.data.data_client import data_client
from indicators import get_trend_signal, _macd, _rsi, get_full_indicator_set, get_kst_momentum, get_volatility_regime, record_trade, get_macd_crossover
from discord_notifier import send_scalp_entry, send_scalp_exit, set_strategy_ready, send_webhook, is_ready

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/strategy_scalp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
EST = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / "scalp_strategy_state.json"
LOCK_FILE = SCRIPT_DIR / "scalp_strategy_state.json.lock"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
TRADES_FILE = KNOWLEDGE_DIR / "trades.json"
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
NYSE_HOLIDAYS = {
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25)
}


class ScalpStrategy:
    """ATM directional option scalping strategy"""
    
    def __init__(self, use_brain: bool = True):
        self.prefix = "[SCALP]"
        self._strategy_ready_sent = False
        self.trades_today = []
        self.last_exit = {}  # ticker: ('direction', datetime)
        self._cache = {}  # Simple in-memory cache: key -> (timestamp, data)
        self.current_trends = {}  # Track current trend for each ticker
        self.brain = None
        self.use_brain = use_brain

        # Initialize brain for trade decisions
        if use_brain and BRAIN_AVAILABLE:
            try:
                self.brain = SnoogansBrain()
                logger.info(f"{self.prefix} Brain loaded - will use historical patterns for trade decisions")
            except Exception as e:
                logger.warning(f"{self.prefix} Brain not available: {e}")
                self.brain = None

        # Initialize adaptive learner for RL-based entry decisions
        self.learner = None
        if LEARNER_AVAILABLE:
            try:
                self.learner = get_learner()
                stats = self.learner.get_stats()
                logger.info(f"{self.prefix} Adaptive learner loaded - {stats['total_states']} learned states")
            except Exception as e:
                logger.warning(f"{self.prefix} Adaptive learner not available: {e}")

        # Load state with error handling
        try:
            self._load_state()
            self.daily_loss_limit = -0.03 * self.equity_history[-1]
            logger.info(f"{self.prefix} Strategy initialized. Equity: ${self.equity_history[-1]:,.2f}")
        except Exception as e:
            logger.error(f"{self.prefix} Failed to initialize: {e}", exc_info=True)
            raise

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

            # Reset daily P&L and trades_today if new day
            if self.last_pnl_date != date.today().isoformat():
                logger.info(f"{self.prefix} New trading day - resetting daily P&L and trades_today")
                self.daily_pnl = 0.0
                self.trades_today = []
                self.last_pnl_date = date.today().isoformat()
                self._save_state()
            
            # Validate and clean positions
            self._validate_and_clean_positions()
                
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

    def _validate_and_clean_positions(self):
        """Validate position data and clean invalid entries"""
        required_keys = {"is_call", "strike_price", "debit", "contracts", "best_mark", "entry_time"}
        keys_to_remove = []
        
        for ticker, pos in self.positions.items():
            missing = required_keys - pos.keys()
            if missing:
                logger.warning(f"{self.prefix} Invalid position for {ticker} - missing keys: {missing}. Removing.")
                keys_to_remove.append(ticker)
                continue
                
            # Add default values for optional fields
            if "chop_count" not in pos:
                pos["chop_count"] = 0
            if "trail_active" not in pos:
                pos["trail_active"] = False
                pos["trail_level"] = None
                
        for ticker in keys_to_remove:
            del self.positions[ticker]

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

    def _save_individual_trade(self, closed_trade: Dict[str, Any]):
        """Save individual closed trade to knowledge base"""
        try:
            if TRADES_FILE.exists():
                with open(TRADES_FILE, 'r') as f:
                    try:
                        all_trades = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"{self.prefix} Corrupted trades file, creating new")
                        all_trades = []
            else:
                all_trades = []
                
            all_trades.append(closed_trade)
            
            with open(TRADES_FILE, 'w') as f:
                json.dump(all_trades, f, indent=4, default=str)
                
        except Exception as e:
            logger.error(f"{self.prefix} Failed to save individual trade: {e}", exc_info=True)

    def is_trading_day(self) -> bool:
        """Check if today is a valid trading day"""
        today = date.today()
        
        # Weekend check
        if today.weekday() >= 5:
            return False
        
        # Holiday check
        if today in NYSE_HOLIDAYS:
            logger.info(f"{self.prefix} Market holiday: {today}")
            return False
        
        if today.year not in {2025, 2026}:
            logger.warning(f"{self.prefix} Holidays not fully defined for {today.year}")
            
        return True

    def is_scanning_active(self) -> bool:
        """Check if we should be scanning for opportunities (8am-4pm ET)"""
        if not self.is_trading_day():
            return False
            
        now = datetime.now(EST).time()
        return time(8, 0) <= now <= time(16, 0)

    def can_enter_trades(self) -> bool:
        """Check if we can enter new trades (9:31am-3pm ET)"""
        if not self.is_trading_day():
            return False
            
        now = datetime.now(EST).time()
        return time(9, 31) <= now < time(15, 0)

    def check_daily_loss_nanny(self) -> bool:
        """Check if daily loss limit has been hit (including unrealized)"""
        try:
            unrealized = sum(
                ((self._get_current_mark(t, p) or p['debit']) - p['debit']) * p['contracts'] * 100
                for t, p in self.positions.items()
            )

            total_pnl = self.daily_pnl + unrealized

            if total_pnl <= self.daily_loss_limit:
                logger.warning(f"{self.prefix} Daily loss limit hit (incl unrealized): "
                             f"${total_pnl:.2f} <= ${self.daily_loss_limit:.2f}")
                return False

        except Exception as e:
            logger.error(f"{self.prefix} Error checking loss nanny: {e}")

        return True

    def is_conservative_mode(self) -> bool:
        """Check if we should be more selective (daily PnL >= $1000)"""
        return self.daily_pnl >= 1000.0

    def _get_adaptive_exits(self, ticker: str, is_call: bool, market_context: dict) -> dict:
        """
        Get adaptive exit parameters from brain based on pattern.
        Returns default values if brain unavailable or insufficient data.
        """
        # Default exit parameters
        defaults = {
            'profit_target_pct': 100.0,  # No hard cap - let winners run with trailing stop
            'stop_loss_pct': 20.0,       # 20% stop loss
            'trail_activation_pct': 10.0,  # Activate 10% trailing at 10% profit
            'confidence': 0,
            'source': 'default'
        }

        if not self.brain:
            return defaults

        try:
            # Determine trade type
            trade_type = 'LONG_CALL' if is_call else 'LONG_PUT'

            # Get optimal exits from brain
            optimal = self.brain.get_optimal_exits(
                ticker=ticker,
                trend=market_context.get('trend'),
                trade_type=trade_type
            )

            # Only use brain values if confidence is high enough
            if optimal.get('confidence', 0) >= 50 and optimal.get('sample_size', 0) >= 10:
                return {
                    'profit_target_pct': min(optimal.get('profit_target_pct', 80.0), 100.0),  # Cap at 100%
                    'stop_loss_pct': min(optimal.get('stop_loss_pct', 15.0), 25.0),  # Cap at 25%
                    'trail_activation_pct': max(8.0, min(optimal.get('profit_target_pct', 80.0) * 0.15, 15.0)),  # 15% of target, max 15%
                    'confidence': optimal.get('confidence', 0),
                    'source': 'brain',
                    'analysis': optimal.get('analysis', '')
                }

            logger.debug(f"{self.prefix} [{ticker}] Brain exit data insufficient "
                        f"(conf={optimal.get('confidence', 0)}, n={optimal.get('sample_size', 0)}) - using defaults")

        except Exception as e:
            logger.debug(f"{self.prefix} [{ticker}] Brain exit query failed: {e}")

        return defaults

    def _calculate_position_size(self, confidence: float = 50.0, win_rate: float = 50.0, vol_regime: dict = None) -> int:
        """
        Scale position size 5-20 contracts based on:
        - Brain confidence (0-100)
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

    def _cached_fetch(self, key: Tuple, fetch_func, ttl: int = 10):
        """Simple cache with TTL to reduce API calls"""
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

    def _fetch_15m_bars(self, ticker: str, now: datetime) -> Optional[pd.DataFrame]:
        """Fetch 15-minute bars for higher timeframe bias"""
        return self._cached_fetch(
            ('bars_15m', ticker),
            lambda: data_client.get_spy_bars(
                now - timedelta(days=7),
                now,
                ticker=ticker,
                timeframe="15Min"
            ),
            ttl=60  # Cache 15m bars for 60 seconds
        )

    def _get_higher_timeframe_bias(self, ticker: str, now: datetime) -> Optional[str]:
        """Get 15m trend as higher timeframe directional bias"""
        bars_15m = self._fetch_15m_bars(ticker, now)
        if bars_15m is None or bars_15m.empty or len(bars_15m) < 20:
            return None

        try:
            htf_trend = get_trend_signal(bars_15m, None, None)
            return htf_trend
        except Exception as e:
            logger.debug(f"{self.prefix} [{ticker}] Error getting HTF trend: {e}")
            return None

    def run_cycle(self):
        """Main strategy cycle - called periodically"""
        now = datetime.now(EST)
        tickers = ["SPY", "QQQ", "IWM"]

        # Log cycle start
        logger.debug(f"{self.prefix} Cycle start: {now.strftime('%H:%M:%S')}")
        mode = "CONSERVATIVE" if self.is_conservative_mode() else "NORMAL"
        logger.info(f"{self.prefix} Daily P&L: ${self.daily_pnl:+.2f} | Open: {len(self.positions)} | Mode: {mode}")

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
        try:
            for ticker in tickers:
                try:
                    self._process_ticker(ticker, now)
                except Exception as e:
                    logger.error(f"{self.prefix} Error processing {ticker}: {e}", exc_info=True)
                    continue

            # Manage existing positions
            self.manage_positions()

            # Update counterfactual tracking for RL learner
            if self.learner:
                try:
                    # Expire old counterfactuals (resolved after 10 minutes)
                    expired = self.learner.expire_old_counterfactuals(max_age_minutes=10)
                    for cf in expired:
                        logger.info(f"{self.prefix} Counterfactual resolved: {cf['ticker']} | "
                                   f"Would have P&L: ${cf['would_have_pnl']:+.2f} | {cf['outcome']}")
                except Exception as e:
                    logger.debug(f"{self.prefix} Counterfactual update error: {e}")

            # Save state
            self._save_state()
            
        except Exception as e:
            logger.error(f"{self.prefix} Cycle error: {e}", exc_info=True)

    def _process_ticker(self, ticker: str, now: datetime):
        """Process a single ticker for entry or monitoring"""
        
        # Fetch bars with caching
        bars = self._cached_fetch(
            ('bars', ticker),
            lambda: data_client.get_spy_bars(now - timedelta(days=7), now, ticker=ticker),
            ttl=10
        )
        
        if bars is None or bars.empty or len(bars) < 30:
            logger.debug(f"{self.prefix} [{ticker}] Insufficient bars")
            return

        # Calculate indicators
        try:
            trend = get_trend_signal(bars, None, None)
            self.current_trends[ticker] = trend  # Store for dashboard display

            macd_line, macd_signal = _macd(bars['close'])  # Returns (float, float)
            is_momentum_bull = macd_line > macd_signal

            # Current RSI
            rsi_value = _rsi(bars['close'])  # Returns a single float

            # RSI momentum: compare current to 3 bars ago for direction
            rsi_prev = _rsi(bars['close'].iloc[:-3]) if len(bars) > 17 else rsi_value
            rsi_rising = rsi_value > rsi_prev + 1  # Rising if up by at least 1 point
            rsi_falling = rsi_value < rsi_prev - 1  # Falling if down by at least 1 point

            # MACD momentum: is MACD line itself positive/negative (not just vs signal)
            macd_bullish = macd_line > 0

            price = bars['close'].iloc[-1]

            # KST momentum (ThinkOrSwim style - 2 consecutive bars of direction)
            kst_data = get_kst_momentum(bars['close'])
            kst_rising = kst_data['kst_rising']
            kst_falling = kst_data['kst_falling']
            kst_direction = kst_data['kst_direction']

        except Exception as e:
            logger.error(f"{self.prefix} [{ticker}] Error calculating indicators: {e}")
            return

        # Fetch option chain with caching
        chain = self._cached_fetch(
            ('chain', ticker),
            lambda: data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker),
            ttl=8
        )

        if chain is None or chain.empty:
            logger.debug(f"{self.prefix} [{ticker}] No option chain")
            return

        # Check volatility regime before entry
        vol_regime = get_volatility_regime(bars)
        logger.debug(f"{self.prefix} [{ticker}] Price: ${price:.2f} | Trend: {trend} | "
                    f"MACD Bull: {is_momentum_bull} | MACD+: {macd_bullish} | "
                    f"RSI: {rsi_value:.1f} | RSI↑: {rsi_rising} | RSI↓: {rsi_falling} | "
                    f"KST: {kst_direction} | ATR: {vol_regime['atr_pct']*100:.2f}%")

        # Entry logic
        if ticker not in self.positions:
            # Volatility gate: log but don't skip (was too restrictive)
            if vol_regime.get("is_low_vol", False):
                logger.info(f"{self.prefix} [{ticker}] LOW VOL detected but proceeding: {vol_regime['reason']}")

            if self.can_enter_trades():
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

                self._check_entry_conditions(
                    ticker, trend, is_momentum_bull, rsi_value, chain, price, now, bars,
                    rsi_rising=rsi_rising, rsi_falling=rsi_falling, macd_bullish=macd_bullish,
                    kst_rising=kst_rising, kst_falling=kst_falling, vol_regime=vol_regime
                )
            else:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: Outside entry hours (9:31am-3pm ET)")
        else:
            logger.info(f"{self.prefix} [{ticker}] NO TRADE: Already have open position")

    def _check_entry_conditions(self, ticker: str, trend: str, is_momentum_bull: bool,
                                rsi_value: float, chain: pd.DataFrame, price: float, now: datetime,
                                bars: pd.DataFrame = None, rsi_rising: bool = False,
                                rsi_falling: bool = False, macd_bullish: bool = False,
                                kst_rising: bool = False, kst_falling: bool = False,
                                vol_regime: dict = None):
        """Check if entry conditions are met - EARLY ENTRY LOGIC with KST filter"""

        conservative = self.is_conservative_mode()
        if conservative:
            logger.info(f"{self.prefix} CONSERVATIVE MODE active (P&L ${self.daily_pnl:+.0f}) - higher standards")

        # Determine direction
        new_dir = None
        if trend == "bull":
            new_dir = 'bull'
        elif trend == "bear":
            new_dir = 'bear'
        else:
            logger.info(f"{self.prefix} [{ticker}] NO TRADE: Trend is CHOP (no clear direction)")
            return

        # Check cooloff period after opposite direction exit
        if ticker in self.last_exit:
            last_dir, last_time = self.last_exit[ticker]
            cooloff_minutes = 10 if conservative else 3  # Reduced from 5 to 3 for faster re-entry
            if (now - last_time) < timedelta(minutes=cooloff_minutes) and new_dir != last_dir:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: Cooloff period after {last_dir} exit")
                return

        # Get full indicator set for learning
        indicators = get_full_indicator_set(bars) if bars is not None else {}

        # Build enhanced market context for learning
        market_context = {
            "trend": trend,
            "rsi": round(rsi_value, 2),
            "rsi_rising": rsi_rising,
            "rsi_falling": rsi_falling,
            "macd_bull": is_momentum_bull,
            "macd_bullish": macd_bullish,
            "price": round(price, 2),
            "hour": now.hour,
            "day_of_week": now.strftime("%A"),
            "conservative_mode": conservative,
            # Enhanced indicators for learning
            "macd_line": indicators.get("macd_line", 0.0),
            "macd_signal": indicators.get("macd_signal", 0.0),
            "atr": indicators.get("atr", 0.0),
            "trend_strength": indicators.get("trend_strength", 0.0),
        }

        # ============ EARLY ENTRY LOGIC ============
        # Goal: Enter BEFORE all lagging indicators confirm, catch moves earlier
        #
        # BULL ENTRY CONDITIONS (any ONE of these triggers entry):
        #   1. STANDARD: trend=bull + MACD crossed + RSI > 45 (lowered from 50)
        #   2. EARLY MOMENTUM: trend=bull + RSI rising + RSI > 40 + MACD line positive
        #   3. STRONG TREND: trend=bull + RSI > 55 (strong momentum, don't wait for MACD)
        #
        # BEAR ENTRY CONDITIONS (mirror logic):
        #   1. STANDARD: trend=bear + MACD crossed + RSI < 55 (raised from 50)
        #   2. EARLY MOMENTUM: trend=bear + RSI falling + RSI < 60 + MACD line negative
        #   3. STRONG TREND: trend=bear + RSI < 45 (strong momentum, don't wait for MACD)

        # Conservative mode: Still require stricter conditions but with adjusted thresholds
        if conservative:
            if trend == "bull" and not (50 <= rsi_value <= 70):
                logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): Bull RSI {rsi_value:.1f} not in 50-70 range")
                return
            if trend == "bear" and not (30 <= rsi_value <= 50):
                logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): Bear RSI {rsi_value:.1f} not in 30-50 range")
                return

        # ============ BULL ENTRY ============
        if trend == "bull":
            # RSI > 60 = overbought, don't enter calls (too late in the move)
            if rsi_value > 60:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: RSI {rsi_value:.1f} > 60 (overbought, too late for calls)")
                return

            # RSI falling = momentum fading, don't enter calls against momentum
            if rsi_falling:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: RSI falling (momentum fading, don't buy calls)")
                return

            # KST filter: require 2 consecutive bars of rising KST for calls (ThinkOrSwim logic)
            if not kst_rising:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: KST not rising (need 2 bars of bullish momentum)")
                return

            entry_reason = None

            # 1. STRONG TREND: RSI 55-60 = strong momentum, enter immediately
            if rsi_value > 55:
                entry_reason = f"STRONG TREND (RSI {rsi_value:.1f} in 55-60 range)"

            # 2. EARLY MOMENTUM: RSI rising toward 50 + MACD line positive
            elif rsi_rising and rsi_value > 40 and macd_bullish:
                entry_reason = f"EARLY MOMENTUM (RSI↑ {rsi_value:.1f}, MACD+)"

            # 3. STANDARD: MACD crossed + RSI > 45 (lowered threshold)
            elif is_momentum_bull and rsi_value > 45:
                entry_reason = f"STANDARD (MACD crossed, RSI {rsi_value:.1f} > 45)"

            if entry_reason:
                logger.info(f"{self.prefix} [{ticker}] BULL ENTRY: {entry_reason}")
                self._attempt_entry(ticker, True, chain, price, market_context, vol_regime)
            else:
                # Log why we didn't enter
                reasons = []
                if rsi_value <= 55:
                    reasons.append(f"RSI {rsi_value:.1f} ≤ 55 (not strong)")
                if not (rsi_rising and rsi_value > 40 and macd_bullish):
                    reasons.append("no early momentum signal")
                if not (is_momentum_bull and rsi_value > 45):
                    reasons.append("MACD not crossed or RSI ≤ 45")
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: Bull trend but {'; '.join(reasons)}")

        # ============ BEAR ENTRY ============
        elif trend == "bear":
            # RSI < 40 = oversold, don't enter puts (too late in the move)
            if rsi_value < 40:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: RSI {rsi_value:.1f} < 40 (oversold, too late for puts)")
                return

            # RSI rising = momentum reversing, don't enter puts against momentum
            if rsi_rising:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: RSI rising (momentum reversing, don't buy puts)")
                return

            # KST filter: require 2 consecutive bars of falling KST for puts (ThinkOrSwim logic)
            if not kst_falling:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: KST not falling (need 2 bars of bearish momentum)")
                return

            entry_reason = None

            # 1. STRONG TREND: RSI 40-45 = strong bearish momentum, enter immediately
            if rsi_value < 45:
                entry_reason = f"STRONG TREND (RSI {rsi_value:.1f} in 40-45 range)"

            # 2. EARLY MOMENTUM: RSI falling toward 50 + MACD line negative
            elif rsi_falling and rsi_value < 60 and not macd_bullish:
                entry_reason = f"EARLY MOMENTUM (RSI↓ {rsi_value:.1f}, MACD-)"

            # 3. STANDARD: MACD crossed bearish + RSI < 55 (raised threshold)
            elif not is_momentum_bull and rsi_value < 55:
                entry_reason = f"STANDARD (MACD crossed, RSI {rsi_value:.1f} < 55)"

            if entry_reason:
                logger.info(f"{self.prefix} [{ticker}] BEAR ENTRY: {entry_reason}")
                self._attempt_entry(ticker, False, chain, price, market_context, vol_regime)
            else:
                # Log why we didn't enter
                reasons = []
                if rsi_value >= 45:
                    reasons.append(f"RSI {rsi_value:.1f} ≥ 45 (not strong)")
                if not (rsi_falling and rsi_value < 60 and not macd_bullish):
                    reasons.append("no early momentum signal")
                if not (not is_momentum_bull and rsi_value < 55):
                    reasons.append("MACD not crossed or RSI ≥ 55")
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: Bear trend but {'; '.join(reasons)}")

    def _get_option_mark(self, row: pd.Series) -> Optional[float]:
        """Calculate mid-market option price"""
        try:
            bid = row.get('bid', 0) or row.get('b', 0) or row.get('bid_price', 0) or 0
            ask = row.get('ask', 0) or row.get('a', 0) or row.get('ask_price', 0) or 0
            
            if bid == 0 or ask == 0:
                return None
                
            return round((bid + ask) / 2, 2)
            
        except Exception as e:
            logger.error(f"{self.prefix} Error calculating option mark: {e}")
            return None

    def _attempt_entry(self, ticker: str, is_call: bool, chain: pd.DataFrame, price: float,
                       market_context: Dict = None, vol_regime: dict = None):
        """Attempt to enter a new ATM option position"""

        # ============ POSITION CORRELATION LIMITS ============
        # Block if we already have a position in the same direction
        same_direction_count = sum(
            1 for t, p in self.positions.items()
            if p["is_call"] == is_call
        )
        if same_direction_count >= 1:
            direction = "CALL (bullish)" if is_call else "PUT (bearish)"
            logger.info(f"{self.prefix} [{ticker}] NO TRADE: Already have {same_direction_count} {direction} position(s) - avoiding correlated risk")
            return

        # Limit total open positions to 2
        if len(self.positions) >= 2:
            logger.info(f"{self.prefix} [{ticker}] NO TRADE: Max 2 positions reached ({list(self.positions.keys())})")
            return

        # Get bars for MACD and brain analysis
        bars = self._cached_fetch(
            ('bars', ticker),
            lambda: data_client.get_spy_bars(
                datetime.now(EST) - timedelta(days=7),
                datetime.now(EST),
                ticker=ticker
            ),
            ttl=10
        )

        # ============ MACD ALIGNMENT FILTER ============
        # REQUIRE MACD alignment with trade direction (not just absence of conflict)
        # Calls: MACD histogram must be positive (bullish momentum)
        # Puts: MACD histogram must be negative (bearish momentum)
        if bars is not None and len(bars) >= 30:
            macd_cross = get_macd_crossover(bars, lookback=3)

            if is_call:  # Buying calls = need bullish MACD
                # Block if MACD just crossed down
                if macd_cross['crossing_down']:
                    logger.info(f"{self.prefix} [{ticker}] NO TRADE (MACD): Buying calls but MACD crossing DOWN "
                               f"({macd_cross['bars_since_cross']} bars ago)")
                    return
                # REQUIRE positive histogram (bullish momentum)
                if macd_cross['macd_momentum'] <= 0:
                    logger.info(f"{self.prefix} [{ticker}] NO TRADE (MACD): Buying calls but MACD histogram negative "
                               f"(hist={macd_cross['macd_momentum']:.3f}) - need bullish momentum")
                    return
            else:  # Buying puts = need bearish MACD
                # Block if MACD just crossed up
                if macd_cross['crossing_up']:
                    logger.info(f"{self.prefix} [{ticker}] NO TRADE (MACD): Buying puts but MACD crossing UP "
                               f"({macd_cross['bars_since_cross']} bars ago)")
                    return
                # REQUIRE negative histogram (bearish momentum)
                if macd_cross['macd_momentum'] >= 0:
                    logger.info(f"{self.prefix} [{ticker}] NO TRADE (MACD): Buying puts but MACD histogram positive "
                               f"(hist={macd_cross['macd_momentum']:.3f}) - need bearish momentum")
                    return

            # Log MACD alignment confirmation
            logger.info(f"{self.prefix} [{ticker}] MACD ALIGNED: hist={macd_cross['macd_momentum']:.3f} "
                       f"({'bullish' if is_call else 'bearish'} momentum confirmed)")

        # Default confidence/win_rate if brain unavailable
        brain_confidence = 50.0
        brain_win_rate = 50.0

        # Query brain for trade recommendation if available
        if self.brain:
            try:

                if bars is not None and len(bars) >= 30:
                    # Use live recommendation (combines indicators + history)
                    recommendation = self.brain.get_live_recommendation(
                        ticker=ticker,
                        bars=bars,
                        is_call=is_call
                    )

                    if not recommendation['should_enter']:
                        reason = recommendation.get('final_reason', recommendation.get('reason', 'Unknown'))
                        warnings = recommendation.get('indicator_warnings', [])
                        logger.info(f"{self.prefix} [{ticker}] Brain says SKIP: {reason}")
                        if warnings:
                            logger.info(f"{self.prefix} [{ticker}] Warnings: {', '.join(warnings)}")
                        return

                    hist = recommendation.get('historical_analysis', {})
                    brain_confidence = hist.get('confidence', 50.0)
                    brain_win_rate = hist.get('avg_win_rate', 50.0)

                    # Conservative mode: require higher confidence and win rate
                    if self.is_conservative_mode():
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
                if self.is_conservative_mode():
                    logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): Brain unavailable - {e}")
                    return
                logger.warning(f"{self.prefix} Brain query failed: {e} - proceeding with trade")

        # RL-based entry decision using adaptive learner
        learner_state_key = None
        if self.learner and market_context:
            try:
                learner_state_key = self.learner.get_state_key(
                    ticker=ticker,
                    trend=market_context.get('trend', 'chop'),
                    rsi=market_context.get('rsi', 50.0),
                    hour=market_context.get('hour', 12),
                    confidence=brain_confidence
                )

                rl_decision = self.learner.should_enter(learner_state_key, brain_win_rate)

                if not rl_decision['should_enter']:
                    logger.info(f"{self.prefix} [{ticker}] RL Learner says SKIP: {rl_decision['reason']}")
                    # Don't completely block - only block if Q-values strongly favor skip
                    q_vals = rl_decision.get('q_values', {})
                    if q_vals.get('skip', 0) > q_vals.get('enter', 0) + 0.2:
                        logger.info(f"{self.prefix} [{ticker}] Strong RL skip signal - blocking entry")
                        # Track as counterfactual for learning
                        try:
                            atm_strike = round(price)
                            type_col = next((c for c in ['type', 'contract_type', 'option_type']
                                           if c in chain.columns), None)
                            if type_col:
                                target_type = 'call' if is_call else 'put'
                                opts = chain[chain[type_col].str.lower() == target_type].copy()
                                if 'strike_price' in opts.columns:
                                    opts['strike_diff'] = abs(opts['strike_price'] - atm_strike)
                                    atm_opt = opts.loc[opts['strike_diff'].idxmin()]
                                    mark = (atm_opt.get('bid_price', 0) + atm_opt.get('ask_price', 0)) / 2
                                    if mark > 0:
                                        self.learner.track_skipped_trade(
                                            ticker=ticker,
                                            state_key=learner_state_key,
                                            entry_price=mark,
                                            strike=atm_strike,
                                            is_call=is_call,
                                            debit=mark,
                                            contracts=10
                                        )
                                        logger.info(f"{self.prefix} [{ticker}] Tracking counterfactual: {target_type.upper()} @ ${mark:.2f}")
                        except Exception as e:
                            logger.debug(f"{self.prefix} [{ticker}] Could not track counterfactual: {e}")
                        return
                    else:
                        logger.info(f"{self.prefix} [{ticker}] Weak RL skip - proceeding anyway")
                else:
                    if rl_decision.get('exploration'):
                        logger.info(f"{self.prefix} [{ticker}] RL Learner: EXPLORATION entry")
                    else:
                        logger.info(f"{self.prefix} [{ticker}] RL Learner says ENTER: {rl_decision['reason']}")

            except Exception as e:
                logger.debug(f"{self.prefix} [{ticker}] RL learner error: {e}")

        try:
            # Find type column
            type_col = next((c for c in ['type', 'contract_type', 'option_type', 
                                        'right', 'call_put', 't'] 
                            if c in chain.columns), None)
            if type_col is None:
                logger.warning(f"{self.prefix} [{ticker}] No type column in chain")
                return

            # Filter for option type
            target_type = 'call' if is_call else 'put'
            opts = chain[chain[type_col].str.lower() == target_type].copy()
            
            if opts.empty:
                logger.debug(f"{self.prefix} [{ticker}] No {target_type}s in chain")
                return

            # Find ATM strike
            strike_col = 'strike_price' if 'strike_price' in opts.columns else 'strike'
            closest_idx = abs(opts[strike_col] - price).argmin()
            row = opts.iloc[closest_idx]
            
            # Get debit
            debit = self._get_option_mark(row)
            
            if debit is None or debit < 0.20:
                logger.debug(f"{self.prefix} [{ticker}] Invalid/low debit: {debit}")
                return

            strike_price = float(row[strike_col])

            # Calculate adaptive position size using brain confidence/win_rate
            contracts = self._calculate_position_size(
                confidence=brain_confidence,
                win_rate=brain_win_rate,
                vol_regime=vol_regime
            )

            # Get adaptive exit parameters from brain
            exit_params = self._get_adaptive_exits(ticker, is_call, market_context or {})

            if exit_params.get('source') == 'brain':
                logger.info(f"{self.prefix} [{ticker}] Using brain exits: "
                           f"PT={exit_params['profit_target_pct']:.0f}%, "
                           f"SL={exit_params['stop_loss_pct']:.0f}%, "
                           f"Trail={exit_params['trail_activation_pct']:.0f}%")

            # Store position with market context for learning
            self.positions[ticker] = {
                "is_call": is_call,
                "strike_price": strike_price,
                "debit": debit,
                "contracts": contracts,
                "best_mark": debit,
                "worst_mark": debit,  # Track worst for max drawdown
                "trail_active": False,
                "trail_level": None,
                "entry_time": datetime.now(EST).isoformat(),
                "chop_count": 0,
                "market_context": market_context or {},
                # Adaptive exit parameters
                "profit_target_pct": exit_params['profit_target_pct'],
                "stop_loss_pct": exit_params['stop_loss_pct'],
                "trail_activation_pct": exit_params['trail_activation_pct'],
                # RL learner state key for feedback
                "learner_state_key": learner_state_key,
            }

            # Send notification ONLY if position was successfully stored
            if ticker in self.positions:
                try:
                    send_scalp_entry(
                        is_call=is_call,
                        strike=strike_price,
                        debit=debit * contracts * 100,
                        underlying=ticker
                    )
                except Exception as e:
                    logger.error(f"{self.prefix} Failed to send entry notification: {e}")

                logger.info(f"{self.prefix} ENTRY: {ticker} LONG ATM {'CALL' if is_call else 'PUT'} "
                           f"{strike_price:.1f} @ ${debit:.2f} x{contracts}")
            else:
                logger.error(f"{self.prefix} [{ticker}] Position storage failed - notification blocked")
            
        except Exception as e:
            logger.error(f"{self.prefix} [{ticker}] Entry error: {e}", exc_info=True)

    def _get_current_mark(self, ticker: str, pos: Dict) -> Optional[float]:
        """Get current market price for an open option position"""
        try:
            now = datetime.now(EST)
            
            chain = self._cached_fetch(
                ('chain', ticker),
                lambda: data_client.get_spy_option_chain(now.strftime("%Y-%m-%d"), ticker=ticker),
                ttl=8
            )
            
            if chain is None or chain.empty:
                return None

            type_col = next((c for c in ['type', 'contract_type', 'option_type', 
                                        'right', 'call_put', 't'] 
                            if c in chain.columns), None)
            if type_col is None:
                return None

            target_type = 'call' if pos["is_call"] else 'put'
            strike_col = 'strike_price' if 'strike_price' in chain.columns else 'strike'
            
            row = chain[
                (chain[strike_col] == pos["strike_price"]) & 
                (chain[type_col].str.lower() == target_type)
            ]
            
            if row.empty:
                return None

            return self._get_option_mark(row.iloc[0])
            
        except Exception as e:
            logger.error(f"{self.prefix} [{ticker}] Error fetching mark: {e}")
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
        """Manage a single scalp position with tight exit logic"""

        # Get current bars for trend and momentum
        bars = self._cached_fetch(
            ('bars', ticker),
            lambda: data_client.get_spy_bars(now - timedelta(days=7), now, ticker=ticker),
            ttl=10
        )

        if bars is None or bars.empty or len(bars) < 20:
            return

        # Calculate indicators
        try:
            trend = get_trend_signal(bars, None, None)
            rsi_value = _rsi(bars['close'])
            macd_line, macd_signal = _macd(bars['close'])
            macd_bull = macd_line > macd_signal
        except Exception as e:
            logger.error(f"{self.prefix} [{ticker}] Error getting indicators: {e}")
            return

        # Get current mark
        mark = self._get_current_mark(ticker, pos)

        if mark is None:
            logger.debug(f"{self.prefix} [{ticker}] No live mark - skipping cycle")
            return

        # Store current mark for dashboard display
        pos["current_mark"] = mark

        # Parse entry time and calculate hold duration
        try:
            entry_dt = datetime.fromisoformat(pos["entry_time"])
        except:
            entry_dt = now

        hold_seconds = (now - entry_dt).total_seconds()
        hold_minutes = hold_seconds / 60

        debit = pos["debit"]
        profit_pct = (mark - debit) / debit if debit > 0 else 0

        logger.debug(f"{self.prefix} [{ticker}] LONG {'CALL' if pos['is_call'] else 'PUT'} "
                    f"{pos['strike_price']:.1f} | ${debit:.2f}→${mark:.2f} "
                    f"({profit_pct*100:+.1f}%) | Hold: {hold_minutes:.1f}m")

        # Update best mark for trailing
        if mark > pos["best_mark"]:
            pos["best_mark"] = mark
            if pos["trail_active"]:
                pos["trail_level"] = pos["best_mark"] * 0.90  # 10% trail
                logger.info(f"{self.prefix} [{ticker}] Trail updated → ${pos['trail_level']:.2f}")

        # Update worst mark for drawdown tracking
        if mark < pos.get("worst_mark", mark):
            pos["worst_mark"] = mark

        # ============ ADAPTIVE SCALP EXIT LOGIC ============
        realized = None
        exit_reason = None

        # Get adaptive exit parameters (with defaults for backward compatibility)
        profit_target = pos.get("profit_target_pct", 20.0) / 100.0  # Convert to decimal
        stop_loss = pos.get("stop_loss_pct", 15.0) / 100.0
        trail_activation = pos.get("trail_activation_pct", 10.0) / 100.0

        # 1. QUICK SCALP: +8% in first 2 minutes = take profit
        if profit_pct >= 0.08 and hold_minutes <= 2:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Quick scalp +{profit_pct*100:.0f}% in {hold_minutes:.1f}m"

        # 2. TRAILING STOP HIT - let winners run until trail is hit
        elif pos["trail_active"] and mark <= pos["trail_level"]:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Trail stop @ ${pos['trail_level']:.2f}"

        # 4. STOP LOSS (adaptive)
        elif profit_pct <= -stop_loss:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Stop loss {profit_pct*100:.0f}% (limit: {-stop_loss*100:.0f}%)"

        # 5. TIME DECAY STOP: After 5min, tighten stop by 30%
        elif hold_minutes > 5 and profit_pct <= -(stop_loss * 0.7):
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Time decay stop {profit_pct*100:.0f}% after {hold_minutes:.0f}m"

        # 6. TREND REVERSAL
        elif (pos["is_call"] and trend == "bear") or (not pos["is_call"] and trend == "bull"):
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Trend reversal → {trend}"

        # 7. MOMENTUM FADE: RSI diverges from position
        elif pos["is_call"] and rsi_value < 40:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Momentum fade RSI={rsi_value:.0f} (call)"
        elif not pos["is_call"] and rsi_value > 60:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Momentum fade RSI={rsi_value:.0f} (put)"

        # 8. MACD CROSSOVER against position
        elif pos["is_call"] and not macd_bull and hold_minutes > 1:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = "MACD bearish cross (call)"
        elif not pos["is_call"] and macd_bull and hold_minutes > 1:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = "MACD bullish cross (put)"

        # 9. CHOP EXIT: 2 consecutive chop cycles (reduced from 3)
        elif trend == "chop":
            pos["chop_count"] = pos.get("chop_count", 0) + 1
            if pos["chop_count"] >= 2:
                realized = (mark - debit) * pos["contracts"] * 100
                exit_reason = "Chop exit (2 cycles)"
        else:
            pos["chop_count"] = 0

        # 10. MAX HOLD: 8 minutes (reduced from 20)
        if realized is None and hold_minutes > 8:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Max hold {hold_minutes:.0f}m"

        # 11. EOD EXIT: 3:30 PM
        if realized is None and now.time() >= time(15, 30):
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = "EOD 3:30 PM"

        # 12. OPTION WORTHLESS
        if realized is None and mark <= 0.01:
            realized = -debit * pos["contracts"] * 100
            exit_reason = "Option worthless"

        # 13. ACTIVATE TRAILING at 10% profit - 10% trail, let winners run
        if realized is None and profit_pct >= 0.10 and not pos["trail_active"]:
            pos["trail_active"] = True
            pos["trail_level"] = pos["best_mark"] * 0.90
            logger.info(f"{self.prefix} [{ticker}] +{profit_pct*100:.0f}% profit - 10% trail active @ ${pos['trail_level']:.2f}")
            if is_ready():
                try:
                    send_webhook(f"**TRAIL** {ticker} +{profit_pct*100:.0f}% → 10% trail @ ${pos['trail_level']:.2f}")
                except:
                    pass

        # Execute exit
        if realized is not None:
            logger.info(f"{self.prefix} [{ticker}] EXIT: {exit_reason} | "
                       f"Entry: ${debit:.2f} → Exit: ${mark:.2f} | "
                       f"P&L: ${realized:+.2f} ({profit_pct*100:+.1f}%) | "
                       f"Hold: {hold_minutes:.1f}m | "
                       f"Trend: {trend} | RSI: {rsi_value:.0f} | MACD Bull: {macd_bull}")
            self.exit_position(ticker, pos, realized, mark, exit_reason)

    def exit_position(self, ticker: str, pos: Dict, realized_pnl: float, final_mark: float,
                      exit_reason: str = "unknown"):
        """Close a position and update state with learning metrics"""
        try:
            # Update P&L and equity (scalp strategy only - no cross-contamination)
            self.daily_pnl += realized_pnl
            self.equity_history.append(self.equity_history[-1] + realized_pnl)

            # Update shared equity (single source of truth)
            update_shared_equity(realized_pnl)

            # Calculate learning metrics
            debit = pos["debit"]
            best_mark = pos.get("best_mark", debit)
            worst_mark = pos.get("worst_mark", debit)
            entry_time_str = pos.get("entry_time", "unknown")

            # Max profit seen (best_mark - debit) / debit * 100 (for long options)
            max_profit_pct = ((best_mark - debit) / debit * 100) if debit > 0 else 0.0

            # Max drawdown seen (debit - worst_mark) / debit * 100
            max_drawdown_pct = ((debit - worst_mark) / debit * 100) if debit > 0 else 0.0

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
                send_scalp_exit(
                    is_call=pos["is_call"],
                    strike=pos["strike_price"],
                    debit=pos["debit"] * pos["contracts"] * 100,
                    pnl=realized_pnl,
                    underlying=ticker
                )
            except Exception as e:
                logger.error(f"{self.prefix} Failed to send exit notification: {e}")

            # Enhanced exit logging with indicators
            market_ctx = pos.get("market_context", {})
            logger.info(f"{self.prefix} EXIT: {ticker} LONG {'CALL' if pos['is_call'] else 'PUT'} "
                       f"{pos['strike_price']:.1f} | P&L: ${realized_pnl:+.2f} | Reason: {exit_reason} | "
                       f"Entry: ${debit:.2f} → Exit: ${final_mark:.2f} | "
                       f"RSI: {market_ctx.get('rsi', 'N/A')} | Trend: {market_ctx.get('trend', 'N/A')}")

            # Archive closed trade with market context and learning metrics
            closed_trade = {
                "ticker": ticker,
                "is_call": pos["is_call"],
                "strike_price": pos["strike_price"],
                "debit": pos["debit"],
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
            self._save_individual_trade(closed_trade)

            # Record trade for statistical tracking (binomial, conditional EV, etc.)
            try:
                record_trade({
                    'pnl': realized_pnl,
                    'win': realized_pnl > 0,
                    'regime': market_ctx.get('trend', 'unknown'),
                    'setup_type': 'long_call' if pos['is_call'] else 'long_put',
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
                    # Normalize P&L to reward: max_risk is debit * contracts * 100
                    max_risk = debit * pos.get("contracts", 10) * 100
                    self.learner.record_trade_outcome(
                        state_key=learner_state_key,
                        pnl=realized_pnl,
                        max_risk=max_risk
                    )
                    logger.debug(f"{self.prefix} [{ticker}] Updated RL learner with P&L=${realized_pnl:+.2f}")
                except Exception as e:
                    logger.debug(f"{self.prefix} [{ticker}] RL learner update failed: {e}")

            # Remove position and set exit cooloff
            del self.positions[ticker]
            self.last_exit[ticker] = ('bull' if pos["is_call"] else 'bear', datetime.now(EST))

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
            "current_trends": self.current_trends,
        }


if __name__ == "__main__":
    logger.info("Starting Scalp Strategy")
    bot = ScalpStrategy()
    
    while True:
        start_time = time_mod.time()
        
        try:
            bot.run_cycle()
        except Exception as e:
            logger.error(f"Critical cycle error: {e}", exc_info=True)
        
        cycle_duration = time_mod.time() - start_time
        logger.debug(f"Cycle completed in {cycle_duration:.2f}s")

        now_time = datetime.now(EST).time()

        # Dynamic sleep based on time of day
        if time(9, 30) <= now_time <= time(16, 0):
            target_sleep = 15.0  # Full market hours - 15 seconds
        elif time(8, 0) <= now_time < time(9, 30):
            target_sleep = 30.0  # Pre-market - 30 seconds
        else:
            target_sleep = 300.0  # Overnight - 5 minutes

        sleep_time = max(1.0, target_sleep - cycle_duration)
        logger.debug(f"Sleeping {sleep_time:.1f}s (target {target_sleep}s)")
        time_mod.sleep(sleep_time)