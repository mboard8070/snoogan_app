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
from indicators import get_trend_signal, _macd, _rsi
from discord_notifier import send_scalp_entry, send_scalp_exit, set_strategy_ready, send_webhook

# Import brain for trade decisions (optional - fails gracefully)
try:
    from code.rag.snoogans_brain import SnoogansBrain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False

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

            # Reset daily P&L if new day
            if self.last_pnl_date != date.today().isoformat():
                logger.info(f"{self.prefix} New trading day - resetting daily P&L")
                self.daily_pnl = 0.0
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
            'closed_trades': self.closed_trades
        }
        
        try:
            with self._state_lock():
                # Write to temporary file first, then atomic rename
                temp_file = STATE_FILE.with_suffix('.json.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state, f, indent=4, default=str)
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
            
            rsi_value = _rsi(bars['close'])  # Returns a single float
            price = bars['close'].iloc[-1]

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

        logger.debug(f"{self.prefix} [{ticker}] Price: ${price:.2f} | Trend: {trend} | "
                    f"MACD Bull: {is_momentum_bull} | RSI: {rsi_value:.2f}")

        # Entry logic
        if ticker not in self.positions:
            if self.can_enter_trades():
                self._check_entry_conditions(ticker, trend, is_momentum_bull, rsi_value, chain, price, now)
            else:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: Outside entry hours (9:31am-3pm ET)")
        else:
            logger.info(f"{self.prefix} [{ticker}] NO TRADE: Already have open position")

    def _check_entry_conditions(self, ticker: str, trend: str, is_momentum_bull: bool,
                                rsi_value: float, chain: pd.DataFrame, price: float, now: datetime):
        """Check if entry conditions are met"""

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
            cooloff_minutes = 10 if conservative else 5  # Longer cooloff in conservative mode
            if (now - last_time) < timedelta(minutes=cooloff_minutes) and new_dir != last_dir:
                logger.info(f"{self.prefix} [{ticker}] NO TRADE: Cooloff period after {last_dir} exit")
                return

        # Build market context for learning
        market_context = {
            "trend": trend,
            "rsi": round(rsi_value, 2),
            "macd_bull": is_momentum_bull,
            "price": round(price, 2),
            "hour": now.hour,
            "day_of_week": now.strftime("%A"),
            "conservative_mode": conservative,
        }

        # Conservative mode: Stricter RSI requirements
        if conservative:
            # Bull: RSI must be 55-70 (strong but not overbought)
            # Bear: RSI must be 30-45 (weak but not oversold)
            if trend == "bull" and not (55 <= rsi_value <= 70):
                logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): Bull RSI {rsi_value:.1f} not in 55-70 range")
                return
            if trend == "bear" and not (30 <= rsi_value <= 45):
                logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): Bear RSI {rsi_value:.1f} not in 30-45 range")
                return

        # Bull entry conditions
        if trend == "bull" and is_momentum_bull and rsi_value > 50:
            self._attempt_entry(ticker, True, chain, price, market_context)
        elif trend == "bull":
            # Bull trend but other conditions not met
            reasons = []
            if not is_momentum_bull:
                reasons.append("MACD bearish")
            if rsi_value <= 50:
                reasons.append(f"RSI too low ({rsi_value:.1f})")
            logger.info(f"{self.prefix} [{ticker}] NO TRADE: Bull trend but {', '.join(reasons)}")

        # Bear entry conditions
        elif trend == "bear" and not is_momentum_bull and rsi_value < 50:
            self._attempt_entry(ticker, False, chain, price, market_context)
        elif trend == "bear":
            # Bear trend but other conditions not met
            reasons = []
            if is_momentum_bull:
                reasons.append("MACD bullish")
            if rsi_value >= 50:
                reasons.append(f"RSI too high ({rsi_value:.1f})")
            logger.info(f"{self.prefix} [{ticker}] NO TRADE: Bear trend but {', '.join(reasons)}")

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
                       market_context: Dict = None):
        """Attempt to enter a new ATM option position"""

        # Query brain for trade recommendation if available
        if self.brain:
            try:
                # Get live bars for the brain to analyze
                bars = self._cached_fetch(
                    ('bars', ticker),
                    lambda: data_client.get_spy_bars(
                        datetime.now(EST) - timedelta(days=7),
                        datetime.now(EST),
                        ticker=ticker
                    ),
                    ttl=10
                )

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
                    confidence = hist.get('confidence', 0)
                    win_rate = hist.get('avg_win_rate', 0)

                    # Conservative mode: require higher confidence and win rate
                    if self.is_conservative_mode():
                        min_confidence = 60
                        min_win_rate = 50
                        if confidence < min_confidence or win_rate < min_win_rate:
                            logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): "
                                       f"Brain confidence {confidence:.0f}% < {min_confidence}% or "
                                       f"win rate {win_rate:.0f}% < {min_win_rate}%")
                            return

                    logger.info(f"{self.prefix} [{ticker}] Brain says ENTER: "
                              f"Win rate: {win_rate:.0f}% | Confidence: {confidence:.0f}%")

            except Exception as e:
                if self.is_conservative_mode():
                    logger.info(f"{self.prefix} [{ticker}] NO TRADE (CONSERVATIVE): Brain unavailable - {e}")
                    return
                logger.warning(f"{self.prefix} Brain query failed: {e} - proceeding with trade")

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
            contracts = 10

            # Store position with market context for learning
            self.positions[ticker] = {
                "is_call": is_call,
                "strike_price": strike_price,
                "debit": debit,
                "contracts": contracts,
                "best_mark": debit,
                "trail_active": False,
                "trail_level": None,
                "entry_time": datetime.now(EST).isoformat(),
                "chop_count": 0,
                "market_context": market_context or {},
            }

            # Send notification
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
                       f"{strike_price:.1f} @ ${debit:.2f}")
            
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
                pos["trail_level"] = pos["best_mark"] * 0.95  # 5% trail
                logger.info(f"{self.prefix} [{ticker}] Trail → ${pos['trail_level']:.2f}")

        # ============ TIGHT SCALP EXIT LOGIC ============
        realized = None
        exit_reason = None

        # 1. QUICK SCALP: +8% in first 2 minutes = take profit
        if profit_pct >= 0.08 and hold_minutes <= 2:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Quick scalp +{profit_pct*100:.0f}% in {hold_minutes:.1f}m"

        # 2. PROFIT TARGET: 20% (reduced from 40%)
        elif profit_pct >= 0.20:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Profit target +{profit_pct*100:.0f}%"

        # 3. TRAILING STOP HIT
        elif pos["trail_active"] and mark <= pos["trail_level"]:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Trail stop @ ${pos['trail_level']:.2f}"

        # 4. TIGHT STOP: 15% loss (reduced from 30%)
        elif profit_pct <= -0.15:
            realized = (mark - debit) * pos["contracts"] * 100
            exit_reason = f"Stop loss {profit_pct*100:.0f}%"

        # 5. TIME DECAY STOP: After 5min, tighten to 10% loss
        elif hold_minutes > 5 and profit_pct <= -0.10:
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

        # 13. ACTIVATE TRAILING: 10% profit starts 5% trail (earlier activation)
        if realized is None and profit_pct >= 0.10 and not pos["trail_active"]:
            pos["trail_active"] = True
            pos["trail_level"] = pos["best_mark"] * 0.95
            logger.info(f"{self.prefix} [{ticker}] +10% - 5% trail active @ ${pos['trail_level']:.2f}")
            try:
                send_webhook(f"**TRAIL** {ticker} +{profit_pct*100:.0f}% → 5% trail @ ${pos['trail_level']:.2f}")
            except:
                pass

        # Execute exit
        if realized is not None:
            logger.info(f"{self.prefix} [{ticker}] EXIT: {exit_reason} | "
                       f"Entry: ${debit:.2f} → Exit: ${mark:.2f} | "
                       f"P&L: ${realized:+.2f} ({profit_pct*100:+.1f}%) | "
                       f"Hold: {hold_minutes:.1f}m | "
                       f"Trend: {trend} | RSI: {rsi_value:.0f} | MACD Bull: {macd_bull}")
            self.exit_position(ticker, pos, realized, mark)

    def exit_position(self, ticker: str, pos: Dict, realized_pnl: float, final_mark: float):
        """Close a position and update state"""
        try:
            # Update P&L and equity
            self.daily_pnl += realized_pnl
            self.equity_history.append(self.equity_history[-1] + realized_pnl)

            # Attempt to sync with main strategy (optional)
            try:
                from strategy import TradingStrategy
                main = TradingStrategy()
                main.daily_pnl += realized_pnl
                main.equity_history.append(main.equity_history[-1] + realized_pnl)
                main._save_state()
            except Exception as e:
                logger.debug(f"{self.prefix} UI sync not available: {e}")

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

            logger.info(f"{self.prefix} EXIT: {ticker} LONG {'CALL' if pos['is_call'] else 'PUT'} "
                       f"{pos['strike_price']:.1f} | P&L: ${realized_pnl:+.2f} (mark ${final_mark:.2f})")

            # Archive closed trade with market context for learning
            closed_trade = {
                "ticker": ticker,
                "is_call": pos["is_call"],
                "strike_price": pos["strike_price"],
                "debit": pos["debit"],
                "pnl": realized_pnl,
                "entry_time": pos["entry_time"],
                "exit_time": datetime.now(EST).isoformat(),
                "market_context": pos.get("market_context", {}),
            }
            
            self.closed_trades.append(closed_trade)
            self._save_individual_trade(closed_trade)

            # Remove position and set exit cooloff
            del self.positions[ticker]
            self.last_exit[ticker] = ('bull' if pos["is_call"] else 'bear', datetime.now(EST))

        except Exception as e:
            logger.error(f"{self.prefix} Error exiting position: {e}", exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status for dashboard display"""
        return {
            "equity": self.equity_history[-1] if self.equity_history else 0,
            "daily_pnl": self.daily_pnl,
            "daily_loss_limit": self.daily_loss_limit,
            "open_positions": len(self.positions),
            "positions": self.positions,
            "trades_today": len(self.trades_today),
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