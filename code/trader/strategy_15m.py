# code/trader/strategy_15m.py – 15-minute vertical credit spread strategy (Alpaca Data)
# REFACTORED: Fixed dashboard issues with proper logging, error handling, and state management
# Updated: Fixed $5 width spreads, lowered min credit to $0.15, widened delta to 20-45
# Added: End-of-day forced closeout at/after 4:00 PM EST

import os
import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from filelock import FileLock, Timeout
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple, List

# Import your existing modules
from code.data.data_client import data_client
from indicators import get_trend_signal
from discord_notifier import send_entry, send_exit, set_strategy_ready

# Configure logging - use INFO for production, DEBUG for troubleshooting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/strategy_15m.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
EST = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).parent.parent.parent
STATE_FILE = PROJECT_ROOT / "strategy_15m_state.json"
LOCK_FILE = PROJECT_ROOT / "strategy_15m_state.json.lock"

# NYSE holidays for 2026
NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # Martin Luther King Jr. Day
    date(2026, 2, 16),   # Washington's Birthday
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving Day
    date(2026, 12, 25)   # Christmas Day
}


class TradingStrategy15m:
    """15-minute timeframe vertical credit spread strategy"""
    
    def __init__(self):
        self.prefix = "[15m]"
        self._strategy_ready_sent = False
        self.trades_today = []
        self.current_trends = {}  # Track current trend for each ticker

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
            'closed_trades': self.closed_trades
        }
        
        try:
            with self._state_lock():
                # Write to temporary file first, then atomic rename
                temp_file = STATE_FILE.with_suffix('.json.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state, f, indent=4)
                temp_file.replace(STATE_FILE)
                
        except Exception as e:
            logger.error(f"{self.prefix} Failed to save state: {e}", exc_info=True)

    def _log_closed_trades_batch(self):
        """Archive closed trades to knowledge base when batch reaches 30"""
        if len(self.closed_trades) < 30:
            return
            
        try:
            timestamp = datetime.now(EST).strftime("%Y%m%d_%H%M%S")
            rag_folder = Path("/data/knowledge")
            rag_folder.mkdir(parents=True, exist_ok=True)
            batch_file = rag_folder / f"trades_batch_15m_{timestamp}.json"
            
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
        if today in NYSE_HOLIDAYS_2026:
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
        """Check if we can enter new trades (9:35am-2:30pm ET)"""
        if not self.is_trading_day():
            return False
            
        now = datetime.now(EST).time()
        return time(9, 35) <= now < time(14, 30)

    def check_daily_loss_nanny(self) -> bool:
        """Check if daily loss limit has been hit"""
        if self.daily_pnl <= self.daily_loss_limit:
            logger.warning(f"{self.prefix} Daily loss limit hit: ${self.daily_pnl:.2f} <= ${self.daily_loss_limit:.2f}")
            return False
        return True

    def is_conservative_mode(self) -> bool:
        """Check if we should be more selective (daily PnL >= $1000)"""
        return self.daily_pnl >= 1000.0

    def run_cycle(self):
        """Main strategy cycle - called periodically"""
        now = datetime.now(EST)
        current_time = now.time()
        tickers = ["SPY", "QQQ", "IWM"]

        # Log cycle start at DEBUG level to reduce noise
        logger.debug(f"{self.prefix} Cycle start: {now.strftime('%H:%M:%S')}")
        mode = "CONSERVATIVE" if self.is_conservative_mode() else "NORMAL"
        logger.info(f"{self.prefix} Daily P&L: ${self.daily_pnl:+.2f} | Open: {len(self.positions)} | Mode: {mode}")

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
                current_value = self._get_current_spread_mark(ticker, pos)
                if current_value is None:
                    logger.warning(f"{self.prefix} [{ticker}] No final mark - closing at $0.00")
                    current_value = 0.0
                    
                realized = (pos["credit"] - current_value) * pos["contracts"] * 100
                self.exit_position(ticker, pos, realized, current_value)
                
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

        # Resample to 15-minute bars
        bars_15m = self._resample_to_15min(minute_bars)
        if bars_15m is None:
            return

        # Get trend signal
        try:
            trend = get_trend_signal(bars_15m, None, None)
            self.current_trends[ticker] = trend  # Store for dashboard display
            price = bars_15m['close'].iloc[-1]
        except Exception as e:
            logger.error(f"{self.prefix} [{ticker}] Error calculating trend: {e}")
            return

        # Fetch option chain
        chain = self._fetch_option_chain(ticker, now.strftime("%Y-%m-%d"))
        if chain is None:
            return

        logger.debug(f"{self.prefix} [{ticker}] Price: ${price:.2f} | Trend: {trend}")

        # Entry logic
        if ticker not in self.positions:
            if trend != "chop" and self.can_enter_trades():
                try:
                    self._attempt_entry(ticker, trend, chain, price, now.strftime("%Y-%m-%d"))
                except Exception as e:
                    logger.error(f"{self.prefix} [{ticker}] Entry error: {e}", exc_info=True)
        else:
            logger.debug(f"{self.prefix} [{ticker}] Monitoring existing position")

    def _fetch_minute_bars(self, ticker: str, now: datetime) -> Optional[pd.DataFrame]:
        """Fetch minute bars with error handling"""
        try:
            minute_bars = data_client.get_spy_bars(
                now - timedelta(days=7), 
                now, 
                ticker=ticker
            )
            
            if minute_bars is None or minute_bars.empty or len(minute_bars) < 20:
                logger.debug(f"{self.prefix} [{ticker}] Insufficient minute bars")
                return None
                
            return minute_bars
            
        except Exception as e:
            logger.error(f"{self.prefix} [{ticker}] Error fetching bars: {e}")
            return None

    def _resample_to_15min(self, minute_bars: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Resample minute bars to 15-minute timeframe"""
        try:
            bars_15m = minute_bars.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if bars_15m.empty or len(bars_15m) < 20:
                logger.debug(f"{self.prefix} Insufficient 15m bars after resampling")
                return None
                
            return bars_15m
            
        except Exception as e:
            logger.error(f"{self.prefix} Error resampling bars: {e}")
            return None

    def _fetch_option_chain(self, ticker: str, expiration: str) -> Optional[pd.DataFrame]:
        """Fetch option chain with error handling"""
        try:
            chain = data_client.get_spy_option_chain(expiration, ticker=ticker)
            
            if chain is None or chain.empty:
                logger.debug(f"{self.prefix} [{ticker}] Empty option chain")
                return None
                
            return chain
            
        except Exception as e:
            logger.error(f"{self.prefix} [{ticker}] Error fetching chain: {e}")
            return None

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
            if for_entry and credit < 0.15:
                return None

            # Return mid-market spread price
            return round((short_bid + short_ask)/2 - (long_bid + long_ask)/2, 2)

        except Exception as e:
            logger.error(f"{self.prefix} Error calculating spread price: {e}")
            return None

    def _attempt_entry(self, ticker: str, trend: str, chain: pd.DataFrame,
                      price: float, expiration_date: str):
        """Attempt to enter a new credit spread position"""

        conservative = self.is_conservative_mode()
        is_put = trend == "bull"  # Bull trend -> put credit spread
        now = datetime.now(EST)

        # Build market context for learning
        market_context = {
            "trend": trend,
            "price": round(price, 2),
            "hour": now.hour,
            "day_of_week": now.strftime("%A"),
            "conservative_mode": conservative,
        }
        
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
            logger.debug(f"{self.prefix} [{ticker}] No {target_type}s in chain")
            return

        opts = opts.sort_values(strike_col)

        # Find short strike candidates (20-45 delta or ~0.7% OTM fallback)
        short_candidates = self._find_short_candidates(opts, is_put, price, strike_col)
        
        if short_candidates.empty:
            logger.debug(f"{self.prefix} [{ticker}] No suitable short strikes")
            return

        # Find valid $5-wide spreads
        candidates = self._find_spread_candidates(
            short_candidates, opts, is_put, strike_col
        )

        if not candidates:
            logger.debug(f"{self.prefix} [{ticker}] No valid $5-wide spreads >= $0.15")
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
        contracts = 10

        # Store position with market context for learning
        self.positions[ticker] = {
            "is_put": is_put,
            "short": short_strike,
            "long": long_strike,
            "credit": best_credit,
            "contracts": contracts,
            "best_value": best_credit,
            "trail_active": False,
            "trail_level": None,
            "entry_time": datetime.now(EST).isoformat(),
            "expiration_date": expiration_date,
            "market_context": market_context,
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
                   f"{short_strike:.1f}/{long_strike:.1f} @ ${best_credit:.2f}")

    def _find_short_candidates(self, opts: pd.DataFrame, is_put: bool, 
                              price: float, strike_col: str) -> pd.DataFrame:
        """Find short strike candidates using delta or OTM percentage"""
        
        delta_col = next((c for c in ['delta', 'd'] if c in opts.columns), None)
        short_candidates = pd.DataFrame()

        # Try delta-based selection first
        if delta_col is not None:
            if is_put:
                short_candidates = opts[
                    (opts[delta_col] >= -0.45) & (opts[delta_col] <= -0.20)
                ]
            else:
                short_candidates = opts[
                    (opts[delta_col] >= 0.20) & (opts[delta_col] <= 0.45)
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

    def _get_current_spread_mark(self, ticker: str, pos: Dict) -> Optional[float]:
        """Get current market price for an open spread position"""
        try:
            expiration = pos.get("expiration_date", datetime.now(EST).strftime("%Y-%m-%d"))
            chain = data_client.get_spy_option_chain(expiration, ticker=ticker)

            if chain is None or chain.empty:
                logger.debug(f"{self.prefix} [{ticker}] Empty chain for {expiration}")
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
        """Manage a single position"""
        
        current_value = self._get_current_spread_mark(ticker, pos)

        if current_value is None:
            logger.info(f"{self.prefix} [{ticker}] No spread mark - skipping cycle")
            return

        # Store current value for dashboard display
        pos["current_value"] = current_value

        credit = pos["credit"]
        unrealized = (credit - current_value) * pos["contracts"] * 100

        logger.info(f"{self.prefix} [{ticker}] {'PUT' if pos['is_put'] else 'CALL'} "
                   f"{pos['short']:.1f}/{pos['long']:.1f} | "
                   f"Entry: ${credit:.2f} | Current: ${current_value:.2f} | "
                   f"Unrealized: ${unrealized:+.2f}")

        # Update best value and trailing stop
        if current_value < pos["best_value"]:
            old_trail = pos.get("trail_level")
            pos["best_value"] = current_value
            
            if pos["trail_active"]:
                pos["trail_level"] = pos["best_value"] * 1.10
                logger.info(f"{self.prefix} [{ticker}] Trailing stop updated: "
                          f"${pos['trail_level']:.2f} (from ${old_trail:.2f if old_trail else 'N/A'})")

        # Check exit conditions
        realized = None

        # 1. Stop loss: 2x credit (100% loss)
        if current_value >= 2.0 * credit:
            realized = (credit - current_value) * pos["contracts"] * 100
            logger.warning(f"{self.prefix} [{ticker}] Stop loss hit (2x credit)")

        # 2. Time-based exit after 2:30 PM
        elif now.time() >= time(14, 30):
            realized = (credit - current_value) * pos["contracts"] * 100
            logger.info(f"{self.prefix} [{ticker}] Time-based exit (after 2:30 PM)")

        # 3. Profit target: 50% of credit (activates trailing)
        elif (credit - current_value) >= 0.5 * credit:
            if not pos["trail_active"]:
                pos["trail_active"] = True
                pos["trail_level"] = pos["best_value"] * 1.10
                logger.info(f"{self.prefix} [{ticker}] 50% profit - trailing stop "
                          f"activated @ ${pos['trail_level']:.2f}")

            # Check if trailing stop hit
            if current_value >= pos["trail_level"]:
                realized = (credit - current_value) * pos["contracts"] * 100
                logger.info(f"{self.prefix} [{ticker}] Trailing stop hit")

        # Exit if any condition triggered
        if realized is not None:
            self.exit_position(ticker, pos, realized, current_value)

    def exit_position(self, ticker: str, pos: Dict, realized_pnl: float, final_mark: float):
        """Close a position and update state"""
        try:
            # Update P&L and equity
            self.daily_pnl += realized_pnl
            self.equity_history.append(self.equity_history[-1] + realized_pnl)

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

            logger.info(f"{self.prefix} EXIT: {ticker} {'PUT' if pos['is_put'] else 'CALL'} "
                       f"{pos['short']:.1f}/{pos['long']:.1f} | "
                       f"P&L: ${realized_pnl:+.2f} (mark ${final_mark:.2f})")

            # Archive closed trade with market context for learning
            closed_trade = {
                "ticker": ticker,
                "is_put": pos["is_put"],
                "short": pos["short"],
                "long": pos["long"],
                "credit": pos["credit"],
                "pnl": realized_pnl,
                "entry_time": pos.get("entry_time", "unknown"),
                "exit_time": datetime.now(EST).isoformat(),
                "market_context": pos.get("market_context", {}),
            }
            self.closed_trades.append(closed_trade)

            # Remove position
            del self.positions[ticker]
            self.trades_today.append("exit")

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