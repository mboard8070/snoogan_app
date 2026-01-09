# code/rag/adaptive_learner.py
# Q-Learning based adaptive entry decision system for Snoogans
# Learns from trade outcomes to improve entry decisions over time

import json
import os
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/home/mboard76/nvidia-workbench/snoogan_app")
STATE_FILE = BASE_DIR / "code" / "rag" / "adaptive_learner_state.json"


class AdaptiveLearner:
    """
    Q-Learning based adaptive learner for trade entry decisions.

    State Space (discretized):
        - ticker: SPY, QQQ, IWM
        - trend: bull, bear, chop
        - rsi_bucket: oversold, low, mid, high, overbought
        - hour_bucket: early (9-10), mid (10-12), afternoon (12-15)
        - confidence_bucket: low (0-40), medium (40-70), high (70-100)

    Action Space:
        - ENTER: Take the trade
        - SKIP: Don't take the trade

    Reward:
        - Normalized PnL from trade outcome
        - For SKIP action: small negative reward (opportunity cost)
    """

    # Actions
    ENTER = "enter"
    SKIP = "skip"

    def __init__(
        self,
        alpha: float = 0.15,           # Learning rate
        gamma: float = 0.0,            # Discount factor (0 = immediate rewards only, makes sense for single trades)
        epsilon: float = 0.20,         # Initial exploration rate
        epsilon_min: float = 0.05,     # Minimum exploration
        epsilon_decay: float = 0.995,  # Decay per trade
        optimistic_init: float = 0.1,  # Optimistic Q-value initialization
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimistic_init = optimistic_init

        # Q-table: state_key -> {action -> q_value}
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {self.ENTER: self.optimistic_init, self.SKIP: 0.0}
        )

        # Statistics
        self.stats = {
            "total_trades": 0,
            "total_skips": 0,
            "enter_wins": 0,
            "enter_losses": 0,
            "total_pnl": 0.0,
            "exploration_count": 0,
            "exploitation_count": 0,
            "counterfactual_good_skips": 0,  # Skips that avoided losses
            "counterfactual_bad_skips": 0,   # Skips that missed wins
        }

        # Counterfactual tracking: pending skipped trades to monitor
        # Format: {ticker: {state_key, entry_price, entry_time, strike, is_call, max_risk}}
        self.pending_counterfactuals: Dict[str, Dict] = {}

        # Decision history for monitoring (last N decisions)
        self.decision_history: List[Dict] = []
        self.max_history = 100

        # Load existing state
        self._load_state()

        logger.info(f"AdaptiveLearner initialized: epsilon={self.epsilon:.3f}, "
                   f"states={len(self.q_table)}, trades={self.stats['total_trades']}")

    def _discretize_rsi(self, rsi: float) -> str:
        """Convert continuous RSI to bucket."""
        if rsi < 30:
            return "oversold"
        elif rsi < 40:
            return "low"
        elif rsi < 60:
            return "mid"
        elif rsi < 70:
            return "high"
        else:
            return "overbought"

    def _discretize_hour(self, hour: int) -> str:
        """Convert hour to trading session bucket."""
        if hour < 10:
            return "early"  # First 30 min after open - volatile
        elif hour < 12:
            return "mid_morning"
        elif hour < 14:
            return "midday"
        else:
            return "afternoon"

    def _discretize_confidence(self, confidence: float) -> str:
        """Convert brain confidence to bucket."""
        if confidence < 40:
            return "low"
        elif confidence < 70:
            return "medium"
        else:
            return "high"

    def get_state_key(
        self,
        ticker: str,
        trend: str,
        rsi: float,
        hour: int,
        confidence: float
    ) -> str:
        """
        Create a discrete state key from market conditions.

        Args:
            ticker: Symbol (SPY, QQQ, IWM)
            trend: Current trend (bull, bear, chop)
            rsi: RSI value (0-100)
            hour: Hour of day (9-16)
            confidence: Brain's confidence (0-100)

        Returns:
            State key string like "SPY|bull|mid|mid_morning|medium"
        """
        rsi_bucket = self._discretize_rsi(rsi)
        hour_bucket = self._discretize_hour(hour)
        conf_bucket = self._discretize_confidence(confidence)

        return f"{ticker}|{trend}|{rsi_bucket}|{hour_bucket}|{conf_bucket}"

    def _get_q_values(self, state_key: str) -> Dict[str, float]:
        """Get Q-values for a state, initializing if needed."""
        if state_key not in self.q_table:
            # Optimistic initialization encourages exploration
            self.q_table[state_key] = {
                self.ENTER: self.optimistic_init,
                self.SKIP: 0.0
            }
        return self.q_table[state_key]

    def should_enter(self, state_key: str, brain_win_rate: float = 50.0) -> Dict:
        """
        Decide whether to enter a trade using epsilon-greedy policy.

        Args:
            state_key: Discretized state from get_state_key()
            brain_win_rate: Brain's estimated win rate for this trade

        Returns:
            dict with:
                - should_enter: bool
                - reason: str explanation
                - q_values: dict of action -> q_value
                - exploration: bool (True if this was a random choice)
        """
        q_values = self._get_q_values(state_key)

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.choice([self.ENTER, self.SKIP])
            exploration = True
            self.stats["exploration_count"] += 1
            reason = f"Exploration (epsilon={self.epsilon:.2f})"
        else:
            # Exploit: choose best action
            exploration = False
            self.stats["exploitation_count"] += 1

            # Incorporate brain's win rate as a prior
            # Boost ENTER Q-value slightly if brain is confident
            adjusted_enter_q = q_values[self.ENTER]
            if brain_win_rate > 60:
                adjusted_enter_q += 0.05  # Small boost for high brain confidence
            elif brain_win_rate < 35:
                adjusted_enter_q -= 0.05  # Penalty for low brain confidence

            if adjusted_enter_q > q_values[self.SKIP]:
                action = self.ENTER
                reason = f"Q[enter]={adjusted_enter_q:.3f} > Q[skip]={q_values[self.SKIP]:.3f}"
            elif q_values[self.SKIP] > adjusted_enter_q:
                action = self.SKIP
                reason = f"Q[skip]={q_values[self.SKIP]:.3f} > Q[enter]={adjusted_enter_q:.3f}"
            else:
                # Tie-breaker: favor ENTER if brain is confident
                action = self.ENTER if brain_win_rate >= 45 else self.SKIP
                reason = f"Q-values tied, brain_wr={brain_win_rate:.0f}%"

        decision = {
            "should_enter": action == self.ENTER,
            "action": action,
            "reason": reason,
            "q_values": dict(q_values),
            "exploration": exploration,
            "state_key": state_key,
            "epsilon": self.epsilon,
            "timestamp": datetime.now().isoformat(),
            "brain_win_rate": brain_win_rate,
        }

        # Track decision in history
        self.decision_history.append(decision)
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)

        return decision

    def track_skipped_trade(
        self,
        ticker: str,
        state_key: str,
        entry_price: float,
        strike: float,
        is_call: bool,
        debit: float,
        contracts: int = 10
    ):
        """
        Track a skipped trade for counterfactual learning.
        Call this when the learner decides to SKIP but the strategy would have entered.

        Args:
            ticker: Symbol (SPY, QQQ, IWM)
            state_key: The state key when skipped
            entry_price: Option price at time of skip
            strike: Strike price
            is_call: True for calls, False for puts
            debit: What the debit would have been
            contracts: Number of contracts
        """
        self.pending_counterfactuals[ticker] = {
            "state_key": state_key,
            "entry_price": entry_price,
            "entry_time": datetime.now().isoformat(),
            "strike": strike,
            "is_call": is_call,
            "debit": debit,
            "contracts": contracts,
            "max_risk": debit * contracts * 100,
            "best_price": entry_price,
            "worst_price": entry_price,
        }
        logger.info(f"Tracking counterfactual: {ticker} {'CALL' if is_call else 'PUT'} {strike} @ ${debit:.2f}")

    def update_counterfactual_prices(self, ticker: str, current_price: float):
        """
        Update the best/worst price seen for a pending counterfactual.
        Call this on each cycle while monitoring a skipped trade.

        Args:
            ticker: Symbol
            current_price: Current option mark price
        """
        if ticker not in self.pending_counterfactuals:
            return

        cf = self.pending_counterfactuals[ticker]
        cf["best_price"] = max(cf["best_price"], current_price)
        cf["worst_price"] = min(cf["worst_price"], current_price)

    def resolve_counterfactual(
        self,
        ticker: str,
        exit_price: float = None,
        reason: str = "timeout"
    ) -> Optional[Dict]:
        """
        Resolve a counterfactual trade and update Q[skip] accordingly.

        Args:
            ticker: Symbol
            exit_price: Final price (if None, uses worst_price for loss estimate)
            reason: Why we're resolving (timeout, trend_change, etc.)

        Returns:
            dict with counterfactual analysis, or None if not tracked
        """
        if ticker not in self.pending_counterfactuals:
            return None

        cf = self.pending_counterfactuals.pop(ticker)
        state_key = cf["state_key"]
        entry_price = cf["entry_price"]
        max_risk = cf["max_risk"]

        # Calculate what P&L would have been
        if exit_price is None:
            # Assume we would have exited at worst price (conservative)
            exit_price = cf["worst_price"]

        would_have_pnl = (exit_price - entry_price) * cf["contracts"] * 100

        # Determine if skipping was good or bad
        if would_have_pnl > 0:
            # We missed a winning trade - skipping was BAD
            skip_reward = -0.15  # Penalty for missing profit
            self.stats["counterfactual_bad_skips"] += 1
            outcome = "MISSED_WIN"
        elif would_have_pnl < -max_risk * 0.1:  # Lost more than 10% of max risk
            # We avoided a losing trade - skipping was GOOD
            skip_reward = 0.10  # Reward for avoiding loss
            self.stats["counterfactual_good_skips"] += 1
            outcome = "AVOIDED_LOSS"
        else:
            # Roughly breakeven - skipping was neutral
            skip_reward = 0.0
            outcome = "NEUTRAL"

        # Update Q[skip] for this state
        q_values = self._get_q_values(state_key)
        old_q = q_values[self.SKIP]
        new_q = old_q + self.alpha * (skip_reward - old_q)
        self.q_table[state_key][self.SKIP] = new_q

        self._save_state()

        result = {
            "ticker": ticker,
            "state_key": state_key,
            "would_have_pnl": round(would_have_pnl, 2),
            "outcome": outcome,
            "skip_reward": skip_reward,
            "old_q_skip": round(old_q, 4),
            "new_q_skip": round(new_q, 4),
            "best_price_seen": cf["best_price"],
            "worst_price_seen": cf["worst_price"],
            "reason": reason,
        }

        logger.info(f"Counterfactual resolved: {ticker} | Would have P&L: ${would_have_pnl:+.2f} | "
                   f"{outcome} | Q[skip]: {old_q:.3f} → {new_q:.3f}")

        return result

    def get_pending_counterfactuals(self) -> Dict[str, Dict]:
        """Get all pending counterfactual trades being monitored."""
        return dict(self.pending_counterfactuals)

    def expire_old_counterfactuals(self, max_age_minutes: int = 10) -> List[Dict]:
        """
        Expire counterfactuals older than max_age_minutes.
        Returns list of resolved counterfactuals.
        """
        now = datetime.now()
        expired = []

        for ticker in list(self.pending_counterfactuals.keys()):
            cf = self.pending_counterfactuals[ticker]
            entry_time = datetime.fromisoformat(cf["entry_time"])
            age = (now - entry_time).total_seconds() / 60

            if age > max_age_minutes:
                result = self.resolve_counterfactual(ticker, reason=f"expired_after_{int(age)}m")
                if result:
                    expired.append(result)

        return expired

    def record_trade_outcome(
        self,
        state_key: str,
        pnl: float,
        max_risk: float,
        action: str = None
    ) -> Dict:
        """
        Update Q-values based on trade outcome.

        Args:
            state_key: The state when the trade was entered
            pnl: Realized P&L (positive for win, negative for loss)
            max_risk: Maximum risk (debit * contracts * 100)
            action: The action taken (defaults to ENTER since we only record actual trades)

        Returns:
            dict with update details
        """
        if action is None:
            action = self.ENTER  # We only record outcomes for trades we entered

        # Normalize reward to [-1, 1] range based on max risk
        # +1 means we made 100% of max risk (rare)
        # -1 means we lost 100% of max risk
        if max_risk > 0:
            reward = max(-1.0, min(1.0, pnl / max_risk))
        else:
            reward = 0.0

        # Get current Q-value
        q_values = self._get_q_values(state_key)
        old_q = q_values[action]

        # Q-learning update: Q(s,a) = Q(s,a) + α * (reward - Q(s,a))
        # Note: gamma=0 means no future state consideration (appropriate for single trades)
        new_q = old_q + self.alpha * (reward - old_q)

        # Update Q-table
        self.q_table[state_key][action] = new_q

        # Update stats
        self.stats["total_trades"] += 1
        self.stats["total_pnl"] += pnl
        if pnl > 0:
            self.stats["enter_wins"] += 1
        else:
            self.stats["enter_losses"] += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Persist state
        self._save_state()

        logger.info(f"Q-update: {state_key} | reward={reward:.3f} | "
                   f"Q[{action}]: {old_q:.3f} -> {new_q:.3f} | epsilon={self.epsilon:.3f}")

        return {
            "state_key": state_key,
            "action": action,
            "pnl": pnl,
            "reward": reward,
            "old_q": old_q,
            "new_q": new_q,
            "epsilon": self.epsilon
        }

    def record_skip_outcome(self, state_key: str, would_have_pnl: float = None):
        """
        Optionally record what would have happened if we skipped.
        This is for counterfactual learning (if you track hypothetical outcomes).

        Args:
            state_key: The state when we skipped
            would_have_pnl: What the P&L would have been if we entered
        """
        if would_have_pnl is None:
            # Small negative reward for skipping (opportunity cost)
            reward = -0.02
        else:
            # If we would have lost, skipping was good
            # If we would have won, skipping was bad
            reward = -0.1 if would_have_pnl > 0 else 0.1

        q_values = self._get_q_values(state_key)
        old_q = q_values[self.SKIP]
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_table[state_key][self.SKIP] = new_q

        self.stats["total_skips"] += 1
        self._save_state()

    def get_stats(self) -> Dict:
        """Get learner statistics."""
        total = self.stats["total_trades"]
        wins = self.stats["enter_wins"]
        good_skips = self.stats.get("counterfactual_good_skips", 0)
        bad_skips = self.stats.get("counterfactual_bad_skips", 0)

        return {
            "total_states": len(self.q_table),
            "total_trades": total,
            "total_skips": self.stats["total_skips"],
            "win_rate": (wins / total * 100) if total > 0 else 0.0,
            "total_pnl": self.stats["total_pnl"],
            "avg_pnl": self.stats["total_pnl"] / total if total > 0 else 0.0,
            "epsilon": self.epsilon,
            "exploration_rate": (
                self.stats["exploration_count"] /
                (self.stats["exploration_count"] + self.stats["exploitation_count"])
                if (self.stats["exploration_count"] + self.stats["exploitation_count"]) > 0
                else self.epsilon
            ),
            "counterfactual_good_skips": good_skips,
            "counterfactual_bad_skips": bad_skips,
            "skip_accuracy": (good_skips / (good_skips + bad_skips) * 100) if (good_skips + bad_skips) > 0 else 0.0,
            "pending_counterfactuals": len(self.pending_counterfactuals),
        }

    def get_recent_decisions(self, n: int = 20) -> List[Dict]:
        """Get the N most recent decisions for monitoring."""
        return self.decision_history[-n:]

    def get_decision_summary(self) -> Dict:
        """Get summary of recent decisions."""
        if not self.decision_history:
            return {"total": 0, "enters": 0, "skips": 0, "explorations": 0, "enter_rate": 0, "exploration_rate": 0}

        recent = self.decision_history[-50:]  # Last 50
        enters = sum(1 for d in recent if d["action"] == self.ENTER)
        skips = sum(1 for d in recent if d["action"] == self.SKIP)
        explorations = sum(1 for d in recent if d.get("exploration", False))

        return {
            "total": len(recent),
            "enters": enters,
            "skips": skips,
            "enter_rate": enters / len(recent) * 100 if recent else 0,
            "explorations": explorations,
            "exploration_rate": explorations / len(recent) * 100 if recent else 0,
        }

    def print_status(self):
        """Print a formatted status report for monitoring."""
        stats = self.get_stats()
        decisions = self.get_decision_summary()

        print("\n" + "=" * 60)
        print("ADAPTIVE LEARNER STATUS")
        print("=" * 60)

        print(f"\n📊 LEARNING PROGRESS")
        print(f"   States learned: {stats['total_states']}")
        print(f"   Total trades: {stats['total_trades']}")
        print(f"   Win rate: {stats['win_rate']:.1f}%")
        print(f"   Total P&L: ${stats['total_pnl']:+.2f}")
        print(f"   Avg P&L: ${stats['avg_pnl']:+.2f}")

        print(f"\n🎲 EXPLORATION vs EXPLOITATION")
        print(f"   Epsilon: {stats['epsilon']:.1%}")
        print(f"   Recent exploration rate: {decisions['exploration_rate']:.1f}%")

        print(f"\n🔮 COUNTERFACTUAL LEARNING")
        print(f"   Good skips (avoided losses): {stats['counterfactual_good_skips']}")
        print(f"   Bad skips (missed wins): {stats['counterfactual_bad_skips']}")
        print(f"   Skip accuracy: {stats['skip_accuracy']:.1f}%")
        print(f"   Pending monitoring: {stats['pending_counterfactuals']}")

        print(f"\n📈 RECENT DECISIONS (last 50)")
        print(f"   Enters: {decisions['enters']} ({decisions['enter_rate']:.0f}%)")
        print(f"   Skips: {decisions['skips']} ({100 - decisions['enter_rate']:.0f}%)")

        print(f"\n🏆 TOP STATES TO TRADE")
        for s in self.get_best_states(3):
            parts = s['state'].split('|')
            print(f"   {parts[0]:4s} {parts[1]:4s} RSI:{parts[2]:10s} | Q={s['q_enter']:+.3f}")

        print(f"\n⚠️  STATES TO AVOID")
        for s in self.get_worst_states(3):
            parts = s['state'].split('|')
            print(f"   {parts[0]:4s} {parts[1]:4s} RSI:{parts[2]:10s} | Q={s['q_enter']:+.3f}")

        print("\n" + "=" * 60)

    def get_best_states(self, n: int = 10) -> list:
        """Get the top N states by Q-value for entering."""
        states = []
        for state_key, q_vals in self.q_table.items():
            states.append({
                "state": state_key,
                "q_enter": q_vals.get(self.ENTER, 0),
                "q_skip": q_vals.get(self.SKIP, 0),
                "advantage": q_vals.get(self.ENTER, 0) - q_vals.get(self.SKIP, 0)
            })

        # Sort by advantage (enter - skip)
        states.sort(key=lambda x: x["advantage"], reverse=True)
        return states[:n]

    def get_worst_states(self, n: int = 10) -> list:
        """Get the bottom N states (where skipping is favored)."""
        states = self.get_best_states(n=1000)  # Get all
        states.reverse()
        return states[:n]

    def _save_state(self):
        """Persist Q-table and stats to disk."""
        try:
            state = {
                "q_table": dict(self.q_table),
                "stats": self.stats,
                "epsilon": self.epsilon,
                "saved_at": datetime.now().isoformat()
            }

            # Write atomically
            temp_file = STATE_FILE.with_suffix(".json.tmp")
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2)
            temp_file.replace(STATE_FILE)

        except Exception as e:
            logger.error(f"Failed to save learner state: {e}")

    def _load_state(self):
        """Load Q-table and stats from disk."""
        if not STATE_FILE.exists():
            logger.info("No existing learner state found, starting fresh")
            return

        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)

            # Restore Q-table
            for key, vals in state.get("q_table", {}).items():
                self.q_table[key] = vals

            # Restore stats
            self.stats.update(state.get("stats", {}))

            # Restore epsilon
            self.epsilon = state.get("epsilon", self.epsilon)

            logger.info(f"Loaded learner state: {len(self.q_table)} states, "
                       f"{self.stats['total_trades']} trades, epsilon={self.epsilon:.3f}")

        except Exception as e:
            logger.error(f"Failed to load learner state: {e}")

    def reset(self):
        """Reset the learner to initial state."""
        self.q_table.clear()
        self.stats = {
            "total_trades": 0,
            "total_skips": 0,
            "enter_wins": 0,
            "enter_losses": 0,
            "total_pnl": 0.0,
            "exploration_count": 0,
            "exploitation_count": 0,
        }
        self.epsilon = 0.20

        if STATE_FILE.exists():
            STATE_FILE.unlink()

        logger.info("Learner reset to initial state")


# Singleton instance
_learner_instance: Optional[AdaptiveLearner] = None


def get_learner() -> AdaptiveLearner:
    """Get or create the singleton AdaptiveLearner instance."""
    global _learner_instance
    if _learner_instance is None:
        _learner_instance = AdaptiveLearner()
    return _learner_instance


def reset_learner():
    """Reset the singleton learner."""
    global _learner_instance
    if _learner_instance:
        _learner_instance.reset()
    _learner_instance = None
