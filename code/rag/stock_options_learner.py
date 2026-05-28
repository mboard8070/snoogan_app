from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict


STATE_FILE = Path(__file__).resolve().parent / "stock_options_learner_state.json"


@dataclass
class Decision:
    should_enter: bool
    action: str
    reason: str
    state_key: str
    q_values: dict


class StockOptionsLearner:
    ENTER = "enter"
    SKIP = "skip"

    def __init__(self, alpha: float = 0.15, epsilon: float = 0.10, epsilon_min: float = 0.03, epsilon_decay: float = 0.995):
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.stats = {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "skips": 0}
        self._load()

    def state_key(self, *, ticker: str, spread_type: str, trend: str, rsi: float, delta: float | None, credit: float, width: float, spread_pct: float) -> str:
        return "|".join([
            ticker,
            spread_type,
            trend,
            self._rsi_bucket(rsi),
            self._delta_bucket(delta),
            self._credit_bucket(credit, width),
            self._liquidity_bucket(spread_pct),
        ])

    def decide(self, state_key: str) -> Decision:
        q = self._q(state_key)
        if q[self.SKIP] > q[self.ENTER] + 0.25:
            return Decision(False, self.SKIP, f"Q[skip]={q[self.SKIP]:.3f} > Q[enter]={q[self.ENTER]:.3f}", state_key, dict(q))
        return Decision(True, self.ENTER, f"Q[enter]={q[self.ENTER]:.3f}, Q[skip]={q[self.SKIP]:.3f}", state_key, dict(q))

    def record_trade_outcome(self, state_key: str, pnl: float, max_risk: float) -> dict:
        reward = max(-1.0, min(1.0, pnl / max_risk)) if max_risk > 0 else 0.0
        q = self._q(state_key)
        old = q[self.ENTER]
        new = old + self.alpha * (reward - old)
        q[self.ENTER] = new
        self.stats["total_trades"] += 1
        self.stats["total_pnl"] += pnl
        if pnl > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._save()
        return {"reward": reward, "old_q": old, "new_q": new}

    def _q(self, state_key: str) -> Dict[str, float]:
        if state_key not in self.q_table:
            self.q_table[state_key] = {self.ENTER: 0.05, self.SKIP: 0.0}
        return self.q_table[state_key]

    def _rsi_bucket(self, rsi: float) -> str:
        if rsi < 35: return "oversold"
        if rsi < 50: return "low"
        if rsi < 65: return "mid"
        return "high"

    def _delta_bucket(self, delta: float | None) -> str:
        if delta is None: return "unknown_delta"
        d = abs(delta)
        if d < 0.20: return "low_delta"
        if d < 0.40: return "target_delta"
        return "high_delta"

    def _credit_bucket(self, credit: float, width: float) -> str:
        ratio = credit / width if width else 0
        if ratio < 0.20: return "low_credit"
        if ratio < 0.35: return "fair_credit"
        return "high_credit"

    def _liquidity_bucket(self, spread_pct: float) -> str:
        if spread_pct <= 0.10: return "tight"
        if spread_pct <= 0.20: return "ok"
        return "wide"

    def _load(self) -> None:
        if not STATE_FILE.exists():
            return
        data = json.loads(STATE_FILE.read_text())
        self.q_table = data.get("q_table", {})
        self.stats.update(data.get("stats", {}))
        self.epsilon = data.get("epsilon", self.epsilon)

    def _save(self) -> None:
        STATE_FILE.write_text(json.dumps({
            "q_table": self.q_table,
            "stats": self.stats,
            "epsilon": self.epsilon,
            "saved_at": datetime.now().isoformat(),
        }, indent=2, sort_keys=True))
