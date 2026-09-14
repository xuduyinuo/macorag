from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _final_turn(rollout: dict[str, Any]) -> dict[str, Any]:
    trajectory = rollout.get("trajectory") or []
    for turn in reversed(trajectory):
        if turn.get("force_final_answer"):
            return turn
    return trajectory[-1] if trajectory else {}


def _is_parse_error(error: str) -> bool:
    return not error.startswith("final_answer_required:")


def compute_protocol_metrics(
    rollouts: list[dict[str, Any]],
    *,
    max_parse_failure_rate: float = 0.01,
    max_missing_answer_tag_rate: float = 0.002,
    min_final_compliance_rate: float = 0.99,
) -> dict[str, Any]:
    total = len(rollouts)
    if total == 0:
        return {
            "count": 0,
            "parse_failure_rate": 0.0,
            "missing_answer_tag_rate": 0.0,
            "final_compliance_rate": 0.0,
            "checkpoint_eligible": False,
        }
    parse_failures = 0
    missing_answer_tags = 0
    final_compliant = 0
    for rollout in rollouts:
        errors = [str(item) for item in rollout.get("parse_errors", [])]
        if any(_is_parse_error(error) for error in errors):
            parse_failures += 1
        final_turn = _final_turn(rollout)
        raw_answer = str((final_turn.get("raw_responses") or {}).get("answer_generator") or "")
        if any("Missing required tag: answer" in error for error in errors) or (
            final_turn and "<answer>" not in raw_answer
        ):
            missing_answer_tags += 1
        answer = final_turn.get("answer") if isinstance(final_turn.get("answer"), dict) else {}
        if not errors and answer.get("can_answer") is True and str(answer.get("answer") or "").strip():
            final_compliant += 1
    parse_rate = parse_failures / total
    missing_rate = missing_answer_tags / total
    final_rate = final_compliant / total
    return {
        "count": total,
        "parse_failure_rate": parse_rate,
        "missing_answer_tag_rate": missing_rate,
        "final_compliance_rate": final_rate,
        "checkpoint_eligible": (
            parse_rate <= max_parse_failure_rate
            and missing_rate <= max_missing_answer_tag_rate
            and final_rate >= min_final_compliance_rate
        ),
    }


@dataclass
class ProtocolWindowMonitor:
    window_size: int = 100
    max_parse_failure_rate: float = 0.02
    bad_windows_to_warn: int = 2
    pending: list[dict[str, Any]] = field(default_factory=list)
    consecutive_bad_windows: int = 0
    completed_windows: int = 0

    def add(self, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
        self.pending.extend(rollouts)
        latest = compute_protocol_metrics(rollouts)
        should_warn = False
        while len(self.pending) >= self.window_size:
            window = self.pending[: self.window_size]
            del self.pending[: self.window_size]
            latest = compute_protocol_metrics(window)
            self.completed_windows += 1
            if latest["parse_failure_rate"] > self.max_parse_failure_rate:
                self.consecutive_bad_windows += 1
                if self.consecutive_bad_windows == self.bad_windows_to_warn:
                    should_warn = True
            else:
                self.consecutive_bad_windows = 0
        return {
            **latest,
            "completed_windows": self.completed_windows,
            "consecutive_bad_windows": self.consecutive_bad_windows,
            "should_warn": should_warn,
        }
