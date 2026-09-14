from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


AGENT_TYPES = ("query", "evidence", "answer")


@dataclass(frozen=True)
class TeacherValidationIssue:
    """One rejected teacher decision; invalid decisions never enter SFT loss."""

    trajectory_id: str
    round_id: int
    agent_type: str
    reason: str


@dataclass
class TeacherValidationStats:
    invalid_query_samples: int = 0
    invalid_evidence_samples: int = 0
    invalid_answer_samples: int = 0
    issues: list[TeacherValidationIssue] = field(default_factory=list)

    @property
    def invalid_sample_count(self) -> int:
        return (
            self.invalid_query_samples
            + self.invalid_evidence_samples
            + self.invalid_answer_samples
        )

    def reject(self, trajectory_id: str, round_id: int, agent_type: str, reason: str) -> None:
        if agent_type not in AGENT_TYPES:
            raise ValueError(f"Unknown agent_type: {agent_type}")
        counter = f"invalid_{agent_type}_samples"
        setattr(self, counter, int(getattr(self, counter)) + 1)
        self.issues.append(
            TeacherValidationIssue(
                trajectory_id=trajectory_id,
                round_id=round_id,
                agent_type=agent_type,
                reason=reason,
            )
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "invalid_query_samples": self.invalid_query_samples,
            "invalid_evidence_samples": self.invalid_evidence_samples,
            "invalid_answer_samples": self.invalid_answer_samples,
            "invalid_sample_count": self.invalid_sample_count,
        }


def validate_query_decision(query: Any) -> str | None:
    if not isinstance(query, dict):
        return "query decision must be an object"
    if not str(query.get("query") or "").strip():
        return "query must be non-empty"
    return None


def validate_evidence_decision(update: Any, observation: Any) -> str | None:
    if not isinstance(update, dict):
        return "evidence decision must be an object"
    if "selected_passage_ids" not in update:
        return "selected_passage_ids is required"
    indices = update.get("selected_passage_ids")
    if not isinstance(indices, list):
        return "selected_passage_ids must be a list"
    passages = observation.get("passages") if isinstance(observation, dict) else None
    if not isinstance(passages, list):
        return "observation.passages must be a list"
    if len(set(indices)) != len(indices):
        return "selected_passage_ids must not contain duplicates"
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(passages):
            return f"selected passage index out of range: {index!r}"
    return None


def validate_answer_decision(answer: Any) -> str | None:
    if not isinstance(answer, dict):
        return "answer decision must be an object"
    can_answer = answer.get("can_answer")
    if not isinstance(can_answer, bool):
        return "can_answer must be a boolean"
    value = answer.get("answer")
    if can_answer and not str(value or "").strip():
        return "Stop decision requires a non-empty answer"
    if not can_answer and value not in (None, ""):
        return "Continue decision must not contain a final answer"
    return None


def validate_round_order(trajectory: Any) -> list[str]:
    """Validate variable-length execution without assuming all three decisions exist."""

    if not isinstance(trajectory, list):
        return ["trajectory must be a list"]
    errors: list[str] = []
    stopped = False
    for offset, turn in enumerate(trajectory):
        if not isinstance(turn, dict):
            errors.append(f"round {offset}: turn must be an object")
            continue
        round_id = turn.get("round", offset)
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id != offset:
            errors.append(f"round {offset}: expected contiguous round id {offset}, got {round_id!r}")
        if stopped:
            errors.append(f"round {offset}: decision exists after the first Stop")
        answer = turn.get("answer")
        if isinstance(answer, dict) and answer.get("can_answer") is True:
            stopped = True
    return errors
