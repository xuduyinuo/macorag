from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from rag import AgentRole


@dataclass
class AgentDecision:
    """One role-conditioned LLM decision and its RL metadata.

    ``decision_token_mask`` is aligned with ``output_token_ids`` only.  Prompt,
    retrieval and padding tokens are never represented by a positive entry.
    """

    agent_type: AgentRole
    prompt: str
    generated_text: str
    input_ids: list[int]
    output_token_ids: list[int]
    old_log_probs: Any
    attention_mask: list[int] = field(default_factory=list)
    decision_token_mask: list[int] = field(default_factory=list)
    round_id: int = 0
    local_reward: float | None = None
    global_reward: float = 0.0
    decision_return: float = 0.0
    advantage: float = 0.0
    is_valid: bool = True
    local_reward_valid: bool = True
    forced_termination: bool = False
    parse_error: str | None = None
    server_logprobs: Any | None = None

    def __post_init__(self) -> None:
        if not self.attention_mask:
            self.attention_mask = [1] * (len(self.input_ids) + len(self.output_token_ids))
        if not self.decision_token_mask:
            self.decision_token_mask = [1] * len(self.output_token_ids)
        if len(self.decision_token_mask) != len(self.output_token_ids):
            raise ValueError("decision_token_mask must align with output_token_ids")
        if any(value not in (0, 1, False, True) for value in self.decision_token_mask):
            raise ValueError("decision_token_mask must be binary")

    # Compatibility aliases used by the existing trainer and rollout code.
    @property
    def role(self) -> AgentRole:
        return self.agent_type

    @property
    def response(self) -> str:
        return self.generated_text

    @property
    def prompt_ids(self) -> list[int]:
        return self.input_ids

    @property
    def completion_ids(self) -> list[int]:
        return self.output_token_ids

    @property
    def old_logprobs(self) -> Any:
        return self.old_log_probs

    @old_logprobs.setter
    def old_logprobs(self, value: Any) -> None:
        self.old_log_probs = value

    @property
    def round_index(self) -> int:
        return self.round_id

    @property
    def terminal_reward(self) -> float:
        return self.global_reward

    @terminal_reward.setter
    def terminal_reward(self, value: float) -> None:
        self.global_reward = float(value)

    @property
    def primary_advantage(self) -> float:
        return self.advantage

    @primary_advantage.setter
    def primary_advantage(self, value: float) -> None:
        self.advantage = float(value)

    @property
    def fallback_advantage(self) -> float:
        return 0.0

    @fallback_advantage.setter
    def fallback_advantage(self, value: float) -> None:
        if abs(float(value)) > 0.0:
            raise ValueError("Paper-faithful credit assignment has no fallback advantage")


@dataclass
class RoundStep:
    round_id: int
    query_decision: AgentDecision | None = None
    retrieved_passages: list[dict[str, Any]] = field(default_factory=list)
    evidence_decision: AgentDecision | None = None
    selected_passages: list[dict[str, Any]] = field(default_factory=list)
    answer_decision: AgentDecision | None = None
    accumulated_evidence: list[dict[str, Any]] = field(default_factory=list)

    def decisions(self) -> Iterable[AgentDecision]:
        for decision in (self.query_decision, self.evidence_decision, self.answer_decision):
            if decision is not None:
                yield decision


@dataclass
class Trajectory:
    question_id: str
    question: str
    gold_answer: str
    gold_passages: list[dict[str, Any]]
    rounds: list[RoundStep] = field(default_factory=list)
    final_answer: str | None = None
    global_reward: float = 0.0
    answer_f1: float = 0.0
    evidence_coverage: float = 0.0
    stopped_by_policy: bool = False
    forced_termination: bool = False

    def decisions(self) -> Iterable[AgentDecision]:
        for step in self.rounds:
            yield from step.decisions()


def decision_output_mask(decision: AgentDecision) -> list[int]:
    """Return the full sequence mask: prompt=0, decision output=1."""

    return [0] * len(decision.input_ids) + list(decision.decision_token_mask)
