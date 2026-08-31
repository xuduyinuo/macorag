from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class AgentRole(str, Enum):
    QUERY_RETRIEVER = "query_retriever"
    EVIDENCE_UPDATER = "evidence_updater"
    ANSWER_GENERATOR = "answer_generator"


@dataclass(frozen=True)
class AnswerPromptContext:
    round_index: int
    max_rounds: int

    def __post_init__(self) -> None:
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if self.round_index < 0 or self.round_index >= self.max_rounds:
            raise ValueError("round_index must satisfy 0 <= round_index < max_rounds")

    @property
    def is_final_round(self) -> bool:
        return self.round_index == self.max_rounds - 1

    @property
    def remaining_rounds(self) -> int:
        return self.max_rounds - self.round_index - 1


@dataclass
class ParsedAction:
    role: AgentRole
    query_retriever: dict[str, Any] | None = None
    update_evidence: dict[str, Any] | None = None
    answer: dict[str, Any] | None = None


@dataclass
class RAGState:
    question: str
    current_sub_goal: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    retrieval_history: list[dict[str, Any]] = field(default_factory=list)
    retrieval_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "current_sub_goal": self.current_sub_goal,
            "evidence": self.evidence,
            "retrieval_history": self.retrieval_history,
            "retrieval_count": self.retrieval_count,
        }


@dataclass
class RAGLoopResult:
    question: str
    dataset: str
    trajectory: list[dict[str, Any]]
    state: RAGState
    final_answer: str | None
    parse_errors: list[str] = field(default_factory=list)


class SharedPolicy(Protocol):
    def generate(
        self,
        *,
        role: AgentRole,
        question: str,
        state: RAGState,
        observation: dict[str, Any] | None = None,
        answer_context: AnswerPromptContext | None = None,
    ) -> str:
        ...


class RetrievalEnv(Protocol):
    def query(self, dataset: str, query: str) -> dict[str, Any]:
        ...
