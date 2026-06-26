from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class AgentRole(str, Enum):
    QUERY_RETRIEVER = "query_retriever"
    EVIDENCE_UPDATER = "evidence_updater"
    ANSWER_GENERATOR = "answer_generator"


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
    ) -> str:
        ...


class RetrievalEnv(Protocol):
    def query(self, dataset: str, query: str) -> dict[str, Any]:
        ...
