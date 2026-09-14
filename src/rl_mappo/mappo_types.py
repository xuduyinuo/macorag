from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    QUERY = "query_retriever"
    EVIDENCE = "evidence_updater"
    ANSWER = "answer_generator"


@dataclass(frozen=True)
class RLSample:
    qid: str
    dataset: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    supporting_facts: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGState:
    question: str
    sub_goal: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    retrieval_history: list[dict[str, Any]] = field(default_factory=list)
    round_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "current_sub_goal": self.sub_goal or None,
            "evidence": self.evidence,
            "retrieval_history": self.retrieval_history,
            "retrieval_count": len(self.retrieval_history),
            "round_index": self.round_index,
        }


@dataclass
class MAPPOTransition:
    role: AgentRole
    round_index: int
    prompt: str
    prompt_ids: list[int]
    action_ids: list[int]
    old_token_logprobs: Any
    central_state: dict[str, Any]
    next_central_state: dict[str, Any]
    reference_token_logprobs: Any = None
    reward: float = 0.0
    old_value: float = 0.0
    next_value: float = 0.0
    advantage: float = 0.0
    return_: float = 0.0
    done: bool = False
    valid: bool = True
    parse_error: str | None = None
    response: str = ""


@dataclass
class Episode:
    qid: str
    dataset: str
    transitions: list[MAPPOTransition] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    global_reward: float = 0.0
    answer_f1: float = 0.0
    evidence_coverage: float = 0.0
    parse_errors: list[str] = field(default_factory=list)
