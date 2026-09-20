from __future__ import annotations

from typing import Any

from .schema import AnswerPromptContext, RAGState
from rl_mappo.mappo_types import AgentRole as MAPPOAgentRole, RAGState as MAPPORAGState
from rl_mappo.protocol import build_prompt


def _shared_state(state: RAGState, *, round_index: int | None = None) -> MAPPORAGState:
    return MAPPORAGState(
        question=state.question,
        sub_goal=str(state.current_sub_goal or ""),
        evidence=[dict(item) for item in state.evidence],
        retrieval_history=[dict(item) for item in state.retrieval_history],
        round_index=(
            max(0, int(state.retrieval_count))
            if round_index is None else int(round_index)
        ),
    )


def build_query_retriever_prompt(*, question: str, state: RAGState) -> str:
    return build_prompt(
        MAPPOAgentRole.QUERY, question=question, state=_shared_state(state),
        final_round=False,
    )


def build_evidence_updater_prompt(
    *,
    question: str,
    state: RAGState,
    observation: dict[str, Any],
) -> str:
    return build_prompt(
        MAPPOAgentRole.EVIDENCE, question=question, state=_shared_state(state),
        observation=observation, final_round=False,
    )


def build_answer_generator_prompt(
    *,
    question: str,
    state: RAGState,
    context: AnswerPromptContext | None = None,
    force_final_answer: bool | None = None,
) -> str:
    if context is not None and force_final_answer is not None:
        raise ValueError("Pass AnswerPromptContext instead of force_final_answer, not both")
    is_final_round = context.is_final_round if context is not None else bool(force_final_answer)
    round_index = context.round_index if context is not None else state.retrieval_count
    return build_prompt(
        MAPPOAgentRole.ANSWER, question=question,
        state=_shared_state(state, round_index=round_index),
        final_round=is_final_round,
    )
