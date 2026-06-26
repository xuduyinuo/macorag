from __future__ import annotations

import json
from typing import Any

from .schema import RAGState


def _json_block(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_query_retriever_prompt(*, question: str, state: RAGState) -> str:
    return (
        "Task: plan the next knowledge-base query.\n"
        "Use only the question and verified facts in <state>. Avoid repeated queries and unsupported intermediate facts.\n"
        f"Question: {question}\n"
        f"<state>{_json_block(state.to_dict())}</state>\n"
        'Return exactly: <query-retriever>{"sub_goal":"...","query":"..."}</query-retriever>'
    )


def build_evidence_updater_prompt(
    *,
    question: str,
    state: RAGState,
    observation: dict[str, Any],
) -> str:
    return (
        "Task: select evidence from the latest observation.\n"
        "Pick only passage IDs from <observation> that support the question, current sub-goal, or a needed reasoning step.\n"
        f"Question: {question}\n"
        f"<state>{_json_block(state.to_dict())}</state>\n"
        f"<observation>{_json_block(observation)}</observation>\n"
        'Return exactly: <update-evidence>{"selected_passage_ids":[],"rationale":"..."}</update-evidence>'
    )


def build_answer_generator_prompt(*, question: str, state: RAGState) -> str:
    return (
        "Task: answer from accumulated evidence.\n"
        "Use selected evidence in <state>. If evidence is insufficient and retrieval budget remains, return can_answer=false.\n"
        'If budget is exhausted, a fallback guess is allowed only with rationale marked "fallback_guess".\n'
        f"Question: {question}\n"
        f"<state>{_json_block(state.to_dict())}</state>\n"
        'Return exactly: <answer>{"can_answer":...,"answer":...,"rationale":"..."}</answer>'
    )
