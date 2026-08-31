from __future__ import annotations

import json
from typing import Any

from prompt_config import load_prompt_contract

from .schema import AnswerPromptContext, RAGState


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
    instructions = load_prompt_contract().instructions["answer"]
    decision_rule = (
        str(instructions["final"] if is_final_round else instructions["normal"]).strip()
    )
    output_example = str(
        instructions["final_output_example"] if is_final_round else instructions["normal_output_example"]
    ).strip()
    round_line = ""
    if context is not None:
        round_line = (
            f"Round: {context.round_index + 1}/{context.max_rounds}; "
            f"remaining retrieval rounds after this answer: {context.remaining_rounds}.\n"
        )
    return (
        "Task: answer from accumulated evidence.\n"
        "Use selected evidence in <state>.\n"
        f"{decision_rule}\n"
        f"{round_line}"
        f"Question: {question}\n"
        f"<state>{_json_block(state.to_dict())}</state>\n"
        f"Return exactly: {output_example}"
    )
