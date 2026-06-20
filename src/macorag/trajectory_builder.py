from __future__ import annotations

import json
from typing import Any

from macorag.retrieval_env import InMemoryRetrievalEnv
from macorag.teacher_api import TeacherClient
from macorag.teacher_protocol import parse_teacher_message


VISIBLE_STATE_KEYS = (
    "qid",
    "dataset",
    "question",
    "evidence",
    "retrieval_history",
    "retrieval_count",
    "retrieval_budget",
)


class TrajectoryBuildError(ValueError):
    pass


def _prompt_from_state(state: dict[str, Any]) -> str:
    visible_state = {
        key: state[key] for key in VISIBLE_STATE_KEYS if key in state
    }
    return (
        "You are a multi-agent RAG teacher. "
        "Use only the visible state and retrieval observations. "
        "Do not use gold answers or hidden evidence.\n"
        f"Question: {state['question']}\n"
        f"Retrieval budget: {state['retrieval_budget']}\n"
        "Visible state JSON:\n"
        f"{json.dumps(visible_state, ensure_ascii=False, sort_keys=True)}\n"
        "Return exactly one JSON object inside each required tag: "
        "<plan>, <retrieval>, <update-evidence>, and <answer>."
    )


def _raw_tag(tag: str, payload: dict[str, Any]) -> str:
    return f"<{tag}>{json.dumps(payload, ensure_ascii=False, sort_keys=True)}</{tag}>"


def _retrieval_top_k(retrieval_action: dict[str, Any]) -> int:
    if "top_k" not in retrieval_action:
        return 5

    top_k = retrieval_action["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TrajectoryBuildError("retrieval top_k must be a positive integer")
    if top_k <= 0:
        raise TrajectoryBuildError("retrieval top_k must be greater than 0")
    return top_k


def _copy_list_field(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key, [])
    if isinstance(value, list):
        return list(value)
    return value


def build_one_trajectory(
    qid: str,
    env: InMemoryRetrievalEnv,
    teacher: TeacherClient,
) -> dict[str, Any]:
    state = env.reset(qid)
    raw_text = teacher.generate(_prompt_from_state(state))
    parsed = parse_teacher_message(raw_text)

    retrieval_action = parsed["retrieval"]
    query = str(retrieval_action["query"])
    top_k = _retrieval_top_k(retrieval_action)
    observation_list = env.step(query, top_k=top_k)

    update_action = parsed["update-evidence"]
    answer_action = parsed["answer"]
    accepted_chunk_ids = _copy_list_field(update_action, "accepted_chunk_ids")
    rejected_chunk_ids = _copy_list_field(update_action, "rejected_chunk_ids")
    added_evidence = (
        list(accepted_chunk_ids)
        if isinstance(accepted_chunk_ids, list)
        else accepted_chunk_ids
    )
    supporting_chunk_ids = _copy_list_field(answer_action, "supporting_chunk_ids")

    return {
        "qid": qid,
        "dataset": state.get("dataset"),
        "trajectory": [
            {
                "t": 0,
                "state": state,
                "agent": "planner",
                "raw_text": _raw_tag("plan", parsed["plan"]),
                "action": {
                    "type": "plan_query",
                    "sub_query": parsed["plan"].get("sub_query"),
                    "rationale": parsed["plan"].get("rationale"),
                },
            },
            {
                "t": 1,
                "agent": "retriever",
                "raw_text": _raw_tag("retrieval", retrieval_action),
                "action": {
                    "type": "retrieve",
                    "query": query,
                    "top_k": top_k,
                },
                "observation": {
                    "retrieved_chunks": observation_list,
                },
            },
            {
                "t": 2,
                "agent": "evidence_updater",
                "raw_text": _raw_tag("update-evidence", update_action),
                "action": {
                    "type": "update_evidence",
                    "accepted_chunk_ids": accepted_chunk_ids,
                    "rejected_chunk_ids": rejected_chunk_ids,
                    "reason": update_action.get("reason"),
                },
                "state_delta": {
                    "added_evidence": added_evidence,
                },
            },
            {
                "t": 3,
                "agent": "answer_generator",
                "raw_text": _raw_tag("answer", answer_action),
                "action": {
                    "type": "final_answer",
                    "answer": answer_action.get("answer"),
                    "supporting_chunk_ids": supporting_chunk_ids,
                },
            },
        ],
    }
