from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from macorag.io_utils import normalize_key


@dataclass
class FilterResult:
    accepted: bool
    reasons: list[str]


def _contains_answer(prediction: str | None, gold: str | None, aliases: list[str]) -> bool:
    if not prediction or not gold:
        return False

    normalized_prediction = normalize_key(prediction)
    candidates = [gold, *aliases]
    return any(
        normalized_candidate in normalized_prediction
        for candidate in candidates
        if (normalized_candidate := normalize_key(candidate))
    )


def _collect_accepted_chunks(steps: list[dict[str, Any]]) -> set[str]:
    accepted_chunks: set[str] = set()
    for step in steps:
        action = step.get("action", {})
        if action.get("type") == "update_evidence":
            accepted_chunks.update(str(chunk_id) for chunk_id in action.get("accepted_chunk_ids", []))
    return accepted_chunks


def _find_final_answer(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in reversed(steps):
        action = step.get("action", {})
        if action.get("type") == "final_answer":
            return action
    return None


def evaluate_trajectory(
    trajectory: dict[str, Any],
    qrels_by_qid: dict[str, dict[str, Any]],
    answers_by_qid: dict[str, dict[str, Any]],
) -> FilterResult:
    qid = trajectory.get("qid")
    steps = list(trajectory.get("steps", []))
    qrels = qrels_by_qid.get(qid, {})
    answer_record = answers_by_qid.get(qid, {})

    reasons: list[str] = []
    if not steps:
        reasons.append("empty_trajectory")

    accepted_chunks = _collect_accepted_chunks(steps)
    if not accepted_chunks:
        reasons.append("no_accepted_evidence")

    gold_chunks = {str(chunk_id) for chunk_id in qrels.get("gold_chunk_ids", [])}
    if gold_chunks and accepted_chunks and accepted_chunks.isdisjoint(gold_chunks):
        reasons.append("no_gold_evidence_overlap")

    final_answer = _find_final_answer(steps)
    if final_answer is None:
        reasons.append("missing_final_answer")
        return FilterResult(accepted=False, reasons=reasons)

    supporting_chunks = {
        str(chunk_id) for chunk_id in final_answer.get("supporting_chunk_ids", [])
    }
    if not supporting_chunks:
        reasons.append("missing_supporting_chunks")
    elif not supporting_chunks.issubset(accepted_chunks):
        reasons.append("supporting_chunks_not_accepted")

    prediction = final_answer.get("answer")
    gold_answer = answer_record.get("answer")
    aliases = list(answer_record.get("aliases", answer_record.get("answer_aliases", [])))
    if not _contains_answer(prediction, gold_answer, aliases):
        reasons.append("answer_mismatch")

    return FilterResult(accepted=not reasons, reasons=reasons)
