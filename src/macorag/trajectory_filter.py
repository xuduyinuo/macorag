from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from macorag.io_utils import normalize_key


@dataclass
class FilterResult:
    accepted: bool
    reasons: list[str]
    evidence_coverage: float = 0.0
    retrieval_efficiency: float = 0.0
    gold_evidence_count: int = 0
    covered_gold_evidence_count: int = 0
    retrieval_count: int = 0


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
            accepted_chunks.update(
                str(chunk_id) for chunk_id in action.get("accepted_chunk_ids", [])
            )
    return accepted_chunks


def _find_final_answer(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in reversed(steps):
        action = step.get("action", {})
        if action.get("type") == "final_answer":
            return action
    return None


def _is_positive_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _has_chunk_identifier(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        chunk_id = value.get("chunk_id", value.get("id"))
        return chunk_id is not None and str(chunk_id).strip() != ""
    return False


def _chunk_id_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        chunk_id = value.get("chunk_id", value.get("id"))
        if chunk_id is not None and str(chunk_id).strip():
            return str(chunk_id)
    return None


def _chunk_ids_from_values(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {
        chunk_id
        for value in values
        if (chunk_id := _chunk_id_from_value(value)) is not None
    }


def _has_retrieved_chunks(value: Any) -> bool:
    return isinstance(value, list) and any(_has_chunk_identifier(item) for item in value)


def _valid_retrieval_observation(observation: Any) -> bool:
    if not isinstance(observation, dict) or not observation:
        return False

    return _has_retrieved_chunks(observation.get("retrieved_chunks"))


def _evaluate_retrieval_steps(steps: list[dict[str, Any]]) -> tuple[int, list[str]]:
    retrieval_count = 0
    reasons: list[str] = []
    for step in steps:
        action = step.get("action", {})
        action_type = action.get("type")
        if action_type not in {"retrieval", "retrieve"}:
            continue

        valid_action = (
            bool(str(action.get("query", "")).strip())
            and "top_k" in action
            and _is_positive_int(action.get("top_k"))
        )
        if not valid_action and "invalid_retrieval_action" not in reasons:
            reasons.append("invalid_retrieval_action")

        observation = step.get("observation")
        valid_observation = _valid_retrieval_observation(observation)
        if not valid_observation and "invalid_retrieval_observation" not in reasons:
            reasons.append("invalid_retrieval_observation")

        if valid_action and valid_observation:
            retrieval_count += 1
    return retrieval_count, reasons


def _collect_retrieved_chunks(steps: list[dict[str, Any]]) -> set[str]:
    retrieved_chunks: set[str] = set()
    for step in steps:
        action_type = step.get("action", {}).get("type")
        if action_type not in {"retrieval", "retrieve"}:
            continue

        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue

        retrieved_chunks.update(_chunk_ids_from_values(observation.get("retrieved_chunks")))
    return retrieved_chunks


def _get_retrieval_budget(
    trajectory: dict[str, Any],
    qrels: dict[str, Any],
) -> int | None:
    candidates = (
        trajectory.get("retrieval_budget"),
        trajectory.get("metadata", {}).get("retrieval_budget"),
        qrels.get("retrieval_budget"),
        qrels.get("metadata", {}).get("retrieval_budget"),
    )
    for candidate in candidates:
        if candidate is not None:
            return int(candidate)
    return None


def _normalized_values(values: list[Any]) -> set[str]:
    return {normalize_key(str(value)) for value in values if str(value).strip()}


def _gold_chunk_ids_from_qrels_and_meta(
    qrels: dict[str, Any],
    chunk_meta_by_chunk_id: dict[str, dict[str, Any]] | None,
) -> set[str]:
    gold_chunk_ids = {str(chunk_id) for chunk_id in qrels.get("gold_chunk_ids", [])}
    if chunk_meta_by_chunk_id is None:
        return gold_chunk_ids

    gold_doc_ids = {str(doc_id) for doc_id in qrels.get("gold_doc_ids", [])}
    gold_titles = _normalized_values(qrels.get("gold_titles", []))
    gold_sentences = _normalized_values(qrels.get("gold_sentences", []))
    gold_sentence_indices = {
        int(index)
        for index in qrels.get("gold_sentence_indices", qrels.get("gold_sent_ids", []))
    }

    for chunk_id, meta in chunk_meta_by_chunk_id.items():
        if str(meta.get("doc_id", "")) in gold_doc_ids:
            gold_chunk_ids.add(str(chunk_id))
        if normalize_key(str(meta.get("title", ""))) in gold_titles:
            gold_chunk_ids.add(str(chunk_id))

        sentence_value = meta.get("sentence", meta.get("text"))
        if sentence_value is not None and normalize_key(str(sentence_value)) in gold_sentences:
            gold_chunk_ids.add(str(chunk_id))

        sentence_index = meta.get("sentence_index", meta.get("sent_id"))
        if sentence_index is not None and int(sentence_index) in gold_sentence_indices:
            gold_chunk_ids.add(str(chunk_id))

    return gold_chunk_ids


def _find_missing_chunk_meta(
    chunk_ids: set[str],
    chunk_meta_by_chunk_id: dict[str, dict[str, Any]] | None,
) -> set[str]:
    if chunk_meta_by_chunk_id is None:
        return set(chunk_ids)
    return {chunk_id for chunk_id in chunk_ids if chunk_id not in chunk_meta_by_chunk_id}


def evaluate_trajectory(
    trajectory: dict[str, Any],
    qrels_by_qid: dict[str, dict[str, Any]],
    answers_by_qid: dict[str, dict[str, Any]],
    chunk_meta_by_chunk_id: dict[str, dict[str, Any]] | None = None,
) -> FilterResult:
    qid = trajectory.get("qid")
    steps = list(trajectory.get("trajectory", []))
    qrels = qrels_by_qid.get(qid, {})
    answer_record = answers_by_qid.get(qid, {})

    reasons: list[str] = []
    if not steps:
        reasons.append("empty_trajectory")

    retrieval_count, retrieval_reasons = _evaluate_retrieval_steps(steps)
    reasons.extend(retrieval_reasons)
    if retrieval_count == 0:
        reasons.append("missing_retrieval")

    retrieval_budget = _get_retrieval_budget(trajectory, qrels)
    if retrieval_budget is not None and retrieval_count > retrieval_budget:
        reasons.append("retrieval_budget_exceeded")

    accepted_chunks = _collect_accepted_chunks(steps)
    retrieved_chunks = _collect_retrieved_chunks(steps)
    if not accepted_chunks:
        reasons.append("no_accepted_evidence")

    gold_chunks = _gold_chunk_ids_from_qrels_and_meta(qrels, chunk_meta_by_chunk_id)
    gold_evidence_count = len(gold_chunks)
    covered_gold_chunks = accepted_chunks & gold_chunks
    covered_gold_evidence_count = len(covered_gold_chunks)
    if gold_chunks and accepted_chunks and not covered_gold_chunks:
        reasons.append("no_gold_evidence_overlap")

    final_answer = _find_final_answer(steps)
    supporting_chunks: set[str] = set()
    if final_answer is None:
        reasons.append("missing_final_answer")
    else:
        supporting_chunks = {
            str(chunk_id) for chunk_id in final_answer.get("supporting_chunk_ids", [])
        }
        if not supporting_chunks:
            reasons.append("missing_supporting_chunks")
        elif not supporting_chunks.issubset(accepted_chunks):
            reasons.append("supporting_chunks_not_accepted")

        if supporting_chunks and not supporting_chunks.issubset(gold_chunks):
            reasons.append("supporting_chunks_not_gold")

        prediction = final_answer.get("answer")
        gold_answer = answer_record.get("answer")
        aliases = list(answer_record.get("aliases", answer_record.get("answer_aliases", [])))
        if not _contains_answer(prediction, gold_answer, aliases):
            reasons.append("answer_mismatch")

    mapped_chunk_ids = accepted_chunks | supporting_chunks | retrieved_chunks
    if _find_missing_chunk_meta(mapped_chunk_ids, chunk_meta_by_chunk_id):
        reasons.append("missing_chunk_meta")

    evidence_coverage = (
        covered_gold_evidence_count / gold_evidence_count
        if gold_evidence_count
        else 0.0
    )
    retrieval_efficiency = (
        covered_gold_evidence_count / retrieval_count if retrieval_count else 0.0
    )

    return FilterResult(
        accepted=not reasons,
        reasons=reasons,
        evidence_coverage=evidence_coverage,
        retrieval_efficiency=retrieval_efficiency,
        gold_evidence_count=gold_evidence_count,
        covered_gold_evidence_count=covered_gold_evidence_count,
        retrieval_count=retrieval_count,
    )
