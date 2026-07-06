from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def _normalize_answer(value: Any) -> str:
    text = str(value or "").lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _tokens(value: Any) -> list[str]:
    normalized = _normalize_answer(value)
    return normalized.split() if normalized else []


def _f1(prediction: Any, ground_truth: Any) -> float:
    prediction_tokens = _tokens(prediction)
    ground_truth_tokens = _tokens(ground_truth)
    if not prediction_tokens or not ground_truth_tokens:
        return 1.0 if prediction_tokens == ground_truth_tokens else 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_answer_f1(prediction: Any, gold_answer: Any, answer_aliases: list[str]) -> float:
    candidates = [gold_answer, *(answer_aliases or [])]
    return max((_f1(prediction, candidate) for candidate in candidates), default=0.0)


def _supporting_fact_texts(sample: dict[str, Any]) -> list[str]:
    facts = sample.get("supporting_facts") or []
    texts: list[str] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        for key in ("doc_id", "title", "text"):
            value = str(fact.get(key) or "").strip()
            if value:
                texts.append(value)
    return texts


def _supporting_fact_groups(sample: dict[str, Any]) -> list[list[str]]:
    facts = sample.get("supporting_facts") or []
    groups: list[list[str]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        group = []
        for key in ("doc_id", "title", "text"):
            value = str(fact.get(key) or "").strip()
            if value:
                group.append(value)
        if group:
            groups.append(group)
    return groups


def _supporting_fact_items(sample: dict[str, Any]) -> list[dict[str, str]]:
    facts = sample.get("supporting_facts") or []
    items: list[dict[str, str]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        item = {
            key: str(fact.get(key) or "").strip()
            for key in ("doc_id", "title", "text")
            if str(fact.get(key) or "").strip()
        }
        if item:
            items.append(item)
    return items


def _matches_supporting_fact(passage: dict[str, Any], support_texts: list[str]) -> bool:
    haystack = _normalize_answer(
        " ".join(str(passage.get(key) or "") for key in ("doc_id", "title", "text"))
    )
    if not haystack:
        return False
    for text in support_texts:
        needle = _normalize_answer(text)
        if needle and (needle in haystack or haystack in needle):
            return True
    return False


def _matches_supporting_fact_item(passage: dict[str, Any], support: dict[str, str]) -> bool:
    passage_doc_id = _normalize_answer(passage.get("doc_id"))
    support_doc_id = _normalize_answer(support.get("doc_id"))
    if passage_doc_id and support_doc_id:
        return passage_doc_id == support_doc_id
    passage_title = _normalize_answer(passage.get("title"))
    support_title = _normalize_answer(support.get("title"))
    if passage_title and support_title:
        return passage_title == support_title
    support_text = str(support.get("text") or "").strip()
    return bool(support_text and _matches_supporting_fact(passage, [support_text]))


def _selected_passages(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for turn in trajectory:
        update = turn.get("update_evidence") or {}
        evidence = update.get("evidence") or []
        if evidence:
            selected.extend(item for item in evidence if isinstance(item, dict))
            continue
        passages = (turn.get("observation") or {}).get("passages") or []
        passage_by_id = {str(item.get("passage_id")): item for item in passages if isinstance(item, dict)}
        for selected_id in update.get("selected_passage_ids", []) or []:
            passage = passage_by_id.get(str(selected_id))
            if passage is not None:
                selected.append(passage)
    return selected


def _covered_supporting_fact_indices(trajectory: list[dict[str, Any]], sample: dict[str, Any]) -> set[int]:
    selected = _selected_passages(trajectory)
    covered: set[int] = set()
    for index, support in enumerate(_supporting_fact_items(sample)):
        if any(_matches_supporting_fact_item(passage, support) for passage in selected):
            covered.add(index)
    return covered


def _covered_supporting_fact_count(trajectory: list[dict[str, Any]], sample: dict[str, Any]) -> int:
    return len(_covered_supporting_fact_indices(trajectory, sample))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _query_reward(trajectory: list[dict[str, Any]], sample: dict[str, Any]) -> float:
    reward = 0.0
    seen_queries: set[str] = set()
    for turn in trajectory:
        query_action = turn.get("query_retriever") or {}
        query = str(query_action.get("query") or "").strip()
        sub_goal = str(query_action.get("sub_goal") or "").strip()
        if query and sub_goal:
            reward += 0.25
        if 3 <= len(query.split()) <= 32:
            reward += 0.25
        normalized_query = _normalize_answer(query)
        if normalized_query and normalized_query not in seen_queries:
            reward += 0.15
            seen_queries.add(normalized_query)
        else:
            reward -= 0.5
    return reward


def _query_penalties(trajectory: list[dict[str, Any]]) -> tuple[float, float]:
    seen_queries: set[str] = set()
    repeated_query_penalty = 0.0
    retrieval_cost = 0.0
    for turn in trajectory:
        query = str((turn.get("query_retriever") or {}).get("query") or "").strip()
        normalized_query = _normalize_answer(query)
        if not normalized_query:
            continue
        retrieval_cost -= 0.1
        if normalized_query in seen_queries:
            repeated_query_penalty -= 0.2
        seen_queries.add(normalized_query)
    return repeated_query_penalty, retrieval_cost


def _retrieval_hit_reward(trajectory: list[dict[str, Any]], sample: dict[str, Any]) -> float:
    support_texts = _supporting_fact_texts(sample)
    for turn in trajectory:
        passages = (turn.get("observation") or {}).get("passages") or []
        if any(isinstance(item, dict) and _matches_supporting_fact(item, support_texts) for item in passages):
            return 0.5
    return 0.0


def _evidence_reward(trajectory: list[dict[str, Any]], sample: dict[str, Any]) -> float:
    reward = 0.0
    covered_support_indices: set[int] = set()
    supporting_facts = _supporting_fact_items(sample)
    for turn in trajectory:
        passages = (turn.get("observation") or {}).get("passages") or []
        passage_by_id = {str(item.get("passage_id")): item for item in passages if isinstance(item, dict)}
        selected_ids = (turn.get("update_evidence") or {}).get("selected_passage_ids") or []
        if not selected_ids:
            reward -= 0.25
            continue
        if len(selected_ids) > 5:
            reward -= 0.15 * (len(selected_ids) - 5)
        for selected_id in selected_ids:
            passage = passage_by_id.get(str(selected_id))
            if passage is None:
                reward -= 0.5
                continue
            matched_index = next(
                (
                    index
                    for index, support in enumerate(supporting_facts)
                    if _matches_supporting_fact_item(passage, support)
                ),
                None,
            )
            if matched_index is None:
                reward -= 0.1
            elif matched_index not in covered_support_indices:
                covered_support_indices.add(matched_index)
                reward += 1.0
    return reward


def compute_rl_rewards(*, rollout: dict[str, Any], sample: dict[str, Any]) -> dict[str, float]:
    trajectory = rollout.get("trajectory") or []
    parse_errors = rollout.get("parse_errors") or []
    final_answer = rollout.get("final_answer")
    if final_answer is None and trajectory:
        final_answer = (trajectory[-1].get("answer") or {}).get("answer")
    answer_f1 = compute_answer_f1(final_answer, sample.get("answer"), sample.get("answer_aliases") or [])
    query_reward = _query_reward(trajectory, sample)
    evidence_reward = _evidence_reward(trajectory, sample)
    answer_reward = 2.0 * answer_f1
    format_reward = 0.0 if parse_errors else 0.25
    support_groups = _supporting_fact_groups(sample)
    support_facts_required = 2 if len(support_groups) >= 2 else len(support_groups)
    support_facts_covered = _covered_supporting_fact_count(trajectory, sample)
    support_coverage = (
        min(1.0, support_facts_covered / support_facts_required)
        if support_facts_required > 0
        else 0.0
    )
    retrieval_hit_reward = _retrieval_hit_reward(trajectory, sample)
    repeated_query_penalty, retrieval_cost = _query_penalties(trajectory)
    last_answer = (trajectory[-1].get("answer") or {}) if trajectory else {}
    premature_answer_penalty = 0.0
    if support_facts_required >= 2 and support_facts_covered < support_facts_required and _truthy(last_answer.get("can_answer")):
        premature_answer_penalty = -1.0 if answer_f1 <= 0.0 else -0.25
    total = (
        (0.25 * query_reward)
        + (0.25 * evidence_reward)
        + answer_reward
        + support_coverage
        + retrieval_hit_reward
        + format_reward
        + repeated_query_penalty
        + retrieval_cost
        + premature_answer_penalty
    )
    if parse_errors:
        total -= float(len(parse_errors))
    return {
        "query_reward": float(query_reward),
        "evidence_reward": float(evidence_reward),
        "answer_f1": float(answer_f1),
        "answer_reward": float(answer_reward),
        "format_reward": float(format_reward),
        "support_facts_required": float(support_facts_required),
        "support_facts_covered": float(support_facts_covered),
        "support_coverage": float(support_coverage),
        "retrieval_hit_reward": float(retrieval_hit_reward),
        "repeated_query_penalty": float(repeated_query_penalty),
        "retrieval_cost": float(retrieval_cost),
        "premature_answer_penalty": float(premature_answer_penalty),
        "total": float(total),
    }
