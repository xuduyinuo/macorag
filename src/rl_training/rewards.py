from __future__ import annotations

import math
import re
import string
from typing import Any, Iterable

from answer_metrics import calculate_f1
from rag.parser import parse_can_answer


def normalize_passage(value: Any) -> str:
    """One text fallback shared by gold, retrieval and selected evidence."""

    text = str(value or "").casefold()
    text = "".join(" " if char in string.punctuation else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


_STABLE_ID_FIELDS = ("doc_id", "context_id", "paragraph_id", "chunk_id")


def passage_identity(passage: dict[str, Any]) -> tuple[str, ...] | None:
    """Prefer stable corpus identifiers; use normalized text only as fallback."""

    for field in _STABLE_ID_FIELDS:
        value = normalize_passage(passage.get(field))
        if value:
            return (field, value)
    title = normalize_passage(passage.get("title"))
    text = normalize_passage(passage.get("text"))
    if title and text:
        return ("title_text", title, text)
    if text:
        return ("text", text)
    if title:
        return ("title", title)
    return None


def _unique_passage_keys(passages: Iterable[dict[str, Any]]) -> set[tuple[str, ...]]:
    return {
        identity
        for passage in passages
        if isinstance(passage, dict)
        for identity in [passage_identity(passage)]
        if identity is not None
    }


def _gold_units(sample_or_passages: Any) -> list[dict[str, Any]]:
    passages = (
        sample_or_passages.get("supporting_facts", [])
        if isinstance(sample_or_passages, dict)
        else sample_or_passages
    ) or []
    # Retrieval acts on documents/passages. Several labeled sentences in one
    # document are one reachable reward unit.
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for passage in passages:
        if not isinstance(passage, dict):
            continue
        identity = passage_identity(passage)
        if identity is not None:
            unique.setdefault(identity, passage)
    return list(unique.values())


def _matching_gold_indices(
    passages: Iterable[dict[str, Any]], gold_passages: Iterable[dict[str, Any]]
) -> set[int]:
    gold = list(gold_passages)
    gold_keys = [passage_identity(item) for item in gold]
    matches: set[int] = set()
    for passage in passages:
        if not isinstance(passage, dict):
            continue
        key = passage_identity(passage)
        if key is not None:
            for index, gold_key in enumerate(gold_keys):
                if key == gold_key:
                    matches.add(index)
            if any(key == gold_key for gold_key in gold_keys):
                continue
        # Cross-schema fallback for retrieval indices that omit some gold fields.
        passage_doc = normalize_passage(passage.get("doc_id"))
        passage_title = normalize_passage(passage.get("title"))
        passage_text = normalize_passage(passage.get("text"))
        for index, gold_item in enumerate(gold):
            gold_doc = normalize_passage(gold_item.get("doc_id"))
            gold_title = normalize_passage(gold_item.get("title"))
            gold_text = normalize_passage(gold_item.get("text"))
            if passage_doc and gold_doc and passage_doc == gold_doc:
                matches.add(index)
            elif passage_title and gold_title and passage_title == gold_title:
                matches.add(index)
            elif not (passage_doc and gold_doc) and passage_text and gold_text and (
                passage_text in gold_text or gold_text in passage_text
            ):
                matches.add(index)
    return matches


def compute_query_reward(
    *,
    retrieved_passages: list[dict[str, Any]],
    previously_retrieved_passages: list[dict[str, Any]],
    gold_passages: list[dict[str, Any]],
    eta_query: float,
) -> float:
    """Paper Eq. r^Q: new gold evidence minus repeated retrieval."""

    current_keys = _unique_passage_keys(retrieved_passages)
    previous_keys = _unique_passage_keys(previously_retrieved_passages)
    new_passages = [
        passage
        for passage in retrieved_passages
        if passage_identity(passage) not in previous_keys
    ]
    gold = _gold_units(gold_passages)
    gain = len(_matching_gold_indices(new_passages, gold)) / max(1, len(gold))
    repetition = len(current_keys & previous_keys) / max(1, len(current_keys))
    reward = gain - float(eta_query) * repetition
    if not math.isfinite(reward):
        raise FloatingPointError("non-finite query reward")
    return float(reward)


def compute_evidence_reward(
    *,
    retrieved_passages: list[dict[str, Any]],
    selected_passages: list[dict[str, Any]],
    gold_passages: list[dict[str, Any]],
    eta_evidence: float,
) -> float:
    """Paper Eq. r^E: retain retrievable gold and reject selected noise."""

    gold = _gold_units(gold_passages)
    retrievable_gold = _matching_gold_indices(retrieved_passages, gold)
    selected_gold = _matching_gold_indices(selected_passages, gold)
    retained = len(selected_gold & retrievable_gold) / max(1, len(retrievable_gold))
    selected_keys = _unique_passage_keys(selected_passages)
    noise_keys = {
        passage_identity(passage)
        for passage in selected_passages
        if passage_identity(passage) is not None
        and not _matching_gold_indices([passage], gold)
    }
    noise = len(noise_keys) / max(1, len(selected_keys))
    reward = retained - float(eta_evidence) * noise
    if not math.isfinite(reward):
        raise FloatingPointError("non-finite evidence reward")
    return float(reward)


def compute_answer_reward(
    *, evidence_sufficient: bool, decision: str | bool, forced_termination: bool = False
) -> float | None:
    """Paper Eq. R_dec. Forced max-round termination has no local reward."""

    if forced_termination:
        return None
    if isinstance(decision, bool):
        is_stop = decision
    else:
        normalized = str(decision).strip().casefold()
        if normalized in {"stop", "true"}:
            is_stop = True
        elif normalized in {"continue", "false"}:
            is_stop = False
        else:
            raise ValueError(f"Invalid answer decision: {decision!r}")
    return 1.0 if is_stop == bool(evidence_sufficient) else -1.0


def compute_answer_f1(prediction: Any, gold_answer: Any, answer_aliases: list[str]) -> float:
    """Reuse the exact gold-only evaluation F1 contract."""

    del answer_aliases
    return calculate_f1(prediction, gold_answer)


def compute_global_reward(
    *,
    final_answer: Any,
    gold_answer: Any,
    final_evidence: list[dict[str, Any]],
    gold_passages: list[dict[str, Any]],
    omega_answer: float,
    omega_evidence: float,
    answer_aliases: list[str] | None = None,
) -> dict[str, float]:
    answer_f1 = compute_answer_f1(final_answer, gold_answer, answer_aliases or [])
    gold = _gold_units(gold_passages)
    coverage = len(_matching_gold_indices(final_evidence, gold)) / max(1, len(gold))
    reward = float(omega_answer) * answer_f1 + float(omega_evidence) * coverage
    if not all(math.isfinite(value) for value in (answer_f1, coverage, reward)):
        raise FloatingPointError("non-finite global reward")
    return {
        "global_reward": reward,
        "answer_f1": float(answer_f1),
        "evidence_coverage": float(coverage),
    }


def _truthy(value: Any) -> bool:
    try:
        return parse_can_answer(value)
    except ValueError:
        return False


def _selected_from_turn(turn: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    passages = [
        item
        for item in ((turn.get("observation") or {}).get("passages") or [])
        if isinstance(item, dict)
    ]
    selected_ids = (turn.get("update_evidence") or {}).get("selected_passage_ids") or []
    by_id = {str(item.get("passage_id")): item for item in passages}
    selected: list[dict[str, Any]] = []
    valid = isinstance(selected_ids, list)
    for selected_id in selected_ids if isinstance(selected_ids, list) else []:
        passage = by_id.get(str(selected_id))
        if passage is None:
            valid = False
        else:
            selected.append(passage)
    return selected, valid


def _support_facts_required(sample: dict[str, Any]) -> int:
    """Compatibility helper used by evaluation/pilot tooling."""

    return len(_gold_units(sample))


def _covered_supporting_fact_count(
    trajectory: list[dict[str, Any]], sample: dict[str, Any]
) -> int:
    """Count unique final evidence units with the Section 3.3 matcher."""

    selected = [
        passage
        for turn in trajectory
        for passage in _selected_from_turn(turn)[0]
    ]
    return len(_matching_gold_indices(selected, _gold_units(sample)))


def compute_action_rewards(
    *,
    rollout: dict[str, Any],
    sample: dict[str, Any],
    eta_query: float = 0.2,
    eta_evidence: float = 0.2,
    omega_answer: float = 1.0,
    omega_evidence: float = 1.0,
    answer_local_reward_weight: float = 1.0,
) -> dict[str, Any]:
    """Score every actual variable-length decision with Section 3.3 formulas."""

    if not 0.0 <= answer_local_reward_weight <= 1.0:
        raise ValueError("answer_local_reward_weight must be between 0 and 1")
    trajectory = rollout.get("trajectory") or []
    gold = _gold_units(sample)
    previous_retrieved: list[dict[str, Any]] = []
    accumulated_evidence: list[dict[str, Any]] = []
    action_rewards: list[dict[str, Any]] = []

    for fallback_round, turn in enumerate(trajectory):
        round_index = int(turn.get("round", fallback_round))
        roles = set(
            turn.get("generated_roles")
            or ("query_retriever", "evidence_updater", "answer_generator")
        )
        failed_role = str(turn.get("parse_error_role") or "")
        passages = [
            item
            for item in ((turn.get("observation") or {}).get("passages") or [])
            if isinstance(item, dict)
        ]

        if "query_retriever" in roles:
            valid = failed_role != "query_retriever" and bool(
                str((turn.get("query_retriever") or {}).get("query") or "").strip()
            )
            reward = (
                compute_query_reward(
                    retrieved_passages=passages,
                    previously_retrieved_passages=previous_retrieved,
                    gold_passages=gold,
                    eta_query=eta_query,
                )
                if valid
                else -1.0
            )
            action_rewards.append({
                "role": "query_retriever", "round_index": round_index,
                "local_reward": reward, "is_valid": valid,
                "local_reward_valid": True, "components": {"query_local_reward": reward},
            })
        previous_retrieved.extend(passages)

        selected, indices_valid = _selected_from_turn(turn)
        if "evidence_updater" in roles:
            valid = failed_role != "evidence_updater" and indices_valid
            reward = (
                compute_evidence_reward(
                    retrieved_passages=passages,
                    selected_passages=selected,
                    gold_passages=gold,
                    eta_evidence=eta_evidence,
                )
                if valid
                else -1.0
            )
            action_rewards.append({
                "role": "evidence_updater", "round_index": round_index,
                "local_reward": reward, "is_valid": valid,
                "local_reward_valid": True, "components": {"evidence_local_reward": reward},
            })
        accumulated_evidence.extend(selected)

        if "answer_generator" in roles:
            forced = bool(turn.get("force_final_answer", False))
            answer = turn.get("answer") or {}
            valid = (
                failed_role != "answer_generator"
                and "can_answer" in answer
                and (
                    not _truthy(answer.get("can_answer"))
                    or bool(str(answer.get("answer") or "").strip())
                )
            )
            if not valid:
                answer_reward: float | None = -1.0
                local_valid = True
            else:
                sufficient = bool(gold) and len(
                    _matching_gold_indices(accumulated_evidence, gold)
                ) == len(gold)
                answer_reward = compute_answer_reward(
                    evidence_sufficient=sufficient,
                    decision=_truthy(answer.get("can_answer")),
                    forced_termination=forced,
                )
                if answer_reward is not None:
                    answer_reward *= answer_local_reward_weight
                local_valid = answer_reward is not None
            action_rewards.append({
                "role": "answer_generator", "round_index": round_index,
                "local_reward": answer_reward, "is_valid": valid,
                "local_reward_valid": local_valid, "forced_termination": forced,
                "components": {"answer_decision_reward": answer_reward},
            })

    final_answer = rollout.get("final_answer")
    if final_answer is None and trajectory:
        last_answer = trajectory[-1].get("answer") or {}
        if _truthy(last_answer.get("can_answer")):
            final_answer = last_answer.get("answer")
    global_terms = compute_global_reward(
        final_answer=final_answer,
        gold_answer=sample.get("answer"),
        answer_aliases=sample.get("answer_aliases") or [],
        final_evidence=accumulated_evidence,
        gold_passages=gold,
        omega_answer=omega_answer,
        omega_evidence=omega_evidence,
    )
    return {
        "action_rewards": action_rewards,
        "terminal_reward": global_terms["global_reward"],
        **global_terms,
    }


def compute_rl_rewards(
    *, rollout: dict[str, Any], sample: dict[str, Any], eta_query: float = 0.2,
    eta_evidence: float = 0.2, omega_answer: float = 1.0,
    omega_evidence: float = 1.0,
) -> dict[str, float]:
    """Aggregate logging view; optimization uses per-decision rewards above."""

    scored = compute_action_rewards(
        rollout=rollout, sample=sample, eta_query=eta_query,
        eta_evidence=eta_evidence, omega_answer=omega_answer,
        omega_evidence=omega_evidence,
    )
    by_role: dict[str, list[float]] = {
        "query_retriever": [], "evidence_updater": [], "answer_generator": [],
    }
    for item in scored["action_rewards"]:
        if item["local_reward"] is not None:
            by_role[item["role"]].append(float(item["local_reward"]))
    role_mean = {
        role: sum(values) / len(values) if values else 0.0
        for role, values in by_role.items()
    }
    gold = _gold_units(sample)
    covered_count = float(scored["evidence_coverage"] * len(gold))
    # Compatibility monitoring fields. They are derived from the paper terms
    # and are not included as extra optimization rewards.
    query_values = by_role["query_retriever"]
    repeated_query_penalty = sum(min(0.0, value) for value in query_values)
    return {
        "query_reward": role_mean["query_retriever"],
        "evidence_reward": role_mean["evidence_updater"],
        "answer_reward": role_mean["answer_generator"],
        "answer_f1": float(scored["answer_f1"]),
        "support_coverage": float(scored["evidence_coverage"]),
        "evidence_coverage": float(scored["evidence_coverage"]),
        "total": float(scored["global_reward"]),
        "global_reward": float(scored["global_reward"]),
        "support_facts_required": float(len(gold)),
        "support_facts_covered": covered_count,
        "repeated_query_penalty": repeated_query_penalty,
        "retrieval_cost": 0.0,
        "premature_answer_penalty": 0.0,
        "retrieval_hit_reward": 1.0 if covered_count > 0 else 0.0,
        "format_reward": 0.0,
    }
