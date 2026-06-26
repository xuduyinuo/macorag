from __future__ import annotations

import re
from typing import Any


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _answer_matches(final_answer: str | None, gold_answer: str | None, answer_aliases: list[str]) -> bool:
    predicted = _normalize_text(final_answer)
    if not predicted:
        return False
    candidates = [_normalize_text(gold_answer), *[_normalize_text(item) for item in answer_aliases]]
    return predicted in {item for item in candidates if item}


def _answer_supported(final_answer: str | None, trajectory: list[dict[str, Any]]) -> bool:
    answer_text = _normalize_text(final_answer)
    if not answer_text:
        return False
    evidence_text = _normalize_text(
        " ".join(
            str(item.get("text") or "")
            for turn in trajectory
            for item in (turn.get("update_evidence", {}).get("evidence", []) or [])
            if isinstance(item, dict)
        )
    )
    if answer_text in {"yes", "no"}:
        return bool(evidence_text)
    return answer_text in evidence_text


def compute_reward_terms(
    *,
    trajectory: list[dict[str, Any]],
    final_answer: str | None,
    gold_answer: str | None,
    answer_aliases: list[str],
    parse_errors: list[str],
) -> dict[str, float]:
    answer_correct = 1.0 if _answer_matches(final_answer, gold_answer, answer_aliases) else 0.0
    evidence_supported = 1.0 if _answer_supported(final_answer, trajectory) else 0.0
    format_valid = 0.0 if parse_errors else 1.0
    retrieval_cost = -float(len(trajectory))
    total = (2.0 * answer_correct) + evidence_supported + format_valid + (0.05 * retrieval_cost)
    if parse_errors:
        total -= 1.0
    return {
        "answer_correct": answer_correct,
        "evidence_supported": evidence_supported,
        "format_valid": format_valid,
        "retrieval_cost": retrieval_cost,
        "total": total,
    }
