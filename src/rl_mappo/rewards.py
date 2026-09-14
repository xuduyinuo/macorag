from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Iterable


def _normalize(text: Any) -> str:
    value = str(text or "").casefold()
    value = "".join(" " if char in string.punctuation else char for char in value)
    return re.sub(r"\s+", " ", value).strip()


def answer_f1(prediction: Any, gold: Any) -> float:
    pred = _normalize(prediction).split()
    target = _normalize(gold).split()
    if not pred or not target:
        return float(pred == target)
    overlap = sum((Counter(pred) & Counter(target)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(target)
    return 2.0 * precision * recall / (precision + recall)


def _identity(passage: dict[str, Any]) -> tuple[str, ...] | None:
    for key in ("doc_id", "context_id", "paragraph_id", "chunk_id"):
        value = _normalize(passage.get(key))
        if value:
            return key, value
    title, text = _normalize(passage.get("title")), _normalize(passage.get("text"))
    if title and text:
        return "title_text", title, text
    return ("text", text) if text else None


def _matches(passages: Iterable[dict[str, Any]], gold: tuple[dict[str, Any], ...]) -> set[int]:
    result: set[int] = set()
    gold_keys = [_identity(item) for item in gold]
    for passage in passages:
        key = _identity(passage)
        for index, (target, target_row) in enumerate(zip(gold_keys, gold)):
            if key is not None and key == target:
                result.add(index)
                continue
            title, target_title = _normalize(passage.get("title")), _normalize(target_row.get("title"))
            text, target_text = _normalize(passage.get("text")), _normalize(target_row.get("text"))
            if title and target_title and title == target_title:
                result.add(index)
            elif text and target_text and (text in target_text or target_text in text):
                result.add(index)
    return result


def query_reward(current: list[dict[str, Any]], previous: list[dict[str, Any]], gold: tuple[dict[str, Any], ...], eta: float) -> float:
    previous_keys = {_identity(x) for x in previous}
    current_keys = {_identity(x) for x in current}
    new = [x for x in current if _identity(x) not in previous_keys]
    gain = len(_matches(new, gold)) / max(1, len(gold))
    repetition = len(current_keys & previous_keys) / max(1, len(current_keys))
    return gain - eta * repetition


def evidence_reward(retrieved: list[dict[str, Any]], selected: list[dict[str, Any]], gold: tuple[dict[str, Any], ...], eta: float) -> float:
    retrievable = _matches(retrieved, gold)
    selected_gold = _matches(selected, gold)
    retained = len(retrievable & selected_gold) / max(1, len(retrievable))
    noise = sum(1 for item in selected if not _matches([item], gold)) / max(1, len(selected))
    return retained - eta * noise


def answer_decision_reward(
    can_answer: bool,
    evidence: list[dict[str, Any]],
    gold: tuple[dict[str, Any], ...],
    *,
    final_round: bool = False,
    non_final_wait_reward: float = 0.2,
    final_answer_bonus: float = 1.0,
) -> float:
    if final_round:
        return final_answer_bonus if can_answer else -1.0
    sufficient = bool(gold) and len(_matches(evidence, gold)) == len(gold)
    if can_answer:
        return 1.0 if sufficient else -1.0
    return non_final_wait_reward if not sufficient else -1.0


def terminal_reward(
    final_answer: str | None,
    evidence: list[dict[str, Any]],
    gold_answer: str,
    gold: tuple[dict[str, Any], ...],
    omega_answer: float,
    omega_evidence: float,
    *,
    gate_evidence_on_valid_answer: bool = False,
) -> tuple[float, float, float]:
    f1 = answer_f1(final_answer, gold_answer)
    coverage = len(_matches(evidence, gold)) / max(1, len(gold))
    evidence_term = 0.0 if gate_evidence_on_valid_answer and not final_answer else omega_evidence * coverage
    return omega_answer * f1 + evidence_term, f1, coverage
