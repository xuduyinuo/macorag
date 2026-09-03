from __future__ import annotations

from collections import Counter
import re
import string
from typing import Any

ANSWER_F1_CONTRACT = "normalized_token_f1_gold_only_v1"


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def calculate_contain(pre_answer: str | None, gold_answer: str | None) -> int:
    if pre_answer is None or str(pre_answer).strip() == "":
        return 0
    if gold_answer is None or str(gold_answer).strip() == "":
        return 0
    normalized_pred = normalize_answer(str(pre_answer))
    normalized_gold = normalize_answer(str(gold_answer))
    return 1 if normalized_gold in normalized_pred or normalized_pred in normalized_gold else 0


def calculate_exact_match(pre_answer: str | None, gold_answer: str | None) -> int:
    if pre_answer is None or str(pre_answer).strip() == "":
        return 0
    if gold_answer is None or str(gold_answer).strip() == "":
        return 0
    return 1 if normalize_answer(str(pre_answer)) == normalize_answer(str(gold_answer)) else 0


def calculate_f1(pre_answer: str | None, gold_answer: str | None) -> float:
    if pre_answer is None or str(pre_answer).strip() == "":
        return 0.0
    if gold_answer is None or str(gold_answer).strip() == "":
        return 0.0
    pred_tokens = normalize_answer(str(pre_answer)).split()
    gold_tokens = normalize_answer(str(gold_answer)).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def calculate_answer_metrics(prediction: Any, gold_answer: Any) -> dict[str, float | int]:
    return {
        "exact_match": calculate_exact_match(prediction, gold_answer),
        "contain": calculate_contain(prediction, gold_answer),
        "f1": calculate_f1(prediction, gold_answer),
    }
