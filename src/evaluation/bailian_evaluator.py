from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import json
import os
import re
import socket
import string
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


class BailianJudgeClient:
    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key_env: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        retries: int,
        retry_sleep_seconds: float,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key. Set environment variable {api_key_env} before evaluation.")
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def infer(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: BaseException | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"])
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                socket.timeout,
                TimeoutError,
                KeyError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_sleep_seconds * attempt)
        raise RuntimeError(f"Bailian judge request failed after {self.retries} attempts: {last_error}")


def calculate_llm_accuracy(client: Any, pre_answer: str, gold_answer: str) -> float:
    system_prompt = "You are an expert evaluator."
    user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {pre_answer}
Gold answer: {gold_answer}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:
"""
    response = client.infer(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return 1.0 if response.strip().lower() == "correct" else 0.0


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


def _coerce_answer(value: Any) -> str:
    return "" if value is None else str(value)


def _evaluate_one(index: int, prediction: dict[str, Any], client: Any) -> tuple[int, float, int, int, float, str | None]:
    try:
        pre_answer = _coerce_answer(prediction.get("pred_answer"))
        gold_answer = _coerce_answer(prediction.get("gold_answer"))
        return (
            index,
            calculate_llm_accuracy(client, pre_answer, gold_answer),
            calculate_contain(pre_answer, gold_answer),
            calculate_exact_match(pre_answer, gold_answer),
            calculate_f1(pre_answer, gold_answer),
            None,
        )
    except Exception as exc:
        pre_answer = _coerce_answer(prediction.get("pred_answer"))
        gold_answer = _coerce_answer(prediction.get("gold_answer"))
        return (
            index,
            0.0,
            calculate_contain(pre_answer, gold_answer),
            calculate_exact_match(pre_answer, gold_answer),
            calculate_f1(pre_answer, gold_answer),
            str(exc),
        )


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        predictions = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Invalid prediction line at {path}: expected a JSON object.")
                predictions.append(payload)
        return predictions
    predictions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError(f"Invalid predictions format at {path}: expected a JSON list.")
    return predictions


def _summarize_scores(
    llm_scores: list[float],
    contain_scores: list[int],
    exact_match_scores: list[int],
    f1_scores: list[float],
) -> dict[str, Any]:
    if not llm_scores:
        return {"llm_accuracy": 0.0, "contain_accuracy": 0.0, "exact_match": 0.0, "f1": 0.0, "num_samples": 0}
    return {
        "llm_accuracy": sum(llm_scores) / len(llm_scores),
        "contain_accuracy": sum(contain_scores) / len(contain_scores),
        "exact_match": sum(exact_match_scores) / len(exact_match_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "num_samples": len(llm_scores),
    }


def evaluate_predictions(
    predictions_path: str | Path,
    *,
    client: Any,
    max_workers: int,
    judge_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(predictions_path)
    predictions = _load_predictions(path)
    if not predictions:
        summary = {"llm_accuracy": 0.0, "contain_accuracy": 0.0, "exact_match": 0.0, "f1": 0.0, "num_samples": 0}
        if judge_metadata is not None:
            summary["judge_metadata"] = judge_metadata
        (path.parent / "evaluation_results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    llm_scores = [0.0] * len(predictions)
    contain_scores = [0] * len(predictions)
    exact_match_scores = [0] * len(predictions)
    f1_scores = [0.0] * len(predictions)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [
            executor.submit(_evaluate_one, index, prediction, client)
            for index, prediction in enumerate(predictions)
        ]
        for future in as_completed(futures):
            index, llm_acc, contain_acc, exact_match, f1, _error = future.result()
            llm_scores[index] = llm_acc
            contain_scores[index] = contain_acc
            exact_match_scores[index] = exact_match
            f1_scores[index] = f1

    summary = _summarize_scores(llm_scores, contain_scores, exact_match_scores, f1_scores)
    if judge_metadata is not None:
        summary["judge_metadata"] = judge_metadata
    (path.parent / "evaluation_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
