from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from answer_metrics import calculate_contain, calculate_exact_match, calculate_f1


def _coerce_answer(value: Any) -> str:
    return "" if value is None else str(value)


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        predictions: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Invalid prediction line at {path}: expected a JSON object.")
                predictions.append(payload)
        return predictions

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid predictions format at {path}: expected a JSON list.")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Invalid predictions format at {path}: expected JSON objects.")
    return payload


def evaluate_predictions(predictions_path: str | Path) -> dict[str, Any]:
    path = Path(predictions_path)
    predictions = _load_predictions(path)
    count = len(predictions)
    contain_total = 0
    exact_match_total = 0
    f1_total = 0.0

    for prediction in predictions:
        pred_answer = _coerce_answer(prediction.get("pred_answer"))
        gold_answer = _coerce_answer(prediction.get("gold_answer"))
        contain_total += calculate_contain(pred_answer, gold_answer)
        exact_match_total += calculate_exact_match(pred_answer, gold_answer)
        f1_total += calculate_f1(pred_answer, gold_answer)

    summary = {
        "contain_accuracy": contain_total / count if count else 0.0,
        "exact_match": exact_match_total / count if count else 0.0,
        "f1": f1_total / count if count else 0.0,
        "num_samples": count,
    }
    (path.parent / "evaluation_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
