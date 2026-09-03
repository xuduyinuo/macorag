from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * quantile)))
    return ordered[index]


def evaluate_training_stability(
    rows: list[dict[str, Any]],
    *,
    window_size: int = 300,
) -> dict[str, Any]:
    if window_size < 100:
        raise ValueError("Training stability window must be at least 100 steps")
    all_steps = sorted(
        [row for row in rows if isinstance(row.get("step"), int) and "kl" in row],
        key=lambda row: int(row["step"]),
    )
    if len(all_steps) < window_size:
        raise ValueError(
            f"Training stability requires {window_size} metric rows, got {len(all_steps)}"
        )
    steps = all_steps[-window_size:]
    step_numbers = [int(row["step"]) for row in steps]
    invalid_rows = []
    for row in steps:
        protocol = row.get("protocol_metrics")
        required_numbers = (row.get("kl"), row.get("clip_fraction"))
        if (
            not isinstance(row.get("did_optimizer_step"), bool)
            or not isinstance(protocol, dict)
            or not isinstance(protocol.get("count"), int)
            or protocol.get("count", 0) <= 0
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in required_numbers
            )
            or any(
                not isinstance(protocol.get(key), (int, float))
                or not math.isfinite(float(protocol[key]))
                for key in ("parse_failure_rate", "missing_answer_tag_rate")
            )
        ):
            invalid_rows.append(int(row["step"]))
    clips = [
        float(row["clip_fraction"])
        if isinstance(row.get("clip_fraction"), (int, float))
        else float("nan")
        for row in steps
    ]
    kl_values = [
        float(row["kl"])
        if isinstance(row.get("kl"), (int, float))
        else float("nan")
        for row in steps
    ]
    previous_kl = statistics.fmean(kl_values[-100:-50])
    final_kl = statistics.fmean(kl_values[-50:])
    protocol_count = sum(
        int((row.get("protocol_metrics") or {}).get("count", 0))
        if isinstance((row.get("protocol_metrics") or {}).get("count"), int)
        else 0
        for row in steps
    )
    parse_count = sum(
        int((row.get("protocol_metrics") or {}).get("count", 0))
        * float((row.get("protocol_metrics") or {}).get("parse_failure_rate", 0.0))
        if isinstance((row.get("protocol_metrics") or {}).get("count"), int)
        and isinstance((row.get("protocol_metrics") or {}).get("parse_failure_rate"), (int, float))
        else 0.0
        for row in steps
    )
    missing_count = sum(
        int((row.get("protocol_metrics") or {}).get("count", 0))
        * float((row.get("protocol_metrics") or {}).get("missing_answer_tag_rate", 0.0))
        if isinstance((row.get("protocol_metrics") or {}).get("count"), int)
        and isinstance((row.get("protocol_metrics") or {}).get("missing_answer_tag_rate"), (int, float))
        else 0.0
        for row in steps
    )
    metrics = {
        "clip_mean": statistics.fmean(clips),
        "clip_p95": _percentile(clips, 0.95),
        "kl_previous50": previous_kl,
        "kl_last50": final_kl,
        "kl_growth": final_kl - previous_kl,
        "parse_failure_rate": parse_count / protocol_count if protocol_count else 0.0,
        "missing_answer_tag_rate": missing_count / protocol_count if protocol_count else 0.0,
        "protocol_count": protocol_count,
    }
    failures = []
    if invalid_rows:
        failures.append("missing_or_invalid_training_metric")
    if len(set(step_numbers)) != len(step_numbers):
        failures.append("duplicate_training_steps")
    if step_numbers != list(
        range(step_numbers[-1] - window_size + 1, step_numbers[-1] + 1)
    ):
        failures.append("nonconsecutive_training_steps")
    if not all(
        math.isfinite(float(metrics[key]))
        for key in ("clip_mean", "clip_p95", "kl_previous50", "kl_last50", "kl_growth")
    ):
        failures.append("nonfinite_stability_metric")
    if protocol_count <= 0:
        failures.append("missing_protocol_metrics")
    if metrics["clip_mean"] > 0.001:
        failures.append("clip_mean")
    if metrics["clip_p95"] > 0.005:
        failures.append("clip_p95")
    if final_kl > 0.1:
        failures.append("kl_last50")
    if metrics["kl_growth"] > max(0.01, previous_kl * 0.2):
        failures.append("kl_growth")
    if metrics["parse_failure_rate"] >= 0.01:
        failures.append("parse_failure_rate")
    if metrics["missing_answer_tag_rate"] >= 0.002:
        failures.append("missing_answer_tag_rate")
    if any(
        row.get("did_optimizer_step") is True and row.get("gradients_finite") is not True
        for row in steps
    ):
        failures.append("nonfinite_successful_update")
    return {"passed": not failures, "failures": failures, "metrics": metrics, "steps": len(steps)}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def compare_validation_runs(sft_dir: str | Path, rl_dir: str | Path) -> dict[str, Any]:
    sft_root, rl_root = Path(sft_dir), Path(rl_dir)
    sft_metrics = _read_json(sft_root / "aggregate_metrics.json")
    rl_metrics = _read_json(rl_root / "aggregate_metrics.json")
    rl_protocol = _read_json(rl_root / "aggregate_protocol_metrics.json")
    sft_contract = _read_json(sft_root / "evaluation_contract.json")
    rl_contract = _read_json(rl_root / "evaluation_contract.json")
    failures = []
    sft_fingerprint = sft_contract.get("contract_fingerprint")
    rl_fingerprint = rl_contract.get("contract_fingerprint")
    if not isinstance(sft_fingerprint, str) or not sft_fingerprint:
        failures.append("missing_sft_evaluation_contract")
    if not isinstance(rl_fingerprint, str) or not rl_fingerprint:
        failures.append("missing_rl_evaluation_contract")
    if sft_fingerprint != rl_fingerprint:
        failures.append("evaluation_contract_mismatch")
    datasets = ("2wiki", "hotpotqa", "musique")
    sft_datasets = sft_metrics.get("datasets") if isinstance(sft_metrics.get("datasets"), dict) else {}
    rl_datasets = rl_metrics.get("datasets") if isinstance(rl_metrics.get("datasets"), dict) else {}

    def dataset_item(payload: dict[str, Any], dataset: str) -> dict[str, Any]:
        item = payload.get(dataset)
        return item if isinstance(item, dict) else {}
    for label, payload in (("sft", sft_metrics), ("rl", rl_metrics)):
        if payload.get("num_samples") != 300:
            failures.append(f"{label}_sample_count")
        dataset_payload = payload.get("datasets")
        if not isinstance(dataset_payload, dict):
            failures.append(f"{label}_datasets_missing")
            continue
        for dataset in datasets:
            item = dataset_payload.get(dataset)
            if not isinstance(item, dict) or item.get("num_samples") != 100:
                failures.append(f"{label}_{dataset}_sample_count")
    numeric_values: list[tuple[str, Any]] = [
        ("sft_macro_f1", sft_metrics.get("macro_f1")),
        ("rl_macro_f1", rl_metrics.get("macro_f1")),
        ("validation_parse_failure_rate", rl_protocol.get("parse_failure_rate")),
        ("validation_missing_answer_tag_rate", rl_protocol.get("missing_answer_tag_rate")),
        ("validation_error_rate", rl_protocol.get("error_rate")),
    ]
    for dataset in datasets:
        numeric_values.extend(
            [
                (f"sft_{dataset}_f1", dataset_item(sft_datasets, dataset).get("f1")),
                (f"rl_{dataset}_f1", dataset_item(rl_datasets, dataset).get("f1")),
            ]
        )
    invalid_numbers = [
        name
        for name, value in numeric_values
        if not isinstance(value, (int, float)) or not math.isfinite(float(value))
    ]
    if invalid_numbers:
        failures.append("nonfinite_or_missing_validation_metric")
    elif float(rl_metrics["macro_f1"]) <= float(sft_metrics["macro_f1"]):
        failures.append("macro_f1_not_improved")
    for dataset in datasets:
        sft_item = dataset_item(sft_datasets, dataset)
        rl_item = dataset_item(rl_datasets, dataset)
        if (
            isinstance(sft_item.get("f1"), (int, float))
            and isinstance(rl_item.get("f1"), (int, float))
            and math.isfinite(float(sft_item["f1"]))
            and math.isfinite(float(rl_item["f1"]))
            and float(rl_item["f1"]) < float(sft_item["f1"])
        ):
            failures.append(f"{dataset}_f1_regression")
    if rl_protocol.get("count") != 300:
        failures.append("validation_protocol_count")
    if rl_protocol.get("error_count") != 0:
        failures.append("validation_generation_errors")
    if isinstance(rl_protocol.get("parse_failure_rate"), (int, float)) and math.isfinite(
        float(rl_protocol["parse_failure_rate"])
    ) and float(rl_protocol["parse_failure_rate"]) >= 0.01:
        failures.append("validation_parse_failure_rate")
    if isinstance(rl_protocol.get("missing_answer_tag_rate"), (int, float)) and math.isfinite(
        float(rl_protocol["missing_answer_tag_rate"])
    ) and float(rl_protocol["missing_answer_tag_rate"]) >= 0.002:
        failures.append("validation_missing_answer_tag_rate")
    return {"passed": not failures, "failures": failures, "sft": sft_metrics, "rl": rl_metrics}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_jsonls(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in paths:
        rows.extend(_read_jsonl(Path(value)))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-metrics", required=True, nargs="+")
    parser.add_argument("--sft-eval", required=True)
    parser.add_argument("--rl-eval", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        training = evaluate_training_stability(_read_jsonls(args.train_metrics))
        validation = compare_validation_runs(args.sft_eval, args.rl_eval)
        report = {"passed": training["passed"] and validation["passed"], "training": training, "validation": validation}
        status = 0 if report["passed"] else 2
    except (AttributeError, FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {"passed": False, "incomplete": True, "error": str(exc)}
        status = 3
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
