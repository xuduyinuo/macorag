#!/usr/bin/env python3
"""Summarize GRPO batch-size benchmark artifacts and select a candidate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _latest_run_dir(runs_root: Path) -> Path | None:
    candidates = sorted(
        path for path in runs_root.glob("*") if (path / "train_metrics.jsonl").is_file()
    )
    return candidates[-1] if candidates else None


def _finite_training_metrics(records: list[dict[str, Any]]) -> bool:
    for record in records:
        for key in ("loss", "kl", "reward_total"):
            value = record.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return False
    return True


def _candidate_summary(candidate_dir: Path) -> dict[str, Any]:
    status_path = candidate_dir / "status.json"
    status = _read_json(status_path) if status_path.is_file() else {"status": "missing"}
    run_dir = _latest_run_dir(candidate_dir / "runs")
    result: dict[str, Any] = {
        "status": status.get("status", "missing"),
        "exit_code": status.get("exit_code"),
        "output_dir": str(run_dir) if run_dir else None,
        "reference_batch_size": status.get("reference_batch_size"),
    }
    if run_dir is None:
        result["valid"] = False
        result["failure_reason"] = status.get("failure_reason", "missing training artifacts")
        return result

    metrics = _read_jsonl(run_dir / "train_metrics.jsonl")
    meta_path = run_dir / "train_meta.json"
    meta = _read_json(meta_path) if meta_path.is_file() else {}
    configured_qids = list(meta.get("selected_qids") or [])
    qids = [record.get("qid") for record in metrics]
    expected_records = int(status.get("sample_count") or len(configured_qids) or len(metrics))
    training_times = [
        sum(
            float(record.get("timing", {}).get(key, 0.0))
            for key in (
                "policy_forward_seconds",
                "reference_forward_seconds",
                "backward_seconds",
            )
        )
        for record in metrics
    ]
    total_times = [float(record.get("timing", {}).get("total_seconds", 0.0)) for record in metrics]
    rollout_times = [float(record.get("timing", {}).get("rollout_seconds", 0.0)) for record in metrics]
    valid_tokens = sum(int(record.get("valid_completion_token_count", 0)) for record in metrics)
    action_count = sum(int(record.get("trainable_action_count", 0)) for record in metrics)
    training_seconds = sum(training_times)
    memory_path = candidate_dir / "gpu_memory_mib.txt"
    memory_samples = [
        int(line.strip())
        for line in memory_path.read_text(encoding="utf-8").splitlines()
        if line.strip().isdigit()
    ] if memory_path.is_file() else []
    finite = _finite_training_metrics(metrics)
    successful = status.get("status") == "success" and status.get("exit_code") == 0
    complete = len(metrics) == expected_records

    result.update(
        {
            "valid": bool(successful and finite and complete and metrics),
            "finite_metrics": finite,
            "metric_records": len(metrics),
            "expected_metric_records": expected_records,
            "selected_qids": qids,
            "configured_selected_qids": configured_qids,
            "total_wall_seconds": sum(total_times),
            "mean_sample_seconds": statistics.fmean(total_times) if total_times else 0.0,
            "median_sample_seconds": statistics.median(total_times) if total_times else 0.0,
            "p90_sample_seconds": _percentile(total_times, 0.9),
            "training_side_seconds": training_seconds,
            "rollout_seconds": sum(rollout_times),
            "valid_completion_tokens": valid_tokens,
            "trainable_actions": action_count,
            "valid_tokens_per_training_second": (
                valid_tokens / training_seconds if training_seconds > 0 else 0.0
            ),
            "skipped_zero_advantage_steps": sum(
                record.get("skipped_update_reason") == "zero_advantage" for record in metrics
            ),
            "peak_gpu_memory_mib": max(memory_samples) if memory_samples else None,
        }
    )
    if not result["valid"]:
        if successful and not complete:
            result["failure_reason"] = (
                f"incomplete metric records: {len(metrics)}/{expected_records}"
            )
        else:
            result["failure_reason"] = status.get("failure_reason") or (
                "non-finite metrics" if not finite else "incomplete candidate"
            )
    return result


def summarize(root: Path) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for candidate_dir in sorted(root.glob("bs*"), key=lambda path: int(path.name[2:])):
        if candidate_dir.name[2:].isdigit():
            candidates[candidate_dir.name[2:]] = _candidate_summary(candidate_dir)

    successful = [
        (int(batch_size), candidate)
        for batch_size, candidate in candidates.items()
        if candidate.get("valid")
    ]
    successful.sort()
    qid_sequences = [candidate["selected_qids"] for _, candidate in successful]
    same_qids = bool(qid_sequences) and all(qids == qid_sequences[0] for qids in qid_sequences[1:])

    selected: int | None = None
    if same_qids:
        for index, (batch_size, candidate) in enumerate(successful):
            if index == 0:
                selected = batch_size
                continue
            previous = successful[index - 1][1]
            if candidate["valid_tokens_per_training_second"] >= (
                previous["valid_tokens_per_training_second"] * 1.05
            ):
                selected = batch_size

    return {
        "benchmark_root": str(root),
        "same_selected_qids": same_qids,
        "selected_batch_size": selected,
        "selection_rule": "largest candidate with >=5% throughput gain over next smaller success",
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    args = parser.parse_args()
    root = args.benchmark_root
    summary = summarize(root)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "benchmark_summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
