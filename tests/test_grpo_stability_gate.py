from __future__ import annotations

import json
from pathlib import Path

from rl_training.stability_gate import _read_jsonls, compare_validation_runs, evaluate_training_stability


def _stable_rows() -> list[dict]:
    return [
        {
            "step": step,
            "clip_fraction": 0.0,
            "kl": 0.02,
            "did_optimizer_step": True,
            "gradients_finite": True,
            "protocol_metrics": {
                "count": 4,
                "parse_failure_rate": 0.0,
                "missing_answer_tag_rate": 0.0,
            },
        }
        for step in range(1, 301)
    ]


def test_training_stability_gate_passes_contract() -> None:
    report = evaluate_training_stability(_stable_rows())
    assert report["passed"] is True
    assert report["metrics"]["clip_mean"] == 0.0


def test_training_stability_gate_rejects_clip_and_kl_growth() -> None:
    rows = _stable_rows()
    for row in rows[-50:]:
        row["clip_fraction"] = 0.01
        row["kl"] = 0.2
    report = evaluate_training_stability(rows)
    assert report["passed"] is False
    assert "clip_mean" in report["failures"]
    assert "kl_last50" in report["failures"]


def test_training_stability_gate_rejects_missing_protocol_and_nonfinite_metrics() -> None:
    rows = _stable_rows()
    for row in rows:
        row["protocol_metrics"] = {"count": 0}
    rows[-1]["kl"] = float("nan")

    report = evaluate_training_stability(rows)

    assert report["passed"] is False
    assert "missing_protocol_metrics" in report["failures"]
    assert "nonfinite_stability_metric" in report["failures"]


def test_training_metrics_can_span_staged_run_files(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"step": 2}) + "\n", encoding="utf-8")

    assert [row["step"] for row in _read_jsonls([str(first), str(second)])] == [1, 2]


def test_training_stability_rejects_duplicate_or_gapped_step_windows() -> None:
    duplicates = _stable_rows()
    for row in duplicates:
        row["step"] = 1
    duplicate_report = evaluate_training_stability(duplicates)
    assert "duplicate_training_steps" in duplicate_report["failures"]
    assert "nonconsecutive_training_steps" in duplicate_report["failures"]

    gapped = _stable_rows()
    gapped[-1]["step"] = 302
    gapped_report = evaluate_training_stability(gapped)
    assert "nonconsecutive_training_steps" in gapped_report["failures"]


def _write_eval(root: Path, f1s: dict[str, float], *, fingerprint: str = "same") -> None:
    root.mkdir(parents=True)
    (root / "aggregate_metrics.json").write_text(
        json.dumps({"macro_f1": sum(f1s.values()) / 3, "num_samples": 300, "datasets": {k: {"f1": v, "num_samples": 100} for k, v in f1s.items()}}),
        encoding="utf-8",
    )
    (root / "aggregate_protocol_metrics.json").write_text(
        json.dumps({"count": 300, "parse_failure_rate": 0.0, "missing_answer_tag_rate": 0.0, "error_count": 0, "error_rate": 0.0}),
        encoding="utf-8",
    )
    (root / "evaluation_contract.json").write_text(
        json.dumps({"contract_fingerprint": fingerprint}), encoding="utf-8"
    )


def test_validation_gate_requires_macro_gain_and_no_dataset_regression(tmp_path: Path) -> None:
    sft = tmp_path / "sft"
    rl = tmp_path / "rl"
    _write_eval(sft, {"2wiki": 0.4, "hotpotqa": 0.4, "musique": 0.4})
    _write_eval(rl, {"2wiki": 0.5, "hotpotqa": 0.39, "musique": 0.5})
    report = compare_validation_runs(sft, rl)
    assert report["passed"] is False
    assert "hotpotqa_f1_regression" in report["failures"]


def test_validation_gate_rejects_nan_missing_contract_and_wrong_counts(tmp_path: Path) -> None:
    sft = tmp_path / "sft"
    rl = tmp_path / "rl"
    _write_eval(sft, {"2wiki": 0.4, "hotpotqa": 0.4, "musique": 0.4}, fingerprint="")
    _write_eval(rl, {"2wiki": float("nan"), "hotpotqa": 0.5, "musique": 0.5}, fingerprint="")
    metrics = json.loads((rl / "aggregate_metrics.json").read_text(encoding="utf-8"))
    metrics["num_samples"] = 299
    (rl / "aggregate_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    report = compare_validation_runs(sft, rl)

    assert report["passed"] is False
    assert "missing_sft_evaluation_contract" in report["failures"]
    assert "missing_rl_evaluation_contract" in report["failures"]
    assert "nonfinite_or_missing_validation_metric" in report["failures"]
    assert "rl_sample_count" in report["failures"]
