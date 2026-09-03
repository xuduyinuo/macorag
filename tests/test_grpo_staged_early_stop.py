from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_training.staged_early_stop import (
    _checkpoint_paths,
    decide_stage,
    load_config,
    run,
)
from rl_training.stability_gate import evaluate_training_stability


def _evaluation(
    value: float,
    *,
    fingerprint: str = "fixed",
    per_dataset: int = 100,
) -> dict:
    return {
        "metrics": {
            "macro_f1": value,
            "num_samples": per_dataset * 3,
            "datasets": {
                dataset: {"f1": value, "num_samples": per_dataset}
                for dataset in ("2wiki", "hotpotqa", "musique")
            },
        },
        "protocol": {
            "count": per_dataset * 3,
            "parse_failure_rate": 0.0,
            "missing_answer_tag_rate": 0.0,
            "error_count": 0,
        },
        "contract": {"contract_fingerprint": fingerprint},
    }


def _config() -> dict:
    return {
        "quality_gate_min_step": 600,
        "patience": 2,
        "min_delta": 0.005,
        "max_macro_regression": 0.01,
        "max_dataset_regression": 0.02,
        "max_parse_failure_rate": 0.01,
        "max_missing_answer_tag_rate": 0.002,
    }


def _stability() -> dict:
    return {"passed": True, "failures": [], "metrics": {}, "steps": 200}


def test_early_dense_stage_continues_without_counting_quality_patience() -> None:
    decision = decide_stage(
        step=200,
        baseline=_evaluation(0.5),
        current=_evaluation(0.501),
        stability=_stability(),
        best_macro_f1=0.5,
        bad_validation_count=0,
        config=_config(),
    )

    assert decision.continue_training is True
    assert decision.improved is False
    assert decision.bad_validation_count == 0


def test_quality_patience_stops_after_two_non_improving_validations() -> None:
    first = decide_stage(
        step=600,
        baseline=_evaluation(0.5),
        current=_evaluation(0.501),
        stability=_stability(),
        best_macro_f1=0.5,
        bad_validation_count=0,
        config=_config(),
    )
    second = decide_stage(
        step=1000,
        baseline=_evaluation(0.5),
        current=_evaluation(0.503),
        stability=_stability(),
        best_macro_f1=0.5,
        bad_validation_count=first.bad_validation_count,
        config=_config(),
    )

    assert first.continue_training is True
    assert first.bad_validation_count == 1
    assert second.continue_training is False
    assert second.reason == "early_stopping_patience"


def test_significant_macro_improvement_updates_best_and_resets_patience() -> None:
    decision = decide_stage(
        step=1000,
        baseline=_evaluation(0.5),
        current=_evaluation(0.51),
        stability=_stability(),
        best_macro_f1=0.5,
        bad_validation_count=1,
        config=_config(),
    )

    assert decision.continue_training is True
    assert decision.improved is True
    assert decision.bad_validation_count == 0


def test_hard_validation_regression_stops_immediately() -> None:
    decision = decide_stage(
        step=200,
        baseline=_evaluation(0.5),
        current=_evaluation(0.45),
        stability=_stability(),
        best_macro_f1=0.5,
        bad_validation_count=0,
        config=_config(),
    )

    assert decision.continue_training is False
    assert decision.reason == "macro_f1_regression"


def test_stability_gate_supports_a_200_step_stage_window() -> None:
    rows = [
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
        for step in range(1, 201)
    ]

    report = evaluate_training_stability(rows, window_size=200)

    assert report["passed"] is True
    assert report["steps"] == 200


def test_checkpoint_discovery_requires_complete_manifest(tmp_path: Path) -> None:
    complete = tmp_path / "run-a" / "checkpoint-200"
    complete.mkdir(parents=True)
    (complete / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (complete / "checkpoint_manifest.json").write_text(
        json.dumps({"global_step": 200}), encoding="utf-8"
    )
    incomplete = tmp_path / "run-b" / "checkpoint-200"
    incomplete.mkdir(parents=True)
    (incomplete / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")

    assert _checkpoint_paths(tmp_path, 200) == {complete.resolve()}


def test_staged_config_requires_strictly_increasing_schedule(tmp_path: Path) -> None:
    path = tmp_path / "staged.yml"
    path.write_text(
        "schedule_steps: [200, 200]\n"
        "stability_window: 200\n"
        "patience: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schedule_steps"):
        load_config(path)


def test_active_controller_schedule_stops_at_3000_samples() -> None:
    config = load_config("config/grpo_staged_early_stop.yml")

    assert config["schedule_steps"] == [
        200,
        400,
        600,
        1000,
        1500,
        2000,
        2500,
        3000,
    ]
    assert config["validation_profiles"]["early_90"]["expected_per_dataset"] == 30
    assert config["validation_profiles"]["late_300"]["steps"] == [
        1000,
        1500,
        2000,
        2500,
        3000,
    ]


def test_controller_automatically_validates_resumes_and_early_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    sft_adapter = tmp_path / "sft-adapter"
    sft_adapter.mkdir()
    train_config = tmp_path / "train.yml"
    train_config.write_text(
        yaml.safe_dump(
            {
                "output_root": "training-output",
                "sft_adapter_path": "sft-adapter",
                "max_steps": 400,
            }
        ),
        encoding="utf-8",
    )
    eval_config = tmp_path / "eval.yml"
    eval_config.write_text("eval_request_workers: 4\n", encoding="utf-8")
    controller_config = tmp_path / "controller.yml"
    controller_config.write_text(
        yaml.safe_dump(
            {
                "repo_root": str(tmp_path),
                "train_config_path": "train.yml",
                "eval_config_path": "eval.yml",
                "sft_adapter_path": "sft-adapter",
                "state_dir": "state",
                "schedule_steps": [200, 400],
                "stability_window": 200,
                "quality_gate_min_step": 200,
                "patience": 2,
                "min_delta": 0.005,
                "max_macro_regression": 0.01,
                "max_dataset_regression": 0.02,
                "max_parse_failure_rate": 0.01,
                "max_missing_answer_tag_rate": 0.002,
            }
        ),
        encoding="utf-8",
    )
    trained_steps = []

    def fake_training_stage(**kwargs):
        step = int(kwargs["target_step"])
        trained_steps.append(step)
        checkpoint = tmp_path / "training-output" / f"run-{step}" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        return checkpoint

    def fake_evaluation(*, adapter_label, output_dir, **kwargs):
        value = 0.5 if adapter_label == "sft" else 0.501
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "aggregate_metrics.json").write_text(
            json.dumps(_evaluation(value)["metrics"]), encoding="utf-8"
        )
        (output_dir / "aggregate_protocol_metrics.json").write_text(
            json.dumps(_evaluation(value)["protocol"]), encoding="utf-8"
        )
        (output_dir / "evaluation_contract.json").write_text(
            json.dumps(_evaluation(value)["contract"]), encoding="utf-8"
        )

    monkeypatch.setattr("rl_training.staged_early_stop._run_training_stage", fake_training_stage)
    monkeypatch.setattr("rl_training.staged_early_stop._run_evaluation", fake_evaluation)
    monkeypatch.setattr("rl_training.staged_early_stop._read_jsonls", lambda paths: [])
    monkeypatch.setattr(
        "rl_training.staged_early_stop.evaluate_training_stability",
        lambda rows, window_size: _stability(),
    )

    state = run(controller_config)

    assert trained_steps == [200, 400]
    assert state["status"] == "early_stopped"
    assert state["stop_reason"] == "early_stopping_patience"
    assert [item["step"] for item in state["completed_stages"]] == [200, 400]
    assert "pending_stage" not in state


def test_controller_uses_small_early_validation_and_lazy_full_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yaml

    (tmp_path / "sft-adapter").mkdir()
    (tmp_path / "train.yml").write_text(
        yaml.safe_dump(
            {
                "output_root": "training-output",
                "sft_adapter_path": "sft-adapter",
                "max_steps": 1500,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "early.yml").write_text("early: true\n", encoding="utf-8")
    (tmp_path / "late.yml").write_text("late: true\n", encoding="utf-8")
    controller_config = tmp_path / "controller.yml"
    controller_config.write_text(
        yaml.safe_dump(
            {
                "repo_root": str(tmp_path),
                "train_config_path": "train.yml",
                "sft_adapter_path": "sft-adapter",
                "state_dir": "state",
                "schedule_steps": [200, 1000, 1500],
                "validation_profiles": {
                    "early_90": {
                        "steps": [200],
                        "eval_config_path": "early.yml",
                        "expected_per_dataset": 30,
                        "baseline_kind": "sft",
                        "quality_gate_min_step": 200,
                    },
                    "late_300": {
                        "steps": [1000, 1500],
                        "eval_config_path": "late.yml",
                        "expected_per_dataset": 100,
                        "baseline_kind": "first_stage",
                        "quality_gate_min_step": 1500,
                    },
                },
                "stability_window": 200,
                "patience": 3,
                "min_delta": 0.005,
                "max_macro_regression": 0.01,
                "max_dataset_regression": 0.02,
                "max_parse_failure_rate": 0.01,
                "max_missing_answer_tag_rate": 0.002,
            }
        ),
        encoding="utf-8",
    )
    evaluation_calls = []

    def fake_training_stage(**kwargs):
        step = int(kwargs["target_step"])
        checkpoint = tmp_path / "training-output" / f"run-{step}" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        return checkpoint

    def fake_evaluation(*, eval_config, adapter_label, output_dir, **kwargs):
        per_dataset = 30 if Path(eval_config).name == "early.yml" else 100
        evaluation_calls.append((adapter_label, per_dataset))
        step = int(adapter_label.rsplit("-", 1)[-1]) if adapter_label.startswith("rl-step-") else 0
        value = 0.5 + (step / 100000)
        payload = _evaluation(value, per_dataset=per_dataset)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "aggregate_metrics.json").write_text(
            json.dumps(payload["metrics"]), encoding="utf-8"
        )
        (output_dir / "aggregate_protocol_metrics.json").write_text(
            json.dumps(payload["protocol"]), encoding="utf-8"
        )
        (output_dir / "evaluation_contract.json").write_text(
            json.dumps(payload["contract"]), encoding="utf-8"
        )

    monkeypatch.setattr("rl_training.staged_early_stop._run_training_stage", fake_training_stage)
    monkeypatch.setattr("rl_training.staged_early_stop._run_evaluation", fake_evaluation)
    monkeypatch.setattr("rl_training.staged_early_stop._read_jsonls", lambda paths: [])
    monkeypatch.setattr(
        "rl_training.staged_early_stop.evaluate_training_stability",
        lambda rows, window_size: _stability(),
    )

    state = run(controller_config)

    assert state["status"] == "complete"
    assert evaluation_calls == [
        ("sft-early_90", 30),
        ("rl-step-200", 30),
        ("rl-step-1000", 100),
        ("rl-step-1500", 100),
    ]
    assert [item["validation_samples"] for item in state["completed_stages"]] == [
        90,
        300,
        300,
    ]
    assert state["baselines"]["early_90"]["kind"] == "sft"
    assert state["baselines"]["late_300"]["step"] == 1000
