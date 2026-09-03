from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stability_gate import _read_jsonls, evaluate_training_stability


STATE_SCHEMA_VERSION = 2
DATASETS = ("2wiki", "hotpotqa", "musique")


@dataclass(frozen=True)
class StageDecision:
    continue_training: bool
    improved: bool
    bad_validation_count: int
    reason: str
    report: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required for staged GRPO early stopping.") from exc
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Staged early-stopping config must be a mapping.")
    config = dict(payload)
    schedule = [int(step) for step in config.get("schedule_steps", [])]
    if not schedule or schedule != sorted(set(schedule)) or schedule[0] <= 0:
        raise ValueError("schedule_steps must be a non-empty, strictly increasing list.")
    if int(config.get("stability_window", 200)) < 100:
        raise ValueError("stability_window must be at least 100.")
    if int(config.get("patience", 2)) < 1:
        raise ValueError("patience must be positive.")
    for key in (
        "min_delta",
        "max_macro_regression",
        "max_dataset_regression",
        "max_parse_failure_rate",
        "max_missing_answer_tag_rate",
    ):
        value = float(config.get(key, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be a non-negative finite value.")
    raw_profiles = config.get("validation_profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raw_profiles = {
            "default": {
                "steps": schedule,
                "eval_config_path": config.get("eval_config_path"),
                "expected_per_dataset": 100,
                "baseline_kind": "sft",
                "quality_gate_min_step": config.get("quality_gate_min_step", schedule[0]),
            }
        }
    profiles: dict[str, dict[str, Any]] = {}
    assigned_steps: list[int] = []
    for name, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Validation profile {name!r} must be a mapping.")
        steps = [int(step) for step in raw_profile.get("steps", [])]
        if not steps or steps != sorted(set(steps)):
            raise ValueError(f"Validation profile {name!r} steps must be strictly increasing.")
        expected_per_dataset = int(raw_profile.get("expected_per_dataset", 0))
        if expected_per_dataset <= 0:
            raise ValueError(f"Validation profile {name!r} expected_per_dataset must be positive.")
        baseline_kind = str(raw_profile.get("baseline_kind", "sft"))
        if baseline_kind not in {"sft", "first_stage"}:
            raise ValueError(f"Validation profile {name!r} has invalid baseline_kind.")
        eval_config_path = str(raw_profile.get("eval_config_path") or "").strip()
        if not eval_config_path:
            raise ValueError(f"Validation profile {name!r} requires eval_config_path.")
        profiles[str(name)] = {
            **raw_profile,
            "steps": steps,
            "eval_config_path": eval_config_path,
            "expected_per_dataset": expected_per_dataset,
            "baseline_kind": baseline_kind,
            "quality_gate_min_step": int(
                raw_profile.get("quality_gate_min_step", config.get("quality_gate_min_step", steps[0]))
            ),
        }
        assigned_steps.extend(steps)
    if sorted(assigned_steps) != schedule or len(set(assigned_steps)) != len(assigned_steps):
        raise ValueError("Validation profile steps must partition schedule_steps exactly.")
    config["schedule_steps"] = schedule
    config["validation_profiles"] = profiles
    return config


def _evaluation_payload(root: str | Path, *, expected_per_dataset: int = 100) -> dict[str, Any]:
    path = Path(root)
    metrics = _read_json(path / "aggregate_metrics.json")
    protocol = _read_json(path / "aggregate_protocol_metrics.json")
    contract = _read_json(path / "evaluation_contract.json")
    datasets = metrics.get("datasets")
    expected_total = expected_per_dataset * len(DATASETS)
    if metrics.get("num_samples") != expected_total or not isinstance(datasets, dict):
        raise ValueError(f"Expected a complete {expected_total}-sample evaluation: {path}")
    for dataset in DATASETS:
        item = datasets.get(dataset)
        if not isinstance(item, dict) or item.get("num_samples") != expected_per_dataset:
            raise ValueError(
                f"Expected {expected_per_dataset} {dataset} validation samples: {path}"
            )
    return {"metrics": metrics, "protocol": protocol, "contract": contract}


def _profile_for_step(config: dict[str, Any], step: int) -> tuple[str, dict[str, Any]]:
    matches = [
        (name, profile)
        for name, profile in config["validation_profiles"].items()
        if step in profile["steps"]
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one validation profile for step {step}.")
    return matches[0]


def decide_stage(
    *,
    step: int,
    baseline: dict[str, Any],
    current: dict[str, Any],
    stability: dict[str, Any],
    best_macro_f1: float,
    bad_validation_count: int,
    config: dict[str, Any],
) -> StageDecision:
    baseline_metrics = baseline["metrics"]
    current_metrics = current["metrics"]
    protocol = current["protocol"]
    failures: list[str] = []
    if (
        baseline["contract"].get("contract_fingerprint")
        != current["contract"].get("contract_fingerprint")
    ):
        failures.append("evaluation_contract_mismatch")
    if not stability.get("passed"):
        failures.extend(f"training:{item}" for item in stability.get("failures", []))
    if protocol.get("error_count") != 0:
        failures.append("validation_generation_errors")
    if float(protocol.get("parse_failure_rate", math.inf)) >= float(
        config["max_parse_failure_rate"]
    ):
        failures.append("validation_parse_failure_rate")
    if float(protocol.get("missing_answer_tag_rate", math.inf)) >= float(
        config["max_missing_answer_tag_rate"]
    ):
        failures.append("validation_missing_answer_tag_rate")

    macro_f1 = float(current_metrics["macro_f1"])
    baseline_macro_f1 = float(baseline_metrics["macro_f1"])
    if macro_f1 < baseline_macro_f1 - float(config["max_macro_regression"]):
        failures.append("macro_f1_regression")
    dataset_deltas: dict[str, float] = {}
    for dataset in DATASETS:
        baseline_f1 = float(baseline_metrics["datasets"][dataset]["f1"])
        current_f1 = float(current_metrics["datasets"][dataset]["f1"])
        dataset_deltas[dataset] = current_f1 - baseline_f1
        if current_f1 < baseline_f1 - float(config["max_dataset_regression"]):
            failures.append(f"{dataset}_f1_regression")

    improved = macro_f1 >= best_macro_f1 + float(config["min_delta"])
    next_bad_count = 0 if improved else int(bad_validation_count)
    if step >= int(config["quality_gate_min_step"]) and not improved:
        next_bad_count += 1
    if next_bad_count >= int(config["patience"]):
        failures.append("early_stopping_patience")
    report = {
        "step": int(step),
        "macro_f1": macro_f1,
        "baseline_macro_f1": baseline_macro_f1,
        "best_macro_f1_before": float(best_macro_f1),
        "macro_delta_from_baseline": macro_f1 - baseline_macro_f1,
        "dataset_deltas_from_baseline": dataset_deltas,
        "improved": improved,
        "bad_validation_count": next_bad_count,
        "stability": stability,
        "failures": failures,
    }
    return StageDecision(
        continue_training=not failures,
        improved=improved,
        bad_validation_count=next_bad_count,
        reason="passed" if not failures else failures[0],
        report=report,
    )


def _is_complete_evaluation(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in (
            "aggregate_metrics.json",
            "aggregate_protocol_metrics.json",
            "evaluation_contract.json",
        )
    )


def _checkpoint_paths(output_root: Path, step: int) -> set[Path]:
    return {
        path.resolve()
        for path in output_root.glob(f"*/checkpoint-{step}")
        if path.is_dir()
        and (path / "COMPLETE").is_file()
        and (path / "checkpoint_manifest.json").is_file()
    }


def _run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, env=env, check=True)


def _run_evaluation(
    *,
    repo_root: Path,
    eval_config: Path,
    adapter_label: str,
    adapter_path: Path,
    output_dir: Path,
    dry_run: bool,
) -> None:
    if _is_complete_evaluation(output_dir):
        return
    env = dict(os.environ)
    env.update(
        {
            "CONFIG_PATH": str(eval_config),
            "ADAPTER_LABEL": adapter_label,
            "ADAPTER_PATH": str(adapter_path),
            "OUTPUT_DIR": str(output_dir),
        }
    )
    _run_command(
        ["bash", str(repo_root / "scripts" / "evaluate_grpo_fixed.sh")],
        env=env,
        dry_run=dry_run,
    )


def _run_training_stage(
    *,
    repo_root: Path,
    train_config: Path,
    output_root: Path,
    target_step: int,
    resume_checkpoint: Path | None,
    checkpoints_before: set[Path] | None,
    dry_run: bool,
) -> Path | None:
    before = (
        checkpoints_before
        if checkpoints_before is not None
        else _checkpoint_paths(output_root, target_step)
    )
    env = dict(os.environ)
    env["CONFIG_PATH"] = str(train_config)
    if resume_checkpoint is None:
        command = [
            "bash",
            str(repo_root / "scripts" / "run_train_grpo.sh"),
            "--run-until-step",
            str(target_step),
        ]
    else:
        env["RESUME_CHECKPOINT"] = str(resume_checkpoint)
        env["RUN_UNTIL_STEP"] = str(target_step)
        command = ["bash", str(repo_root / "scripts" / "run_train_grpo_resume.sh")]
    _run_command(command, env=env, dry_run=dry_run)
    if dry_run:
        return None
    created = _checkpoint_paths(output_root, target_step) - before
    if len(created) != 1:
        raise RuntimeError(
            f"Expected exactly one new checkpoint-{target_step}, found {sorted(map(str, created))}"
        )
    return created.pop()


def _initial_state(
    config: dict[str, Any],
    *,
    profile_name: str,
    profile: dict[str, Any],
    baseline_dir: Path,
    sft_adapter: Path,
) -> dict[str, Any]:
    baseline = _evaluation_payload(
        baseline_dir,
        expected_per_dataset=int(profile["expected_per_dataset"]),
    )
    initial_best = {
        "kind": "sft",
        "profile": profile_name,
        "step": 0,
        "macro_f1": float(baseline["metrics"]["macro_f1"]),
        "adapter_path": str(sft_adapter),
        "evaluation_dir": str(baseline_dir),
    }
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "running",
        "schedule_steps": list(config["schedule_steps"]),
        "completed_stages": [],
        "baselines": {
            profile_name: dict(initial_best),
        },
        "best_by_profile": {profile_name: dict(initial_best)},
        "bad_validation_counts": {profile_name: 0},
        "best": initial_best,
    }


def run(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    repo_root = Path(config.get("repo_root", ".")).resolve()
    train_config = (repo_root / str(config["train_config_path"])).resolve()
    state_dir = (repo_root / str(config["state_dir"])).resolve()
    state_path = state_dir / "controller_state.json"
    history_path = state_dir / "validation_history.jsonl"
    best_path = state_dir / "best_checkpoint.json"
    sft_adapter = (repo_root / str(config["sft_adapter_path"])).resolve()
    first_profile_name, first_profile = _profile_for_step(
        config, config["schedule_steps"][0]
    )
    if first_profile["baseline_kind"] != "sft":
        raise ValueError("The first validation profile must use the SFT baseline.")
    baseline_dir = state_dir / "evaluations" / f"sft-{first_profile_name}"
    first_eval_config = (
        repo_root / str(first_profile["eval_config_path"])
    ).resolve()

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required for staged GRPO early stopping.") from exc
    train_payload = yaml.safe_load(train_config.read_text(encoding="utf-8")) or {}
    output_root = (repo_root / str(train_payload["output_root"])).resolve()
    if int(train_payload.get("max_steps", 0)) < max(config["schedule_steps"]):
        raise ValueError("Training max_steps must cover the entire staged schedule.")
    configured_sft_adapter = (repo_root / str(train_payload.get("sft_adapter_path", ""))).resolve()
    if configured_sft_adapter != sft_adapter:
        raise ValueError("Controller and training configs must use the same SFT adapter.")

    _run_evaluation(
        repo_root=repo_root,
        eval_config=first_eval_config,
        adapter_label=f"sft-{first_profile_name}",
        adapter_path=sft_adapter,
        output_dir=baseline_dir,
        dry_run=dry_run,
    )
    if dry_run and not _is_complete_evaluation(baseline_dir):
        return {"status": "dry_run", "next_step": config["schedule_steps"][0]}

    if state_path.is_file():
        state = _read_json(state_path)
        if int(state.get("schema_version", 0)) != STATE_SCHEMA_VERSION:
            raise ValueError("Controller state schema is incompatible with validation profiles.")
        if state.get("schedule_steps") != config["schedule_steps"]:
            raise ValueError("Controller schedule does not match the persisted state.")
        if state.get("status") != "running":
            return state
    else:
        state = _initial_state(
            config,
            profile_name=first_profile_name,
            profile=first_profile,
            baseline_dir=baseline_dir,
            sft_adapter=sft_adapter,
        )
        if not dry_run:
            _write_json(state_path, state)
            _write_json(best_path, state["best"])

    for target_step in config["schedule_steps"]:
        if any(int(item["step"]) == target_step for item in state["completed_stages"]):
            continue
        resume_checkpoint = (
            Path(state["completed_stages"][-1]["checkpoint"])
            if state["completed_stages"]
            else None
        )
        pending = state.get("pending_stage")
        if pending and int(pending.get("step", -1)) != target_step:
            raise ValueError("Persisted pending stage does not match the next schedule step.")
        checkpoint: Path | None = None
        if pending and pending.get("checkpoint"):
            candidate = Path(str(pending["checkpoint"])).resolve()
            if candidate in _checkpoint_paths(output_root, target_step):
                checkpoint = candidate
        checkpoints_before = {
            Path(value).resolve() for value in (pending or {}).get("checkpoints_before", [])
        }
        if checkpoint is None and pending:
            recovered = _checkpoint_paths(output_root, target_step) - checkpoints_before
            if len(recovered) == 1:
                checkpoint = recovered.pop()
        if checkpoint is None:
            if not pending:
                checkpoints_before = _checkpoint_paths(output_root, target_step)
                if not dry_run:
                    state["pending_stage"] = {
                        "step": target_step,
                        "checkpoints_before": sorted(map(str, checkpoints_before)),
                    }
                    _write_json(state_path, state)
            checkpoint = _run_training_stage(
                repo_root=repo_root,
                train_config=train_config,
                output_root=output_root,
                target_step=target_step,
                resume_checkpoint=resume_checkpoint,
                checkpoints_before=checkpoints_before,
                dry_run=dry_run,
            )
        if dry_run:
            return {"status": "dry_run", "next_step": target_step}
        assert checkpoint is not None
        state["pending_stage"]["checkpoint"] = str(checkpoint)
        _write_json(state_path, state)
        profile_name, profile = _profile_for_step(config, target_step)
        eval_config = (repo_root / str(profile["eval_config_path"])).resolve()
        evaluation_dir = (
            state_dir / "evaluations" / profile_name / f"rl-step-{target_step}"
        )
        _run_evaluation(
            repo_root=repo_root,
            eval_config=eval_config,
            adapter_label=f"rl-step-{target_step}",
            adapter_path=checkpoint,
            output_dir=evaluation_dir,
            dry_run=False,
        )
        metrics_paths = [
            str(Path(item["checkpoint"]).parent / "train_metrics.jsonl")
            for item in state["completed_stages"]
        ] + [str(checkpoint.parent / "train_metrics.jsonl")]
        stability = evaluate_training_stability(
            _read_jsonls(metrics_paths),
            window_size=int(config["stability_window"]),
        )
        current = _evaluation_payload(
            evaluation_dir,
            expected_per_dataset=int(profile["expected_per_dataset"]),
        )
        profile_bootstrap = profile_name not in state["baselines"]
        if profile_bootstrap:
            if profile["baseline_kind"] != "first_stage":
                raise ValueError(f"Missing SFT baseline for validation profile {profile_name}.")
            bootstrap = {
                "kind": "rl",
                "profile": profile_name,
                "step": target_step,
                "macro_f1": float(current["metrics"]["macro_f1"]),
                "adapter_path": str(checkpoint),
                "evaluation_dir": str(evaluation_dir),
            }
            state["baselines"][profile_name] = dict(bootstrap)
            state["best_by_profile"][profile_name] = dict(bootstrap)
            state["bad_validation_counts"][profile_name] = 0
        baseline_record = state["baselines"][profile_name]
        baseline = _evaluation_payload(
            baseline_record["evaluation_dir"],
            expected_per_dataset=int(profile["expected_per_dataset"]),
        )
        decision_config = {**config, **profile}
        decision = decide_stage(
            step=target_step,
            baseline=baseline,
            current=current,
            stability=stability,
            best_macro_f1=float(state["best_by_profile"][profile_name]["macro_f1"]),
            bad_validation_count=int(state["bad_validation_counts"][profile_name]),
            config=decision_config,
        )
        stage = {
            "step": target_step,
            "validation_profile": profile_name,
            "validation_samples": int(profile["expected_per_dataset"]) * len(DATASETS),
            "checkpoint": str(checkpoint),
            "evaluation_dir": str(evaluation_dir),
            "decision": decision.report,
        }
        state["completed_stages"].append(stage)
        state.pop("pending_stage", None)
        state["bad_validation_counts"][profile_name] = decision.bad_validation_count
        if profile_bootstrap or decision.improved:
            profile_best = {
                "kind": "rl",
                "profile": profile_name,
                "step": target_step,
                "macro_f1": float(current["metrics"]["macro_f1"]),
                "adapter_path": str(checkpoint),
                "evaluation_dir": str(evaluation_dir),
            }
            state["best_by_profile"][profile_name] = profile_best
            state["best"] = profile_best
            _write_json(best_path, profile_best)
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(stage, ensure_ascii=False, sort_keys=True) + "\n")
        if not decision.continue_training:
            state["status"] = "early_stopped"
            state["stop_reason"] = decision.reason
        elif target_step == config["schedule_steps"][-1]:
            state["status"] = "complete"
        _write_json(state_path, state)
        if state["status"] != "running":
            return state
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run staged GRPO with fixed validation and early stopping.")
    parser.add_argument("--config", default="config/grpo_staged_early_stop.yml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    state = run(args.config, dry_run=args.dry_run)
    print(json.dumps(state, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
