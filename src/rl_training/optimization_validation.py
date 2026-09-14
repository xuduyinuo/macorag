"""Sequential, resumable fixed-300 confirmation and single-variable LR pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

import yaml

from answer_metrics import calculate_f1, normalize_answer
from .answer_reward_ablation import (
    REPO_ROOT, DATASETS, _read, _rows, _write, _sha, _server, _command,
    _environment, _checkpoints, _assert_evaluation_adapter, _training_rows,
)
from .rewards import _covered_supporting_fact_count, _support_facts_required
from .staged_early_stop import _evaluation_payload
from .stability_gate import evaluate_training_stability
from .data import load_rl_samples, select_balanced_samples
from .checkpointing import fingerprint_dataset

ROOT = REPO_ROOT / "outputs/grpo_Meta-Llama-3-8B/optimization_validation_300_dual_gpu_accepted"
OLD = REPO_ROOT / "outputs/grpo_Meta-Llama-3-8B/answer_local_ablation_200"


def accepted_parallel_evidence(smoke_root, plan):
    """Record explicit acceptance of a NEW protocol; never rewrite a failed smoke."""
    smoke_root = Path(smoke_root).resolve()
    previous = _read(smoke_root / "plan.json")
    smoke_result = _read(smoke_root / "smoke.json")
    if (previous["evaluation"] != plan["evaluation"]
            or previous["manifest_sha256"] != plan["manifest_sha256"]
            or previous["adapter_hashes"] != plan["adapter_hashes"]):
        raise ValueError("Accepted smoke configuration/data/adapters differ from this plan")
    # Only this orchestration/acceptance code is allowed to change. The actual
    # evaluator, prompts, metric, model and retrieval code must still match.
    for path, fingerprint in previous["sources"].items():
        if path != "src/rl_training/optimization_validation.py" and plan["sources"].get(path) != fingerprint:
            raise ValueError(f"Accepted smoke source changed: {path}")
    for key in ("metric_sha256", "prompt_sha256"):
        if previous[key] != plan[key]:
            raise ValueError(f"Accepted smoke contract changed: {key}")
    full = _rows(plan["manifest"])
    manifest = {(r["dataset"], r["qid"]): r for ds in DATASETS
                for r in [s for s in full if s["dataset"] == ds][:4]}
    fingerprints = {str(path.relative_to(smoke_root)): _sha(path)
                    for path in (smoke_root / "plan.json", smoke_root / "smoke.json")}
    metrics, predictions = {}, {}
    for name in ("smoke_serial", "smoke_parallel"):
        output = smoke_root / "evaluations" / name
        _assert_evaluation_adapter(output, plan["sft_adapter"])
        fingerprints[str((output / "evaluation_contract.json").relative_to(smoke_root))] = _sha(output / "evaluation_contract.json")
        rows = []
        for ds in DATASETS:
            path = output / ds / "predictions.jsonl"
            rows.extend(_rows(path))
            fingerprints[str(path.relative_to(smoke_root))] = _sha(path)
        if len(rows) != 12 or {(r["dataset"], r["qid"]) for r in rows} != set(manifest):
            raise ValueError("Accepted smoke requires the exact 12-QID subset")
        metrics[name], predictions[name] = {}, {}
        for row in rows:
            key = (row["dataset"], row["qid"])
            sample = manifest[key]
            if (row.get("error") or row["gold_answer"] != str(sample["answer"]).strip()
                    or row["question"] != str(sample["question"]).strip()):
                raise ValueError("Accepted smoke has errors or question/gold mismatch")
            metrics[name][key] = (calculate_f1(row["pred_answer"], row["gold_answer"]),
                                  normalize_answer(row["pred_answer"]) == normalize_answer(row["gold_answer"]),
                                  row["retrieval_count"],
                                  _covered_supporting_fact_count(row["trajectory"], sample),
                                  len(row["parse_errors"]))
            predictions[name][key] = row
    if metrics["smoke_serial"] != metrics["smoke_parallel"]:
        raise ValueError("Accepted smoke has per-question metric differences, not only text differences")
    if smoke_result["transport_errors"] != 0:
        raise ValueError("Cannot accept a smoke with transport errors")
    differences = [list(key) for key in manifest if any(
        predictions["smoke_serial"][key].get(field) != predictions["smoke_parallel"][key].get(field)
        for field in ("pred_answer", "trajectory", "parse_errors"))]
    return {"mode": "user_accepted_new_parallel_protocol", "source_dir": str(smoke_root),
            "original_strict_smoke_passed": smoke_result["passed"],
            "per_question_metrics_equal": True, "sample_count": 12,
            "different_predictions_or_trajectories": differences, "artifact_sha256": fingerprints,
            "limitations": "Text equality is not guaranteed; rebuild all three fixed-300 baselines. "
                           "A 12-question check does not establish full-set equivalence."}


def freeze(root, *, accepted_smoke_dir=None):
    old = _read(OLD / "plan.json")
    for path, expected in old["source_fingerprints"].items():
        if path in ("src/rl_training/rewards.py", "src/rl_training/trainer.py",
                    "src/rl_training/train_grpo.py", "src/rl_training/policy.py"):
            if _sha(REPO_ROOT / path) != expected:
                raise ValueError(f"Training source changed since ablation: {path}")
    adapters = {"sft": old["sft_adapter"]}
    for name in old["variants"]:
        checkpoints = _checkpoints(old, name)
        if not checkpoints or checkpoints[-1][0] != 200:
            raise ValueError(f"Missing completed ablation: {name}")
        adapters[name] = str(checkpoints[-1][1])
    train = dict(old["variants"]["no_answer_local"])
    train.update(learning_rate=3e-6, output_root=str(root / "low_lr/training"),
                 vllm_port=8002, vllm_gpu_indices="1", gpu_indices="0")
    evaluation = yaml.safe_load((REPO_ROOT / "config/eval_grpo_fixed.yml").read_text())
    evaluation.update(eval_request_workers=8, eval_generate_batch_size=1,
                      eval_generate_batch_wait_ms=20, vllm_timeout=300,
                      vllm_base_urls=["http://127.0.0.1:8002", "http://127.0.0.1:8003"])
    server = dict(train, vllm_max_num_seqs=8)
    server_secondary = dict(server, vllm_port=8003, vllm_gpu_indices="0")
    manifest = REPO_ROOT / evaluation["data_root"] / "manifest.jsonl"
    if Counter(r["dataset"] for r in _rows(manifest)) != Counter({ds: 100 for ds in DATASETS}):
        raise ValueError("Expected fixed 100 questions per dataset")
    samples, _ = load_rl_samples(data_root=REPO_ROOT / train["rl_data_root"],
                                 data_files=list(train["rl_data_files"] or []),
                                 max_samples=train["max_samples"],
                                 data_sampling_strategy=train["data_sampling_strategy"],
                                 data_sampling_seed=train["data_sampling_seed"])
    samples = select_balanced_samples(samples, max_total_samples=train["max_total_samples"], seed=train["seed"])
    if fingerprint_dataset(samples) != old["dataset_fingerprint"]:
        raise ValueError("Training dataset changed since ablation")
    if {(s.dataset, s.qid) for s in samples} & {(r["dataset"], r["qid"]) for r in _rows(manifest)}:
        raise ValueError("Training/fixed-validation overlap")
    plan = {"adapters": adapters, "server_start_timeout": 900, "steps": 200,
            "inference_protocol": "dual_gpu_single_prompt_8workers_v1",
            "sft_adapter": old["sft_adapter"], "dataset_fingerprint": old["dataset_fingerprint"],
            "training_prefix": old["training_prefix"], "variants": {"low_lr": train},
            "evaluation": evaluation, "server": server, "server_secondary": server_secondary,
            "manifest": str(manifest),
            "manifest_sha256": _sha(manifest),
            "metric_sha256": _sha(REPO_ROOT / "src/answer_metrics.py"),
            "prompt_sha256": _sha(REPO_ROOT / "config/prompts.yml"),
            "adapter_hashes": {name: _sha(Path(path) / "adapter_model.safetensors")
                               for name, path in adapters.items()},
            "sources": {str(path.relative_to(REPO_ROOT)): _sha(path)
                        for directory in ("src/evaluation", "src/rl_training", "src/rag")
                        for path in sorted((REPO_ROOT / directory).rglob("*.py"))}}
    plan_path = root / "plan.json"
    if accepted_smoke_dir is None and plan_path.exists():
        accepted_smoke_dir = (_read(plan_path).get("parallel_protocol_acceptance") or {}).get("source_dir")
    if accepted_smoke_dir is not None:
        plan["parallel_protocol_acceptance"] = accepted_parallel_evidence(accepted_smoke_dir, plan)
    if plan_path.exists() and _read(plan_path) != plan:
        raise ValueError("Frozen experiment changed; use a new --run-dir")
    _write(plan_path, plan)
    if "parallel_protocol_acceptance" in plan:
        _write(root / "parallel_protocol_acceptance.json", plan["parallel_protocol_acceptance"])
    for name, config in (("evaluation", evaluation), ("server", server),
                         ("server_secondary", server_secondary), ("low_lr", train)):
        path = root / "configs" / f"{name}.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(config, sort_keys=False)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Frozen config changed: {path}")
        path.write_text(content)
    return plan


@contextmanager
def evaluation_servers(root, plan, adapter, label):
    """Start independent single-prompt replicas; clean up only owned groups."""
    contexts = [_server(plan, root / "configs" / f"{name}.yml", adapter,
                        root / "logs" / f"{label}-{name}.log")
                for name in ("server", "server_secondary")]
    with ExitStack() as stack:
        error = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(context.__enter__) for context in contexts]
            for context, future in zip(contexts, futures):
                try:
                    future.result()
                    stack.callback(context.__exit__, None, None, None)
                except BaseException as exc:
                    error = exc
        if error is not None:
            raise error
        yield


def evaluate(root, name, adapter, *, extra=()):
    output = root / "evaluations" / name
    env = _environment(root / "configs/evaluation.yml")
    env.update(ADAPTER_LABEL=name, ADAPTER_PATH=str(adapter), OUTPUT_DIR=str(output))
    started = time.monotonic()
    _command(["bash", str(REPO_ROOT / "scripts/evaluate_grpo_fixed.sh"), *extra],
             env=env, log=root / "logs" / f"{name}-evaluation.log")
    _assert_evaluation_adapter(output, adapter)
    return time.monotonic() - started


def summarize(root, plan, name):
    output = root / "evaluations" / name
    payload = _evaluation_payload(output)
    run_config = _read(output / "run_config.json")
    for key in ("eval_request_workers", "eval_generate_batch_size", "vllm_base_urls"):
        if run_config.get(key) != plan["evaluation"][key]:
            raise ValueError(f"Evaluation parallel protocol mismatch: {name}/{key}")
    manifest = {(r["dataset"], r["qid"]): r for r in _rows(plan["manifest"])}
    rows = [r for ds in DATASETS for r in _rows(output / ds / "predictions.jsonl")]
    if len(rows) != 300 or {(r["dataset"], r["qid"]) for r in rows} != set(manifest):
        raise ValueError("Evaluation QIDs differ from fixed-300 manifest")
    coverage = []
    scores = {}
    for row in rows:
        key = (row["dataset"], row["qid"])
        sample = manifest[key]
        if row["gold_answer"] != str(sample["answer"]).strip() or row["question"] != str(sample["question"]).strip():
            raise ValueError("Gold/question mismatch")
        required = _support_facts_required(sample)
        if required:
            coverage.append(_covered_supporting_fact_count(row["trajectory"], sample) / required)
        scores[key] = calculate_f1(row["pred_answer"], row["gold_answer"])
    return {"metrics": payload["metrics"], "protocol": payload["protocol"],
            "contract": payload["contract"]["contract_fingerprint"],
            "mean_rounds": sum(r["retrieval_count"] for r in rows) / len(rows),
            "mean_coverage": sum(coverage) / len(coverage) if coverage else None,
            "full_support_count": sum(c >= 1 for c in coverage),
            "throughput": _read(output / "throughput.json")}, scores


def confirmation(root, plan):
    stats, scores = {}, {}
    for name, adapter in plan["adapters"].items():
        output = root / "evaluations" / name
        if not (output / "evaluation_contract.json").exists():
            with evaluation_servers(root, plan, adapter, name):
                evaluate(root, name, adapter)
        _assert_evaluation_adapter(output, adapter)
        stats[name], scores[name] = summarize(root, plan, name)
    if len({s["contract"] for s in stats.values()}) != 1:
        raise ValueError("Evaluation contracts differ")
    # This is a pilot selection gate, not a claim of statistical significance.
    f1 = {name: s["metrics"]["macro_f1"] for name, s in stats.items()}
    passed = (f1["no_answer_local"] > f1["aligned_control"]
              and f1["no_answer_local"] >= f1["sft"]
              and all(s["protocol"]["error_count"] == 0 for s in stats.values()))
    paired = {}
    for reference in ("sft", "aligned_control"):
        delta = [scores["no_answer_local"][key] - scores[reference][key] for key in scores[reference]]
        paired[reference] = {"improved": sum(d > 0 for d in delta),
                             "regressed": sum(d < 0 for d in delta),
                             "same": sum(d == 0 for d in delta)}
    result = {"evaluations": stats, "paired": paired, "proceed_low_lr": passed,
              "inference_protocol": plan.get("inference_protocol"),
              "gate": "off > control and off >= SFT, zero transport errors; exploratory only"}
    _write(root / "confirmation.json", result)
    return result


def smoke(root, plan):
    adapter = plan["adapters"]["sft"]
    # A fixed manifest cannot be truncated via --max-samples: its integrity
    # checker correctly rejects that. Give this separate smoke its own manifest.
    source = _rows(plan["manifest"])
    selected = [r for ds in DATASETS for r in [s for s in source if s["dataset"] == ds][:4]]
    smoke_root = root / "smoke_manifest"
    smoke_root.mkdir(parents=True, exist_ok=True)
    manifest = smoke_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    _write(smoke_root / "manifest_meta.json", {
        "manifest_fingerprint": _sha(manifest), "per_dataset": 4,
        "counts_by_dataset": dict(Counter(r["dataset"] for r in selected)),
        "qids": [r["qid"] for r in selected], "parent_manifest_sha256": plan["manifest_sha256"],
    })
    timings = {}
    with evaluation_servers(root, plan, adapter, "smoke"):
        for name, workers, endpoints in (
            ("smoke_serial", 4, ["http://127.0.0.1:8002"]),
            ("smoke_parallel", 8, ["http://127.0.0.1:8002", "http://127.0.0.1:8003"]),
        ):
            timings[name] = evaluate(root, name, adapter, extra=[
                "--data-root", str(smoke_root),
                "--manifest-meta-path", str(smoke_root / "manifest_meta.json"),
                "--eval-request-workers", str(workers),
                "--eval-generate-batch-size", "1", "--vllm-base-urls", *endpoints,
            ])
    rows = {name: {(r["dataset"], r["qid"]): r for ds in DATASETS
                   for r in _rows(root / "evaluations" / name / ds / "predictions.jsonl")}
            for name in timings}
    left, right = rows["smoke_serial"], rows["smoke_parallel"]
    if set(left) != set(right) or len(left) != 12:
        raise ValueError("Smoke QID mismatch")
    differences = [list(key) for key in left if any(left[key].get(field) != right[key].get(field)
                   for field in ("pred_answer", "trajectory", "parse_errors"))]
    errors = sum(bool(r.get("error")) for group in rows.values() for r in group.values())
    result = {"wall_seconds": timings, "different_predictions_or_trajectories": differences,
              "transport_errors": errors, "passed": not errors and not differences}
    _write(root / "smoke.json", result)
    if not result["passed"]:
        raise RuntimeError("Parallel smoke differs or has errors; inspect smoke.json before full evaluation")
    return result


def low_lr(root, plan):
    if not confirmation(root, plan)["proceed_low_lr"]:
        return {"status": "stopped_at_confirmation_gate"}
    config = root / "configs/low_lr.yml"
    checkpoints = _checkpoints(plan, "low_lr")
    checkpoint = checkpoints[-1][1] if checkpoints else None
    complete = bool(checkpoints and checkpoints[-1][0] == 200)
    if not complete:
        with _server(plan, config, plan["sft_adapter"], root / "logs/low_lr-training-server.log"):
            command = ["bash", str(REPO_ROOT / "scripts/run_train_grpo.sh")]
            if checkpoint:
                command += ["--resume-from-checkpoint", str(checkpoint)]
            _command(command, env=_environment(config), log=root / "logs/low_lr-training.log")
        checkpoints = _checkpoints(plan, "low_lr")
        if not checkpoints or checkpoints[-1][0] != 200:
            raise RuntimeError("Low-LR pilot did not reach step 200")
        checkpoint = checkpoints[-1][1]
    if not (root / "evaluations/low_lr/evaluation_contract.json").exists():
        with evaluation_servers(root, plan, checkpoint, "low_lr-eval"):
            evaluate(root, "low_lr", checkpoint)
    _assert_evaluation_adapter(root / "evaluations/low_lr", checkpoint)
    stats, _ = summarize(root, plan, "low_lr")
    rows = _training_rows(checkpoint, output_root=plan["variants"]["low_lr"]["output_root"])
    if [[r["dataset"], r["qid"]] for r in rows] != plan["training_prefix"]:
        raise ValueError("Low-LR pilot training prefix changed")
    result = {"evaluation": stats, "stability": evaluate_training_stability(rows, window_size=200),
              "checkpoint": str(checkpoint), "status": "review_before_efficiency_pilot"}
    _write(root / "low_lr.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "confirm", "low-lr", "all"), default="all")
    parser.add_argument("--run-dir", type=Path, default=ROOT)
    parser.add_argument("--accept-parallel-smoke", type=Path,
                        help="Explicitly accept a prior dual-replica smoke as a new protocol; keeps strict failure intact")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = freeze(root, accepted_smoke_dir=args.accept_parallel_smoke)
        accepted = "parallel_protocol_acceptance" in plan
        result = {"status": "prepared", "root": str(root)}
        if args.stage == "smoke" or (args.stage == "all" and not accepted):
            result = _read(root / "smoke.json") if (root / "smoke.json").exists() else smoke(root, plan)
            if not result["passed"]:
                raise RuntimeError("Existing smoke failed")
        if args.stage in ("confirm", "all", "low-lr"):
            if not accepted and (not (root / "smoke.json").exists() or not _read(root / "smoke.json")["passed"]):
                raise RuntimeError("Run and pass --stage smoke before expensive validation")
            result = confirmation(root, plan)
        if args.stage in ("low-lr", "all"):
            result = low_lr(root, plan)
        if args.stage != "prepare":
            _write(root / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
