"""Isolated, resumable paired pilot: aligned control vs no Answer local shaping.

The launcher owns its vLLM server. Before every evaluation it either starts the
exact disk adapter or uses the server just synchronized by that arm's trainer.
It never assumes an unrelated live server is serving the adapter in a label.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import requests
import yaml

from answer_metrics import ANSWER_F1_CONTRACT, calculate_f1
from .checkpointing import fingerprint_config, fingerprint_dataset, load_full_checkpoint_metadata
from .config import parse_args as parse_train_args
from .data import epoch_sample_order, load_rl_samples, select_balanced_samples
from .rewards import _covered_supporting_fact_count, _support_facts_required
from .staged_early_stop import _evaluation_payload, _is_complete_evaluation

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("2wiki", "hotpotqa", "musique")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(config_path, *, run_dir=None):
    spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    # Canonicalize tuple defaults to lists so YAML/JSON round trips preserve
    # equality when the frozen plan is checked on restart.
    base = json.loads(json.dumps(vars(parse_train_args(["--config", str(REPO_ROOT / spec["train_config"])]))))
    evaluation = yaml.safe_load((REPO_ROOT / spec["eval_config"]).read_text(encoding="utf-8"))
    root = (REPO_ROOT / (run_dir or spec["run_dir"])).resolve()
    steps = int(spec["steps"])
    if steps <= 0 or steps > int(base["max_steps"]):
        raise ValueError("Pilot steps must be positive and within the original max_steps horizon")
    if base["resume_from_checkpoint"]:
        raise ValueError("The paired pilot must start from SFT, not an existing RL checkpoint")
    if len(str(base["gpu_indices"]).split(",")) != 1 or not base["use_vllm_generation"] or base["vllm_sync_mode"] != "lora":
        raise ValueError("This pilot requires a single training GPU and the LoRA generation server")
    if set(str(base["gpu_indices"]).split(",")) & set(str(base["vllm_gpu_indices"]).split(",")):
        raise ValueError("Training and generation GPUs must be separate")
    if base["vllm_sync_every_steps"] != 1 or not base["vllm_sync_after_step"]:
        raise ValueError("Evaluation requires synchronization after every update")
    if evaluation["vllm_transport"] != "training_server":
        raise ValueError("This pilot requires the training_server evaluation transport")
    port = int(spec["server_port"])
    if not 1 <= port <= 65535:
        raise ValueError("Invalid server_port")
    sft = (REPO_ROOT / base["sft_adapter_path"]).resolve()
    adapter_files = {name: _sha(sft / name) for name in (
        "adapter_model.safetensors", "adapter_config.json", "prompt_contract.json",
    )}
    base.update(seed=int(spec["seed"]), sft_adapter_path=str(sft),
                vllm_lora_adapter_path=str(sft), vllm_host="127.0.0.1", vllm_port=port,
                run_until_step=steps, save_steps=min(50, steps), save_total_limit=4,
                log_all_group_rollouts=True, check_only=False)
    # Do not shorten max_steps/max_total_samples: that would change the LR
    # schedule and the selected/shuffled prefix instead of only the reward.
    samples, _ = load_rl_samples(data_root=REPO_ROOT / base["rl_data_root"],
                                 data_files=list(base["rl_data_files"] or []),
                                 max_samples=base["max_samples"],
                                 data_sampling_strategy=base["data_sampling_strategy"],
                                 data_sampling_seed=base["data_sampling_seed"])
    samples = select_balanced_samples(samples, max_total_samples=base["max_total_samples"],
                                      seed=base["seed"])
    if steps > len(samples):
        raise ValueError("Pilot must fit within the first training epoch")
    prefix = epoch_sample_order(samples, seed=base["seed"], epoch=1)[:steps]
    manifest = REPO_ROOT / evaluation["data_root"] / "manifest.jsonl"
    validation = _rows(manifest)
    if Counter(r["dataset"] for r in validation) != Counter({ds: 30 for ds in DATASETS}):
        raise ValueError("The pilot requires the fixed 90-question validation set (30 per dataset)")
    if {(s.dataset, s.qid) for s in samples} & {(r["dataset"], r["qid"]) for r in validation}:
        raise ValueError("Training/validation QID overlap")
    evaluation.update(eval_request_workers=4, vllm_base_urls=[f"http://127.0.0.1:{port}"])
    for key in ("model_path", "max_rounds", "max_prompt_length", "max_completion_length",
                "prompt_config_path", "retrieval_top_k"):
        if evaluation[key] != base[key]:
            raise ValueError(f"Training/evaluation contract mismatch: {key}")
    variants = {}
    for name, weight in spec["variants"].items():
        if not name.replace("_", "").isalnum() or not 0 <= float(weight) <= 1:
            raise ValueError("Invalid variant name or Answer local weight")
        variants[name] = {**base, "answer_local_reward_weight": float(weight),
                          "output_root": str(root / name / "training")}
    if sorted(v["answer_local_reward_weight"] for v in variants.values()) != [0.0, 1.0]:
        raise ValueError("Expected exactly two variants: Answer local weights 1 and 0")
    sources = ["src/answer_metrics.py", "src/rl_training/rewards.py", "src/rl_training/trainer.py",
               "src/rl_training/train_grpo_macorag.py", "src/rl_training/answer_reward_ablation.py",
               "src/rl_training/policy.py", "src/evaluation/evaluate_rag_model.py",
               "src/evaluation/local_evaluator.py", base["prompt_config_path"]]
    return {
        "schema_version": 1, "run_dir": str(root), "steps": steps,
        "answer_f1_contract": ANSWER_F1_CONTRACT, "sft_adapter": str(sft),
        "sft_files": adapter_files, "source_fingerprints": {str(p): _sha(REPO_ROOT / p) for p in sources},
        "dataset_fingerprint": fingerprint_dataset(samples), "num_training_samples": len(samples),
        "training_prefix": [[s.dataset, s.qid] for s in prefix],
        "training_prefix_counts": dict(Counter(s.dataset for s in prefix)),
        "manifest": str(manifest), "manifest_sha256": _sha(manifest),
        "eval_config": evaluation, "variants": variants,
        "server_start_timeout": int(spec["server_start_timeout"]),
    }


def prepare(plan):
    root = Path(plan["run_dir"])
    plan_file = root / "plan.json"
    if plan_file.exists() and _read(plan_file) != plan:
        raise ValueError("Pilot configuration/data/code changed; use a new --run-dir, do not mix experiments")
    _write(plan_file, plan)
    for name, config in {**plan["variants"], "evaluation": plan["eval_config"]}.items():
        path = root / "configs" / f"{name}.yml"
        if path.exists() and yaml.safe_load(path.read_text()) != config:
            raise ValueError(f"Frozen config was modified: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _environment(config):
    return {**os.environ, "CONFIG_PATH": str(config), "PYTHON": sys.executable,
            "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}


def _command(command, *, env, log):
    print(f"+ {' '.join(command)}\n  log: {log}", flush=True)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("a", encoding="utf-8") as stream:
        subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


@contextmanager
def _server(plan, config_path, adapter, log):
    config = yaml.safe_load(Path(config_path).read_text())
    port = int(config["vllm_port"])
    # Never reuse or terminate someone else's server.
    with socket.socket() as probe:
        # Match the HTTP server's restart semantics: TIME_WAIT connections
        # from our previous stage must not look like a live listener. Do not
        # enable SO_REUSEPORT; an existing listening service must still fail.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is occupied; this pilot will not take over an existing service") from exc
    command = ["bash", str(REPO_ROOT / "scripts/run_grpo_vllm_server.sh"),
               "--lora-adapter-path", str(adapter)]
    print(f"Starting owned vLLM server on port {port}; log: {log}", flush=True)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=_environment(config_path),
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + plan["server_start_timeout"]
            with requests.Session() as session:
                session.trust_env = False
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"vLLM server exited; inspect {log}")
                    try:
                        response = session.get(f"http://127.0.0.1:{port}/health/", timeout=2)
                        response.raise_for_status()
                        health = response.json()
                    except (requests.RequestException, ValueError):
                        health = None
                    if health is not None:
                        if (Path(health.get("lora_adapter_path", "")).resolve() != Path(adapter).resolve()
                                or Path(health.get("model", "")).resolve() != (REPO_ROOT / config["model_path"]).resolve()
                                or not health.get("supports_lora_param_update")):
                            raise RuntimeError("Owned server adapter/model/capability mismatch")
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"vLLM server startup timeout; inspect {log}")
                    time.sleep(2)
            yield
        finally:
            # Only signal the process group created above, never a broad pkill.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def _checkpoints(plan, name):
    config = plan["variants"][name]
    expected = fingerprint_config(argparse.Namespace(**config))
    found = []
    for path in Path(config["output_root"]).glob("*/checkpoint-*"):
        if not (path / "COMPLETE").is_file():
            continue
        metadata = load_full_checkpoint_metadata(path)
        if metadata["config_fingerprint"] != expected or metadata["dataset_fingerprint"] != plan["dataset_fingerprint"]:
            raise ValueError(f"Checkpoint does not belong to this arm: {path}")
        if int(metadata["global_step"]) > plan["steps"]:
            raise ValueError(f"Checkpoint beyond pilot target: {path}")
        found.append((int(metadata["global_step"]), path))
    return sorted(found)


def _evaluate(plan, name, adapter):
    root = Path(plan["run_dir"])
    env = _environment(root / "configs/evaluation.yml")
    env.update(ADAPTER_LABEL=name, ADAPTER_PATH=str(adapter), OUTPUT_DIR=str(root / "evaluations" / name))
    _command(["bash", str(REPO_ROOT / "scripts/evaluate_grpo_fixed.sh")], env=env,
             log=root / "logs" / f"{name}-evaluation.log")


def summarize_evaluation(path, manifest):
    payload = _evaluation_payload(path, expected_per_dataset=30)
    rows = [r for ds in DATASETS for r in _rows(Path(path) / ds / "predictions.jsonl")]
    by_id = {(r["dataset"], r["qid"]): r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != set(manifest):
        raise ValueError("Validation QIDs must match the frozen manifest exactly")
    coverage = []
    for key, row in by_id.items():
        sample = manifest[key]
        # The evaluation data loader trims surrounding whitespace before
        # generating/storing predictions; compare against that same contract.
        if row["gold_answer"] != str(sample["answer"]).strip() or row["question"] != str(sample["question"]).strip():
            raise ValueError("Validation gold/question differs from the frozen manifest")
        required = _support_facts_required(sample)
        if required:
            coverage.append(_covered_supporting_fact_count(row["trajectory"], sample) / required)
    f1s = {key: calculate_f1(r.get("pred_answer"), r.get("gold_answer")) for key, r in by_id.items()}
    return {
        "macro_f1": sum(f1s.values()) / len(rows),
        "datasets": payload["metrics"]["datasets"],
        "mean_retrieval_rounds": sum(r["retrieval_count"] for r in rows) / len(rows),
        "mean_support_doc_coverage": sum(coverage) / len(coverage) if coverage else None,
        "full_support_doc_count": sum(c >= 1 for c in coverage),
        "support_supervised_count": len(coverage),
        "protocol": payload["protocol"],
    }, f1s, payload["contract"]["contract_fingerprint"]


def _assert_evaluation_adapter(path, adapter):
    identity = _read(Path(path) / "evaluation_contract.json").get("adapter_identity", {})
    if Path(identity.get("path", "")).resolve() != Path(adapter).resolve():
        raise ValueError(f"Evaluation adapter path mismatch: {path}")
    for name in ("adapter_model.safetensors", "adapter_config.json", "prompt_contract.json"):
        if identity.get("files", {}).get(name) != _sha(Path(adapter) / name):
            raise ValueError(f"Evaluation adapter contents mismatch: {path}")


def _training_rows(checkpoint, *, output_root):
    """Follow the actual resume chain, excluding work lost after a checkpoint."""
    parts = []
    visited = set()
    checkpoint = Path(checkpoint).resolve()
    while True:
        if checkpoint in visited or Path(output_root).resolve() not in checkpoint.parents:
            raise ValueError("Invalid/cyclic checkpoint resume chain")
        visited.add(checkpoint)
        end = int(_read(checkpoint / "checkpoint_manifest.json")["global_step"])
        resume = _read(checkpoint.parent / "resume_meta.json")
        start = int(resume["resume_global_step"])
        part = [r for r in _rows(checkpoint.parent / "train_metrics.jsonl")
                if "kl" in r and start < int(r["step"]) <= end]
        if sorted(r["step"] for r in part) != list(range(start + 1, end + 1)):
            raise ValueError("Missing or duplicate committed training steps")
        parts.extend(part)
        previous = resume.get("resume_from_checkpoint")
        if not previous:
            if start != 0:
                raise ValueError("Training chain must originate at SFT step 0")
            break
        checkpoint = (REPO_ROOT / previous).resolve()
        if int(_read(checkpoint / "checkpoint_manifest.json")["global_step"]) != start:
            raise ValueError("Resume chain step mismatch")
    return sorted(parts, key=lambda r: r["step"])


def report(plan):
    root = Path(plan["run_dir"])
    manifest = {(r["dataset"], r["qid"]): r for r in _rows(plan["manifest"])}
    summary = {"answer_f1_contract": ANSWER_F1_CONTRACT, "steps": plan["steps"], "evaluations": {}}
    baseline_f1s = None
    baseline_contract = None
    actual_orders = []
    scores_by_arm = {}
    for name in ("sft", *plan["variants"]):
        stats, f1s, contract = summarize_evaluation(root / "evaluations" / name, manifest)
        if name == "sft":
            _assert_evaluation_adapter(root / "evaluations" / name, plan["sft_adapter"])
            baseline_f1s, baseline_contract = f1s, contract
        else:
            if contract != baseline_contract:
                raise ValueError("Evaluation contracts differ across arms")
            changes = [f1s[k] - baseline_f1s[k] for k in f1s]
            stats.update(delta_from_sft=sum(changes) / len(changes),
                         improved_questions=sum(d > 1e-9 for d in changes),
                         regressed_questions=sum(d < -1e-9 for d in changes))
            checkpoints = _checkpoints(plan, name)
            if not checkpoints or checkpoints[-1][0] != plan["steps"]:
                raise ValueError("A completed checkpoint is required to report an arm")
            _assert_evaluation_adapter(root / "evaluations" / name, checkpoints[-1][1])
            rows = _training_rows(checkpoints[-1][1], output_root=plan["variants"][name]["output_root"])
            if [r["step"] for r in rows] != list(range(1, plan["steps"] + 1)):
                raise ValueError("Training logs overlap or have missing steps; cannot claim a paired comparison")
            order = [[r["dataset"], r["qid"]] for r in rows]
            if order != plan["training_prefix"]:
                raise ValueError("Actual training sequence differs from frozen prefix")
            actual_orders.append(order)
            stats["optimizer_updates"] = sum(r["did_optimizer_step"] for r in rows)
            stats["zero_advantage_skips"] = sum(r.get("skipped_update_reason") == "zero_advantage" for r in rows)
            scores_by_arm[name] = f1s
        summary["evaluations"][name] = stats
    summary["paired_training_order_verified"] = all(x == actual_orders[0] for x in actual_orders)
    control = next(n for n, c in plan["variants"].items() if c["answer_local_reward_weight"] == 1)
    ablated = next(n for n, c in plan["variants"].items() if c["answer_local_reward_weight"] == 0)
    changes = [scores_by_arm[ablated][k] - scores_by_arm[control][k] for k in baseline_f1s]
    summary["ablated_minus_control"] = {
        "macro_f1_delta": sum(changes) / len(changes),
        "improved_questions": sum(d > 1e-9 for d in changes),
        "regressed_questions": sum(d < -1e-9 for d in changes),
        **{key + "_delta": summary["evaluations"][ablated][key] - summary["evaluations"][control][key]
           for key in ("mean_retrieval_rounds", "mean_support_doc_coverage", "full_support_doc_count")},
    }
    _write(root / "comparison.json", summary)
    return summary


def run(plan):
    root = Path(plan["run_dir"])
    first = next(iter(plan["variants"]))
    first_config = root / "configs" / f"{first}.yml"
    if not _is_complete_evaluation(root / "evaluations/sft"):
        with _server(plan, first_config, plan["sft_adapter"], root / "logs/sft-server.log"):
            _evaluate(plan, "sft", plan["sft_adapter"])
    for name in plan["variants"]:
        config_path = root / "configs" / f"{name}.yml"
        checkpoints = _checkpoints(plan, name)
        checkpoint = checkpoints[-1][1] if checkpoints else None
        complete = bool(checkpoints and checkpoints[-1][0] == plan["steps"])
        evaluation_dir = root / "evaluations" / name
        if complete and _is_complete_evaluation(evaluation_dir):
            continue
        # Resume evaluation by loading its exact checkpoint, not the last arm
        # left in a shared service. Partial training first boots SFT; the trainer
        # restores and synchronizes its full checkpoint before any rollout.
        server_adapter = checkpoint if complete else plan["sft_adapter"]
        with _server(plan, config_path, server_adapter, root / "logs" / f"{name}-server.log"):
            if not complete:
                command = ["bash", str(REPO_ROOT / "scripts/run_train_grpo.sh")]
                if checkpoint:
                    command += ["--resume-from-checkpoint", str(checkpoint)]
                _command(command, env=_environment(config_path), log=root / "logs" / f"{name}-training.log")
                checkpoints = _checkpoints(plan, name)
                if not checkpoints or checkpoints[-1][0] != plan["steps"]:
                    raise RuntimeError(f"{name} did not reach the requested checkpoint")
                checkpoint = checkpoints[-1][1]
            _evaluate(plan, name, checkpoint)
    return report(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config/grpo_answer_reward_ablation.yml"))
    parser.add_argument("--run-dir", help="Use a new directory for another independent pilot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read-only plan; no GPU, server, or artifact writes")
    mode.add_argument("--prepare-only", action="store_true", help="Freeze configs/data identities; do not launch")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    plan = build_plan(args.config, run_dir=args.run_dir)
    brief = {k: plan[k] for k in ("run_dir", "steps", "answer_f1_contract", "sft_adapter",
                                  "num_training_samples", "training_prefix_counts")}
    brief["variants"] = {k: v["answer_local_reward_weight"] for k, v in plan["variants"].items()}
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    root = Path(plan["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(plan)
        if args.prepare_only:
            return
        result = report(plan) if args.report_only else run(plan)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
