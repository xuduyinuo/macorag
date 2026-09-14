#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REQUIRED_ADAPTER_FILES = ("adapter_config.json", "prompt_contract.json")


def _valid_adapter(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_ADAPTER_FILES)


def resolve_adapter(pointer: Path, explicit: str) -> None:
    if explicit.strip():
        adapter = Path(explicit).resolve()
    elif pointer.is_file():
        adapter = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
    else:
        raise SystemExit(
            "Missing shared SFT adapter. Run ablation/05_wo_grpo.sh first, "
            "or set SFT_ADAPTER_PATH explicitly."
        )
    if not _valid_adapter(adapter):
        raise SystemExit(f"Invalid SFT adapter {adapter}: required files={REQUIRED_ADAPTER_FILES}")
    print(adapter)


def record_adapter(root: Path, pointer: Path) -> None:
    candidates = [path for path in root.glob("*/adapter") if _valid_adapter(path)]
    if not candidates:
        raise SystemExit(f"No completed SFT adapter found below {root}")
    adapter = max(candidates, key=lambda path: path.parent.stat().st_mtime).resolve()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(f".{pointer.name}.tmp")
    temporary.write_text(str(adapter) + "\n", encoding="utf-8")
    os.replace(temporary, pointer)
    print(adapter)


def latest_adapter(root: Path) -> None:
    candidates = [path for path in root.glob("*/adapter") if _valid_adapter(path)]
    if not candidates:
        raise SystemExit(f"No completed adapter found below {root}")
    print(max(candidates, key=lambda path: path.parent.stat().st_mtime).resolve())


def adapter_model_identity(adapter: Path, expected: Path) -> None:
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise SystemExit(f"Adapter config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    declared = str(payload.get("base_model_name_or_path") or "").strip()
    if not declared:
        raise SystemExit(f"Adapter does not declare base_model_name_or_path: {config_path}")
    declared_path = Path(declared)
    try:
        same_model = declared_path.samefile(expected)
    except OSError:
        same_model = declared_path.resolve() == expected.resolve()
    if not same_model:
        raise SystemExit(
            f"Adapter base model mismatch: declared={declared!r}, expected={str(expected)!r}"
        )
    # Preserve the adapter's spelling because the upstream evaluator currently
    # compares this metadata field as a raw string.
    print(declared)


def evaluation_complete(output: Path, manifest: Path) -> None:
    expected_by_dataset: dict[str, list[str]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            expected_by_dataset.setdefault(str(row["dataset"]), []).append(str(row["qid"]))
    if set(expected_by_dataset) != {"2wiki", "hotpotqa", "musique"}:
        raise SystemExit(1)
    for dataset, expected_qids in expected_by_dataset.items():
        prediction_path = output / dataset / "predictions.jsonl"
        metrics_path = output / dataset / "evaluation_results.json"
        if not prediction_path.is_file() or not metrics_path.is_file():
            raise SystemExit(1)
        prediction_qids = [
            str(json.loads(line)["qid"])
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if prediction_qids != expected_qids or int(metrics.get("num_samples", -1)) != len(expected_qids):
            raise SystemExit(1)
    aggregate_path = output / "aggregate_metrics.json"
    if not aggregate_path.is_file():
        raise SystemExit(1)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    expected_total = sum(len(qids) for qids in expected_by_dataset.values())
    if int(aggregate.get("num_samples", -1)) != expected_total:
        raise SystemExit(1)
    print(f"Evaluation already complete: {expected_total} samples at {output}")


def register_base(model: Path, output: Path, data_manifest: Path) -> None:
    if not model.is_dir():
        raise SystemExit(f"Base model directory not found: {model}")
    manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": "wo_sft_grpo",
        "training": "none",
        "model_path": str(model.resolve()),
        "dataset_contract": manifest["contract"],
        "dataset_quotas": manifest["quotas"],
        "note": "Base-model control; the 1000-question training split is intentionally unused.",
    }
    target = output / "run_manifest.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve-adapter")
    resolve.add_argument("--pointer", type=Path, required=True)
    resolve.add_argument("--explicit", default="")
    record = subparsers.add_parser("record-adapter")
    record.add_argument("--root", type=Path, required=True)
    record.add_argument("--pointer", type=Path, required=True)
    latest = subparsers.add_parser("latest-adapter")
    latest.add_argument("--root", type=Path, required=True)
    identity = subparsers.add_parser("adapter-model-identity")
    identity.add_argument("--adapter", type=Path, required=True)
    identity.add_argument("--expected", type=Path, required=True)
    complete = subparsers.add_parser("evaluation-complete")
    complete.add_argument("--output", type=Path, required=True)
    complete.add_argument("--manifest", type=Path, required=True)
    register = subparsers.add_parser("register-base")
    register.add_argument("--model", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--data-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "resolve-adapter":
        resolve_adapter(args.pointer, args.explicit)
    elif args.command == "record-adapter":
        record_adapter(args.root, args.pointer)
    elif args.command == "latest-adapter":
        latest_adapter(args.root)
    elif args.command == "adapter-model-identity":
        adapter_model_identity(args.adapter, args.expected)
    elif args.command == "evaluation-complete":
        evaluation_complete(args.output, args.manifest)
    else:
        register_base(args.model, args.output, args.data_manifest)


if __name__ == "__main__":
    main()
