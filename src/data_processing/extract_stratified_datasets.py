from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from data_processing.stratified_extraction import audit_existing_pair, extract_pair


DATASETS = ("2wiki", "hotpotqa", "musique")


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not resolve repository root")


def load_extraction_config(
    path: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Extraction config must be a mapping: {config_path}")
    root = Path(repo_root or _resolve_repo_root()).resolve()
    required = ("source_root", "output_root", "split", "seed", "expected_total", "datasets")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing extraction config keys: {', '.join(missing)}")
    split = str(payload["split"])
    if split not in {"train", "dev"}:
        raise ValueError(f"Extraction split must be train or dev, got {split!r}")
    datasets = payload["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError(f"datasets must contain exactly: {', '.join(DATASETS)}")
    expected_total = int(payload["expected_total"])
    normalized_datasets: dict[str, dict[str, dict[str, int]]] = {}
    for dataset in DATASETS:
        dataset_config = datasets[dataset]
        quotas = dataset_config.get("quotas") if isinstance(dataset_config, dict) else None
        if not isinstance(quotas, dict) or not quotas:
            raise ValueError(f"Missing quotas for dataset={dataset}")
        normalized_quotas = {str(key): int(value) for key, value in quotas.items()}
        if any(value < 0 for value in normalized_quotas.values()):
            raise ValueError(f"Negative quota for dataset={dataset}")
        total = sum(normalized_quotas.values())
        if total != expected_total:
            raise ValueError(
                f"dataset={dataset} quota total {total} != expected_total {expected_total}"
            )
        normalized_datasets[dataset] = {"quotas": normalized_quotas}

    def resolve(value: Any) -> Path:
        candidate = Path(str(value))
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    return {
        "schema_version": int(payload.get("schema_version", 1)),
        "repo_root": root,
        "source_root": resolve(payload["source_root"]),
        "output_root": resolve(payload["output_root"]),
        "split": split,
        "seed": int(payload["seed"]),
        "expected_total": expected_total,
        "datasets": normalized_datasets,
    }


def _dry_run_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": config["split"],
        "seed": config["seed"],
        "source_root": str(config["source_root"]),
        "output_root": str(config["output_root"]),
        "total_quota": sum(
            sum(dataset["quotas"].values()) for dataset in config["datasets"].values()
        ),
        "datasets": config["datasets"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract paired stratified MACORAG datasets")
    parser.add_argument("--train-config", default="config/extract_train.yml")
    parser.add_argument("--eval-config", default="config/extract_eval.yml")
    parser.add_argument("--repo-root", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--audit-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else _resolve_repo_root()
    train = load_extraction_config(Path(args.train_config), repo_root=repo_root)
    evaluation = load_extraction_config(Path(args.eval_config), repo_root=repo_root)
    if train["split"] != "train" or evaluation["split"] != "dev":
        raise ValueError("paired extraction requires train split followed by dev split")
    if train["seed"] != evaluation["seed"]:
        raise ValueError("paired extraction configs must use the same seed")
    if train["source_root"] != evaluation["source_root"]:
        raise ValueError("paired extraction configs must use the same source_root")

    if args.dry_run:
        result = {
            "train": _dry_run_summary(train),
            "evaluation": _dry_run_summary(evaluation),
        }
    elif args.audit_existing:
        result = audit_existing_pair(train, evaluation)
    else:
        result = extract_pair(train, evaluation)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
