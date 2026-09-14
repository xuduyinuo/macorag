#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.fixed_manifest import DATASETS, build_fixed_manifest


PER_DATASET = 500
SEED = 20260905


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _valid(output_dir: Path) -> bool:
    manifest_path = output_dir / "manifest.jsonl"
    meta_path = output_dir / "manifest_meta.json"
    if not manifest_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = _read_json(meta_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (
        meta.get("seed") == SEED
        and meta.get("per_dataset") == PER_DATASET
        and meta.get("counts_by_dataset") == {dataset: PER_DATASET for dataset in DATASETS}
        and sum(1 for line in manifest_path.open("r", encoding="utf-8") if line.strip())
        == PER_DATASET * len(DATASETS)
    )


def _add_stratum_audit(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.jsonl"
    meta_path = output_dir / "manifest_meta.json"
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[str(row["dataset"])][str(row["sampling_stratum"])] += 1
    meta = _read_json(meta_path)
    meta["sampling_contract"] = {
        "method": "proportional_stratified_without_replacement",
        "2wiki": "question_type",
        "hotpotqa": "difficulty/question_type",
        "musique": "hop_composition_type",
    }
    meta["counts_by_dataset_and_stratum"] = {
        dataset: dict(counts[dataset]) for dataset in DATASETS
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed 500-per-dataset stratified evaluation set.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = repo_root / "ablation/data/eval_500_each"
    reused = not args.force and _valid(output_dir)
    if not reused:
        build_fixed_manifest(
            repo_root / "data/eval_1000_stratified_v2",
            output_dir,
            per_dataset=PER_DATASET,
            seed=SEED,
        )
    meta = _add_stratum_audit(output_dir)
    print(json.dumps({
        "manifest": str(output_dir / "manifest.jsonl"),
        "counts_by_dataset": meta["counts_by_dataset"],
        "reused": reused,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
