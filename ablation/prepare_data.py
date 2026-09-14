#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from rl_training.data import load_rl_samples
from sft_training.data import build_training_data


QUOTAS = {"2wiki": 400, "hotpotqa": 400, "musique": 200}
SEED = 20260905


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise SystemExit(f"Expected JSON object in {path}")
                rows.append(payload)
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            digest.update(encoded)
            handle.write(encoded.decode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _line_count(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _is_prepared(repo_root: Path) -> bool:
    manifest_path = repo_root / "ablation/data/manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("contract") != "macorag_framework_ablation_1000":
        return False
    if manifest.get("selection_seed") != SEED or manifest.get("quotas") != QUOTAS:
        return False
    for dataset, quota in QUOTAS.items():
        rl_path = repo_root / "ablation/data/rl_train_1000" / dataset / f"{dataset}_train.jsonl"
        sft_path = repo_root / "ablation/data/sft_train_1000" / f"{dataset}_sft.jsonl"
        if _line_count(rl_path) != quota or _line_count(sft_path) != quota:
            return False
    return True


def _select_rl(repo_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = repo_root / "data/rl_train_2000_stratified_v2"
    summary: dict[str, Any] = {}
    for dataset, quota in QUOTAS.items():
        source_path = source_root / dataset / f"{dataset}_train.jsonl"
        selected, selection = load_rl_samples(
            data_root=source_root,
            data_files=[str(source_path)],
            max_samples=quota,
            data_sampling_strategy="proportional_stratified",
            data_sampling_seed=SEED,
        )
        selected_qids = {sample.qid for sample in selected}
        source_rows = _read_jsonl(source_path)
        rows_by_qid = {str(row.get("qid") or ""): row for row in source_rows}
        output_rows = [rows_by_qid[sample.qid] for sample in selected]
        if len(output_rows) != quota or len(selected_qids) != quota:
            raise SystemExit(f"RL quota failure for {dataset}: expected {quota}")
        output_path = output_root / "rl_train_1000" / dataset / f"{dataset}_train.jsonl"
        summary[dataset] = {
            "count": len(output_rows),
            "sha256": _write_jsonl_atomic(output_path, output_rows),
            "source": str(source_path.relative_to(repo_root)),
            "output": str(output_path.relative_to(repo_root)),
            "strata": selection.get("counts_by_dataset_and_stratum", {}).get(dataset, {}),
        }
    return summary


def _select_sft(repo_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = repo_root / "data/sft/teacher_qwen_plus_trajectory_train_v2"
    selected = build_training_data(
        source_root,
        max_samples_by_dataset=QUOTAS,
        data_sampling_seed=SEED,
    )
    selected_qids = {
        dataset: {sample.qid for sample in selected.samples if sample.dataset == dataset}
        for dataset in QUOTAS
    }
    summary: dict[str, Any] = {}
    target_root = output_root / "sft_train_1000"
    for dataset, quota in QUOTAS.items():
        source_path = source_root / f"{dataset}_sft.jsonl"
        output_rows = [
            row for row in _read_jsonl(source_path)
            if str(row.get("qid") or "") in selected_qids[dataset]
        ]
        if len(output_rows) != quota or len(selected_qids[dataset]) != quota:
            raise SystemExit(f"SFT quota failure for {dataset}: expected {quota}")
        output_path = target_root / f"{dataset}_sft.jsonl"
        summary[dataset] = {
            "count": len(output_rows),
            "sha256": _write_jsonl_atomic(output_path, output_rows),
            "source": str(source_path.relative_to(repo_root)),
            "output": str(output_path.relative_to(repo_root)),
        }
    for filename in ("run_config.json", "summary.json"):
        payload = json.loads((source_root / filename).read_text(encoding="utf-8"))
        _write_json_atomic(target_root / filename, payload)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed 400/400/200 ablation datasets.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true", help="Rebuild an already valid fixed split.")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_root = repo_root / "ablation/data"
    if not args.force and _is_prepared(repo_root):
        print(json.dumps({"manifest": str(output_root / "manifest.json"), "quotas": QUOTAS, "reused": True}, ensure_ascii=False))
        return
    manifest = {
        "schema_version": 1,
        "contract": "macorag_framework_ablation_1000",
        "selection_seed": SEED,
        "quotas": QUOTAS,
        "total": sum(QUOTAS.values()),
        "rl": _select_rl(repo_root, output_root),
        "sft": _select_sft(repo_root, output_root),
    }
    _write_json_atomic(output_root / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output_root / "manifest.json"), "quotas": QUOTAS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
