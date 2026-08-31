from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rl_training.data import STRATA_BY_DATASET, sampling_stratum


DATASETS = ("2wiki", "hotpotqa", "musique")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seed(seed: int, *parts: str) -> int:
    value = "\0".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _read_rows(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*.jsonl")):
        if path.name == "corpus.jsonl":
            continue
        content = path.read_bytes()
        hashes[str(path.relative_to(root))] = _hash_bytes(content)
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = str(row.get("dataset") or path.parent.name or path.stem).strip()
            if dataset in DATASETS:
                grouped[dataset].append(row)
    return grouped, hashes


def _select_dataset(rows: list[dict[str, Any]], dataset: str, limit: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATA_BY_DATASET[dataset]}
    for row in rows:
        buckets[sampling_stratum(dataset, row)].append(dict(row))
    source_counts = {name: len(bucket) for name, bucket in buckets.items()}
    if sum(source_counts.values()) < limit:
        raise ValueError(f"Not enough {dataset} rows for fixed validation: {sum(source_counts.values())} < {limit}")
    for name, bucket in buckets.items():
        random.Random(_seed(seed, dataset, name)).shuffle(bucket)
    selected_counts = Counter()
    selected: list[dict[str, Any]] = []
    total = sum(source_counts.values())
    canonical = STRATA_BY_DATASET[dataset]
    for position in range(1, limit + 1):
        available = [name for name in canonical if buckets[name]]
        chosen = max(
            available,
            key=lambda name: (
                position * source_counts[name] - selected_counts[name] * total,
                -canonical.index(name),
            ),
        )
        row = buckets[chosen].pop()
        row["dataset"] = dataset
        row["sampling_stratum"] = chosen
        selected.append(row)
        selected_counts[chosen] += 1
    return selected


def build_fixed_manifest(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    per_dataset: int = 100,
    seed: int = 20260831,
) -> dict[str, Any]:
    root = Path(data_root)
    output = Path(output_dir)
    grouped, source_hashes = _read_rows(root)
    selected = [
        row
        for dataset in DATASETS
        for row in _select_dataset(grouped.get(dataset, []), dataset, int(per_dataset), int(seed))
    ]
    qids = [str(row.get("qid") or "").strip() for row in selected]
    if any(not qid for qid in qids) or len(set(qids)) != len(qids):
        raise ValueError("Fixed validation requires unique non-empty qids.")
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in selected
    )
    meta = {
        "seed": int(seed),
        "per_dataset": int(per_dataset),
        "counts_by_dataset": dict(Counter(str(row["dataset"]) for row in selected)),
        "source_hashes": source_hashes,
        "manifest_fingerprint": _hash_bytes(serialized.encode("utf-8")),
        "qids": qids,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.jsonl").write_text(serialized, encoding="utf-8")
    (output / "manifest_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/eval_1000_stratified_v2")
    parser.add_argument("--output-dir", default="data/eval_300_grpo_fixed")
    parser.add_argument("--per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args(argv)
    meta = build_fixed_manifest(args.data_root, args.output_dir, per_dataset=args.per_dataset, seed=args.seed)
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
