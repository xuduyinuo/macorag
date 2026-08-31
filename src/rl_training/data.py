from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RLSample:
    qid: str
    dataset: str
    question: str
    answer: str
    answer_aliases: list[str]
    supporting_facts: list[dict[str, Any]]
    context_doc_ids: list[str]
    metadata: dict[str, Any]
    sampling_stratum: str = ""

    def to_reward_sample(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "answer_aliases": self.answer_aliases,
            "supporting_facts": self.supporting_facts,
        }


STRATA_BY_DATASET: dict[str, tuple[str, ...]] = {
    "2wiki": ("compositional", "comparison", "bridge_comparison", "inference"),
    "hotpotqa": ("hard/bridge", "hard/comparison"),
    "musique": ("2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"),
}


def _derive_sampling_seed(seed: int, *parts: str) -> int:
    payload = "\0".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_proportional_prefix(
    samples: list[RLSample],
    *,
    max_samples: int | None,
    seed: int,
) -> list[RLSample]:
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None.")
    if not samples:
        return []

    dataset = samples[0].dataset
    if any(sample.dataset != dataset for sample in samples):
        raise ValueError("Proportional prefix requires exactly one dataset.")
    canonical = STRATA_BY_DATASET.get(dataset)
    if canonical is None:
        raise ValueError(f"Unsupported proportional sampling dataset: {dataset}")

    buckets: dict[str, list[RLSample]] = {stratum: [] for stratum in canonical}
    for sample in samples:
        if sample.sampling_stratum not in buckets:
            raise ValueError(
                f"Unknown sampling stratum for {dataset}: {sample.sampling_stratum!r}"
            )
        buckets[sample.sampling_stratum].append(sample)
    source_counts = {stratum: len(bucket) for stratum, bucket in buckets.items()}
    for stratum, bucket in buckets.items():
        random.Random(_derive_sampling_seed(seed, dataset, stratum)).shuffle(bucket)

    total = len(samples)
    selected_counts = {stratum: 0 for stratum in canonical}
    schedule: list[RLSample] = []
    for position in range(1, total + 1):
        available = [stratum for stratum in canonical if buckets[stratum]]
        chosen = max(
            available,
            key=lambda stratum: (
                position * source_counts[stratum]
                - selected_counts[stratum] * total,
                -canonical.index(stratum),
            ),
        )
        schedule.append(buckets[chosen].pop())
        selected_counts[chosen] += 1

    limit = total if max_samples is None else min(max_samples, total)
    return schedule[:limit]


def select_balanced_samples(
    samples: list[RLSample],
    *,
    max_total_samples: int | None,
    seed: int,
) -> list[RLSample]:
    if max_total_samples is not None and max_total_samples < 0:
        raise ValueError("max_total_samples must be non-negative or None.")
    buckets: dict[str, list[RLSample]] = defaultdict(list)
    for sample in samples:
        buckets[sample.dataset].append(sample)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    limit = len(samples) if max_total_samples is None else min(max_total_samples, len(samples))
    selected: list[RLSample] = []
    dataset_names = sorted(buckets)
    while len(selected) < limit:
        added = False
        for dataset in dataset_names:
            bucket = buckets[dataset]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
                added = True
        if not added:
            break
    return selected


def epoch_sample_order(
    samples: list[RLSample],
    *,
    seed: int,
    epoch: int,
) -> list[RLSample]:
    ordered = list(samples)
    random.Random(seed + epoch).shuffle(ordered)
    return ordered


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _candidate_files(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return sorted(
        path
        for path in data_root.rglob("*.jsonl")
        if path.is_file() and path.name != "corpus.jsonl"
    )


def _resolve_files(data_root: Path, data_files: list[str] | tuple[str, ...]) -> list[Path]:
    if data_files:
        paths = []
        for item in data_files:
            path = Path(item)
            if not path.is_absolute():
                path = data_root / path
            paths.append(path)
        return paths
    return _candidate_files(data_root)


def _build_sample(row: dict[str, Any], fallback_dataset: str) -> RLSample | None:
    qid = str(row.get("qid") or "").strip()
    dataset = str(row.get("dataset") or fallback_dataset or "").strip()
    question = str(row.get("question") or "").strip()
    answer = row.get("answer", row.get("gold_answer"))
    answer = str(answer or "").strip()
    supporting_facts = row.get("supporting_facts")
    if not qid or not dataset or not question or not answer or not isinstance(supporting_facts, list):
        return None
    aliases = row.get("answer_aliases")
    if not isinstance(aliases, list):
        aliases = []
    context_doc_ids = row.get("context_doc_ids")
    if not isinstance(context_doc_ids, list):
        context_doc_ids = []
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return RLSample(
        qid=qid,
        dataset=dataset,
        question=question,
        answer=answer,
        answer_aliases=[str(item) for item in aliases if item is not None],
        supporting_facts=[item for item in supporting_facts if isinstance(item, dict)],
        context_doc_ids=[str(item) for item in context_doc_ids if item is not None],
        metadata=metadata,
    )


def sampling_stratum(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "2wiki":
        value = str(row.get("question_type") or "").strip()
    elif dataset == "hotpotqa":
        level = str((row.get("metadata") or {}).get("level") or "").strip()
        question_type = str(row.get("question_type") or "").strip()
        value = f"{level}/{question_type}" if level and question_type else ""
    elif dataset == "musique":
        qid = str(row.get("qid") or "")
        value = qid.split("__", 1)[0] if "__" in qid else ""
    else:
        raise ValueError(f"Unsupported proportional sampling dataset: {dataset}")
    if not value:
        raise ValueError(
            f"Missing sampling stratum for {dataset}/{row.get('qid', '')}"
        )
    if value not in STRATA_BY_DATASET[dataset]:
        raise ValueError(f"Unknown sampling stratum for {dataset}: {value!r}")
    return value


_sampling_stratum = sampling_stratum


def load_rl_samples(
    *,
    data_root: str | Path,
    data_files: list[str] | tuple[str, ...] | None = None,
    max_samples: int | None = None,
    data_sampling_strategy: str = "head",
    data_sampling_seed: int = 20260826,
) -> tuple[list[RLSample], dict[str, Any]]:
    if data_sampling_strategy not in {"head", "proportional_stratified"}:
        raise ValueError(
            "data_sampling_strategy must be 'head' or 'proportional_stratified'; "
            f"got {data_sampling_strategy!r}."
        )
    if type(data_sampling_seed) is not int:
        raise TypeError("data_sampling_seed must be an integer.")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None.")

    root = Path(data_root)
    files = _resolve_files(root, tuple(data_files or ()))
    if not files:
        raise FileNotFoundError(f"No RL jsonl files found under {root}")

    samples: list[RLSample] = []
    skipped = 0
    counts_by_dataset: dict[str, int] = {}
    all_by_dataset: dict[str, list[RLSample]] = defaultdict(list)
    source_files: list[str] = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"RL data file not found: {path}")
        source_files.append(str(path))
        fallback_dataset = path.stem.replace("_rl", "").replace("_train", "")
        for row in _read_jsonl(path):
            sample = _build_sample(row, fallback_dataset)
            if sample is None:
                skipped += 1
                continue
            if data_sampling_strategy == "proportional_stratified":
                sample = replace(
                    sample,
                    sampling_stratum=_sampling_stratum(sample.dataset, row),
                )
                all_by_dataset[sample.dataset].append(sample)
                continue
            dataset_count = counts_by_dataset.get(sample.dataset, 0)
            if max_samples is None or dataset_count < max_samples:
                samples.append(sample)
                counts_by_dataset[sample.dataset] = dataset_count + 1

    counts_by_dataset_and_stratum: dict[str, dict[str, int]] = {}
    if data_sampling_strategy == "proportional_stratified":
        for dataset in sorted(all_by_dataset):
            selected = select_proportional_prefix(
                all_by_dataset[dataset],
                max_samples=max_samples,
                seed=data_sampling_seed,
            )
            samples.extend(selected)
            counts_by_dataset[dataset] = len(selected)
            counts_by_dataset_and_stratum[dataset] = dict(
                Counter(sample.sampling_stratum for sample in selected)
            )

    if not samples:
        raise ValueError(f"No valid RL samples found in {root}")
    summary = {
        "data_root": str(root),
        "source_files": source_files,
        "loaded_samples": len(samples),
        "skipped_samples": skipped,
        "counts_by_dataset": counts_by_dataset,
        "max_samples": max_samples,
        "max_samples_per_dataset": max_samples,
        "data_sampling_strategy": data_sampling_strategy,
        "data_sampling_seed": data_sampling_seed,
        "counts_by_dataset_and_stratum": counts_by_dataset_and_stratum,
    }
    return samples, summary
