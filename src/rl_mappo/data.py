from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .mappo_types import RLSample


_MUSIQUE_HOPS = {"2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"}
_MUSIQUE_RARE_HOPS = {"4hop2", "4hop3"}


def sample_stratum(sample: RLSample) -> str:
    """Return the stable dataset-specific stratum used for sampling/holdout."""
    dataset = sample.dataset.casefold()
    qid_prefix = sample.qid.split("__", 1)[0].casefold()
    if dataset == "musique" and qid_prefix in _MUSIQUE_HOPS:
        return qid_prefix
    question_type = str(sample.metadata.get("question_type") or "").strip().casefold()
    level = str(sample.metadata.get("level") or "").strip().casefold()
    hop_count = str(sample.metadata.get("hop_count") or "").strip().casefold()
    if dataset == "hotpotqa" and (level or question_type):
        return "/".join(item for item in (level, question_type) if item)
    if question_type:
        return question_type
    if hop_count:
        return f"{hop_count}hop"
    return "default"


def _allocate_counts(
    sizes: dict[str, int], total: int, *, weights: dict[str, float] | None = None,
) -> dict[str, int]:
    """Allocate an exact bounded total proportionally across named strata."""
    total = min(total, sum(sizes.values()))
    if total <= 0:
        return {name: 0 for name in sizes}
    weights = weights or {}
    mass = {name: size * float(weights.get(name, 1.0)) for name, size in sizes.items()}
    denominator = sum(mass.values())
    raw = {name: total * mass[name] / denominator for name in sizes}
    result = {name: min(sizes[name], int(raw[name])) for name in sizes}
    remaining = total - sum(result.values())
    while remaining:
        candidates = [name for name in sizes if result[name] < sizes[name]]
        if not candidates:
            break
        candidates.sort(
            key=lambda name: (raw[name] - result[name], mass[name], name), reverse=True,
        )
        for name in candidates:
            if remaining == 0:
                break
            result[name] += 1
            remaining -= 1
    return result


def _select_dataset_samples(
    values: list[RLSample], *, dataset: str, limit: int | None,
    seed: int, musique_rare_hop_oversample_factor: float,
) -> list[RLSample]:
    if limit is None or limit >= len(values):
        result = list(values)
        random.Random(f"{seed}:mappo-sampling:{dataset}").shuffle(result)
        return result
    by_stratum: dict[str, list[RLSample]] = defaultdict(list)
    for sample in values:
        by_stratum[sample_stratum(sample)].append(sample)
    weights = {
        name: musique_rare_hop_oversample_factor
        if dataset.casefold() == "musique" and name in _MUSIQUE_RARE_HOPS else 1.0
        for name in by_stratum
    }
    counts = _allocate_counts(
        {name: len(items) for name, items in by_stratum.items()}, limit,
        weights=weights,
    )
    selected: list[RLSample] = []
    for name, items in sorted(by_stratum.items()):
        shuffled = list(items)
        random.Random(f"{seed}:mappo-sampling:{dataset}:{name}").shuffle(shuffled)
        selected.extend(shuffled[:counts[name]])
    random.Random(f"{seed}:mappo-sampling:{dataset}:selected").shuffle(selected)
    return selected


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            yield value


def load_samples(
    root: str | Path, *, max_per_dataset: int | None, max_total: int | None,
    seed: int, musique_rare_hop_oversample_factor: float = 1.0,
) -> list[RLSample]:
    root = Path(root)
    files = sorted(path for path in root.rglob("*.jsonl") if path.name != "corpus.jsonl")
    if not files:
        raise FileNotFoundError(f"No training JSONL files under {root}")
    by_dataset: dict[str, list[RLSample]] = defaultdict(list)
    for path in files:
        fallback = path.stem.replace("_train", "").replace("_rl", "")
        for row in _rows(path):
            dataset = str(row.get("dataset") or fallback).strip()
            qid = str(row.get("qid") or "").strip()
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer", row.get("gold_answer")) or "").strip()
            facts = row.get("supporting_facts")
            if not (dataset and qid and question and answer and isinstance(facts, list)):
                continue
            aliases = row.get("answer_aliases") or []
            metadata = dict(row.get("metadata") or {})
            for key in ("question_type", "hop_count"):
                if row.get(key) is not None and key not in metadata:
                    metadata[key] = row[key]
            sample = RLSample(
                qid=qid, dataset=dataset, question=question, answer=answer,
                answer_aliases=tuple(str(x) for x in aliases),
                supporting_facts=tuple(dict(x) for x in facts if isinstance(x, dict)),
                metadata=metadata,
            )
            by_dataset[dataset].append(sample)
    for dataset, values in list(by_dataset.items()):
        by_dataset[dataset] = _select_dataset_samples(
            values, dataset=dataset, limit=max_per_dataset, seed=seed,
            musique_rare_hop_oversample_factor=musique_rare_hop_oversample_factor,
        )
    selected: list[RLSample] = []
    names = sorted(by_dataset)
    while any(by_dataset.values()) and (max_total is None or len(selected) < max_total):
        for name in names:
            if by_dataset[name] and (max_total is None or len(selected) < max_total):
                selected.append(by_dataset[name].pop())
    if not selected:
        raise ValueError("No valid RL samples were loaded")
    return selected


def epoch_order(samples: list[RLSample], *, seed: int, epoch: int) -> list[RLSample]:
    result = list(samples)
    random.Random(seed + epoch).shuffle(result)
    return result


def split_train_validation(
    samples: list[RLSample], *, validation_ratio: float, seed: int,
    validation_samples_per_dataset: int | None = None,
) -> tuple[list[RLSample], list[RLSample]]:
    """Create a deterministic dataset-and-question-stratified holdout."""
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1)")
    if validation_ratio == 0.0 and not validation_samples_per_dataset:
        return list(samples), []
    by_dataset: dict[str, list[RLSample]] = defaultdict(list)
    for sample in samples:
        by_dataset[sample.dataset].append(sample)
    validation_ids: set[tuple[str, str]] = set()
    for dataset, values in sorted(by_dataset.items()):
        requested = (
            validation_samples_per_dataset
            if validation_samples_per_dataset is not None
            else max(1, int(len(values) * validation_ratio + 0.5))
        )
        count = min(requested, max(0, len(values) - 1))
        by_stratum: dict[str, list[RLSample]] = defaultdict(list)
        for sample in values:
            by_stratum[sample_stratum(sample)].append(sample)
        counts = _allocate_counts(
            {name: len(items) for name, items in by_stratum.items()}, count,
        )
        for name, candidates in sorted(by_stratum.items()):
            candidates = list(candidates)
            random.Random(
                f"{seed}:mappo-validation:{dataset}:{name}"
            ).shuffle(candidates)
            validation_ids.update(
                (item.dataset, item.qid) for item in candidates[:counts[name]]
            )
    train = [item for item in samples if (item.dataset, item.qid) not in validation_ids]
    validation = [item for item in samples if (item.dataset, item.qid) in validation_ids]
    if not train:
        raise ValueError("Validation split left no MAPPO training samples")
    return train, validation
