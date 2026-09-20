from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .mappo_types import RLSample


DATASET_ALIASES = {
    "2wikimultihopqa": "2wiki",
    "2wiki": "2wiki",
    "hotpotqa": "hotpotqa",
    "musique": "musique",
}


def sample_stratum(sample: RLSample) -> str:
    sampling = sample.metadata.get("sampling") or {}
    value = sampling.get("stratum") if isinstance(sampling, dict) else None
    if isinstance(value, list):
        return "/".join(str(item) for item in value)
    return str(value or (sampling.get("difficulty") if isinstance(sampling, dict) else "") or "default")


def _rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            yield line_number, row


def _supporting_facts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("supporting_evidence must be a non-empty list")
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("every supporting_evidence item must be an object")
        title = str(item.get("title") or "").strip()
        text = str(item.get("text") or "").strip()
        if not title or not text:
            raise ValueError("supporting evidence requires title and text")
        result.append({
            "title": title,
            "text": text,
            "corpus_doc_ids": [str(x) for x in item.get("corpus_doc_ids", [])],
            "evidence_index": item.get("evidence_index"),
        })
    return tuple(result)


def load_split(path: str | Path) -> list[RLSample]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"RL split not found: {source}")
    samples: list[RLSample] = []
    seen: set[str] = set()
    for line_number, row in _rows(source):
        qid = str(row.get("id") or "").strip()
        dataset_source = str(row.get("dataset") or "").strip().casefold()
        dataset = DATASET_ALIASES.get(dataset_source)
        question = str(row.get("question") or "").strip()
        answers = row.get("golden_answers")
        if not qid or not dataset or not question:
            raise ValueError(f"Invalid id/dataset/question in {source}:{line_number}")
        if qid in seen:
            raise ValueError(f"Duplicate sample id {qid!r} in {source}:{line_number}")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"golden_answers must be non-empty in {source}:{line_number}")
        normalized_answers = tuple(str(item).strip() for item in answers if str(item).strip())
        if not normalized_answers:
            raise ValueError(f"golden_answers contains no usable answer in {source}:{line_number}")
        metadata = dict(row.get("metadata") or {})
        metadata.update({
            "source_dataset": dataset_source,
            "source_id": row.get("source_id"),
            "source_split": row.get("source_split"),
            "sampling": row.get("sampling") or {},
        })
        samples.append(RLSample(
            qid=qid,
            dataset=dataset,
            question=question,
            answer=normalized_answers[0],
            answer_aliases=normalized_answers[1:],
            supporting_facts=_supporting_facts(row.get("supporting_evidence")),
            metadata=metadata,
        ))
        seen.add(qid)
    if not samples:
        raise ValueError(f"No RL samples found in {source}")
    return samples


def shuffled_training_samples(
    path: str | Path, *, seed: int, limit: int | None = None,
) -> list[RLSample]:
    """Load and shuffle before applying any optional debug limit."""
    samples = load_split(path)
    random.Random(seed).shuffle(samples)
    return samples if limit is None else samples[:limit]


def stratified_validation_samples(
    path: str | Path, *, seed: int, limit: int | None = None,
) -> list[RLSample]:
    """Load a deterministic proportional subset over dataset and stratum.

    Largest-remainder allocation preserves the fixed validation split's
    dataset/difficulty mixture as closely as an integer-sized subset permits.
    Selected rows are returned in source order so validation reporting remains
    stable and easy to compare across runs.
    """
    samples = load_split(path)
    if limit is None or limit >= len(samples):
        return samples
    if limit <= 0:
        raise ValueError("validation limit must be positive")

    groups: dict[tuple[str, str], list[tuple[int, RLSample]]] = {}
    for index, sample in enumerate(samples):
        groups.setdefault((sample.dataset, sample_stratum(sample)), []).append(
            (index, sample)
        )

    def proportional_quotas(
        sizes: dict[Any, int], requested: int,
    ) -> dict[Any, int]:
        population = sum(sizes.values())
        exact = {
            key: requested * size / population for key, size in sizes.items()
        }
        result = {key: int(value) for key, value in exact.items()}
        remainder_order = sorted(
            sizes,
            key=lambda key: (-(exact[key] - result[key]), key),
        )
        for key in remainder_order[:requested - sum(result.values())]:
            result[key] += 1
        return result

    dataset_sizes = Counter(item.dataset for item in samples)
    dataset_quotas = proportional_quotas(dict(dataset_sizes), limit)
    quotas: dict[tuple[str, str], int] = {}
    for dataset, dataset_quota in dataset_quotas.items():
        dataset_groups = {
            key: len(rows) for key, rows in groups.items() if key[0] == dataset
        }
        quotas.update(proportional_quotas(dataset_groups, dataset_quota))

    chosen: list[tuple[int, RLSample]] = []
    for key, rows in groups.items():
        ranked = sorted(
            rows,
            key=lambda item: hashlib.sha256(
                f"{seed}:rl-v2:validation:{item[1].qid}".encode("utf-8")
            ).digest(),
        )
        chosen.extend(ranked[:quotas[key]])
    chosen.sort(key=lambda item: item[0])
    return [sample for _, sample in chosen]


def epoch_order(samples: list[RLSample], *, seed: int, epoch: int) -> list[RLSample]:
    result = list(samples)
    random.Random(f"{seed}:rl-v2:epoch:{epoch}").shuffle(result)
    return result


def split_manifest(samples: list[RLSample], path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "count": len(samples),
        "datasets": dict(sorted(Counter(item.dataset for item in samples).items())),
        "ids": [item.qid for item in samples],
    }
