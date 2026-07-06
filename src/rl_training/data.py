from __future__ import annotations

import json
from dataclasses import dataclass
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

    def to_reward_sample(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "answer_aliases": self.answer_aliases,
            "supporting_facts": self.supporting_facts,
        }


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


def load_rl_samples(
    *,
    data_root: str | Path,
    data_files: list[str] | tuple[str, ...] | None = None,
    max_samples: int | None = None,
) -> tuple[list[RLSample], dict[str, Any]]:
    root = Path(data_root)
    files = _resolve_files(root, tuple(data_files or ()))
    if not files:
        raise FileNotFoundError(f"No RL jsonl files found under {root}")

    samples: list[RLSample] = []
    skipped = 0
    counts_by_dataset: dict[str, int] = {}
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
            dataset_count = counts_by_dataset.get(sample.dataset, 0)
            if max_samples is None or dataset_count < max_samples:
                samples.append(sample)
                counts_by_dataset[sample.dataset] = dataset_count + 1

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
    }
    return samples, summary
