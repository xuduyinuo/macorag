from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvalSample:
    qid: str
    dataset: str
    question: str
    answer: str
    answer_aliases: list[str]
    supporting_facts: list[dict[str, Any]]
    metadata: dict[str, Any]


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
    return sorted(path for path in data_root.rglob("*.jsonl") if path.is_file() and path.name != "corpus.jsonl")


def _resolve_files(data_root: Path, data_files: list[str] | tuple[str, ...]) -> list[Path]:
    if data_files:
        paths = []
        for item in data_files:
            path = Path(item)
            if not path.is_absolute():
                path = data_root / path
            if path.is_dir():
                paths.extend(_candidate_files(path))
                continue
            if path.name == "corpus.jsonl":
                continue
            paths.append(path)
        return paths
    return _candidate_files(data_root)


def _build_sample(row: dict[str, Any], fallback_dataset: str) -> EvalSample | None:
    qid = str(row.get("qid") or "").strip()
    dataset = str(row.get("dataset") or fallback_dataset or "").strip()
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not answer:
        answer = str(row.get("gold_answer") or "").strip()
    supporting_facts = row.get("supporting_facts")
    if not qid or not dataset or not question or not answer or not isinstance(supporting_facts, list):
        return None
    aliases = row.get("answer_aliases")
    if not isinstance(aliases, list):
        aliases = []
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return EvalSample(
        qid=qid,
        dataset=dataset,
        question=question,
        answer=answer,
        answer_aliases=[str(item) for item in aliases if item is not None],
        supporting_facts=[item for item in supporting_facts if isinstance(item, dict)],
        metadata=metadata,
    )


def load_eval_samples(
    *,
    data_root: str | Path,
    data_files: list[str] | tuple[str, ...] | None = None,
    max_samples: int | None = None,
) -> tuple[list[EvalSample], dict[str, Any]]:
    root = Path(data_root)
    explicit_files = tuple(data_files or ())
    files = _resolve_files(root, explicit_files)
    if not files:
        raise FileNotFoundError(f"No evaluation jsonl files found under {root}")
    if explicit_files:
        missing_files = [path for path in files if not path.exists()]
        if missing_files:
            missing = ", ".join(str(path) for path in missing_files)
            raise FileNotFoundError(f"Evaluation data file not found: {missing}")

    samples: list[EvalSample] = []
    skipped = 0
    counts_by_dataset: dict[str, int] = {}
    source_files: list[str] = []
    for path in files:
        if max_samples is not None and len(samples) >= max_samples:
            break
        source_files.append(str(path))
        fallback_dataset = path.parent.name or path.stem.replace("_dev", "")
        for row in _read_jsonl(path):
            sample = _build_sample(row, fallback_dataset)
            if sample is None:
                skipped += 1
                continue
            if max_samples is None or len(samples) < max_samples:
                samples.append(sample)
                counts_by_dataset[sample.dataset] = counts_by_dataset.get(sample.dataset, 0) + 1
            if max_samples is not None and len(samples) >= max_samples:
                break

    if not samples:
        raise ValueError(f"No valid evaluation samples found in {root}")
    summary = {
        "data_root": str(root),
        "source_files": source_files,
        "loaded_samples": len(samples),
        "skipped_samples": skipped,
        "counts_by_dataset": counts_by_dataset,
        "max_samples": max_samples,
    }
    return samples, summary
