from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class SourceExample:
    qid: str
    question: str
    answers: tuple[str, ...]
    dataset: str
    source_dataset: str
    split: str
    metadata: dict[str, Any]

    @property
    def primary_answer(self) -> str:
        return self.answers[0]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def normalize_source_row(row: dict[str, Any], aliases: dict[str, str]) -> SourceExample:
    qid = str(row.get("id") or row.get("qid") or "").strip()
    question = str(row.get("question") or "").strip()
    source_dataset = str(row.get("dataset") or "").strip().lower()
    dataset = aliases.get(source_dataset, source_dataset)
    raw_answers = row.get("golden_answers")
    if raw_answers is None:
        raw_answers = [row.get("answer"), *(row.get("answer_aliases") or [])]
    answers = tuple(dict.fromkeys(str(item).strip() for item in raw_answers if str(item or "").strip()))
    if not qid or not question or not answers or not dataset:
        raise ValueError(f"Invalid source row qid={qid!r}: id, question, dataset and answers are required")
    return SourceExample(
        qid=qid,
        question=question,
        answers=answers,
        dataset=dataset,
        source_dataset=source_dataset,
        split=str(row.get("source_split") or row.get("split") or "train"),
        metadata={
            "source_id": row.get("source_id"),
            "sampling": row.get("sampling") or {},
            "metadata": row.get("metadata") or {},
            "supporting_evidence_titles": row.get("supporting_evidence_titles") or [],
        },
    )


def load_source_examples(
    path: str | Path,
    *,
    aliases: dict[str, str],
    limits: dict[str, int] | None = None,
    total_limit: int | None = None,
) -> list[SourceExample]:
    examples: list[SourceExample] = []
    counts: Counter[str] = Counter()
    qids: set[str] = set()
    for row in iter_jsonl(path):
        example = normalize_source_row(row, aliases)
        if example.qid in qids:
            raise ValueError(f"Duplicate source qid: {example.qid}")
        qids.add(example.qid)
        if limits and example.dataset not in limits:
            continue
        if limits and counts[example.dataset] >= int(limits[example.dataset]):
            continue
        examples.append(example)
        counts[example.dataset] += 1
        if total_limit is not None and len(examples) >= total_limit:
            break
    if not examples:
        raise ValueError(f"No usable source examples found in {path}")
    if limits and total_limit is None:
        missing = {dataset: limit - counts[dataset] for dataset, limit in limits.items() if counts[dataset] < limit}
        if missing:
            raise ValueError(f"Source does not satisfy candidate limits: {missing}")
    return examples


def source_summary(examples: list[SourceExample]) -> dict[str, Any]:
    counts = Counter(item.dataset for item in examples)
    return {
        "total": len(examples),
        "datasets": dict(sorted(counts.items())),
        "unique_qids": len({item.qid for item in examples}),
        "multi_answer_examples": sum(len(item.answers) > 1 for item in examples),
    }
