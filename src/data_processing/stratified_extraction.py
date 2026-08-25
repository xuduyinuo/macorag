from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_processing.io_utils import read_jsonl


@dataclass(frozen=True)
class SelectionResult:
    rows: list[dict[str, Any]]
    source_indices: list[int]
    qids: list[str]
    quota_actual: dict[str, int]
    eligible_count: int
    excluded_by_reason: dict[str, int]


def normalize_question(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value).casefold()).split())


def derive_seed(seed: int, *parts: str) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def eligibility_error(row: dict[str, Any], *, split: str) -> str | None:
    if str(row.get("split") or "") != split:
        return "wrong_split"
    if not str(row.get("qid") or "").strip():
        return "missing_qid"
    if not str(row.get("question") or "").strip():
        return "missing_question"
    if not str(row.get("answer") or "").strip():
        return "missing_answer"
    if not isinstance(row.get("supporting_facts"), list):
        return "invalid_supporting_facts"
    if split == "train" and row.get("usable_for_sft") is not True:
        return "not_usable_for_sft"
    if row.get("usable_for_retrieval_eval") is not True:
        return "not_usable_for_retrieval_eval"
    if row.get("quality_flags"):
        return "quality_flags"
    return None


def stratum_key(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "2wiki":
        return str(row.get("question_type") or "")
    if dataset == "hotpotqa":
        level = str((row.get("metadata") or {}).get("level") or "")
        return f"{level}/{row.get('question_type') or ''}"
    if dataset == "musique":
        return str(row.get("qid") or "").split("__", 1)[0]
    raise ValueError(f"Unsupported dataset: {dataset}")


def select_rows(
    *,
    source_path: Path,
    dataset: str,
    split: str,
    quotas: dict[str, int],
    seed: int,
) -> SelectionResult:
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    seen_questions: set[str] = set()

    for source_index, row in enumerate(read_jsonl(source_path)):
        error = eligibility_error(row, split=split)
        if error is not None:
            excluded[error] += 1
            continue
        normalized = normalize_question(str(row.get("question") or ""))
        if normalized in seen_questions:
            excluded["duplicate_question"] += 1
            continue
        seen_questions.add(normalized)
        stratum = stratum_key(dataset, row)
        if stratum not in quotas:
            excluded["unconfigured_stratum"] += 1
            continue
        buckets[stratum].append((source_index, row))

    selected: list[tuple[int, dict[str, Any]]] = []
    actual: dict[str, int] = {}
    for stratum, required in quotas.items():
        if not isinstance(required, int) or required < 0:
            raise ValueError(f"Invalid quota for {dataset}/{stratum}: {required!r}")
        candidates = buckets.get(stratum, [])
        if len(candidates) < required:
            raise ValueError(
                f"Underfilled stratum {dataset}/{stratum}: "
                f"required={required}, available={len(candidates)}"
            )
        rng = random.Random(derive_seed(seed, dataset, split, stratum))
        chosen = rng.sample(candidates, required)
        selected.extend(chosen)
        actual[stratum] = len(chosen)

    random.Random(derive_seed(seed, dataset, split, "final")).shuffle(selected)
    return SelectionResult(
        rows=[row for _, row in selected],
        source_indices=[index for index, _ in selected],
        qids=[str(row["qid"]) for _, row in selected],
        quota_actual=actual,
        eligible_count=sum(len(bucket) for bucket in buckets.values()),
        excluded_by_reason=dict(sorted(excluded.items())),
    )
