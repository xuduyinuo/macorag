from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_processing.io_utils import read_json, read_jsonl, write_json, write_jsonl


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


def required_doc_ids(rows: list[dict[str, Any]]) -> set[str]:
    required: set[str] = set()
    for row in rows:
        for value in row.get("context_doc_ids") or []:
            doc_id = str(value or "").strip()
            if doc_id:
                required.add(doc_id)
        for fact in row.get("supporting_facts") or []:
            if not isinstance(fact, dict):
                continue
            doc_id = str(fact.get("doc_id") or "").strip()
            if doc_id:
                required.add(doc_id)
    return required


def scope_corpus(source: Path, required: set[str]) -> list[dict[str, Any]]:
    scoped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(source):
        doc_id = str(row.get("doc_id") or "")
        if doc_id not in required:
            continue
        if doc_id in seen:
            raise ValueError(f"duplicate corpus doc_id: {doc_id}")
        seen.add(doc_id)
        scoped.append(row)
    missing = sorted(required - seen)
    if missing:
        raise ValueError(f"missing required corpus docs: {', '.join(missing)}")
    return scoped


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Source path is outside repository root: {path}") from exc


def write_dataset_output(
    *,
    output_root: Path,
    repo_root: Path,
    dataset: str,
    split: str,
    selection: SelectionResult,
    source_examples: Path,
    source_corpus: Path,
    quotas: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    examples_path = dataset_dir / f"{dataset}_{split}.jsonl"
    corpus_path = dataset_dir / "corpus.jsonl"
    summary_path = dataset_dir / "extract_summary.json"

    required = required_doc_ids(selection.rows)
    corpus_rows = scope_corpus(source_corpus, required)
    write_jsonl(examples_path, selection.rows)
    write_jsonl(corpus_path, corpus_rows)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "source_examples": _repo_relative(source_examples, repo_root),
        "source_corpus": _repo_relative(source_corpus, repo_root),
        "source_sha256": {
            "examples": sha256_file(source_examples),
            "corpus": sha256_file(source_corpus),
        },
        "output_examples": examples_path.relative_to(output_root).as_posix(),
        "output_corpus": corpus_path.relative_to(output_root).as_posix(),
        "configured_quota": dict(quotas),
        "actual_quota": dict(selection.quota_actual),
        "eligible_count": selection.eligible_count,
        "excluded_by_reason": dict(selection.excluded_by_reason),
        "selected_source_indices": list(selection.source_indices),
        "selected_qids": list(selection.qids),
        "example_count": len(selection.rows),
        "unique_qid_count": len(set(selection.qids)),
        "unique_question_count": len(
            {normalize_question(str(row.get("question") or "")) for row in selection.rows}
        ),
        "required_corpus_count": len(required),
        "corpus_count": len(corpus_rows),
        "output_sha256": {
            "examples": sha256_file(examples_path),
            "corpus": sha256_file(corpus_path),
        },
    }
    write_json(summary_path, summary)
    return summary


def validate_dataset_output(
    dataset_dir: Path,
    *,
    dataset: str,
    split: str,
    quotas: dict[str, int],
) -> dict[str, Any]:
    examples_path = dataset_dir / f"{dataset}_{split}.jsonl"
    corpus_path = dataset_dir / "corpus.jsonl"
    summary_path = dataset_dir / "extract_summary.json"
    summary = read_json(summary_path)

    if sha256_file(examples_path) != summary["output_sha256"]["examples"]:
        raise ValueError(f"example SHA256 mismatch for {dataset}/{split}")
    if sha256_file(corpus_path) != summary["output_sha256"]["corpus"]:
        raise ValueError(f"corpus SHA256 mismatch for {dataset}/{split}")

    rows = list(read_jsonl(examples_path))
    corpus_rows = list(read_jsonl(corpus_path))
    errors = Counter(
        error
        for row in rows
        if (error := eligibility_error(row, split=split)) is not None
    )
    if errors:
        raise ValueError(f"ineligible output rows for {dataset}/{split}: {dict(errors)}")

    qids = [str(row.get("qid") or "") for row in rows]
    questions = [normalize_question(str(row.get("question") or "")) for row in rows]
    if len(set(qids)) != len(qids):
        raise ValueError(f"duplicate qids in {dataset}/{split}")
    if len(set(questions)) != len(questions):
        raise ValueError(f"duplicate normalized questions in {dataset}/{split}")

    actual = Counter(stratum_key(dataset, row) for row in rows)
    if dict(actual) != dict(quotas):
        raise ValueError(
            f"quota mismatch for {dataset}/{split}: expected={dict(quotas)}, actual={dict(actual)}"
        )

    required = required_doc_ids(rows)
    corpus_ids = [str(row.get("doc_id") or "") for row in corpus_rows]
    if len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError(f"duplicate corpus ids in {dataset}/{split}")
    if set(corpus_ids) != required:
        missing = sorted(required - set(corpus_ids))
        extra = sorted(set(corpus_ids) - required)
        raise ValueError(
            f"scoped corpus mismatch for {dataset}/{split}: missing={missing}, extra={extra}"
        )

    audit = {
        "dataset": dataset,
        "split": split,
        "example_count": len(rows),
        "unique_qid_count": len(set(qids)),
        "unique_question_count": len(set(questions)),
        "actual_quota": dict(actual),
        "required_corpus_count": len(required),
        "corpus_count": len(corpus_rows),
        "examples_sha256": sha256_file(examples_path),
        "corpus_sha256": sha256_file(corpus_path),
    }
    expected_summary_fields = {
        "example_count": audit["example_count"],
        "unique_qid_count": audit["unique_qid_count"],
        "unique_question_count": audit["unique_question_count"],
        "required_corpus_count": audit["required_corpus_count"],
        "corpus_count": audit["corpus_count"],
    }
    for key, value in expected_summary_fields.items():
        if summary.get(key) != value:
            raise ValueError(f"summary {key} mismatch for {dataset}/{split}")
    return audit
