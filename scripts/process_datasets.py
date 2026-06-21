#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from macorag.dataset_builders import (
    build_2wiki_example_from_row,
    build_hotpot_example_from_row,
    build_musique_canonical_from_rows,
)
from macorag.io_utils import normalize_key, normalize_text, sha1_text, write_json, write_jsonl
from macorag.linearrag_adapter import build_linearrag_dataset
from macorag.schemas import CorpusDoc, Example


DATASETS = ("hotpotqa", "2wiki", "musique")
SPLITS = ("train", "dev")


class DatasetProcessingError(RuntimeError):
    pass


def _doc_id(dataset: str, title: str, text: str) -> str:
    key = f"{normalize_key(title)} {normalize_text(text)}"
    return f"{dataset}:{sha1_text(key)}"


def _sentence_records(sentences: list[str]) -> list[dict[str, Any]]:
    return [
        {"sent_id": sent_id, "text": sentence}
        for sent_id, sentence in enumerate(sentences)
    ]


def _corpus_record(dataset: str, index: int, doc: CorpusDoc) -> dict[str, Any]:
    return {
        "chunk_id": f"{dataset}:chunk:{index}",
        "doc_id": doc.doc_id,
        "dataset": doc.dataset,
        "title": doc.title,
        "text": doc.text,
        "sentences": _sentence_records(doc.sentences),
    }


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _jsonable(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return _jsonable(value.as_py())
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _nested_or_flat(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    prefix = f"{key}."
    nested = {
        column[len(prefix) :]: row[column]
        for column in row
        if column.startswith(prefix)
    }
    return nested or None


def _normalize_qa_row(row: dict[str, Any]) -> dict[str, Any]:
    row = _jsonable(row)
    normalized = dict(row)
    supporting_facts = _nested_or_flat(row, "supporting_facts")
    context = _nested_or_flat(row, "context")
    if supporting_facts is not None:
        normalized["supporting_facts"] = supporting_facts
    if context is not None:
        normalized["context"] = context
    if normalized.get("evidences") is None:
        normalized["evidences"] = []
    return normalized


def _read_parquet_rows(paths: Iterable[Path], *, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            path_rows = table.to_pylist()
        except Exception as pyarrow_error:
            try:
                import pandas as pd

                frame = pd.read_parquet(path, engine="fastparquet")
                path_rows = frame.to_dict(orient="records")
            except Exception as pandas_error:
                errors.append(
                    f"{path}: pyarrow={pyarrow_error!r}; fastparquet={pandas_error!r}"
                )
                continue

        for row in path_rows:
            rows.append(_normalize_qa_row(row))
            if limit is not None and len(rows) >= limit:
                return rows

    if not rows and errors:
        raise DatasetProcessingError("failed to read parquet files: " + "; ".join(errors))
    return rows


def _context_corpus(dataset: str, rows: list[dict[str, Any]]) -> list[CorpusDoc]:
    corpus_by_doc_id: dict[str, CorpusDoc] = {}
    missing_context_sentences = 0
    for row in rows:
        context = row.get("context") or {}
        titles = context.get("title", [])
        sentence_groups = context.get("sentences")
        if sentence_groups is None:
            missing_context_sentences += 1
            continue
        for title, sentences in zip(titles, sentence_groups, strict=False):
            sentence_list = [str(sentence) for sentence in sentences]
            text = normalize_text(" ".join(sentence_list))
            doc_id = _doc_id(dataset, str(title), text)
            if doc_id not in corpus_by_doc_id:
                corpus_by_doc_id[doc_id] = CorpusDoc(
                    doc_id=doc_id,
                    dataset=dataset,
                    title=str(title),
                    text=text,
                    sentences=sentence_list,
                    source=f"{dataset}_context",
                    metadata={},
                )

    if rows and missing_context_sentences == len(rows):
        raise DatasetProcessingError(
            f"{dataset} parquet rows did not include context.sentences; "
            "install a compatible parquet/datasets reader before processing this dataset"
        )
    return list(corpus_by_doc_id.values())


def _supporting_fact_pairs(row: dict[str, Any]) -> list[tuple[str, int]]:
    supporting_facts = row.get("supporting_facts") or {}
    if isinstance(supporting_facts, dict):
        return [
            (str(title), int(sent_id))
            for title, sent_id in zip(
                supporting_facts.get("title", []) or [],
                supporting_facts.get("sent_id", []) or [],
                strict=False,
            )
            if str(title).strip()
        ]

    pairs: list[tuple[str, int]] = []
    for item in supporting_facts or []:
        if isinstance(item, dict):
            title = str(item.get("title") or "")
            sent_id = item.get("sent_id")
            if title.strip() and sent_id is not None:
                pairs.append((title, int(sent_id)))
        elif len(item) >= 2:
            title = str(item[0])
            if title.strip():
                pairs.append((title, int(item[1])))
    return pairs


def _needed_context_titles(rows: list[dict[str, Any]]) -> set[str]:
    titles: set[str] = set()
    for row in rows:
        context = row.get("context") or {}
        for title in context.get("title", []) or []:
            if str(title).strip():
                titles.add(str(title))
        for title, _sent_id in _supporting_fact_pairs(row):
            titles.add(title)
    return titles


def _external_corpus_by_title(
    dataset: str,
    data_root: Path,
    titles: set[str],
) -> dict[str, list[str]]:
    if not titles:
        return {}
    if dataset == "2wiki":
        return _two_wiki_corpus_by_title(data_root, titles)
    if dataset == "hotpotqa":
        return _hotpot_beir_corpus_by_title(data_root, titles)
    return {}


def _two_wiki_corpus_by_title(data_root: Path, titles: set[str]) -> dict[str, list[str]]:
    source = data_root / "2wiki" / "corpus" / "2wiki_corpus.jsonl"
    if not source.exists():
        return {}
    by_title: dict[str, list[str]] = {}
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            title = str(row.get("title") or "")
            if title in titles and title not in by_title:
                by_title[title] = [str(sentence) for sentence in row.get("sentences", [])]
    return by_title


def _hotpot_beir_corpus_by_title(data_root: Path, titles: set[str]) -> dict[str, list[str]]:
    source = data_root / "hotpotqa" / "beir_corpus" / "corpus" / "corpus-00000-of-00001.parquet"
    if not source.exists():
        return {}
    import pandas as pd

    frame = pd.read_parquet(source, engine="fastparquet", columns=["title", "text"])
    by_title: dict[str, list[str]] = {}
    for row in frame.itertuples(index=False):
        title = str(row.title or "")
        if title in titles and title not in by_title:
            text = normalize_text(str(row.text or ""))
            by_title[title] = _split_sentences(text)
    return by_title


def _split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    return sentences or [text]


def _fill_missing_context_sentences(
    dataset: str,
    rows: list[dict[str, Any]],
    data_root: Path,
) -> None:
    rows_needing_external = []
    for row in rows:
        context = row.get("context") or {}
        titles = [str(title) for title in context.get("title", []) or []]
        title_set = set(titles)
        missing_support_titles = [
            title for title, _sent_id in _supporting_fact_pairs(row)
            if title not in title_set
        ]
        if context.get("sentences") is None or missing_support_titles:
            rows_needing_external.append(row)

    if not rows_needing_external:
        return

    corpus_by_title = _external_corpus_by_title(
        dataset,
        data_root,
        _needed_context_titles(rows_needing_external),
    )
    for row in rows_needing_external:
        context = row.get("context") or {}
        titles = [str(title) for title in context.get("title", []) or []]
        sentence_groups = context.get("sentences")
        if sentence_groups is None:
            sentence_groups = [corpus_by_title.get(title, []) for title in titles]
        else:
            sentence_groups = [
                [str(sentence) for sentence in sentences]
                for sentences in sentence_groups
            ]

        while len(sentence_groups) < len(titles):
            sentence_groups.append([])

        for index, title in enumerate(titles):
            if not sentence_groups[index] and title in corpus_by_title:
                sentence_groups[index] = corpus_by_title[title]

        existing_titles = set(titles)
        for title, _sent_id in _supporting_fact_pairs(row):
            if title in existing_titles:
                continue
            external_sentences = corpus_by_title.get(title)
            if external_sentences:
                titles.append(title)
                sentence_groups.append(external_sentences)
                existing_titles.add(title)

        context["title"] = titles
        context["sentences"] = sentence_groups
        row["context"] = context

    if rows and all(not (row.get("context") or {}).get("sentences") for row in rows):
        raise DatasetProcessingError(
            f"{dataset} parquet rows did not include context.sentences; "
            "external corpus fallback did not recover any context sentences"
        )


def _dedupe_corpus(corpus_items: Iterable[CorpusDoc]) -> list[CorpusDoc]:
    corpus_by_doc_id: dict[str, CorpusDoc] = {}
    for doc in corpus_items:
        corpus_by_doc_id.setdefault(doc.doc_id, doc)
    return list(corpus_by_doc_id.values())


def _write_canonical(
    *,
    dataset: str,
    processed_root: Path,
    examples_by_split: dict[str, list[Example]],
    corpus: list[CorpusDoc],
) -> None:
    dataset_dir = processed_root / dataset
    for split, examples in examples_by_split.items():
        write_jsonl(
            dataset_dir / f"examples.{split}.jsonl",
            [example.to_dict() for example in examples],
        )
    write_jsonl(
        dataset_dir / "corpus.jsonl",
        [_corpus_record(dataset, index, doc) for index, doc in enumerate(corpus)],
    )


def _align_examples_to_corpus(examples: list[Example], corpus: list[CorpusDoc]) -> None:
    corpus_by_title: dict[str, CorpusDoc] = {}
    corpus_ids = {doc.doc_id for doc in corpus}
    for doc in corpus:
        corpus_by_title.setdefault(doc.title, doc)

    for example in examples:
        quality_flags = set(example.quality_flags)
        for fact in example.supporting_facts:
            doc = corpus_by_title.get(fact.title)
            if doc is None:
                quality_flags.add("support_doc_not_in_corpus")
                continue

            fact.doc_id = doc.doc_id
            if fact.sent_id is not None and fact.text is None:
                if 0 <= fact.sent_id < len(doc.sentences):
                    fact.text = doc.sentences[fact.sent_id]
            if fact.text is None:
                quality_flags.add("missing_supporting_fact_text")

        doc_ids = [
            doc.doc_id
            for fact in example.supporting_facts
            if (doc := corpus_by_title.get(fact.title)) is not None
        ]
        for doc_id in example.context_doc_ids:
            if doc_id in corpus_ids and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        example.context_doc_ids = list(dict.fromkeys(doc_ids))

        has_complete_support = bool(example.supporting_facts) and all(
            fact.doc_id in corpus_ids and fact.text is not None
            for fact in example.supporting_facts
        )
        is_usable = bool(example.answer and has_complete_support)
        example.usable_for_sft = is_usable
        example.usable_for_retrieval_eval = is_usable
        example.quality_flags = sorted(quality_flags)


def _process_musique(
    *,
    data_root: Path,
    processed_root: Path,
    linearrag_root: Path,
    sample_root: Path,
    splits: list[str],
    per_dataset: int,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    examples_by_split: dict[str, list[Example]] = {}
    all_corpus: list[CorpusDoc] = []
    split_summary: dict[str, Any] = {}
    for split in splits:
        source = data_root / "musique" / f"musique_ans_v1.0_{split}.jsonl"
        if not source.exists():
            continue
        rows = _jsonl_rows(source)
        if limit is not None:
            rows = rows[:limit]
        report = build_musique_canonical_from_rows(rows, split=split)
        examples = report["examples"]
        corpus = report["corpus"]
        examples_by_split[split] = examples
        all_corpus.extend(corpus)
        split_summary[split] = {
            "source": str(source),
            "examples": len(examples),
            "corpus": len(corpus),
            "errors": len(report["errors"]),
        }

    return _finalize_dataset(
        dataset="musique",
        processed_root=processed_root,
        linearrag_root=linearrag_root,
        sample_root=sample_root,
        examples_by_split=examples_by_split,
        corpus=_dedupe_corpus(all_corpus),
        per_dataset=per_dataset,
        seed=seed,
        split_summary=split_summary,
    )


def _process_qa_parquet_dataset(
    *,
    dataset: str,
    data_root: Path,
    processed_root: Path,
    linearrag_root: Path,
    sample_root: Path,
    splits: list[str],
    per_dataset: int,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    source_dir = data_root / ("hotpotqa/fullwiki" if dataset == "hotpotqa" else "2wiki/qa")
    builder = build_hotpot_example_from_row if dataset == "hotpotqa" else build_2wiki_example_from_row
    split_name_by_file = {"validation": "dev", "dev": "dev", "train": "train"}
    examples_by_split: dict[str, list[Example]] = {}
    all_corpus: list[CorpusDoc] = []
    split_summary: dict[str, Any] = {}
    for split in splits:
        file_prefixes = [
            source_split
            for source_split, target_split in split_name_by_file.items()
            if target_split == split
        ]
        paths = sorted(
            path
            for prefix in file_prefixes
            for path in source_dir.glob(f"{prefix}-*.parquet")
        )
        if not paths:
            continue
        rows = [_normalize_qa_row(row) for row in _read_parquet_rows(paths, limit=limit)]
        _fill_missing_context_sentences(dataset, rows, data_root)
        corpus = _context_corpus(dataset, rows)
        examples = [builder(row, split=split) for row in rows]
        _align_examples_to_corpus(examples, corpus)
        examples_by_split[split] = examples
        all_corpus.extend(corpus)
        split_summary[split] = {
            "sources": [str(path) for path in paths],
            "examples": len(examples),
            "corpus": len(corpus),
        }

    return _finalize_dataset(
        dataset=dataset,
        processed_root=processed_root,
        linearrag_root=linearrag_root,
        sample_root=sample_root,
        examples_by_split=examples_by_split,
        corpus=_dedupe_corpus(all_corpus),
        per_dataset=per_dataset,
        seed=seed,
        split_summary=split_summary,
    )


def _finalize_dataset(
    *,
    dataset: str,
    processed_root: Path,
    linearrag_root: Path,
    sample_root: Path,
    examples_by_split: dict[str, list[Example]],
    corpus: list[CorpusDoc],
    per_dataset: int,
    seed: int,
    split_summary: dict[str, Any],
) -> dict[str, Any]:
    if not examples_by_split:
        raise DatasetProcessingError(f"no input files found for {dataset}")
    if not corpus:
        raise DatasetProcessingError(f"no corpus documents built for {dataset}")

    _write_canonical(
        dataset=dataset,
        processed_root=processed_root,
        examples_by_split=examples_by_split,
        corpus=corpus,
    )
    all_examples = [
        example
        for split in SPLITS
        for example in examples_by_split.get(split, [])
    ]
    build_linearrag_dataset(dataset, all_examples, corpus, linearrag_root)

    return {
        **split_summary,
        "corpus_total": len(corpus),
    }


def process_datasets(
    *,
    data_root: str | Path,
    processed_root: str | Path,
    linearrag_root: str | Path,
    sample_root: str | Path,
    datasets: list[str],
    splits: list[str],
    per_dataset: int,
    seed: int,
    limit: int | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    processed_root = Path(processed_root)
    linearrag_root = Path(linearrag_root)
    sample_root = Path(sample_root)
    summary: dict[str, Any] = {}
    for dataset in datasets:
        if dataset == "musique":
            summary[dataset] = _process_musique(
                data_root=data_root,
                processed_root=processed_root,
                linearrag_root=linearrag_root,
                sample_root=sample_root,
                splits=splits,
                per_dataset=per_dataset,
                seed=seed,
                limit=limit,
            )
        elif dataset in {"hotpotqa", "2wiki"}:
            summary[dataset] = _process_qa_parquet_dataset(
                dataset=dataset,
                data_root=data_root,
                processed_root=processed_root,
                linearrag_root=linearrag_root,
                sample_root=sample_root,
                splits=splits,
                per_dataset=per_dataset,
                seed=seed,
                limit=limit,
            )
        else:
            raise DatasetProcessingError(f"unsupported dataset: {dataset}")

    write_json(processed_root / "processing_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical, LinearRAG, and teacher sampling inputs.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--linearrag-root", default="linearrag_dataset")
    parser.add_argument("--sample-root", default="trajectories/samples")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--per-dataset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit per split for smoke tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = process_datasets(
        data_root=args.data_root,
        processed_root=args.processed_root,
        linearrag_root=args.linearrag_root,
        sample_root=args.sample_root,
        datasets=args.datasets,
        splits=args.splits,
        per_dataset=args.per_dataset,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
