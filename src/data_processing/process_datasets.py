#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from data_processing.dataset_builders import (
    build_2wiki_example_from_row,
    build_hotpot_example_from_row,
    build_musique_canonical_from_rows,
)
from data_processing.io_utils import normalize_key, normalize_text, sha1_text, write_json, write_jsonl
from data_processing.schemas import CorpusDoc, Example


DATASETS = ("hotpotqa", "2wiki", "musique")
SPLITS = ("train", "dev")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = _resolve_repo_root()
DEFAULT_PROCESSING_CONFIG = REPO_ROOT / "config" / "process_datasets.yml"


def _load_yaml_config(path: Union[str, Path]) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    if config_path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError(f"Unsupported config type: {config_path.suffix}")

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyYAML is required to load .yml/.yaml config. Install it with `pip install PyYAML`."
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format at {config_path}: expected mapping.")
    return {key.replace("-", "_"): value for key, value in data.items()}


def _coerce_list(value: Any, *, fallback: list[str], name: str) -> list[str]:
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"{name} must be a list; got {type(value)}")


def _coalesce(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value

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


def _read_parquet_rows(paths: Iterable[Path], *, limit: Optional[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in tqdm(tuple(paths), desc="Reading parquet shards", unit="file"):
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

        for row in tqdm(path_rows, desc=f"Normalize rows: {path.name}", unit="row", leave=False):
            rows.append(_normalize_qa_row(row))
            if limit is not None and len(rows) >= limit:
                return rows

    if not rows and errors:
        raise DatasetProcessingError("failed to read parquet files: " + "; ".join(errors))
    return rows


def _context_corpus(dataset: str, rows: list[dict[str, Any]]) -> list[CorpusDoc]:
    corpus_by_doc_id: dict[str, CorpusDoc] = {}
    missing_context_sentences = 0
    for row in tqdm(rows, desc=f"{dataset} context corpus", unit="row"):
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
    for row in tqdm(rows, desc="Collecting context titles", unit="row"):
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
        for line in tqdm(file, desc="Reading 2wiki corpus", unit="row"):
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
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="Reading BEIR corpus", unit="row"):
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
    for row in tqdm(rows, desc="Identifying missing context", unit="row"):
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
    for row in tqdm(rows_needing_external, desc="Filling missing context", unit="row"):
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

    for example in tqdm(examples, desc="Aligning examples", unit="item"):
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
    splits: list[str],
    limit: Optional[int],
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
        examples = [
            example
            for example in tqdm(
                report["examples"],
                desc=f"musique {split} examples",
                unit="item",
            )
        ]
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
        examples_by_split=examples_by_split,
        corpus=_dedupe_corpus(all_corpus),
        split_summary=split_summary,
    )


def _process_qa_parquet_dataset(
    *,
    dataset: str,
    data_root: Path,
    processed_root: Path,
    splits: list[str],
    limit: Optional[int],
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
        examples = []
        for row in tqdm(rows, desc=f"{dataset} {split} examples", unit="item"):
            examples.append(builder(row, split=split))
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
        examples_by_split=examples_by_split,
        corpus=_dedupe_corpus(all_corpus),
        split_summary=split_summary,
    )


def _finalize_dataset(
    *,
    dataset: str,
    processed_root: Path,
    examples_by_split: dict[str, list[Example]],
    corpus: list[CorpusDoc],
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

    return {
        **split_summary,
        "corpus_total": len(corpus),
    }


def process_datasets(
    *,
    data_root: Union[str, Path],
    processed_root: Union[str, Path],
    datasets: list[str],
    splits: list[str],
    limit: Optional[int] = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    processed_root = Path(processed_root)
    summary: dict[str, Any] = {}
    for dataset in tqdm(datasets, desc="Processing datasets", unit="dataset"):
        if dataset == "musique":
            summary[dataset] = _process_musique(
                data_root=data_root,
                processed_root=processed_root,
                splits=splits,
                limit=limit,
            )
        elif dataset in {"hotpotqa", "2wiki"}:
            summary[dataset] = _process_qa_parquet_dataset(
                dataset=dataset,
                data_root=data_root,
                processed_root=processed_root,
                splits=splits,
                limit=limit,
            )
        else:
            raise DatasetProcessingError(f"unsupported dataset: {dataset}")

    write_json(processed_root / "processing_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build processed corpus/examples.",
    )
    parser.add_argument("--config", default=str(DEFAULT_PROCESSING_CONFIG))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit per split for smoke tests.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_yaml_config(args.config)

    data_root = _coalesce(args.data_root, config.get("data_root", "data"))
    processed_root = _coalesce(args.processed_root, config.get("processed_root", "data/processed"))
    datasets = _coerce_list(args.datasets, fallback=_coalesce(config.get("datasets"), list(DATASETS)), name="datasets")
    splits = _coerce_list(args.splits, fallback=_coalesce(config.get("splits"), list(SPLITS)), name="splits")
    limit = _coalesce(
        args.limit,
        config.get("limit"),
    )
    if limit is not None:
        limit = int(limit)

    summary = process_datasets(
        data_root=data_root,
        processed_root=processed_root,
        datasets=datasets,
        splits=splits,
        limit=limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
