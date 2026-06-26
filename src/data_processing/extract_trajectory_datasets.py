#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional, Union

from data_processing.io_utils import read_jsonl, write_json, write_jsonl


DATASETS = ("hotpotqa", "2wiki", "musique")


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current.parents[2]


def _coerce_int(value: Any) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError("sample_size must be positive")
    return value


def _coerce_list(value: Any, *, fallback: list[str], name: str) -> list[str]:
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"{name} must be a list; got {type(value)}")


def _coerce_str(value: Any, *, fallback: str, name: str) -> str:
    if value is None:
        return fallback
    if not str(value).strip():
        raise ValueError(f"{name} cannot be empty")
    return str(value)


def _coalesce(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


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
            "PyYAML is required to load .yml/.yaml config. Install it with `python -m pip install PyYAML`."
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format at {config_path}: expected mapping.")

    return {key.replace("-", "_"): value for key, value in data.items()}


def _read_and_sample(path: Path, size: int, rng: random.Random) -> tuple[list[dict[str, Any]], list[int]]:
    """Reservoir sample rows from a jsonl file."""
    selected: list[dict[str, Any]] = []
    selected_indices: list[int] = []

    for idx, row in enumerate(read_jsonl(path)):
        if len(selected) < size:
            selected.append(row)
            selected_indices.append(idx)
            continue

        replace = rng.randint(0, idx)
        if replace < size:
            selected[replace] = row
            selected_indices[replace] = idx

    return selected, selected_indices


def _collect_needed_doc_ids(examples: list[dict[str, Any]]) -> set[str]:
    doc_ids: set[str] = set()

    for example in examples:
        for doc_id in example.get("context_doc_ids", []):
            if isinstance(doc_id, str) and doc_id.strip():
                doc_ids.add(doc_id)

        for fact in example.get("supporting_facts", []):
            if not isinstance(fact, dict):
                continue
            doc_id = fact.get("doc_id")
            if isinstance(doc_id, str) and doc_id.strip():
                doc_ids.add(doc_id)

    return doc_ids


def _filter_corpus(corpus_path: Path, doc_ids: set[str]) -> list[dict[str, Any]]:
    if not doc_ids:
        return []
    target_doc_ids = {str(doc_id) for doc_id in doc_ids}
    return [row for row in read_jsonl(corpus_path) if str(row.get("doc_id")) in target_doc_ids]


def _resolve_split_path(dataset_dir: Path, dataset: str, split: str) -> Path:
    candidates = [
        dataset_dir / f"{dataset}_{split}.jsonl",
        dataset_dir / f"examples.{split}.jsonl",
        dataset_dir / f"{split}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {split} split for dataset={dataset}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _extract_dataset(
    *,
    dataset_dir: Path,
    dataset: str,
    split: str,
    sample_size: int,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    split_path = _resolve_split_path(dataset_dir, dataset, split)
    examples, source_indices = _read_and_sample(split_path, sample_size, rng)

    needed_doc_ids = _collect_needed_doc_ids(examples)
    corpus_rows = _filter_corpus(dataset_dir / "corpus.jsonl", needed_doc_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_examples = output_dir / f"{dataset}_{split}.jsonl"
    out_corpus = output_dir / "corpus.jsonl"
    write_jsonl(out_examples, examples)
    write_jsonl(out_corpus, corpus_rows)

    summary = {
        "dataset": dataset,
        "split": split,
        "requested_sample_size": sample_size,
        "actual_sample_size": len(examples),
        "source_split_file": str(split_path),
        "selected_source_indices_count": len(source_indices),
        "output_examples": str(out_examples),
        "output_corpus": str(out_corpus),
        "selected_source_indices": source_indices,
        "corpus_docs_needed": len(needed_doc_ids),
        "corpus_docs_extracted": len(corpus_rows),
        "seed": seed,
    }

    write_json(
        output_dir / "extract_summary.json",
        summary,
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    repo_root = _resolve_repo_root()

    parser = argparse.ArgumentParser(
        description=(
            "Extract train/dev/test samples and corresponding corpus chunks for "
            "trajectory construction datasets."
        )
    )
    parser.add_argument("--config", default=str(repo_root / "config" / "extract_trajectory_datasets_train.yml"))
    parser.add_argument(
        "--sample-size",
        type=_coerce_int,
        default=None,
        help="Number of samples per dataset to extract from split.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory under repo/data to store extracted datasets.",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "dev", "validation", "test"],
        help="Input split to sample from in processed datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for per-dataset sampling.",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Processed data root containing hotpotqa/2wiki/musique.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to extract; default from config or hotpotqa,2wiki,musique.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_yaml_config(args.config)

    source_root = _coerce_str(
        args.source_root,
        fallback=_coerce_str(
            config.get("source_root"),
            fallback="data/processed",
            name="source_root",
        ),
        name="source_root",
    )
    split = _coalesce(args.split, config.get("split", "train"))
    sample_size = _coerce_int(_coalesce(args.sample_size, config.get("sample_size", 1000)))
    output_root = Path(_coerce_str(
        args.output_dir,
        fallback=_coerce_str(config.get("output_dir"), fallback="/data/trajectory_dataset", name="output_dir"),
        name="output_dir",
    ))
    seed = int(_coalesce(args.seed, config.get("seed", 42)))
    datasets = _coerce_list(
        args.datasets,
        fallback=_coerce_list(
            config.get("datasets"),
            fallback=list(DATASETS),
            name="datasets",
        ),
        name="datasets",
    )

    source_root = Path(source_root)
    summary: dict[str, Any] = {
        "source_root": str(source_root),
        "split": split,
        "sample_size": sample_size,
        "seed": seed,
        "datasets": list(datasets),
        "outputs": {},
    }

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    for dataset in datasets:
        dataset_dir = source_root / dataset
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

        corpus_path = dataset_dir / "corpus.jsonl"
        if not corpus_path.exists():
            raise FileNotFoundError(f"Missing corpus file for dataset={dataset}: {corpus_path}")

        dataset_out = output_root / dataset
        summary["outputs"][dataset] = _extract_dataset(
            dataset_dir=dataset_dir,
            dataset=dataset,
            split=split,
            sample_size=sample_size,
            output_dir=dataset_out,
            seed=seed,
        )

    write_json(output_root / "trajectory_extract_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
