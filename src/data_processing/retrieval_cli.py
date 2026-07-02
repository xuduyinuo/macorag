from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Union

from data_processing.process_datasets import DATASETS
from data_processing.retrieval import (
    RETRIEVAL_DEFAULT_ROOT,
    RETRIEVAL_DEFAULT_SPLITS,
    build_linearrag_assets,
    build_linear_rag_index,
    query_linear_rag,
)


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = _resolve_repo_root()
DEFAULT_RETRIEVAL_CONFIG = REPO_ROOT / "config" / "build_retrieval.yml"


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


def _coalesce(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


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


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return bool(fallback)
    return bool(value)


def _coerce_int(value: Any, fallback: int) -> int:
    if value is None:
        return int(fallback)
    return int(value)


def _coerce_query(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"query must be a string or list of strings; got {type(value)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/search a LinearRAG-compatible retrieval environment from processed data."
    )
    parser.add_argument("--config", default=str(DEFAULT_RETRIEVAL_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=False)

    build = subparsers.add_parser("build", help="Build retrieval assets and optional indexes.")
    build.add_argument(
        "--processed-root",
        default=None,
        help="Processed directory containing corpus.jsonl and split files "
        "(examples.<split>.jsonl or <dataset>_<split>.jsonl).",
    )
    build.add_argument(
        "--retrieval-root",
        default=None,
        help="Output directory for linearrag-style files and index cache.",
    )
    build.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=None,
        help="Datasets to process.",
    )
    build.add_argument(
        "--splits",
        nargs="+",
        choices=RETRIEVAL_DEFAULT_SPLITS,
        default=None,
        help="Input splits to include in questions.json.",
    )
    build.add_argument("--build-index", action="store_true", default=None, help="Build LinearRAG indexes now.")
    build.add_argument("--embedding-model", default=None)
    build.add_argument("--spacy-model", default=None)
    build.add_argument("--max-workers", type=int, default=None)
    build.add_argument("--batch-size", type=int, default=None)
    build.add_argument("--retrieval-top-k", type=int, default=None)
    build.add_argument("--use-vectorized-retrieval", action="store_true", default=None)
    build.add_argument(
        "--co-locate-index",
        action="store_true",
        default=None,
        help="Store built retrieval indexes in the same directory as dataset data.",
    )

    query = subparsers.add_parser("query", help="Query a built retrieval environment.")
    query.add_argument("query", nargs="*", help="Query string.")
    query.add_argument("--dataset", choices=DATASETS, default=None)
    query.add_argument(
        "--retrieval-root",
        default=None,
        help="Directory where linearrag-style files and index were built.",
    )
    query.add_argument("--embedding-model", default=None)
    query.add_argument("--spacy-model", default=None)
    query.add_argument("--top-k", type=int, default=None)
    query.add_argument("--max-workers", type=int, default=None)
    query.add_argument("--batch-size", type=int, default=None)
    query.add_argument("--use-vectorized-retrieval", action="store_true", default=None)

    return parser


def _read_chunks(retrieval_root: str, dataset: str) -> list[str]:
    path = Path(retrieval_root) / dataset / "chunks.json"
    with path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected list in {path}")
    return [str(item) for item in chunks]


def _parse_command(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.command is not None:
        return args.command
    return str(_coalesce(config.get("command"), "build"))


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_yaml_config(args.config)
    command = _parse_command(args, config)

    if command == "build":
        processed_root = _coalesce(
            getattr(args, "processed_root", None),
            config.get("processed_root", "data/processed"),
        )
        retrieval_root = _coalesce(
            getattr(args, "retrieval_root", None),
            config.get("retrieval_root", RETRIEVAL_DEFAULT_ROOT),
        )
        datasets = _coerce_list(
            getattr(args, "datasets", None),
            fallback=_coalesce(config.get("datasets"), list(DATASETS)),
            name="datasets",
        )
        splits = _coerce_list(
            getattr(args, "splits", None),
            fallback=_coalesce(config.get("splits"), list(RETRIEVAL_DEFAULT_SPLITS)),
            name="splits",
        )
        build_index = _coerce_bool(
            getattr(args, "build_index", None),
            bool(config.get("build_index", False)),
        )

        summary = build_linearrag_assets(
            processed_root=processed_root,
            retrieval_root=retrieval_root,
            datasets=datasets,
            splits=splits,
        )

        if build_index:
            embedding_model = _coalesce(
                getattr(args, "embedding_model", None),
                config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"),
            )
            spacy_model = _coalesce(
                getattr(args, "spacy_model", None),
                config.get("spacy_model", "en_core_web_trf"),
            )
            max_workers = _coerce_int(
                getattr(args, "max_workers", None),
                _coalesce(config.get("max_workers"), 16),
            )
            batch_size = _coerce_int(
                getattr(args, "batch_size", None),
                _coalesce(config.get("batch_size"), 128),
            )
            retrieval_top_k = _coerce_int(
                getattr(args, "retrieval_top_k", None),
                _coerce_int(config.get("retrieval_top_k"), 5),
            )
            use_vectorized_retrieval = _coerce_bool(
                getattr(args, "use_vectorized_retrieval", None),
                bool(config.get("use_vectorized_retrieval", False)),
            )
            co_locate_index = _coerce_bool(
                getattr(args, "co_locate_index", None),
                bool(config.get("co_locate_index", False)),
            )
            for dataset in datasets:
                chunks = _read_chunks(retrieval_root, dataset)
                build_linear_rag_index(
                    retrieval_root=retrieval_root,
                    dataset=dataset,
                    chunks=chunks,
                    embedding_model=embedding_model,
                    spacy_model=spacy_model,
                    max_workers=max_workers,
                    batch_size=batch_size,
                    retrieval_top_k=retrieval_top_k,
                    use_vectorized_retrieval=use_vectorized_retrieval,
                    co_locate_index=co_locate_index,
                )

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if command == "query":
        query_tokens = _coerce_query(_coalesce(getattr(args, "query", None), config.get("query")))
        if not query_tokens:
            raise ValueError("query must be provided via CLI positional args or `query` in config.")

        dataset = _coalesce(getattr(args, "dataset", None), config.get("dataset"))
        if not dataset:
            raise ValueError("dataset must be provided in query mode.")

        result = query_linear_rag(
            retrieval_root=_coalesce(
                getattr(args, "retrieval_root", None),
                config.get("retrieval_root", RETRIEVAL_DEFAULT_ROOT),
            ),
            dataset=dataset,
            query=" ".join(query_tokens),
            embedding_model=_coalesce(
                getattr(args, "embedding_model", None),
                config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"),
            ),
            spacy_model=_coalesce(
                getattr(args, "spacy_model", None),
                config.get("spacy_model", "en_core_web_trf"),
            ),
            top_k=_coerce_int(getattr(args, "top_k", None), _coalesce(config.get("top_k"), 5)),
            max_workers=_coerce_int(
                getattr(args, "max_workers", None),
                _coalesce(config.get("max_workers"), 16),
            ),
            batch_size=_coerce_int(
                getattr(args, "batch_size", None),
                _coalesce(config.get("batch_size"), 128),
            ),
            use_vectorized_retrieval=_coerce_bool(
                getattr(args, "use_vectorized_retrieval", None),
                bool(config.get("use_vectorized_retrieval", False)),
            ),
        )
        for rank, (passage, score) in enumerate(
            zip(result.passages, result.scores),
            start=1,
        ):
            print(f"[{rank:02d}] {score:.4f}\n{passage}\n")
        return 0

    raise ValueError(f"Unsupported command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
