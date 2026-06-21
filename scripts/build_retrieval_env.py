#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_processing.process_datasets import DATASETS as PROCESS_DATASETS
from retrieval_env import (
    RETRIEVAL_DEFAULT_SPLITS,
    PROCESSED_DATASETS,
    build_linearrag_assets,
    build_linear_rag_index,
    query_linear_rag,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/search a LinearRAG-compatible retrieval environment from processed data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build retrieval assets and optional indexes.")
    build.add_argument(
        "--processed-root",
        default="data/processed",
        help="Processed directory containing corpus.jsonl and examples.*.jsonl.",
    )
    build.add_argument(
        "--retrieval-root",
        default="linearrag_dataset",
        help="Output directory for linearrag-style files and index cache.",
    )
    build.add_argument(
        "--datasets",
        nargs="+",
        choices=PROCESS_DATASETS,
        default=list(PROCESSED_DATASETS),
        help="Datasets to process.",
    )
    build.add_argument(
        "--splits",
        nargs="+",
        choices=RETRIEVAL_DEFAULT_SPLITS,
        default=list(RETRIEVAL_DEFAULT_SPLITS),
        help="Input splits to include in questions.json.",
    )
    build.add_argument("--build-index", action="store_true", help="Build LinearRAG indexes now.")
    build.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    build.add_argument("--spacy-model", default="en_core_web_trf")
    build.add_argument("--max-workers", type=int, default=16)
    build.add_argument("--batch-size", type=int, default=128)
    build.add_argument("--retrieval-top-k", type=int, default=5)
    build.add_argument("--use-vectorized-retrieval", action="store_true")

    query = subparsers.add_parser("query", help="Query a built retrieval environment.")
    query.add_argument("query", nargs="+", help="Query string.")
    query.add_argument("--dataset", choices=PROCESS_DATASETS, required=True)
    query.add_argument(
        "--retrieval-root",
        default="linearrag_dataset",
        help="Directory where linearrag-style files and index were built.",
    )
    query.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    query.add_argument("--spacy-model", default="en_core_web_trf")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--max-workers", type=int, default=16)
    query.add_argument("--batch-size", type=int, default=128)
    query.add_argument("--use-vectorized-retrieval", action="store_true")

    return parser


def _read_chunks(retrieval_root: str, dataset: str) -> list[str]:
    path = Path(retrieval_root) / dataset / "chunks.json"
    with path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected list in {path}")
    return [str(item) for item in chunks]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        summary = build_linearrag_assets(
            processed_root=args.processed_root,
            retrieval_root=args.retrieval_root,
            datasets=args.datasets,
            splits=args.splits,
        )

        if args.build_index:
            for dataset in args.datasets:
                chunks = _read_chunks(args.retrieval_root, dataset)
                build_linear_rag_index(
                    retrieval_root=args.retrieval_root,
                    dataset=dataset,
                    chunks=chunks,
                    embedding_model=args.embedding_model,
                    spacy_model=args.spacy_model,
                    max_workers=args.max_workers,
                    batch_size=args.batch_size,
                    retrieval_top_k=args.retrieval_top_k,
                    use_vectorized_retrieval=args.use_vectorized_retrieval,
                )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    query_text = " ".join(args.query)
    result = query_linear_rag(
        retrieval_root=args.retrieval_root,
        dataset=args.dataset,
        query=query_text,
        embedding_model=args.embedding_model,
        spacy_model=args.spacy_model,
        top_k=args.top_k,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
    )

    for rank, (passage, score) in enumerate(
        zip(result.passages, result.scores),
        start=1,
    ):
        print(f"[{rank:02d}] {score:.4f}\n{passage}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
