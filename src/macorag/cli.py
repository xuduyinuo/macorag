from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="macorag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_canonical = subparsers.add_parser("build-canonical")
    build_canonical.add_argument("--data-root", default="data")
    build_canonical.add_argument("--output-root", default="data/processed")

    build_linearrag = subparsers.add_parser("build-linearrag")
    build_linearrag.add_argument("--processed-root", default="data/processed")
    build_linearrag.add_argument("--output-root", default="linearrag_dataset")

    sample = subparsers.add_parser("sample")
    sample.add_argument("--processed-root", default="data/processed")
    sample.add_argument("--output-root", default="trajectories")
    sample.add_argument("--per-dataset", type=int, default=1000)
    sample.add_argument("--seed", type=int, default=7)

    generate = subparsers.add_parser("generate-trajectories")
    generate.add_argument("--linearrag-root", default="linearrag_dataset")
    generate.add_argument("--sample-root", default="trajectories")
    generate.add_argument("--raw-output-name", default="raw_teacher_trajectories.jsonl")

    filter_cmd = subparsers.add_parser("filter-trajectories")
    filter_cmd.add_argument("--trajectory-root", default="trajectories")
    filter_cmd.add_argument("--linearrag-root", default="linearrag_dataset")
    filter_cmd.add_argument("--filtered-output-name", default="filtered_sft_trajectories.jsonl")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0
