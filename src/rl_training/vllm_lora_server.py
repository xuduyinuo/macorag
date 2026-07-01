from __future__ import annotations

import argparse


def parse_server_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MACORAG vLLM LoRA hot-sync server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--lora-name", required=True)
    parser.add_argument("--lora-int-id", type=int, required=True)
    parser.add_argument("--lora-adapter-path", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_server_args()
    raise SystemExit(
        "vLLM LoRA server runtime is not implemented yet; parser and launcher are available for tests."
    )


if __name__ == "__main__":
    main()
