#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess


def query_free_memory() -> list[tuple[int, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Unable to query GPU memory with nvidia-smi: {exc}") from exc
    devices: list[tuple[int, int]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        raw_index, raw_free = (part.strip() for part in line.split(",", 1))
        devices.append((int(raw_index), int(raw_free)))
    if not devices:
        raise SystemExit("nvidia-smi returned no GPUs.")
    return devices


def select_gpu(devices: list[tuple[int, int]], *, min_free_mib: int) -> int:
    if not devices:
        raise ValueError("At least one GPU memory record is required.")
    selected_index, selected_free = max(devices, key=lambda item: (item[1], -item[0]))
    if selected_free < min_free_mib:
        state = ", ".join(f"gpu{index}={free}MiB" for index, free in devices)
        raise RuntimeError(
            f"No GPU has the required {min_free_mib} MiB free for Qwen-7B vLLM; {state}. "
            "Free a GPU or explicitly choose one with EVAL_GPU_INDEX."
        )
    return selected_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a GPU with enough free memory for ablation evaluation.")
    parser.add_argument("--min-free-mib", type=int, default=18000)
    args = parser.parse_args()
    devices = query_free_memory()
    try:
        selected_index = select_gpu(devices, min_free_mib=args.min_free_mib)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(selected_index)


if __name__ == "__main__":
    main()
