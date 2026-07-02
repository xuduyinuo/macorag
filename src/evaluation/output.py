from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_dir(output_root: str | Path, *, timestamp: str | None = None) -> Path:
    """在评估根目录下创建一次运行专属的时间戳目录。"""
    run_name = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
