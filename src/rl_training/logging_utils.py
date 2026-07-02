from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def make_timestamped_run_dir(output_root: str | Path, timestamp: str | None = None) -> Path:
    """按 LinearRAG 风格生成运行目录：保存根目录/时间戳。"""
    run_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(output_root) / run_timestamp


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写入小型运行元信息；训练主流程只关心事件本身，不处理文件细节。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
