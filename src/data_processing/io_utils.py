from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeAlias

PathLike: TypeAlias = str | Path


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_key(value: str) -> str:
    return normalize_text(value).casefold()


def sha1_text(value: str) -> str:
    normalized = normalize_text(value)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def ensure_parent(path: PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: PathLike, items: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False))
            file.write("\n")


def read_json(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: PathLike, data: Any) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")
