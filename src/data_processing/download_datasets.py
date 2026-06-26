from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


ROOT = _resolve_repo_root()


@dataclass(frozen=True)
class DownloadItem:
    name: str
    url: str
    path: str
    size: Optional[int]
    source_note: Optional[str] = None


FILES: tuple[DownloadItem, ...] = (
    DownloadItem(
        "framolfese/2WikiMultihopQA README",
        "https://huggingface.co/datasets/framolfese/2WikiMultihopQA/resolve/main/README.md",
        "data/2wiki/qa/README.md",
        5462,
    ),
    DownloadItem(
        "framolfese/2WikiMultihopQA test",
        "https://huggingface.co/datasets/framolfese/2WikiMultihopQA/resolve/main/data/test-00000-of-00001.parquet",
        "data/2wiki/qa/data/test-00000-of-00001.parquet",
        27956501,
    ),
    DownloadItem(
        "framolfese/2WikiMultihopQA train shard 0",
        "https://huggingface.co/datasets/framolfese/2WikiMultihopQA/resolve/main/data/train-00000-of-00002.parquet",
        "data/2wiki/qa/data/train-00000-of-00002.parquet",
        165708170,
    ),
    DownloadItem(
        "framolfese/2WikiMultihopQA train shard 1",
        "https://huggingface.co/datasets/framolfese/2WikiMultihopQA/resolve/main/data/train-00001-of-00002.parquet",
        "data/2wiki/qa/data/train-00001-of-00002.parquet",
        164873439,
    ),
    DownloadItem(
        "framolfese/2WikiMultihopQA validation",
        "https://huggingface.co/datasets/framolfese/2WikiMultihopQA/resolve/main/data/validation-00000-of-00001.parquet",
        "data/2wiki/qa/data/validation-00000-of-00001.parquet",
        29505064,
    ),
    DownloadItem(
        "hotpotqa/hotpot_qa README",
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/README.md",
        "data/hotpotqa/fullwiki/README.md",
        9522,
    ),
    DownloadItem(
        "hotpotqa/hotpot_qa fullwiki train shard 0",
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/train-00000-of-00002.parquet",
        "data/hotpotqa/fullwiki/fullwiki/train-00000-of-00002.parquet",
        165624177,
    ),
    DownloadItem(
        "hotpotqa/hotpot_qa fullwiki train shard 1",
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/train-00001-of-00002.parquet",
        "data/hotpotqa/fullwiki/fullwiki/train-00001-of-00002.parquet",
        166162479,
    ),
    DownloadItem(
        "hotpotqa/hotpot_qa fullwiki validation",
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/validation-00000-of-00001.parquet",
        "data/hotpotqa/fullwiki/fullwiki/validation-00000-of-00001.parquet",
        28041820,
    ),
    DownloadItem(
        "hotpotqa/hotpot_qa fullwiki test",
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/test-00000-of-00001.parquet",
        "data/hotpotqa/fullwiki/fullwiki/test-00000-of-00001.parquet",
        27558644,
    ),
    DownloadItem(
        "BeIR/hotpotqa README",
        "https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/README.md",
        "data/hotpotqa/beir_corpus/README.md",
        10434,
    ),
    DownloadItem(
        "BeIR/hotpotqa corpus",
        "https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/corpus/corpus-00000-of-00001.parquet",
        "data/hotpotqa/beir_corpus/corpus/corpus-00000-of-00001.parquet",
        975977704,
    ),
    DownloadItem(
        "BeIR/hotpotqa queries",
        "https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/queries/queries-00000-of-00001.parquet",
        "data/hotpotqa/beir_corpus/queries/queries-00000-of-00001.parquet",
        8454589,
    ),
    DownloadItem(
        "bdsaglam/musique README",
        "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/README.md",
        "data/musique/README.md",
        359,
    ),
    DownloadItem(
        "bdsaglam/musique ans dev",
        "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_ans_v1.0_dev.jsonl",
        "data/musique/musique_ans_v1.0_dev.jsonl",
        30439728,
    ),
    DownloadItem(
        "bdsaglam/musique ans train",
        "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_ans_v1.0_train.jsonl",
        "data/musique/musique_ans_v1.0_train.jsonl",
        241046755,
    ),
    DownloadItem(
        "bdsaglam/musique full dev",
        "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_full_v1.0_dev.jsonl",
        "data/musique/musique_full_v1.0_dev.jsonl",
        59422562,
    ),
    DownloadItem(
        "bdsaglam/musique full train",
        "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_full_v1.0_train.jsonl",
        "data/musique/musique_full_v1.0_train.jsonl",
        476696984,
    ),
    DownloadItem(
        "Alab-NII/2wikimultihop para_with_hyperlink",
        "https://www.dropbox.com/s/wlhw26kik59wbh8/para_with_hyperlink.zip?dl=1",
        "data/2wiki/para_with_hyperlink/para_with_hyperlink.zip",
        None,
        "Linked from https://github.com/Alab-NII/2wikimultihop README.md",
    ),
)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(item: DownloadItem) -> dict[str, Any]:
    target = ROOT / item.path
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item.size
    if target.exists() and (expected is None or target.stat().st_size == expected):
        status = "already_present"
    else:
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--connect-timeout",
            "30",
            "--retry",
            "8",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "-o",
            str(target),
            item.url,
        ]
        print(f"Downloading {item.name} -> {item.path}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        status = "downloaded"

    actual_size = target.stat().st_size if target.exists() else None
    ok = bool(target.exists()) and (expected is None or actual_size == expected)
    return {
        "name": item.name,
        "path": item.path,
        "url": item.url,
        "source_note": item.source_note,
        "expected_size": expected,
        "actual_size": actual_size,
        "sha256": sha256sum(target) if target.exists() and ok else None,
        "status": status,
        "ok": ok,
    }


def main() -> None:
    results = []
    for item in tqdm(FILES, desc="Downloading dataset files", unit="file"):
        results.append(download(item))

    manifest = ROOT / "data" / "DOWNLOAD_MANIFEST.json"
    manifest.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    failures = [result for result in results if not result["ok"]]
    print(f"Wrote {manifest.relative_to(ROOT)}")
    if failures:
        print(json.dumps(failures, indent=2, ensure_ascii=False))
        raise SystemExit(1)
    print(f"Verified {len(results)} files")


if __name__ == "__main__":
    main()
