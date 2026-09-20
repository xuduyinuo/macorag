from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets README",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/README.md",
        "data/README_flashrag_datasets.md",
        11310,
    ),
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets nq train",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/nq/train.jsonl",
        "data/nq/train.jsonl",
        9960189,
    ),
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets nq dev",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/nq/dev.jsonl",
        "data/nq/dev.jsonl",
        1073443,
    ),
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets nq test",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/nq/test.jsonl",
        "data/nq/test.jsonl",
        487676,
    ),
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets triviaqa train",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/triviaqa/train.jsonl",
        "data/triviaqa/train.jsonl",
        32952174,
    ),
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets triviaqa dev",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/triviaqa/dev.jsonl",
        "data/triviaqa/dev.jsonl",
        3714793,
    ),
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets triviaqa test",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/triviaqa/test.jsonl",
        "data/triviaqa/test.jsonl",
        4797585,
    ),
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets popqa test",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/popqa/test.jsonl",
        "data/popqa/test.jsonl",
        8526089,
    ),
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets bamboogle test",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/bamboogle/test.jsonl",
        "data/bamboogle/test.jsonl",
        17031,
    ),
    DownloadItem(
        "RUC-NLPIR/FlashRAG_datasets wiki18_100w retrieval corpus",
        "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/retrieval-corpus/wiki18_100w.zip",
        "data/wikipedia/wiki18_100w.zip",
        5130719280,
        "Shared open-domain Wikipedia corpus used by FlashRAG for NQ/TriviaQA/PopQA/Bamboogle retrieval.",
    ),
)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Files at or above this size are fetched with several parallel range requests.
# A single curl connection through the proxy sustains ~0.5 MB/s, while 6 parallel
# connections reach ~2.5 MB/s; more than ~8 connections makes it slower again.
PARALLEL_THRESHOLD = 256 * 1024 * 1024
PART_SIZE = 64 * 1024 * 1024
PARALLEL_WORKERS = 6


def _curl(argv: list[str]) -> None:
    subprocess.run(argv, cwd=ROOT, check=True)


def _download_single(url: str, target: Path) -> None:
    _curl(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--no-progress-meter",
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
            url,
        ]
    )


def _download_range(url: str, start: int, end: int, dest: Path, attempts: int = 5) -> None:
    expected = end - start + 1
    for attempt in range(1, attempts + 1):
        if dest.exists() and dest.stat().st_size == expected:
            return
        dest.unlink(missing_ok=True)
        try:
            _curl(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--no-progress-meter",
                    "--connect-timeout",
                    "30",
                    "--retry",
                    "4",
                    "--retry-delay",
                    "5",
                    "-r",
                    f"{start}-{end}",
                    "-o",
                    str(dest),
                    url,
                ]
            )
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(f"  retry {attempt}/{attempts - 1}: bytes {start}-{end}", flush=True)
            continue
        if dest.exists() and dest.stat().st_size == expected:
            return
    raise RuntimeError(f"incomplete range {start}-{end} for {url}")


def _download_parallel(url: str, target: Path, expected: int) -> None:
    parts_dir = target.parent / f".{target.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [
        (index, start, min(start + PART_SIZE, expected) - 1)
        for index, start in enumerate(range(0, expected, PART_SIZE))
    ]
    pending = [
        (index, start, end)
        for index, start, end in ranges
        if not (parts_dir / f"part-{index:05d}").exists()
        or (parts_dir / f"part-{index:05d}").stat().st_size != end - start + 1
    ]
    cached = len(ranges) - len(pending)
    if cached:
        print(f"  resuming: {cached}/{len(ranges)} parts already complete", flush=True)

    completed = cached
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {
            pool.submit(_download_range, url, start, end, parts_dir / f"part-{index:05d}"): index
            for index, start, end in pending
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"  part {completed}/{len(ranges)} done", flush=True)

    print(f"  assembling {target.name} from {len(ranges)} parts", flush=True)
    assembling = target.parent / f".{target.name}.assembling"
    with assembling.open("wb") as out:
        for index, _start, _end in ranges:
            with (parts_dir / f"part-{index:05d}").open("rb") as part:
                shutil.copyfileobj(part, out, 4 * 1024 * 1024)
    assembling.replace(target)
    shutil.rmtree(parts_dir, ignore_errors=True)


def download(item: DownloadItem) -> dict[str, Any]:
    target = ROOT / item.path
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item.size
    if target.exists() and (expected is None or target.stat().st_size == expected):
        status = "already_present"
    else:
        print(f"Downloading {item.name} -> {item.path}", flush=True)
        if expected is not None and expected >= PARALLEL_THRESHOLD:
            _download_parallel(item.url, target, expected)
        else:
            _download_single(item.url, target)
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


def _load_manifest_entries(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(entries, list):
        return {}
    return {
        entry["path"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }


def select_items(only: Optional[list[str]]) -> list[DownloadItem]:
    if not only:
        return list(FILES)
    needles = [needle.lower() for needle in only]
    return [
        item
        for item in FILES
        if any(needle in item.path.lower() or needle in item.name.lower() for needle in needles)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify dataset files into data/.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="SUBSTRING",
        help=(
            "Only process items whose name or path contains SUBSTRING "
            "(repeatable). Matched entries are merged into the existing manifest."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the items that would be processed and exit.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    items = select_items(args.only)
    if args.list:
        for item in items:
            print(f"{item.path}\t{item.size if item.size is not None else '?'}\t{item.name}")
        return
    if not items:
        raise SystemExit("no download items matched the given filters")

    results = []
    for item in tqdm(items, desc="Downloading dataset files", unit="file"):
        results.append(download(item))

    manifest = ROOT / "data" / "DOWNLOAD_MANIFEST.json"
    merged = _load_manifest_entries(manifest)
    for result in results:
        merged[result["path"]] = result
    manifest.write_text(
        json.dumps(list(merged.values()), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    failures = [result for result in results if not result["ok"]]
    print(f"Wrote {manifest.relative_to(ROOT)}")
    if failures:
        print(json.dumps(failures, indent=2, ensure_ascii=False))
        raise SystemExit(1)
    print(f"Verified {len(results)} files")


if __name__ == "__main__":
    main()
