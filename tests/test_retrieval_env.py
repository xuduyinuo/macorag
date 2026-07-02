from __future__ import annotations

import json
from pathlib import Path

from data_processing.retrieval import build_linearrag_assets
from data_processing.retrieval import _chunk_text_from_row, _resolve_example_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_assets_supports_dataset_prefix_split_files(tmp_path: Path) -> None:
    source = tmp_path / "data" / "processed" / "toyqa"
    _write_jsonl(
        source / "toyqa_train.jsonl",
        [
            {
                "qid": "t1",
                "question": "Who leads the city?",
                "answer": "Ada",
                "question_type": "bridge",
                "hop_count": 1,
            }
        ],
    )
    _write_jsonl(
        source / "toyqa_dev.jsonl",
        [
            {
                "qid": "t2",
                "question": "Where is the city?",
                "answer": "Nowhere",
                "question_type": "bridge",
                "hop_count": 1,
            }
        ],
    )
    _write_jsonl(
        source / "corpus.jsonl",
        [
            {
                "chunk_id": "toyqa:chunk:0",
                "doc_id": "d0",
                "dataset": "toyqa",
                "title": "City Hall",
                "text": "The city is led by Ada.",
                "source": "manual",
            }
        ],
    )

    summary = build_linearrag_assets(
        processed_root=tmp_path / "data" / "processed",
        retrieval_root=tmp_path / "data" / "retrieval",
        datasets=["toyqa"],
        splits=["train", "dev"],
    )

    assert summary["toyqa"]["questions"] == 2
    assert summary["toyqa"]["chunks"] == 1
    assert _resolve_example_path(source, "toyqa", "train").name == "toyqa_train.jsonl"

    target = tmp_path / "data" / "retrieval" / "toyqa"
    assert (target / "questions.json").exists()
    assert (target / "chunks.json").exists()
    assert (target / "chunk_metadata.jsonl").exists()


def test_chunk_text_falls_back_to_sentences(tmp_path: Path) -> None:
    row = {
        "title": "T",
        "sentences": [
            {"text": "First sentence."},
            {"text": "Second sentence."},
        ],
    }
    assert (
        _chunk_text_from_row(row)
        == "First sentence.\nSecond sentence."
    )


def test_retrieval_scripts_use_data_processing_entrypoints() -> None:
    root = Path(__file__).resolve().parents[1]
    build_script = (root / "scripts" / "build_retrieval.sh").read_text(encoding="utf-8")
    query_script = (root / "scripts" / "query_retrieval.sh").read_text(encoding="utf-8")

    assert "python -m data_processing.retrieval_cli" in build_script
    assert "config/build_retrieval.yml" in build_script
    assert "python -m data_processing.retrieval_cli" in query_script
    assert "config/query_retrieval.yml" in query_script
