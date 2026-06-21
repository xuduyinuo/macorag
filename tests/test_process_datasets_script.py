import importlib.util
import json
from pathlib import Path

import pytest

from macorag.io_utils import read_json, read_jsonl


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "process_datasets.py"
    spec = importlib.util.spec_from_file_location("process_datasets", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_process_musique_dataset_builds_teacher_sampling_inputs(tmp_path):
    module = _load_script_module()
    data_root = tmp_path / "data"
    musique_root = data_root / "musique"
    musique_root.mkdir(parents=True)
    row = {
        "id": "2hop__1_2",
        "question": "Who is the spouse of the Green performer?",
        "answer": "Miquette Giraudy",
        "answer_aliases": [],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 5,
                "title": "Miquette Giraudy",
                "paragraph_text": "Miquette Giraudy is the spouse.",
            },
            {
                "idx": 10,
                "title": "Steve Hillage",
                "paragraph_text": "Steve Hillage performed with Green.",
            },
        ],
        "question_decomposition": [
            {
                "id": 1,
                "question": "Green >> performer",
                "answer": "Steve Hillage",
                "paragraph_support_idx": 10,
            },
            {
                "id": 2,
                "question": "#1 >> spouse",
                "answer": "Miquette Giraudy",
                "paragraph_support_idx": 5,
            },
        ],
    }
    (musique_root / "musique_ans_v1.0_train.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = module.process_datasets(
        data_root=data_root,
        processed_root=tmp_path / "processed",
        linearrag_root=tmp_path / "linearrag",
        sample_root=tmp_path / "samples",
        datasets=["musique"],
        splits=["train"],
        per_dataset=1,
        seed=7,
    )

    assert summary["musique"]["train"]["examples"] == 1
    examples = list(read_jsonl(tmp_path / "processed" / "musique" / "examples.train.jsonl"))
    corpus = list(read_jsonl(tmp_path / "processed" / "musique" / "corpus.jsonl"))
    questions = read_json(tmp_path / "linearrag" / "musique" / "questions.json")
    chunks = read_json(tmp_path / "linearrag" / "musique" / "chunks.json")
    samples = list(read_jsonl(tmp_path / "samples" / "musique.train.1.jsonl"))

    assert examples[0]["qid"] == "2hop__1_2"
    assert len(corpus) == 2
    assert questions[0]["id"] == "2hop__1_2"
    assert chunks[0]["chunk_id"].startswith("musique:chunk:")
    assert samples[0]["qid"] == "2hop__1_2"


def test_process_qa_dataset_fills_missing_context_sentences_from_external_corpus(
    tmp_path,
    monkeypatch,
):
    module = _load_script_module()
    source_dir = tmp_path / "data" / "hotpotqa" / "fullwiki"
    source_dir.mkdir(parents=True)
    (source_dir / "train-00000-of-00001.parquet").write_bytes(b"placeholder")

    monkeypatch.setattr(
        module,
        "_read_parquet_rows",
        lambda paths, limit: [
            {
                "id": "q1",
                "question": "Where was Alice born?",
                "answer": "Paris",
                "type": "bridge",
                "level": "easy",
                "supporting_facts": {"title": ["Alice"], "sent_id": [0]},
                "context": {"title": ["Alice"], "sentences": None},
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "_external_corpus_by_title",
        lambda dataset, data_root, titles: {"Alice": ["Alice was born in Paris."]},
    )

    summary = module.process_datasets(
        data_root=tmp_path / "data",
        processed_root=tmp_path / "processed",
        linearrag_root=tmp_path / "linearrag",
        sample_root=tmp_path / "samples",
        datasets=["hotpotqa"],
        splits=["train"],
        per_dataset=1,
        seed=7,
    )

    assert summary["hotpotqa"]["train"]["examples"] == 1
    examples = list(read_jsonl(tmp_path / "processed" / "hotpotqa" / "examples.train.jsonl"))
    samples = list(read_jsonl(tmp_path / "samples" / "hotpotqa.train.1.jsonl"))
    assert examples[0]["supporting_facts"][0]["text"] == "Alice was born in Paris."
    assert samples[0]["qid"] == "q1"


def test_process_2wiki_dataset_normalizes_missing_evidences(tmp_path, monkeypatch):
    module = _load_script_module()
    source_dir = tmp_path / "data" / "2wiki" / "qa"
    source_dir.mkdir(parents=True)
    (source_dir / "train-00000-of-00001.parquet").write_bytes(b"placeholder")

    monkeypatch.setattr(
        module,
        "_read_parquet_rows",
        lambda paths, limit: [
            {
                "id": "q1",
                "question": "Where was Alice born?",
                "answer": "Paris",
                "type": "bridge",
                "evidences": None,
                "supporting_facts": {"title": ["Alice"], "sent_id": [0]},
                "context": {"title": ["Alice"], "sentences": None},
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "_external_corpus_by_title",
        lambda dataset, data_root, titles: {"Alice": ["Alice was born in Paris."]},
    )

    summary = module.process_datasets(
        data_root=tmp_path / "data",
        processed_root=tmp_path / "processed",
        linearrag_root=tmp_path / "linearrag",
        sample_root=tmp_path / "samples",
        datasets=["2wiki"],
        splits=["train"],
        per_dataset=1,
        seed=7,
    )

    examples = list(read_jsonl(tmp_path / "processed" / "2wiki" / "examples.train.jsonl"))
    assert summary["2wiki"]["train"]["examples"] == 1
    assert examples[0]["metadata"]["evidences"] == []
