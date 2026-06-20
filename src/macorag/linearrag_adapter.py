from __future__ import annotations

from pathlib import Path

from macorag.io_utils import write_json, write_jsonl
from macorag.schemas import CorpusDoc, Example


def _chunk_id(dataset: str, index: int) -> str:
    return f"{dataset}:chunk:{index}"


def _dedupe_stable(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _chunk_record(dataset: str, index: int, doc: CorpusDoc) -> dict[str, object]:
    return {
        "chunk_id": _chunk_id(dataset, index),
        "chunk_index": index,
        "doc_id": doc.doc_id,
        "title": doc.title,
        "text": doc.text,
        "dataset": dataset,
        "source": doc.source,
    }


def _gold_sentences(example: Example) -> list[dict[str, object]]:
    seen: set[tuple[str | None, int | None, str, str]] = set()
    sentences: list[dict[str, object]] = []
    for fact in example.supporting_facts:
        if fact.text is None:
            continue

        key = (fact.doc_id, fact.sent_id, fact.title, fact.text)
        if key in seen:
            continue

        seen.add(key)
        sentences.append(
            {
                "doc_id": fact.doc_id,
                "sent_id": fact.sent_id,
                "title": fact.title,
                "text": fact.text,
            }
        )
    return sentences


def _build_qrel(
    example: Example,
    chunk_id_by_doc_id: dict[str, str],
) -> dict[str, object]:
    gold_doc_ids = _dedupe_stable(
        [
            fact.doc_id
            for fact in example.supporting_facts
            if fact.doc_id is not None and fact.doc_id != ""
        ]
    )
    gold_chunk_ids = _dedupe_stable(
        [
            chunk_id_by_doc_id[fact.doc_id]
            for fact in example.supporting_facts
            if fact.doc_id is not None
            and fact.doc_id != ""
            and fact.doc_id in chunk_id_by_doc_id
        ]
    )
    gold_titles = _dedupe_stable(
        [
            fact.title
            for fact in example.supporting_facts
            if fact.title is not None and fact.title != ""
        ]
    )

    return {
        "qid": example.qid,
        "gold_doc_ids": gold_doc_ids,
        "gold_chunk_ids": gold_chunk_ids,
        "gold_titles": gold_titles,
        "gold_sentences": _gold_sentences(example),
    }


def build_linearrag_dataset(
    dataset: str,
    examples: list[Example],
    corpus: list[CorpusDoc],
    output_root: str | Path,
) -> None:
    output_dir = Path(output_root) / dataset
    chunk_id_by_doc_id = {
        doc.doc_id: _chunk_id(dataset, index) for index, doc in enumerate(corpus)
    }

    write_json(
        output_dir / "questions.json",
        [
            {
                "id": example.qid,
                "question": example.question,
                "answer": example.answer,
                "dataset": example.dataset,
                "split": example.split,
            }
            for example in examples
        ],
    )
    chunk_records = [
        _chunk_record(dataset, index, doc) for index, doc in enumerate(corpus)
    ]
    write_json(output_dir / "chunks.json", chunk_records)
    write_jsonl(
        output_dir / "chunk_meta.jsonl",
        [
            {
                "chunk_id": _chunk_id(dataset, index),
                "chunk_index": index,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "dataset": dataset,
            }
            for index, doc in enumerate(corpus)
        ],
    )
    write_jsonl(
        output_dir / "qrels.jsonl",
        [_build_qrel(example, chunk_id_by_doc_id) for example in examples],
    )
