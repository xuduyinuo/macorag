from __future__ import annotations

from pathlib import Path

from macorag.io_utils import write_json, write_jsonl
from macorag.schemas import CorpusDoc, Example


def _chunk_id(dataset: str, index: int) -> str:
    return f"{dataset}:chunk:{index}"


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
        [{"qid": example.qid, "question": example.question} for example in examples],
    )
    write_json(output_dir / "chunks.json", [doc.to_chunk_text() for doc in corpus])
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
        [
            {
                "qid": example.qid,
                "gold_doc_ids": [
                    fact.doc_id
                    for fact in example.supporting_facts
                    if fact.doc_id is not None
                ],
                "gold_chunk_ids": [
                    chunk_id_by_doc_id[fact.doc_id]
                    for fact in example.supporting_facts
                    if fact.doc_id is not None and fact.doc_id in chunk_id_by_doc_id
                ],
                "gold_titles": [fact.title for fact in example.supporting_facts],
                "gold_sentences": [
                    fact.text
                    for fact in example.supporting_facts
                    if fact.text is not None
                ],
            }
            for example in examples
        ],
    )
