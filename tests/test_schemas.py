import json

import pytest

from macorag.schemas import CorpusDoc, Example, SupportingFact


def test_example_to_json_roundtrip():
    example = Example(
        qid="q1",
        dataset="2wiki",
        split="train",
        question="Who founded the company?",
        answer="Alice",
        answer_aliases=["A. Smith"],
        question_type="bridge",
        hop_count=2,
        supporting_facts=[
            SupportingFact(
                doc_id="d1",
                title="Alice",
                sent_id=0,
                text="Alice founded the company.",
                source="gold",
            )
        ],
        evidence_chain=[],
        context_doc_ids=["d1"],
        usable_for_sft=True,
        usable_for_retrieval_eval=True,
        quality_flags=[],
        metadata={"level": "medium"},
    )

    loaded = Example.from_json(example.to_json())

    assert loaded.qid == "q1"
    assert loaded.supporting_facts[0].title == "Alice"
    assert loaded.metadata["level"] == "medium"


def test_corpus_doc_chunk_text_includes_title():
    doc = CorpusDoc(
        doc_id="d1",
        dataset="hotpotqa",
        title="Document Title",
        text="Document body.",
        sentences=["Document body."],
        source="beir",
        metadata={},
    )

    assert doc.to_chunk_text() == "Document Title\nDocument body."


def test_example_from_json_parses_boolean_strings_strictly():
    payload = {
        "qid": "q1",
        "dataset": "2wiki",
        "split": "train",
        "question": "Who founded the company?",
        "answer": "Alice",
        "usable_for_sft": "false",
        "usable_for_retrieval_eval": "0",
    }

    loaded = Example.from_json(json.dumps(payload))

    assert loaded.usable_for_sft is False
    assert loaded.usable_for_retrieval_eval is False


def test_example_from_json_rejects_unknown_boolean_strings():
    payload = {
        "qid": "q1",
        "dataset": "2wiki",
        "split": "train",
        "question": "Who founded the company?",
        "answer": "Alice",
        "usable_for_sft": "maybe",
        "usable_for_retrieval_eval": False,
    }

    with pytest.raises(ValueError):
        Example.from_json(payload)
