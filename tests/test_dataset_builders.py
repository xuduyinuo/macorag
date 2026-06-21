import pytest

from data_processing.dataset_builders import (
    build_2wiki_example_from_row,
    build_hotpot_example_from_row,
    build_musique_canonical_from_rows,
)


def test_musique_answerable_rows_build_examples_and_corpus():
    row = {
        "id": "2hop__1_2",
        "question": "Where was Alice's birthplace founded?",
        "answer": "France",
        "answer_aliases": ["alias"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "Alice",
                "paragraph_text": "Alice was born in Paris.",
            },
            {
                "idx": 5,
                "title": "Paris",
                "paragraph_text": "Paris is the capital of France.",
            },
            {
                "idx": 10,
                "title": "France",
                "paragraph_text": "France was founded over many centuries.",
            },
        ],
        "question_decomposition": [
            {
                "question": "Where was Alice's birthplace founded?",
                "answer": "France",
                "paragraph_support_idx": 10,
            },
            {
                "question": "Where is Paris?",
                "answer": "France",
                "paragraph_support_idx": 5,
            },
        ],
    }

    report = build_musique_canonical_from_rows([row], split="train")

    assert report["errors"] == []
    assert len(report["examples"]) == 1
    assert report["examples"][0].qid == "2hop__1_2"
    assert report["examples"][0].hop_count == 2
    assert report["examples"][0].answer_aliases == ["alias"]
    assert report["examples"][0].metadata["answerable"] is True
    assert len(report["examples"][0].supporting_facts) == 2
    assert len(report["examples"][0].evidence_chain) == 2
    assert len(report["corpus"]) == 3
    assert all(doc.source == "musique_context" for doc in report["corpus"])
    assert any("2hop__1_2" in doc.metadata["linked_qids"] for doc in report["corpus"])
    assert report["examples"][0].usable_for_sft is True


def test_hotpot_row_maps_supporting_facts():
    row = {
        "id": "h1",
        "question": "Where was Alice born?",
        "answer": "Paris",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {"title": ["Alice"], "sent_id": [0]},
        "context": {"title": ["Alice"], "sentences": [["Alice was born in Paris."]]},
    }

    example = build_hotpot_example_from_row(row, split="train")

    assert example.dataset == "hotpotqa"
    assert example.supporting_facts[0].text == "Alice was born in Paris."
    assert example.metadata["level"] == "easy"


def test_hotpot_row_accepts_pyarrow_struct_scalars():
    pa = pytest.importorskip("pyarrow")
    supporting_facts = pa.scalar(
        {"title": ["Alice"], "sent_id": [0]},
        type=pa.struct(
            [
                ("title", pa.list_(pa.string())),
                ("sent_id", pa.list_(pa.int64())),
            ]
        ),
    )
    context = pa.scalar(
        {"title": ["Alice"], "sentences": [["Alice was born in Paris."]]},
        type=pa.struct(
            [
                ("title", pa.list_(pa.string())),
                ("sentences", pa.list_(pa.list_(pa.string()))),
            ]
        ),
    )
    row = {
        "id": "h_arrow",
        "question": "Where was Alice born?",
        "answer": "Paris",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": supporting_facts,
        "context": context,
    }

    example = build_hotpot_example_from_row(row, split="train")

    assert example.supporting_facts[0].text == "Alice was born in Paris."


def test_hotpot_row_accepts_pyarrow_list_struct_scalars():
    pa = pytest.importorskip("pyarrow")
    supporting_facts = pa.scalar(
        [{"title": "Alice", "sent_id": 0}],
        type=pa.list_(
            pa.struct(
                [
                    ("title", pa.string()),
                    ("sent_id", pa.int64()),
                ]
            )
        ),
    )
    context = pa.scalar(
        [{"title": "Alice", "sentences": ["Alice was born in Paris."]}],
        type=pa.list_(
            pa.struct(
                [
                    ("title", pa.string()),
                    ("sentences", pa.list_(pa.string())),
                ]
            )
        ),
    )
    row = {
        "id": "h_arrow_list",
        "question": "Where was Alice born?",
        "answer": "Paris",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": supporting_facts,
        "context": context,
    }

    example = build_hotpot_example_from_row(row, split="train")

    assert example.supporting_facts[0].text == "Alice was born in Paris."


def test_hotpot_blank_answer_is_not_usable():
    row = {
        "id": "h_blank",
        "question": "Where was Alice born?",
        "answer": " \n\t ",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {"title": ["Alice"], "sent_id": [0]},
        "context": {"title": ["Alice"], "sentences": [["Alice was born in Paris."]]},
    }

    example = build_hotpot_example_from_row(row, split="train")

    assert example.answer is None
    assert example.usable_for_sft is False
    assert example.usable_for_retrieval_eval is False


def test_musique_duplicate_paragraph_keeps_linked_qids_unique():
    row = {
        "id": "2hop__1_2",
        "question": "Where is Paris?",
        "answer": "France",
        "answerable": True,
        "paragraphs": [
            {
                "idx": 5,
                "title": "Paris",
                "paragraph_text": "Paris is the capital of France.",
            },
            {
                "idx": 6,
                "title": "Paris",
                "paragraph_text": "Paris is the capital of France.",
            },
        ],
        "question_decomposition": [
            {
                "question": "Where is Paris?",
                "answer": "France",
                "paragraph_support_idx": 5,
            }
        ],
    }

    report = build_musique_canonical_from_rows([row], split="train")

    assert len(report["corpus"]) == 1
    assert report["corpus"][0].metadata["linked_qids"] == ["2hop__1_2"]


def test_2wiki_row_maps_evidences_to_chain():
    row = {
        "id": "w1",
        "question": "Who is related to Bob?",
        "answer": "Alice",
        "type": "inference",
        "evidences": [["Bob", "sibling", "Alice"]],
        "supporting_facts": {"title": ["Bob"], "sent_id": [0]},
        "context": {"title": ["Bob"], "sentences": [["Bob is Alice's sibling."]]},
    }

    example = build_2wiki_example_from_row(row, split="train")

    assert example.dataset == "2wiki"
    assert example.evidence_chain[0].sub_question == "Bob sibling ?"
    assert example.evidence_chain[0].answer == "Alice"


def test_2wiki_evidence_only_row_is_usable():
    row = {
        "id": "w2",
        "question": "Who is related to Bob?",
        "answer": "Alice",
        "type": "inference",
        "evidences": [["Bob", "sibling", "Alice"]],
        "supporting_facts": {"title": [], "sent_id": []},
        "context": {"title": [], "sentences": []},
    }

    example = build_2wiki_example_from_row(row, split="train")

    assert len(example.evidence_chain) == 1
    assert example.usable_for_sft is True
    assert example.usable_for_retrieval_eval is True
