from macorag.dataset_builders import (
    build_2wiki_example_from_row,
    build_hotpot_example_from_row,
    build_musique_canonical_from_rows,
)


def test_musique_answerable_rows_build_examples_and_corpus():
    row = {
        "id": "m1",
        "question": "Where was Alice's birthplace founded?",
        "answer": "France",
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
    assert report["examples"][0].qid == "m1"
    assert report["examples"][0].hop_count == 2
    assert len(report["examples"][0].supporting_facts) == 2
    assert len(report["examples"][0].evidence_chain) == 2
    assert len(report["corpus"]) == 3
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
