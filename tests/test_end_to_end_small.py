from macorag.dataset_builders import build_musique_canonical_from_rows
from macorag.io_utils import read_json, read_jsonl
from macorag.linearrag_adapter import build_linearrag_dataset


def test_small_musique_to_linearrag_end_to_end(tmp_path):
    rows = [
        {
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
    ]
    report = build_musique_canonical_from_rows(rows, split="train")
    build_linearrag_dataset(
        "musique",
        report["examples"],
        report["corpus"],
        tmp_path,
    )

    questions = read_json(tmp_path / "musique" / "questions.json")
    qrels = list(read_jsonl(tmp_path / "musique" / "qrels.jsonl"))

    assert report["errors"] == []
    assert questions[0]["id"] == "2hop__1_2"
    assert len(qrels[0]["gold_doc_ids"]) == 2
