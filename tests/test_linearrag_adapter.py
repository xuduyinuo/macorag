import json

from macorag.io_utils import read_json
from macorag.linearrag_adapter import build_linearrag_dataset
from macorag.retrieval_env import InMemoryRetrievalEnv
from macorag.schemas import CorpusDoc, Example, SupportingFact


def test_build_linearrag_dataset_writes_questions_chunks_and_qrels(tmp_path):
    example = Example(
        qid="h1",
        dataset="hotpotqa",
        split="train",
        question="Where was Alice born?",
        answer="Paris",
        answer_aliases=[],
        question_type="bridge",
        hop_count=2,
        supporting_facts=[
            SupportingFact(
                doc_id="doc-a",
                title="Alice",
                sent_id=0,
                text="Alice was born in Paris.",
                source="gold",
            )
        ],
        evidence_chain=[],
        context_doc_ids=["doc-a"],
        usable_for_sft=True,
        usable_for_retrieval_eval=True,
        quality_flags=[],
        metadata={},
    )
    corpus = [
        CorpusDoc(
            doc_id="doc-a",
            dataset="hotpotqa",
            title="Alice",
            text="Alice was born in Paris.",
            sentences=["Alice was born in Paris."],
            source="hotpot_context",
            metadata={},
        )
    ]

    build_linearrag_dataset("hotpotqa", [example], corpus, tmp_path)

    output_dir = tmp_path / "hotpotqa"
    assert (output_dir / "questions.json").is_file()
    assert (output_dir / "chunks.json").is_file()
    assert (output_dir / "chunk_meta.jsonl").is_file()
    assert (output_dir / "qrels.jsonl").is_file()

    questions = read_json(output_dir / "questions.json")
    assert questions[0] == {
        "id": "h1",
        "question": "Where was Alice born?",
        "answer": "Paris",
        "dataset": "hotpotqa",
        "split": "train",
    }
    assert "qid" not in questions[0]
    assert read_json(output_dir / "chunks.json") == [
        {
            "chunk_id": "hotpotqa:chunk:0",
            "chunk_index": 0,
            "doc_id": "doc-a",
            "title": "Alice",
            "text": "Alice was born in Paris.",
            "sentences": [{"sent_id": 0, "text": "Alice was born in Paris."}],
            "dataset": "hotpotqa",
            "source": "hotpot_context",
        }
    ]

    chunk_meta = [
        json.loads(line) for line in (output_dir / "chunk_meta.jsonl").read_text().splitlines()
    ]
    assert chunk_meta == [
        {
            "chunk_id": "hotpotqa:chunk:0",
            "chunk_index": 0,
            "doc_id": "doc-a",
            "title": "Alice",
            "source": "hotpot_context",
            "dataset": "hotpotqa",
        }
    ]

    qrels = [
        json.loads(line) for line in (output_dir / "qrels.jsonl").read_text().splitlines()
    ]
    assert qrels == [
        {
            "qid": "h1",
            "gold_doc_ids": ["doc-a"],
            "gold_chunk_ids": ["hotpotqa:chunk:0"],
            "gold_titles": ["Alice"],
            "gold_sentences": [
                {
                    "doc_id": "doc-a",
                    "sent_id": 0,
                    "title": "Alice",
                    "text": "Alice was born in Paris.",
                }
            ],
        }
    ]


def test_build_linearrag_dataset_chunks_feed_in_memory_retrieval_env(tmp_path):
    example = Example(
        qid="h1",
        dataset="hotpotqa",
        split="train",
        question="Where was Alice born?",
        answer="Paris",
        answer_aliases=[],
        question_type="bridge",
        hop_count=2,
        supporting_facts=[],
        evidence_chain=[],
        context_doc_ids=["doc-a"],
        usable_for_sft=True,
        usable_for_retrieval_eval=True,
        quality_flags=[],
        metadata={},
    )
    corpus = [
        CorpusDoc(
            doc_id="doc-a",
            dataset="hotpotqa",
            title="Alice",
            text="Alice was born in Paris.",
            sentences=["Alice was born in Paris."],
            source="hotpot_context",
            metadata={},
        )
    ]

    build_linearrag_dataset("hotpotqa", [example], corpus, tmp_path)

    questions = read_json(tmp_path / "hotpotqa" / "questions.json")
    chunks = read_json(tmp_path / "hotpotqa" / "chunks.json")
    env = InMemoryRetrievalEnv(questions=questions, chunks=chunks, retrieval_budget=2)

    state = env.reset("h1")
    observation = env.step("Alice Paris", top_k=1)

    assert state["qid"] == "h1"
    assert observation[0]["chunk_id"] == "hotpotqa:chunk:0"
    assert observation[0]["doc_id"] == "doc-a"
    assert "score" in observation[0]


def test_build_linearrag_dataset_deduplicates_and_filters_qrels(tmp_path):
    example = Example(
        qid="h2",
        dataset="hotpotqa",
        split="train",
        question="Who is connected to Alice?",
        answer="Bob",
        answer_aliases=[],
        question_type="bridge",
        hop_count=2,
        supporting_facts=[
            SupportingFact(
                doc_id="doc-a",
                title="Alice",
                sent_id=0,
                text="Alice first sentence.",
            ),
            SupportingFact(
                doc_id="doc-a",
                title="Alice",
                sent_id=0,
                text="Alice first sentence.",
            ),
            SupportingFact(
                doc_id="doc-a",
                title="Alice",
                sent_id=1,
                text="Alice duplicate doc sentence.",
            ),
            SupportingFact(
                doc_id=None,
                title="Missing Doc",
                sent_id=2,
                text="Fact without a doc id.",
            ),
            SupportingFact(
                doc_id="",
                title="Empty Doc",
                sent_id=5,
                text="Fact with an empty doc id.",
            ),
            SupportingFact(
                doc_id="doc-b",
                title="",
                sent_id=3,
                text="Fact with an empty title.",
            ),
            SupportingFact(
                doc_id="doc-missing",
                title="Missing Corpus",
                sent_id=4,
                text="Fact with a doc outside corpus.",
            ),
            SupportingFact(
                doc_id="doc-b",
                title="Bob",
                sent_id=6,
                text=None,
            ),
        ],
        evidence_chain=[],
        context_doc_ids=["doc-a", "doc-b", "doc-missing"],
        usable_for_sft=True,
        usable_for_retrieval_eval=True,
        quality_flags=[],
        metadata={},
    )
    corpus = [
        CorpusDoc(
            doc_id="doc-a",
            dataset="hotpotqa",
            title="Alice",
            text="Alice body.",
            sentences=["Alice body."],
            source="hotpot_context",
            metadata={},
        ),
        CorpusDoc(
            doc_id="doc-b",
            dataset="hotpotqa",
            title="Bob",
            text="Bob body.",
            sentences=["Bob body."],
            source="hotpot_context",
            metadata={},
        ),
    ]

    build_linearrag_dataset("hotpotqa", [example], corpus, tmp_path)

    qrels = [
        json.loads(line)
        for line in (tmp_path / "hotpotqa" / "qrels.jsonl").read_text().splitlines()
    ]
    assert qrels == [
        {
            "qid": "h2",
            "gold_doc_ids": ["doc-a", "doc-b", "doc-missing"],
            "gold_chunk_ids": ["hotpotqa:chunk:0", "hotpotqa:chunk:1"],
            "gold_titles": ["Alice", "Missing Doc", "Empty Doc", "Missing Corpus", "Bob"],
            "gold_sentences": [
                {
                    "doc_id": "doc-a",
                    "sent_id": 0,
                    "title": "Alice",
                    "text": "Alice first sentence.",
                },
                {
                    "doc_id": "doc-a",
                    "sent_id": 1,
                    "title": "Alice",
                    "text": "Alice duplicate doc sentence.",
                },
                {
                    "doc_id": None,
                    "sent_id": 2,
                    "title": "Missing Doc",
                    "text": "Fact without a doc id.",
                },
                {
                    "doc_id": "",
                    "sent_id": 5,
                    "title": "Empty Doc",
                    "text": "Fact with an empty doc id.",
                },
                {
                    "doc_id": "doc-b",
                    "sent_id": 3,
                    "title": "",
                    "text": "Fact with an empty title.",
                },
                {
                    "doc_id": "doc-missing",
                    "sent_id": 4,
                    "title": "Missing Corpus",
                    "text": "Fact with a doc outside corpus.",
                },
            ],
        }
    ]
