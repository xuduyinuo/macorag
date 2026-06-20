import json

from macorag.io_utils import read_json
from macorag.linearrag_adapter import build_linearrag_dataset
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

    assert read_json(output_dir / "questions.json") == [
        {"qid": "h1", "question": "Where was Alice born?"}
    ]
    assert read_json(output_dir / "chunks.json") == ["Alice\nAlice was born in Paris."]

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
            "gold_sentences": ["Alice was born in Paris."],
        }
    ]
