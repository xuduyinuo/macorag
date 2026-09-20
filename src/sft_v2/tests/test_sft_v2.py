from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest

from sft_v2.generate_trajectories import (
    _deterministic_example_order,
    _materialize_train_validation_reserve_splits,
)
from sft_v2.config import load_config
from sft_v2.prompts import answer_messages, evidence_messages, load_prompt_contract, query_messages
from sft_v2.retrieval import JsonlOffsetStore, RetrievedPassage, UnifiedE5FaissRetriever, split_contents
from sft_v2.source import normalize_source_row
from sft_v2.trajectory import (
    TrajectoryGenerator,
    _normalize_answer,
    _shuffle_and_reindex_passages,
    answer_matches_any,
)


ROOT = Path(__file__).resolve().parents[1]


def test_source_adapter_maps_new_schema_and_dataset_name() -> None:
    example = normalize_source_row(
        {
            "id": "2wikimultihopqa:q1",
            "question": "Who wrote the book?",
            "golden_answers": ["A. Writer", "Writer"],
            "dataset": "2wikimultihopqa",
            "source_split": "train",
        },
        {"2wikimultihopqa": "2wiki"},
    )
    assert example.qid == "2wikimultihopqa:q1"
    assert example.dataset == "2wiki"
    assert example.answers == ("A. Writer", "Writer")


def test_prompt_contract_never_receives_gold_labels() -> None:
    contract = load_prompt_contract(ROOT / "prompts.yml")
    state = {"evidence": [], "retrieval_history": [], "retrieval_count": 0}
    observation = {"passages": [{"passage_id": "P0", "text": "public evidence"}]}
    rendered = json.dumps(
        [
            query_messages(contract, question="Question?", state=state),
            evidence_messages(contract, question="Question?", state=state, observation=observation),
            answer_messages(contract, question="Question?", state=state, round_index=0, max_rounds=4),
        ]
    ).casefold()
    assert "secret-gold-value" not in rendered
    assert "golden_answers" not in rendered


def test_contents_parser_and_random_access_offsets(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"id": "0", "contents": '"Title A"\nBody A'}) + "\n"
        + json.dumps({"id": "1", "contents": '"Title B"\nBody B'}) + "\n",
        encoding="utf-8",
    )
    store = JsonlOffsetStore(corpus, tmp_path / "offsets.u64")
    store.open()
    assert store.row_count == 2
    assert store.get(1)["id"] == "1"
    assert split_contents(store.get(0)["contents"]) == ("Title A", "Body A")


def test_answer_matches_any_alias() -> None:
    assert answer_matches_any("The Writer", ("A. Writer", "Writer"))


def test_answer_agent_rejects_rationale_field() -> None:
    with pytest.raises(ValueError, match="unexpected fields: rationale"):
        _normalize_answer({"can_answer": False, "answer": None, "rationale": "not allowed"})


def test_formal_config_has_exact_1252_1000_200_52_contract() -> None:
    config = load_config(ROOT / "config" / "teacher_trajectory.yml")
    assert config.accepted_target_total == 1252
    assert config.resolved_train_target_total == 1000
    assert config.validation_target_total == 200
    assert config.reserve_target_total == 52
    assert config.sample_workers == 8
    assert config.retrieval_batch_size == 8
    assert config.retrieval_batch_wait_ms == 1000
    assert config.role_validation_retries == 2
    assert config.shuffle_source_examples is True
    assert config.shuffle_retrieved_passages is True


def test_global_candidate_shuffle_is_deterministic_and_mixes_datasets() -> None:
    aliases = {"hotpotqa": "hotpotqa", "musique": "musique"}
    examples = [
        normalize_source_row(
            {
                "id": f"hotpotqa:{index}",
                "question": "Question?",
                "golden_answers": ["answer"],
                "dataset": "hotpotqa",
            },
            aliases,
        )
        for index in range(20)
    ] + [
        normalize_source_row(
            {
                "id": f"musique:{index}",
                "question": "Question?",
                "golden_answers": ["answer"],
                "dataset": "musique",
            },
            aliases,
        )
        for index in range(20)
    ]
    first = _deterministic_example_order(examples, seed=42, enabled=True)
    second = _deterministic_example_order(examples, seed=42, enabled=True)
    assert [item.qid for item in first] == [item.qid for item in second]
    assert {item.dataset for item in first[:10]} == {"hotpotqa", "musique"}
    assert [item.qid for item in first] != [item.qid for item in examples]


def test_retrieval_shuffle_is_reproducible_and_reassigns_pointers() -> None:
    passages = [
        RetrievedPassage(f"P{index}", str(index), f"T{index}", f"B{index}", 1.0 - index / 10, index)
        for index in range(5)
    ]
    first = _shuffle_and_reindex_passages(
        passages, seed=42, qid="q1", round_index=0, query="query", enabled=True
    )
    second = _shuffle_and_reindex_passages(
        passages, seed=42, qid="q1", round_index=0, query="query", enabled=True
    )
    assert [item.corpus_id for item in first] == [item.corpus_id for item in second]
    assert [item.pointer for item in first] == [f"P{index}" for index in range(5)]
    assert {item.corpus_id for item in first} == {str(index) for index in range(5)}
    assert [item.corpus_id for item in first] != [str(index) for index in range(5)]


def test_concurrent_searches_are_combined_into_one_retrieval_batch() -> None:
    import numpy as np

    class FakeEncoder:
        def __init__(self):
            self.calls = []

        def encode(self, queries):
            self.calls.append(list(queries))
            return np.asarray([[float(index)] for index, _ in enumerate(queries)], dtype=np.float32)

    class FakeIndex:
        ntotal = 100

        def search(self, vectors, k):
            size = len(vectors)
            return (
                np.ones((size, 1), dtype=np.float32),
                np.arange(size, dtype=np.int64).reshape(size, 1),
            )

    class FakeCorpus:
        def get(self, row_index):
            return {"id": str(row_index), "contents": f'"T{row_index}"\nB{row_index}'}

        def close(self):
            return None

    retriever = UnifiedE5FaissRetriever.__new__(UnifiedE5FaissRetriever)
    retriever.top_k = 1
    retriever.retrieval_batch_size = 4
    retriever.batch_wait_seconds = 0.1
    from queue import Queue
    import threading

    retriever._search_queue = Queue()
    retriever._closed = False
    retriever.retrieval_queries = 0
    retriever.retrieval_batches = 0
    retriever.max_observed_batch_size = 0
    retriever.encoder = FakeEncoder()
    retriever.index = FakeIndex()
    retriever.corpus = FakeCorpus()
    retriever._batch_worker = threading.Thread(target=retriever._batch_worker_loop, daemon=True)
    retriever._batch_worker.start()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(retriever.search, ["q0", "q1", "q2", "q3"]))
    stats = retriever.batch_stats()
    retriever.close()
    assert len(results) == 4
    assert stats["queries"] == 4
    assert stats["batches"] == 1
    assert stats["average_batch_size"] == 4.0
    assert len(retriever.encoder.calls) == 1


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, messages):
        self.calls += 1
        if self.calls == 1:
            return {"sub_goal": "find author", "query": "book author"}, {"id": "q"}
        if self.calls == 2:
            return {"selected_passage_ids": ["P0"]}, {"id": "e"}
        return {"can_answer": True, "answer": "Alice"}, {"id": "a"}


class _FakeRetriever:
    def search(self, query):
        return [RetrievedPassage("P0", "17", "Book", "The author is Alice.", 0.9, 17)]


class _AlwaysContinueClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, messages):
        role = self.calls % 3
        self.calls += 1
        if role == 0:
            return {"sub_goal": "find author", "query": f"book author {self.calls}"}, {}
        if role == 1:
            return {"selected_passage_ids": ["P0"]}, {}
        return {"can_answer": False, "answer": None}, {}


def test_three_agent_trajectory_uses_pointer_contract(tmp_path: Path) -> None:
    from sft_v2.config import TeacherConfig

    example = normalize_source_row(
        {"id": "hotpotqa:q", "question": "Who is the author?", "golden_answers": ["Alice"], "dataset": "hotpotqa"},
        {"hotpotqa": "hotpotqa"},
    )
    config = TeacherConfig(
        source_path=tmp_path / "source",
        output_dir=tmp_path / "out",
        corpus_path=tmp_path / "corpus",
        index_path=tmp_path / "index",
        index_manifest_path=tmp_path / "manifest",
        corpus_offsets_path=tmp_path / "offsets",
        retrieval_model_path="unused",
        prompt_path=ROOT / "prompts.yml",
        dataset_aliases={"hotpotqa": "hotpotqa"},
        max_rounds=4,
    )
    generator = TrajectoryGenerator(
        config=config,
        prompts=load_prompt_contract(ROOT / "prompts.yml"),
        client=_FakeClient(),
        retriever=_FakeRetriever(),
    )
    result = generator.generate(example)
    assert result["status"] == "accepted"
    turn = result["sample"]["trajectory"][0]
    assert turn["update_evidence"]["selected_passage_ids"] == ["P0"]
    assert "rationale" not in turn["update_evidence"]
    assert turn["answer"]["answer"] == "Alice"
    assert "rationale" not in turn["answer"]


def test_final_round_rejects_can_answer_false(tmp_path: Path) -> None:
    from sft_v2.config import TeacherConfig

    example = normalize_source_row(
        {"id": "hotpotqa:q", "question": "Who is the author?", "golden_answers": ["Alice"], "dataset": "hotpotqa"},
        {"hotpotqa": "hotpotqa"},
    )
    config = TeacherConfig(
        source_path=tmp_path / "source",
        output_dir=tmp_path / "out",
        corpus_path=tmp_path / "corpus",
        index_path=tmp_path / "index",
        index_manifest_path=tmp_path / "manifest",
        corpus_offsets_path=tmp_path / "offsets",
        retrieval_model_path="unused",
        prompt_path=ROOT / "prompts.yml",
        dataset_aliases={"hotpotqa": "hotpotqa"},
        max_rounds=4,
        role_validation_retries=0,
    )
    generator = TrajectoryGenerator(
        config=config,
        prompts=load_prompt_contract(ROOT / "prompts.yml"),
        client=_AlwaysContinueClient(),
        retriever=_FakeRetriever(),
    )
    with pytest.raises(ValueError, match="can_answer=true on the final round"):
        generator.generate(example)


class _RetryFinalAnswerClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, messages):
        self.calls += 1
        responses = [
            {"sub_goal": "find author", "query": "book author"},
            {"selected_passage_ids": ["P0"]},
            {"can_answer": False, "answer": None},
            {"can_answer": True, "answer": "Alice"},
        ]
        return responses[self.calls - 1], {"call": self.calls}


def test_final_round_refusal_retries_only_answer_action(tmp_path: Path) -> None:
    from sft_v2.config import TeacherConfig

    example = normalize_source_row(
        {"id": "hotpotqa:q", "question": "Who is the author?", "golden_answers": ["Alice"], "dataset": "hotpotqa"},
        {"hotpotqa": "hotpotqa"},
    )
    client = _RetryFinalAnswerClient()
    stages = []
    config = TeacherConfig(
        source_path=tmp_path / "source", output_dir=tmp_path / "out",
        corpus_path=tmp_path / "corpus", index_path=tmp_path / "index",
        index_manifest_path=tmp_path / "manifest", corpus_offsets_path=tmp_path / "offsets",
        retrieval_model_path="unused", prompt_path=ROOT / "prompts.yml",
        dataset_aliases={"hotpotqa": "hotpotqa"}, max_rounds=1,
    )
    result = TrajectoryGenerator(
        config=config, prompts=load_prompt_contract(ROOT / "prompts.yml"),
        client=client, retriever=_FakeRetriever(), progress_callback=stages.append,
    ).generate(example)
    assert result["status"] == "accepted"
    assert client.calls == 4
    assert stages.count("retrieval") == 1
    assert stages.count("answer_retry") == 1
    trace = result["sample"]["trajectory"][0]["teacher_traces"]["answer_generator"]
    assert trace["validation_retry_count"] == 1


class _RetryLeakingQueryClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, messages):
        self.calls += 1
        responses = [
            {"sub_goal": "leak", "query": "Alice"},
            {"sub_goal": "find author", "query": "book author"},
            {"selected_passage_ids": ["P0"]},
            {"can_answer": True, "answer": "Alice"},
        ]
        return responses[self.calls - 1], {"call": self.calls}


def test_answer_leaking_query_retries_only_query_action(tmp_path: Path) -> None:
    from sft_v2.config import TeacherConfig

    example = normalize_source_row(
        {"id": "hotpotqa:q", "question": "Who is the author?", "golden_answers": ["Alice"], "dataset": "hotpotqa"},
        {"hotpotqa": "hotpotqa"},
    )
    client = _RetryLeakingQueryClient()
    stages = []
    config = TeacherConfig(
        source_path=tmp_path / "source", output_dir=tmp_path / "out",
        corpus_path=tmp_path / "corpus", index_path=tmp_path / "index",
        index_manifest_path=tmp_path / "manifest", corpus_offsets_path=tmp_path / "offsets",
        retrieval_model_path="unused", prompt_path=ROOT / "prompts.yml",
        dataset_aliases={"hotpotqa": "hotpotqa"}, max_rounds=1,
    )
    result = TrajectoryGenerator(
        config=config, prompts=load_prompt_contract(ROOT / "prompts.yml"),
        client=client, retriever=_FakeRetriever(), progress_callback=stages.append,
    ).generate(example)
    assert result["status"] == "accepted"
    assert client.calls == 4
    assert stages.count("query_retry") == 1
    assert stages.count("retrieval") == 1


def test_global_split_materializes_exact_counts_without_dataset_quota(tmp_path: Path) -> None:
    from sft_v2.config import TeacherConfig

    config = TeacherConfig(
        source_path=tmp_path / "source",
        output_dir=tmp_path / "out",
        corpus_path=tmp_path / "corpus",
        index_path=tmp_path / "index",
        index_manifest_path=tmp_path / "manifest",
        corpus_offsets_path=tmp_path / "offsets",
        retrieval_model_path="unused",
        prompt_path=ROOT / "prompts.yml",
        dataset_aliases={"a": "a", "b": "b"},
        candidate_limits_by_dataset={"a": 3, "b": 3},
        accepted_target_total=4,
        train_target_total=2,
        validation_target_total=1,
        seed=7,
    )
    accepted = [
        {"qid": "a0", "dataset": "a"},
        {"qid": "b0", "dataset": "b"},
        {"qid": "a1", "dataset": "a"},
        {"qid": "b1", "dataset": "b"},
    ]
    order = {"a0": 0, "b0": 1, "a1": 2, "b1": 3}
    train, validation, reserve = _materialize_train_validation_reserve_splits(
        config, accepted, source_order=order
    )
    assert len(train) == 2
    assert len(validation) == 1
    assert len(reserve) == 1
    assert {row["sft_split"] for row in train} == {"train"}
    assert {row["sft_split"] for row in validation} == {"validation"}
    assert {row["sft_split"] for row in reserve} == {"reserve"}
    assert sum(1 for _ in (config.output_dir / "train_sft.jsonl").open()) == 2
    assert sum(1 for _ in (config.output_dir / "validation_sft.jsonl").open()) == 1
    assert sum(1 for _ in (config.output_dir / "reserve_sft.jsonl").open()) == 1
