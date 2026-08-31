from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from rag import AnswerPromptContext

from data_processing.generate_teacher_sft import (
    SFTConfig,
    _retrieve,
    build_answer_messages,
    build_planning_messages,
    build_trajectory_sample,
    build_update_messages,
    query_has_unseen_intermediate_terms,
    answer_supported_by_evidence,
    finalize_teacher_output,
    generate_sft_dataset,
    normalize_observation,
    normalize_retrieval_action,
    normalize_update_evidence,
    parse_teacher_json,
    read_examples,
    validate_trajectory_sample,
)


FORBIDDEN_PROMPT_TERMS = (
    "multi-agent",
    "多智能体",
    "agent_role",
    "planner_retriever",
    "evidence_answerer",
    "你是",
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_read_examples_limits_each_dataset(tmp_path: Path) -> None:
    source_root = tmp_path / "trajectory_test"
    _write_jsonl(
        source_root / "toyqa" / "toyqa_train.jsonl",
        [
            {"qid": "q1", "dataset": "toyqa", "question": "Q1?", "answer": "A1"},
            {"qid": "q2", "dataset": "toyqa", "question": "Q2?", "answer": "A2"},
        ],
    )

    config = SFTConfig(
        datasets=["toyqa"],
        source_root=source_root,
        output_dir=tmp_path / "sft",
        max_examples_per_dataset=1,
    )

    rows = read_examples(config)

    assert [row["qid"] for row in rows] == ["q1"]


def test_parse_teacher_json_accepts_fenced_json() -> None:
    parsed = parse_teacher_json(
        """```json
        {"plan": {"sub_query": "Who directed The Tripper?"}, "answer": {"answer": "American"}}
        ```"""
    )

    assert parsed["plan"]["sub_query"] == "Who directed The Tripper?"
    assert parsed["answer"]["answer"] == "American"


def test_teacher_sft_script_loads_gitignored_env_file() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "generate_teacher_sft.sh").read_text(encoding="utf-8")
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")

    assert "ENV_FILE" in script
    assert 'source "${ENV_FILE}"' in script
    assert "-m data_processing.generate_teacher_sft" in script
    assert "config/generate_teacher_sft.yml" in script
    assert "CONFIG_PATH" in script
    assert ".env" in gitignore.splitlines()


def test_final_update_prompt_requires_answer() -> None:
    messages = build_update_messages(
        example={
            "qid": "q1",
            "dataset": "toyqa",
            "question": "Who fathered the leader?",
            "answer": "Estêvão da Gama",
            "supporting_facts": [{"title": "Vasco da Gama", "text": "His father was Estêvão da Gama."}],
        },
        state={"evidence": [], "retrieval_history": [], "retrieval_count": 3},
        plan={"retrieval": {"query": "Vasco da Gama father"}},
        observation={"passages": ["Vasco da Gama\nHis father was Estêvão da Gama."], "scores": [0.8]},
    )

    content = "\n".join(message["content"] for message in messages)
    assert messages[0]["content"].startswith("Task: select evidence from the latest observation.")
    assert "You are a retrieval-augmented reasoning assistant" not in content
    assert "final retrieval round" not in messages[1]["content"]
    assert "supporting_facts" not in messages[1]["content"]
    assert "gold" not in messages[1]["content"].lower()
    assert "Keep rationale fields short" in messages[1]["content"]
    assert '"answer"' not in messages[1]["content"]
    assert not any(term in content for term in FORBIDDEN_PROMPT_TERMS)


def test_planning_prompt_omits_top_k_and_rationale() -> None:
    messages = build_planning_messages(
        example={"qid": "q1", "dataset": "toyqa", "question": "Who directed The Tripper?"},
        state={"evidence": [], "retrieval_history": [], "retrieval_count": 0},
    )

    content = messages[1]["content"]
    all_content = "\n".join(message["content"] for message in messages)
    assert messages[0]["content"].startswith("Task: plan the next knowledge-base query.")
    assert "You are a retrieval-augmented reasoning assistant" not in all_content
    assert "query_retriever" in content
    assert "top_k" not in content
    assert "rationale" not in content
    assert "Do not include inferred intermediate answers" in content
    assert not any(term in all_content for term in FORBIDDEN_PROMPT_TERMS)


def test_answer_prompt_uses_accumulated_state_only() -> None:
    messages = build_answer_messages(
        example={"qid": "q1", "dataset": "toyqa", "question": "Who directed The Tripper?"},
        state={
            "evidence": [{"text": "The Tripper was directed by David Arquette."}],
            "retrieval_history": [{"query": "The Tripper director"}],
            "retrieval_count": 1,
        },
        context=AnswerPromptContext(round_index=1, max_rounds=4),
    )

    content = "\n".join(message["content"] for message in messages)
    assert messages[0]["content"].startswith("Task: answer from accumulated evidence.")
    assert "Use selected evidence in <state>" in content
    assert "You are a retrieval-augmented reasoning assistant" not in content
    assert "<observation>" not in content
    assert '"answer"' in content
    assert "can_answer=false" in content
    assert not any(term in content for term in FORBIDDEN_PROMPT_TERMS)


def test_finalize_teacher_output_rejects_false_answer_on_last_round() -> None:
    teacher_output = {
        "plan": {},
        "retrieval": {},
        "update_evidence": {"selected": [], "rationale": "No enough evidence."},
        "answer": {"can_answer": False, "answer": None, "rationale": "Still not enough evidence."},
    }

    with pytest.raises(ValueError, match="final answer"):
        finalize_teacher_output(
            teacher_output,
            example={"answer": "Estêvão da Gama"},
            context=AnswerPromptContext(round_index=3, max_rounds=4),
        )


def test_finalize_teacher_output_normalizes_forced_final_answer() -> None:
    teacher_output = {
        "answer": {
            "can_answer": True,
            "answer": "Wrong",
            "rationale": "This contains an inconsistent historical detour.",
        }
    }

    finalized = finalize_teacher_output(
        teacher_output,
        example={"answer": "Estêvão da Gama"},
        context=AnswerPromptContext(round_index=3, max_rounds=4),
    )

    assert finalized["answer"]["answer"] == "Wrong"
    assert finalized["answer"]["can_answer"] is True
    assert "inconsistent historical detour" in finalized["answer"]["rationale"]


def test_final_teacher_answer_prompt_has_strict_nonempty_answer_semantics() -> None:
    messages = build_answer_messages(
        example={"qid": "q1", "dataset": "toyqa", "question": "Q?", "answer": "SECRET"},
        state={"evidence": [], "retrieval_history": [], "retrieval_count": 4},
        context=AnswerPromptContext(round_index=3, max_rounds=4),
    )

    content = "\n".join(message["content"] for message in messages)
    assert "can_answer=true" in content
    assert "non-empty" in content
    assert "fallback_guess:" in content
    assert "SECRET" not in content


def test_build_trajectory_sample_groups_rounds_in_one_record() -> None:
    example = {
        "qid": "q1",
        "dataset": "toyqa",
        "split": "train",
        "question": "Who directed The Tripper?",
        "answer": "David Arquette",
        "question_type": "bridge",
        "hop_count": 2,
    }
    turns = [
        {
            "round": 0,
            "state": {"retrieval_count": 0},
            "plan": {"sub_query": "Who directed The Tripper?"},
            "retrieval": {"query": "The Tripper director"},
            "observation": {"passages": ["The Tripper was directed by David Arquette."], "scores": [0.9]},
            "update_evidence": {"selected": [{"rank": 1, "evidence": "directed by David Arquette"}]},
            "answer": {"can_answer": True, "answer": "David Arquette"},
        }
    ]

    sample = build_trajectory_sample(example=example, turns=turns)

    assert sample["qid"] == "q1"
    assert sample["trajectory"] == turns
    assert sample["final_answer"] == "David Arquette"
    assert "messages" not in sample


def test_build_trajectory_sample_truncates_after_first_final_answer() -> None:
    example = {"qid": "q1", "dataset": "toyqa", "question": "Q?", "answer": "A"}
    sample = build_trajectory_sample(
        example=example,
        turns=[
            {"round": 0, "answer": {"can_answer": False, "answer": None}},
            {"round": 1, "answer": {"can_answer": True, "answer": "A"}},
            {"round": 2, "answer": {"can_answer": True, "answer": "late duplicate"}},
        ],
    )

    assert [turn["round"] for turn in sample["trajectory"]] == [0, 1]
    assert sample["final_answer"] == "A"


def test_generate_sft_dataset_writes_one_record_per_question(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "trajectory_test"
    _write_jsonl(
        source_root / "toyqa" / "toyqa_train.jsonl",
        [
            {
                "qid": "q1",
                "dataset": "toyqa",
                "split": "train",
                "question": "Q1?",
                "answer": "A1",
                "hop_count": 2,
            }
        ],
    )

    def fake_retrieve(config, dataset, query):
        return {"query": query, "passages": ["Evidence A1"], "scores": [1.0]}

    monkeypatch.setattr("data_processing.generate_teacher_sft._retrieve", fake_retrieve)

    summary = generate_sft_dataset(
        SFTConfig(
            datasets=["toyqa"],
            source_root=source_root,
            output_dir=tmp_path / "sft",
            max_examples_per_dataset=1,
            max_rounds=3,
            dry_run=True,
            resume=False,
        )
    )

    rows = [json.loads(line) for line in (tmp_path / "sft" / "toyqa_sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["samples_processed"] == 1
    assert summary["samples_seen"] == 1
    assert summary["samples_skipped_existing"] == 0
    assert summary["samples_written"] == 1
    assert summary["samples_filtered"] == 0
    assert summary["filter_reasons"] == {}
    assert summary["dataset_stats"]["toyqa"]["samples_seen"] == 1
    assert summary["dataset_stats"]["toyqa"]["samples_written"] == 1
    assert summary["dataset_stats"]["toyqa"]["samples_filtered"] == 0
    assert summary["dataset_stats"]["toyqa"]["filter_reasons"] == {}
    assert summary["contains_messages"] is False
    assert len(rows) == 1
    assert rows[0]["qid"] == "q1"
    assert len(rows[0]["trajectory"]) == 2
    assert rows[0]["final_answer"] == "A1"
    assert not (tmp_path / "sft" / "teacher_traces.jsonl").exists()


def test_generate_sft_dataset_reports_filter_reasons(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "trajectory_test"
    _write_jsonl(
        source_root / "toyqa" / "toyqa_train.jsonl",
        [
            {
                "qid": "q1",
                "dataset": "toyqa",
                "split": "train",
                "question": "Where was Adam Gontier born?",
                "answer": "Canada",
                "hop_count": 1,
            },
            {
                "qid": "q2",
                "dataset": "toyqa",
                "split": "train",
                "question": "Q2?",
                "answer": "A2",
                "hop_count": 1,
            },
        ],
    )

    def fake_retrieve(config, dataset, query):
        return {"query": query, "passages": ["Evidence A2"], "scores": [1.0]}

    def fake_dry_plan(example, state, top_k):
        if example["qid"] == "q1":
            query = "Canada birthplace"
        else:
            query = "Q2"
        return {
            "plan": {"sub_goal": query, "sub_query": query},
            "retrieval": {"query": query},
        }

    monkeypatch.setattr("data_processing.generate_teacher_sft._retrieve", fake_retrieve)
    monkeypatch.setattr("data_processing.generate_teacher_sft._dry_plan", fake_dry_plan)

    summary = generate_sft_dataset(
        SFTConfig(
            datasets=["toyqa"],
            source_root=source_root,
            output_dir=tmp_path / "sft",
            max_examples_per_dataset=2,
            max_rounds=1,
            dry_run=True,
            resume=False,
        )
    )

    assert summary["samples_processed"] == 2
    assert summary["samples_seen"] == 2
    assert summary["samples_skipped_existing"] == 0
    assert summary["samples_written"] == 1
    assert summary["samples_filtered"] == 1
    assert summary["filter_reasons"]["unseen_intermediate_query"] == 1
    assert summary["dataset_stats"]["toyqa"]["samples_written"] == 1
    assert summary["dataset_stats"]["toyqa"]["samples_filtered"] == 1
    assert summary["dataset_stats"]["toyqa"]["filter_reasons"]["unseen_intermediate_query"] == 1
    filtered_rows = [
        json.loads(line)
        for line in (tmp_path / "sft" / "teacher_filtered.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(filtered_rows) == 1
    assert filtered_rows[0]["stage"] == "planning"
    assert filtered_rows[0]["filter_reasons"] == ["unseen_intermediate_query"]
    assert filtered_rows[0]["trajectory"][0]["raw_responses"]["query_retriever"]


def test_generate_sft_dataset_reports_dataset_stats(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "trajectory_test"
    _write_jsonl(
        source_root / "alpha" / "alpha_train.jsonl",
        [{"qid": "a1", "dataset": "alpha", "split": "train", "question": "Alpha?", "answer": "Alpha"}],
    )
    _write_jsonl(
        source_root / "beta" / "beta_train.jsonl",
        [{"qid": "b1", "dataset": "beta", "split": "train", "question": "Where?", "answer": "Canada"}],
    )

    def fake_retrieve(config, dataset, query):
        return {"query": query, "passages": [f"{dataset} evidence"], "scores": [1.0]}

    def fake_dry_plan(example, state, top_k):
        query = "Canada country" if example["dataset"] == "beta" else example["question"]
        return {"plan": {"sub_goal": query, "sub_query": query}, "retrieval": {"query": query}}

    monkeypatch.setattr("data_processing.generate_teacher_sft._retrieve", fake_retrieve)
    monkeypatch.setattr("data_processing.generate_teacher_sft._dry_plan", fake_dry_plan)

    summary = generate_sft_dataset(
        SFTConfig(
            datasets=["alpha", "beta"],
            source_root=source_root,
            output_dir=tmp_path / "sft",
            max_examples_per_dataset=1,
            max_rounds=1,
            dry_run=True,
            resume=False,
        )
    )

    assert summary["samples_seen"] == 2
    assert summary["samples_written"] == 1
    assert summary["samples_filtered"] == 1
    assert summary["dataset_stats"]["alpha"]["samples_seen"] == 1
    assert summary["dataset_stats"]["alpha"]["samples_written"] == 1
    assert summary["dataset_stats"]["alpha"]["samples_filtered"] == 0
    assert summary["dataset_stats"]["beta"]["samples_seen"] == 1
    assert summary["dataset_stats"]["beta"]["samples_written"] == 0
    assert summary["dataset_stats"]["beta"]["samples_filtered"] == 1
    assert summary["dataset_stats"]["beta"]["filter_reasons"]["unseen_intermediate_query"] == 1


def test_generate_sft_dataset_counts_filtered_samples_not_error_events(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "trajectory_test"
    _write_jsonl(
        source_root / "toyqa" / "toyqa_train.jsonl",
        [
            {"qid": "q1", "dataset": "toyqa", "split": "train", "question": "Q1?", "answer": "A1"},
            {"qid": "q2", "dataset": "toyqa", "split": "train", "question": "Q2?", "answer": "A2"},
        ],
    )

    def fake_retrieve(config, dataset, query):
        answer = "A1" if str(query).startswith("Q1") else "A2"
        return {"query": query, "passages": [f"{query}\n{answer} evidence."], "scores": [1.0]}

    def fake_dry_plan(example, state, top_k):
        query = example["question"]
        return {"plan": {"sub_goal": query, "sub_query": query}, "retrieval": {"query": query}}

    def fake_dry_update(example, state, observation):
        return {
            "update_evidence": {"selected_passage_ids": [0], "rationale": "Evidence selected."},
            "answer": {"can_answer": True, "answer": example["answer"], "rationale": "Supported."},
        }

    def fake_validate(sample):
        if sample["qid"] == "q1":
            return ["round 0 has leaked evidence", "round 0 retrieval_count mismatch"]
        return []

    monkeypatch.setattr("data_processing.generate_teacher_sft._retrieve", fake_retrieve)
    monkeypatch.setattr("data_processing.generate_teacher_sft._dry_plan", fake_dry_plan)
    monkeypatch.setattr("data_processing.generate_teacher_sft._dry_update", fake_dry_update)
    monkeypatch.setattr("data_processing.generate_teacher_sft.validate_trajectory_sample", fake_validate)

    summary = generate_sft_dataset(
        SFTConfig(
            datasets=["toyqa"],
            source_root=source_root,
            output_dir=tmp_path / "sft",
            max_examples_per_dataset=2,
            max_rounds=1,
            dry_run=True,
            resume=False,
        )
    )

    assert summary["samples_processed"] == 2
    assert summary["samples_written"] == 1
    assert summary["samples_filtered"] == 1
    assert sum(summary["filter_reasons"].values()) == 2
    assert summary["filtered_records"] == 1
    assert summary["filter_reason_occurrences"] == summary["filter_reasons"]
    assert summary["filtered_output"].endswith("teacher_filtered.jsonl")

    filtered_rows = [
        json.loads(line)
        for line in (tmp_path / "sft" / "teacher_filtered.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(filtered_rows) == 1
    record = filtered_rows[0]
    assert record["dataset"] == "toyqa"
    assert record["qid"] == "q1"
    assert record["stage"] == "validation"
    assert record["validation_errors"] == [
        "round 0 has leaked evidence",
        "round 0 retrieval_count mismatch",
    ]
    assert record["answer_exact_match"] == 1
    assert record["answer_contain"] == 1
    assert record["answer_f1"] == 1.0
    assert record["evidence_supported"] is True
    assert record["trajectory"][0]["raw_responses"]["query_retriever"]
    assert record["prompt_contract_fingerprint"]
    assert "retrieval_index_fingerprint" in record


def test_generate_sft_dataset_reports_resume_skips_separately(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "trajectory_test"
    output_dir = tmp_path / "sft"
    _write_jsonl(
        source_root / "toyqa" / "toyqa_train.jsonl",
        [
            {"qid": "q1", "dataset": "toyqa", "split": "train", "question": "Q1?", "answer": "A1"},
            {"qid": "q2", "dataset": "toyqa", "split": "train", "question": "Q2?", "answer": "A2"},
        ],
    )
    _write_jsonl(
        output_dir / "toyqa_sft.jsonl",
        [{"qid": "q1", "dataset": "toyqa", "trajectory": [], "final_answer": "A1"}],
    )

    def fake_retrieve(config, dataset, query):
        return {"query": query, "passages": ["Q2\nA2 evidence."], "scores": [1.0]}

    def fake_dry_plan(example, state, top_k):
        return {"plan": {"sub_goal": "Find Q2", "sub_query": "Q2"}, "retrieval": {"query": "Q2"}}

    def fake_dry_update(example, state, observation):
        return {
            "update_evidence": {"selected_passage_ids": [0], "rationale": "Supports answer."},
            "answer": {"can_answer": True, "answer": "A2", "rationale": "Evidence supports it."},
        }

    monkeypatch.setattr("data_processing.generate_teacher_sft._retrieve", fake_retrieve)
    monkeypatch.setattr("data_processing.generate_teacher_sft._dry_plan", fake_dry_plan)
    monkeypatch.setattr("data_processing.generate_teacher_sft._dry_update", fake_dry_update)

    summary = generate_sft_dataset(
        SFTConfig(
            datasets=["toyqa"],
            source_root=source_root,
            output_dir=output_dir,
            max_examples_per_dataset=2,
            max_rounds=1,
            dry_run=True,
            resume=True,
        )
    )

    assert summary["samples_seen"] == 2
    assert summary["samples_processed"] == 1
    assert summary["samples_skipped_existing"] == 1
    assert summary["samples_written"] == 1
    assert summary["samples_filtered"] == 0
    assert summary["filter_reasons"] == {}
    assert summary["total_samples"] == 2


def test_generate_sft_dataset_writes_separate_sft_file_per_dataset(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "trajectory_test"
    output_dir = tmp_path / "sft"
    _write_jsonl(
        source_root / "alpha" / "alpha_train.jsonl",
        [{"qid": "a1", "dataset": "alpha", "split": "train", "question": "Alpha?", "answer": "Alpha"}],
    )
    _write_jsonl(
        source_root / "beta" / "beta_train.jsonl",
        [{"qid": "b1", "dataset": "beta", "split": "train", "question": "Beta?", "answer": "Beta"}],
    )

    def fake_retrieve(config, dataset, query):
        answer = "Alpha" if dataset == "alpha" else "Beta"
        return {"query": query, "passages": [f"{dataset}\n{answer} evidence."], "scores": [1.0]}

    def fake_dry_update(example, state, observation):
        return {
            "update_evidence": {"selected_passage_ids": [0], "rationale": "Supports answer."},
            "answer": {"can_answer": True, "answer": example["answer"], "rationale": "Evidence supports it."},
        }

    monkeypatch.setattr("data_processing.generate_teacher_sft._retrieve", fake_retrieve)
    monkeypatch.setattr("data_processing.generate_teacher_sft._dry_update", fake_dry_update)

    summary = generate_sft_dataset(
        SFTConfig(
            datasets=["alpha", "beta"],
            source_root=source_root,
            output_dir=output_dir,
            max_examples_per_dataset=1,
            max_rounds=1,
            dry_run=True,
            resume=False,
        )
    )

    alpha_rows = [json.loads(line) for line in (output_dir / "alpha_sft.jsonl").read_text(encoding="utf-8").splitlines()]
    beta_rows = [json.loads(line) for line in (output_dir / "beta_sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not (output_dir / "sft.jsonl").exists()
    assert [row["qid"] for row in alpha_rows] == ["a1"]
    assert [row["qid"] for row in beta_rows] == ["b1"]
    assert summary["outputs"] == {
        "alpha": str(output_dir / "alpha_sft.jsonl"),
        "beta": str(output_dir / "beta_sft.jsonl"),
    }


def test_generate_sft_dataset_appends_each_success_before_later_error(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "trajectory_test"
    output_dir = tmp_path / "sft"
    _write_jsonl(
        source_root / "toyqa" / "toyqa_train.jsonl",
        [
            {"qid": "q1", "dataset": "toyqa", "split": "train", "question": "Q1?", "answer": "A1"},
            {"qid": "q2", "dataset": "toyqa", "split": "train", "question": "Q2?", "answer": "A2"},
        ],
    )

    def fake_retrieve(config, dataset, query):
        if query == "Q2?":
            raise RuntimeError("teacher rejected sample")
        return {"query": query, "passages": ["Q1\nA1 evidence."], "scores": [1.0]}

    def fake_dry_update(example, state, observation):
        return {
            "update_evidence": {"selected_passage_ids": [0], "rationale": "Supports answer."},
            "answer": {"can_answer": True, "answer": example["answer"], "rationale": "Evidence supports it."},
        }

    monkeypatch.setattr("data_processing.generate_teacher_sft._retrieve", fake_retrieve)
    monkeypatch.setattr("data_processing.generate_teacher_sft._dry_update", fake_dry_update)

    summary = generate_sft_dataset(
        SFTConfig(
            datasets=["toyqa"],
            source_root=source_root,
            output_dir=output_dir,
            max_examples_per_dataset=2,
            max_rounds=1,
            dry_run=True,
            resume=False,
            sft_sample_workers=2,
        )
    )

    rows = [json.loads(line) for line in (output_dir / "toyqa_sft.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [json.loads(line) for line in (output_dir / "teacher_errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["qid"] for row in rows] == ["q1"]
    assert summary["samples_written"] == 1
    assert summary["samples_failed"] == 1
    assert summary["dataset_stats"]["toyqa"]["samples_failed"] == 1
    assert errors[0]["qid"] == "q2"
    assert errors[0]["dataset"] == "toyqa"
    assert errors[0]["stage"] == "retrieval"
    assert "teacher rejected sample" in errors[0]["message"]


def test_generate_sft_dataset_stops_after_target_valid_per_dataset(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "trajectory_test"
    output_dir = tmp_path / "sft"
    _write_jsonl(
        source_root / "toyqa" / "toyqa_train.jsonl",
        [
            {"qid": "q1", "dataset": "toyqa", "split": "train", "question": "Q1?", "answer": "A1"},
            {"qid": "q2", "dataset": "toyqa", "split": "train", "question": "Q2?", "answer": "A2"},
            {"qid": "q3", "dataset": "toyqa", "split": "train", "question": "Q3?", "answer": "A3"},
        ],
    )
    retrieved: list[str] = []

    def fake_retrieve(config, dataset, query):
        retrieved.append(query)
        return {"query": query, "passages": [f"{query}\nA{query[1]} evidence."], "scores": [1.0]}

    def fake_dry_update(example, state, observation):
        return {
            "update_evidence": {"selected_passage_ids": [0], "rationale": "Supports answer."},
            "answer": {"can_answer": True, "answer": example["answer"], "rationale": "Evidence supports it."},
        }

    monkeypatch.setattr("data_processing.generate_teacher_sft._retrieve", fake_retrieve)
    monkeypatch.setattr("data_processing.generate_teacher_sft._dry_update", fake_dry_update)

    summary = generate_sft_dataset(
        SFTConfig(
            datasets=["toyqa"],
            source_root=source_root,
            output_dir=output_dir,
            max_examples_per_dataset=3,
            target_valid_per_dataset=1,
            max_rounds=1,
            dry_run=True,
            resume=False,
            sft_sample_workers=1,
            retrieval_workers=1,
        )
    )

    rows = [json.loads(line) for line in (output_dir / "toyqa_sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert len(retrieved) == 1
    assert summary["samples_seen"] == 1
    assert summary["samples_written"] == 1
    assert summary["dataset_stats"]["toyqa"]["target_reached"] is True
    assert summary["dataset_stats"]["toyqa"]["target_valid_per_dataset"] == 1


def test_retrieve_suppresses_nested_retrieval_output(monkeypatch, capsys) -> None:
    def noisy_query_linear_rag(**kwargs):
        print("[passage] Loaded 60 records")
        print("Retrieving: 100%", file=sys.stderr)
        return SimpleNamespace(query=kwargs["query"], passages=["Evidence"], scores=[1.0])

    monkeypatch.setattr("data_processing.generate_teacher_sft.query_linear_rag", noisy_query_linear_rag)

    observation = _retrieve(SFTConfig(retrieval_backend="linear_rag"), "toyqa", "Q?")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert observation["passages"][0]["text"] == "Evidence"


def test_observation_and_update_use_zero_based_passage_ids() -> None:
    observation = normalize_observation(
        {
            "query": "The Tripper director",
            "passages": ["The Tripper\nThe Tripper was directed by David Arquette."],
            "scores": [0.9],
        }
    )
    update = normalize_update_evidence(
        {
            "selected_passage_ids": [0],
            "rationale": "Useful evidence.",
        },
        observation=observation,
        round_index=0,
    )

    assert observation["passages"][0]["passage_id"] == 0
    assert observation["passages"][0]["title"] == "The Tripper"
    assert update["selected_passage_ids"] == [0]
    assert update["evidence"][0]["passage_id"] == 0
    assert "rank" not in update["evidence"][0]


def test_rationale_is_kept_short() -> None:
    observation = normalize_observation(
        {"query": "q", "passages": ["Title\nEvidence text"], "scores": [0.5]}
    )
    update = normalize_update_evidence(
        {"selected_passage_ids": [0], "rationale": "one two three four five six seven eight"},
        observation=observation,
        round_index=0,
    )

    assert update["rationale"] == "one two three four five six"


def test_retrieval_action_top_k_matches_observation_length() -> None:
    action = normalize_retrieval_action(
        {"query": "David Arquette nationality", "top_k": 3},
        observation={"passages": [{}, {}, {}, {}, {}]},
    )

    assert action["top_k"] == 5
    assert "rationale" not in action


def test_validate_trajectory_sample_rejects_future_state_and_gold_text() -> None:
    sample = {
        "qid": "q1",
        "trajectory": [
            {
                "round": 0,
                "state": {
                    "evidence": [{"text": "future"}],
                    "retrieval_history": [{"query": "future"}],
                    "retrieval_count": 1,
                },
                "retrieval": {"top_k": 1},
                "observation": {"passages": [{"passage_id": 0, "text": "Evidence"}]},
                "update_evidence": {"selected_passage_ids": [0], "evidence": []},
                "answer": {"can_answer": False, "answer": None, "rationale": "gold answer"},
            }
        ],
    }

    errors = validate_trajectory_sample(sample)

    assert any("retrieval_count" in error for error in errors)
    assert any("retrieval_history" in error for error in errors)
    assert any("state.evidence" in error for error in errors)
    assert any("forbidden" in error for error in errors)


def test_validate_trajectory_ignores_forbidden_term_inside_retrieved_text() -> None:
    sample = {
        "qid": "q1",
        "question": "Where was the person born?",
        "gold_answer": "Australia",
        "answer_aliases": [],
        "final_answer": "Australia",
        "trajectory": [
            {
                "round": 0,
                "state": {"evidence": [], "retrieval_history": [], "retrieval_count": 0},
                "query_retriever": {"sub_goal": "find birthplace", "query": "person birthplace"},
                "retrieval": {"query": "person birthplace", "top_k": 1},
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "title": "Gold Coast",
                            "text": "The person was born in Australia.",
                        }
                    ]
                },
                "update_evidence": {
                    "selected_passage_ids": [0],
                    "evidence": [
                        {"passage_id": 0, "title": "Gold Coast", "text": "The person was born in Australia."}
                    ],
                    "rationale": "supports birthplace",
                },
                "answer": {"can_answer": True, "answer": "Australia", "rationale": "supported"},
            }
        ],
    }

    assert validate_trajectory_sample(sample) == []


def test_validate_trajectory_still_rejects_forbidden_authored_rationale() -> None:
    sample = {
        "question": "Q?",
        "gold_answer": "A1",
        "final_answer": "A1",
        "trajectory": [
            {
                "round": 0,
                "state": {"evidence": [], "retrieval_history": [], "retrieval_count": 0},
                "query_retriever": {"sub_goal": "find answer", "query": "Q"},
                "retrieval": {"query": "Q", "top_k": 1},
                "observation": {"passages": [{"passage_id": 0, "text": "A1"}]},
                "update_evidence": {
                    "selected_passage_ids": [0],
                    "evidence": [{"passage_id": 0, "text": "A1"}],
                    "rationale": "copied from gold answer",
                },
                "answer": {"can_answer": True, "answer": "A1", "rationale": "supported"},
            }
        ],
    }

    assert any("forbidden" in error for error in validate_trajectory_sample(sample))


def test_query_leakage_uses_answer_forms_not_title_case() -> None:
    state = {"evidence": [], "retrieval_history": [], "retrieval_count": 0}

    assert not query_has_unseen_intermediate_terms(
        "Birth Place biography",
        question="What country was the actor born in?",
        state=state,
        answer_forms=["United States", "USA"],
    )
    assert query_has_unseen_intermediate_terms(
        "United States birthplace",
        question="What country was the actor born in?",
        state=state,
        answer_forms=["United States", "USA"],
    )
    assert not query_has_unseen_intermediate_terms(
        "United States birthplace",
        question="Was the actor born in the United States?",
        state=state,
        answer_forms=["United States", "USA"],
    )


def test_metric_normalized_exact_match_admits_article_variant_but_not_semantic_only_answer() -> None:
    base = {
        "question": "Who directed the film?",
        "gold_answer": "David Arquette",
        "answer_aliases": [],
        "trajectory": [
            {
                "round": 0,
                "state": {"evidence": [], "retrieval_history": [], "retrieval_count": 0},
                "query_retriever": {"sub_goal": "find director", "query": "film director"},
                "retrieval": {"query": "film director", "top_k": 1},
                "observation": {"passages": [{"passage_id": 0, "text": "The director was David Arquette."}]},
                "update_evidence": {
                    "selected_passage_ids": [0],
                    "evidence": [{"passage_id": 0, "text": "The director was David Arquette."}],
                    "rationale": "supports director",
                },
                "answer": {"can_answer": True, "answer": "The David Arquette", "rationale": "supported"},
            }
        ],
    }
    article_variant = {**base, "final_answer": "The David Arquette"}
    semantic_only = {
        **base,
        "gold_answer": "United States",
        "final_answer": "America",
        "trajectory": [
            {
                **base["trajectory"][0],
                "answer": {"can_answer": True, "answer": "America", "rationale": "supported"},
            }
        ],
    }

    assert validate_trajectory_sample(article_variant) == []
    assert any("exact match" in error for error in validate_trajectory_sample(semantic_only))


def test_evidence_support_accepts_dataset_alias_for_metric_correct_answer() -> None:
    assert answer_supported_by_evidence(
        {"can_answer": True, "answer": "United States"},
        [{"title": "Biography", "text": "Her birthplace was America."}],
        accepted_answers=["United States", "America"],
    )
    assert answer_supported_by_evidence(
        {"can_answer": True, "answer": "US"},
        [{"title": "Biography", "text": "Her birthplace was the U.S."}],
        accepted_answers=["US", "U.S."],
    )


def test_evidence_support_does_not_use_gold_aliases_for_metric_wrong_answer() -> None:
    assert not answer_supported_by_evidence(
        {"can_answer": True, "answer": "Canada"},
        [{"title": "Biography", "text": "Her birthplace was Paris."}],
        accepted_answers=["Paris", "City of Paris"],
    )


def test_validate_trajectory_sample_rejects_missing_final_answer() -> None:
    sample = {
        "qid": "q1",
        "final_answer": None,
        "trajectory": [
            {
                "round": 0,
                "state": {"evidence": [], "retrieval_history": [], "retrieval_count": 0},
                "retrieval": {"top_k": 1},
                "observation": {"passages": [{"passage_id": 0, "text": "Evidence"}]},
                "update_evidence": {"selected_passage_ids": [0], "evidence": []},
                "answer": {"can_answer": False, "answer": None, "rationale": ""},
            }
        ],
    }

    errors = validate_trajectory_sample(sample)

    assert any("final_answer" in error for error in errors)


def test_validate_trajectory_sample_rejects_query_with_unseen_intermediate_terms() -> None:
    sample = {
        "qid": "q1",
        "question": "Where was Adam Gontier born?",
        "gold_answer": "Canada",
        "answer_aliases": ["Canadian"],
        "final_answer": "Canada",
        "trajectory": [
            {
                "round": 0,
                "state": {"evidence": [], "retrieval_history": [], "retrieval_count": 0},
                "retrieval": {"query": "Canada country", "top_k": 1},
                "observation": {"passages": [{"passage_id": 0, "title": "Canada", "text": "Canada is a country."}]},
                "update_evidence": {"selected_passage_ids": [0], "evidence": [{"passage_id": 0, "text": "Canada is a country."}]},
                "answer": {"can_answer": True, "answer": "Canada", "rationale": ""},
            }
        ],
    }

    errors = validate_trajectory_sample(sample)

    assert any("unseen intermediate" in error for error in errors)


def test_query_rejects_unseen_intermediate_answer_terms() -> None:
    state = {"evidence": [], "retrieval_history": [], "retrieval_count": 0}

    assert query_has_unseen_intermediate_terms(
        "Canada national anthem",
        question="Are Adam Gontier and Hayley Williams from the same country?",
        state=state,
        answer_forms=["Canada", "Canadian"],
    )
    assert not query_has_unseen_intermediate_terms(
        "Adam Gontier nationality",
        question="Are Adam Gontier and Hayley Williams from the same country?",
        state=state,
        answer_forms=["Canada", "Canadian"],
    )
    assert not query_has_unseen_intermediate_terms(
        "Canada national anthem",
        question="Are Adam Gontier and Hayley Williams from the same country?",
        state={"evidence": [{"text": "Adam Gontier is a Canadian singer."}], "retrieval_history": [], "retrieval_count": 1},
        answer_forms=["Canada", "Canadian"],
    )


def test_yes_no_answer_requires_comparison_entities_in_evidence() -> None:
    answer = {"can_answer": True, "answer": "no"}
    weak_evidence = [
        {"title": "Adam Gontier", "text": "Adam Wade Gontier is a Canadian singer."},
        {"title": "Canada", "text": "Canada is a country in North America."},
    ]
    strong_evidence = [
        {"title": "Adam Gontier", "text": "Adam Wade Gontier is a Canadian singer."},
        {"title": "Hayley Williams", "text": "Hayley Nichole Williams is an American singer."},
    ]

    question = "Are Adam Gontier and Hayley Williams from the same country?"
    assert not answer_supported_by_evidence(answer, weak_evidence, question=question)
    assert answer_supported_by_evidence(answer, strong_evidence, question=question)
