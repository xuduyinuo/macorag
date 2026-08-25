from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_processing.io_utils import write_jsonl
from data_processing.stratified_extraction import (
    derive_seed,
    eligibility_error,
    normalize_question,
    select_rows,
    stratum_key,
)


def make_row(
    dataset: str,
    split: str,
    qid: str,
    stratum: str | None,
    *,
    level: str | None = None,
    question: str | None = None,
) -> dict:
    row = {
        "qid": qid,
        "dataset": dataset,
        "split": split,
        "question": question or f"Question for {qid}?",
        "answer": f"answer-{qid}",
        "answer_aliases": [],
        "question_type": stratum,
        "hop_count": 2,
        "supporting_facts": [
            {
                "doc_id": f"{dataset}:support:{qid}",
                "title": f"Support {qid}",
                "text": f"Evidence for {qid}.",
            }
        ],
        "evidence_chain": [],
        "context_doc_ids": [f"{dataset}:support:{qid}", f"{dataset}:distractor:{qid}"],
        "usable_for_sft": True,
        "usable_for_retrieval_eval": True,
        "quality_flags": [],
        "metadata": {},
    }
    if level is not None:
        row["metadata"]["level"] = level
    return row


def test_normalize_question_is_case_punctuation_and_space_stable() -> None:
    assert normalize_question("  Who's   Alice?  ") == "who s alice"


def test_training_requires_both_flags_and_empty_quality_flags() -> None:
    row = make_row("2wiki", "train", "q1", "inference")
    assert eligibility_error(row, split="train") is None

    row["usable_for_retrieval_eval"] = False
    assert eligibility_error(row, split="train") == "not_usable_for_retrieval_eval"

    row["usable_for_retrieval_eval"] = True
    row["quality_flags"] = ["missing_supporting_fact_text"]
    assert eligibility_error(row, split="train") == "quality_flags"


def test_evaluation_requires_dev_and_retrieval_usability_only() -> None:
    row = make_row("hotpotqa", "dev", "q1", "bridge", level="hard")
    row["usable_for_sft"] = False
    assert eligibility_error(row, split="dev") is None

    row["split"] = "train"
    assert eligibility_error(row, split="dev") == "wrong_split"


def test_eligibility_rejects_invalid_required_fields_in_stable_order() -> None:
    row = make_row("2wiki", "train", "q1", "inference")
    row["qid"] = ""
    row["answer"] = ""
    assert eligibility_error(row, split="train") == "missing_qid"

    row["qid"] = "q1"
    assert eligibility_error(row, split="train") == "missing_answer"

    row["answer"] = "answer"
    row["supporting_facts"] = None
    assert eligibility_error(row, split="train") == "invalid_supporting_facts"


def test_dataset_specific_stratum_keys() -> None:
    assert stratum_key("2wiki", make_row("2wiki", "train", "w", "inference")) == "inference"
    assert (
        stratum_key("hotpotqa", make_row("hotpotqa", "train", "h", "bridge", level="hard"))
        == "hard/bridge"
    )
    assert stratum_key("musique", make_row("musique", "train", "4hop2__1_2", None)) == "4hop2"

    with pytest.raises(ValueError, match="Unsupported dataset"):
        stratum_key("unknown", make_row("unknown", "train", "q", None))


def test_derived_seed_is_stable_and_partitioned() -> None:
    first = derive_seed(20260826, "2wiki", "train", "inference")
    assert first == derive_seed(20260826, "2wiki", "train", "inference")
    assert first != derive_seed(20260826, "2wiki", "dev", "inference")


def _write_selection_source(path: Path) -> None:
    rows = [
        make_row("2wiki", "train", f"c{i}", "comparison") for i in range(8)
    ] + [make_row("2wiki", "train", f"i{i}", "inference") for i in range(6)]
    rows.append(
        make_row(
            "2wiki",
            "train",
            "duplicate-c0",
            "comparison",
            question=rows[0]["question"].upper(),
        )
    )
    invalid = make_row("2wiki", "train", "invalid", "comparison")
    invalid["quality_flags"] = ["bad"]
    rows.append(invalid)
    write_jsonl(path, rows)


def test_select_rows_fulfills_quota_deduplicates_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "2wiki_train.jsonl"
    _write_selection_source(source)
    kwargs = {
        "source_path": source,
        "dataset": "2wiki",
        "split": "train",
        "quotas": {"comparison": 3, "inference": 2},
        "seed": 20260826,
    }

    first = select_rows(**kwargs)
    second = select_rows(**kwargs)

    assert first == second
    assert first.quota_actual == {"comparison": 3, "inference": 2}
    assert len(first.rows) == 5
    assert len(first.source_indices) == 5
    assert len(set(first.qids)) == 5
    assert len({normalize_question(row["question"]) for row in first.rows}) == 5
    assert first.excluded_by_reason == {"duplicate_question": 1, "quality_flags": 1}


def test_select_rows_uses_seed_to_change_selection(tmp_path: Path) -> None:
    source = tmp_path / "2wiki_train.jsonl"
    _write_selection_source(source)
    common = {
        "source_path": source,
        "dataset": "2wiki",
        "split": "train",
        "quotas": {"comparison": 3, "inference": 2},
    }
    selections = {
        tuple(select_rows(**common, seed=seed).qids)
        for seed in range(1, 8)
    }
    assert len(selections) > 1


def test_select_rows_rejects_underfilled_stratum(tmp_path: Path) -> None:
    source = tmp_path / "2wiki_train.jsonl"
    write_jsonl(source, [make_row("2wiki", "train", "q", "inference")])

    with pytest.raises(ValueError, match=r"inference.*required=2.*available=1"):
        select_rows(
            source_path=source,
            dataset="2wiki",
            split="train",
            quotas={"inference": 2},
            seed=20260826,
        )


def test_test_fixture_rows_are_json_serializable() -> None:
    json.dumps(make_row("2wiki", "train", "q", "comparison"))
