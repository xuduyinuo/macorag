from __future__ import annotations

import json
from pathlib import Path

import pytest

from sft_v2.policy_prompts import load_policy_prompt_contract
from sft_v2.sft_data import load_split, select_trajectory_subset, trajectory_to_decisions


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
CONTRACT = load_policy_prompt_contract(ROOT / "policy_prompts.yml")


def _row(*, final_round: bool = False) -> dict:
    round_index = 0
    return {
        "qid": "test:1",
        "dataset": "test",
        "question": "Who wrote Example Book?",
        "max_rounds": 1 if final_round else 4,
        "trajectory": [
            {
                "round": round_index,
                "state": {
                    "current_sub_goal": None,
                    "evidence": [],
                    "retrieval_history": [],
                    "retrieval_count": round_index,
                },
                "query_retriever": {
                    "sub_goal": "Identify the author of Example Book",
                    "query": "Example Book author",
                },
                "observation": {
                    "passages": [
                        {
                            "passage_id": "P2",
                            "title": "Example Book",
                            "text": "Example Book was written by Alice Example.",
                            "score": 0.99,
                        }
                    ]
                },
                "update_evidence": {
                    "selected_passage_ids": ["P2"],
                    "evidence": [
                        {
                            "passage_id": "P2",
                            "title": "Example Book",
                            "text": "Example Book was written by Alice Example.",
                            "score": 0.99,
                        }
                    ],
                },
                "answer": {"can_answer": True, "answer": "Alice Example"},
            }
        ],
    }


def test_policy_few_shots_cover_decision_boundaries_without_rationale() -> None:
    answer_targets = [item["assistant"] for item in CONTRACT.few_shots["answer_generator"]]
    assert any('"can_answer":false' in item for item in answer_targets)
    assert sum('"can_answer":true' in item for item in answer_targets) >= 2
    assert any(item.get("final_round") for item in CONTRACT.few_shots["answer_generator"])
    evidence_targets = [item["assistant"] for item in CONTRACT.few_shots["evidence_updater"]]
    assert any('"selected_passage_ids":[]' in item for item in evidence_targets)
    assert any('"P0","P1"' in item for item in evidence_targets)
    assert all("rationale" not in item.lower() for item in answer_targets + evidence_targets)


def test_trajectory_renders_three_target_only_actions() -> None:
    decisions = trajectory_to_decisions(_row(), CONTRACT)
    assert [item.role for item in decisions] == [
        "query_retriever",
        "evidence_updater",
        "answer_generator",
    ]
    assert decisions[1].target == (
        '<update-evidence>{"selected_passage_ids":["P2"]}</update-evidence>'
    )
    assert decisions[2].target == (
        '<answer>{"can_answer":true,"answer":"Alice Example"}</answer>'
    )
    assert all("rationale" not in item.target.lower() for item in decisions)
    assert "score" not in decisions[1].target
    assert "Alice Example" not in json.dumps(decisions[0].messages, ensure_ascii=False)


def test_unknown_evidence_pointer_is_rejected() -> None:
    row = _row()
    row["trajectory"][0]["update_evidence"]["selected_passage_ids"] = ["P0"]
    with pytest.raises(ValueError, match="not in current observation"):
        trajectory_to_decisions(row, CONTRACT)


def test_unselected_expanded_evidence_is_rejected() -> None:
    row = _row()
    row["trajectory"][0]["update_evidence"]["selected_passage_ids"] = []
    with pytest.raises(ValueError, match="exactly match"):
        trajectory_to_decisions(row, CONTRACT)


def test_final_round_cannot_abstain() -> None:
    row = _row(final_round=True)
    row["trajectory"][0]["answer"] = {"can_answer": False, "answer": None}
    with pytest.raises(ValueError, match="final round"):
        trajectory_to_decisions(row, CONTRACT)


def test_real_precomputed_splits_are_disjoint_and_complete() -> None:
    data_root = REPO_ROOT / "data_v2" / "teacher_deepseek_v41_flash"
    train = load_split(data_root / "train_sft.jsonl", CONTRACT)
    validation = load_split(data_root / "validation_sft.jsonl", CONTRACT)
    validation_selected = select_trajectory_subset(validation, limit=100, seed=42)
    assert train.trajectory_count == 1000
    assert validation.trajectory_count == 200
    assert validation_selected.trajectory_count == 100
    assert set(validation_selected.answer_counts) == {"can_answer_false", "can_answer_true"}
    assert {
        item.qid for item in select_trajectory_subset(validation, limit=100, seed=42).decisions
    } == {item.qid for item in validation_selected.decisions}
    assert set(train.answer_counts) == {"can_answer_false", "can_answer_true"}
    assert set(validation.answer_counts) == {"can_answer_false", "can_answer_true"}
    train_qids = {item.qid for item in train.decisions}
    validation_qids = {item.qid for item in validation.decisions}
    assert train_qids.isdisjoint(validation_qids)
