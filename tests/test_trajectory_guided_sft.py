from __future__ import annotations

import json
import re

from prompt_config import system_prompt_for
from sft_training.collator import MacoRAGSFTCollator
from sft_training.data import TrajectoryRecord, trajectory_to_sft_records
from sft_training.dataset import _tokenize_records
from sft_training.formatting import compact_decision_prompt
from sft_training.trajectory_parser import TeacherValidationStats


def _turn(round_id: int, *, stop: bool = False) -> dict:
    return {
        "round": round_id,
        "state": {
            "question": "Q?",
            "current_sub_goal": None,
            "evidence": [],
            "retrieval_history": [],
            "retrieval_count": round_id,
        },
        "query_retriever": {"sub_goal": "find fact", "query": f"query {round_id}"},
        "retrieval": {"query": f"query {round_id}", "top_k": 2},
        "observation": {
            "passages": [
                {"passage_id": 0, "title": "A", "text": "fact A"},
                {"passage_id": 1, "title": "B", "text": "fact B"},
            ]
        },
        "update_evidence": {"selected_passage_ids": [1], "rationale": "B is useful"},
        "answer": {
            "can_answer": stop,
            "answer": "final" if stop else None,
            "rationale": "enough" if stop else "continue",
        },
    }


def test_two_round_trajectory_becomes_six_decision_samples_and_stops() -> None:
    records = trajectory_to_sft_records(
        {"qid": "q", "dataset": "toy", "question": "Q?", "trajectory": [_turn(0), _turn(1, stop=True), _turn(2)]}
    )
    assert [record.agent_type for record in records] == [
        "query", "evidence", "answer", "query", "evidence", "answer"
    ]


def test_each_decision_uses_its_role_instruction() -> None:
    records = trajectory_to_sft_records(
        {"qid": "q", "dataset": "toy", "question": "Q?", "trajectory": [_turn(0, stop=True)]}
    )
    assert [record.role_instruction for record in records] == [
        system_prompt_for("query_retriever"),
        system_prompt_for("evidence_updater"),
        system_prompt_for("answer_generator"),
    ]


def test_invalid_evidence_index_is_skipped_without_crashing_trajectory() -> None:
    turn = _turn(0, stop=True)
    turn["update_evidence"]["selected_passage_ids"] = [2]
    stats = TeacherValidationStats()
    records = trajectory_to_sft_records(
        {"qid": "q", "dataset": "toy", "question": "Q?", "trajectory": [turn]},
        validation_stats=stats,
    )
    assert [record.agent_type for record in records] == ["query", "answer"]
    assert stats.invalid_evidence_samples == 1


def test_continue_and_stop_are_preserved_as_distinct_targets() -> None:
    records = trajectory_to_sft_records(
        {"qid": "q", "dataset": "toy", "question": "Q?", "trajectory": [_turn(0), _turn(1, stop=True)]}
    )
    answers = [json.loads(re.search(r"<answer>(.*)</answer>", r.target_text).group(1)) for r in records if r.agent_type == "answer"]
    assert answers == [
        {"can_answer": False, "answer": None, "rationale": "continue"},
        {"can_answer": True, "answer": "final", "rationale": "enough"},
    ]


def test_safe_compaction_keeps_candidate_count_order_and_ids() -> None:
    passages = [
        {"passage_id": index, "title": str(index), "text": "x" * 300}
        for index in range(5)
    ]
    prompt = f"Question: Q?\n<state>{{\"evidence\":[],\"retrieval_history\":[]}}</state>\n<observation>{json.dumps({'passages': passages})}</observation>"
    compacted = compact_decision_prompt(prompt, token_count=len, max_tokens=800)
    observation = json.loads(re.search(r"<observation>(.*)</observation>", compacted.text).group(1))
    assert [item["passage_id"] for item in observation["passages"]] == list(range(5))
    assert compacted.truncated_passage_texts == 5


def test_target_only_mask_survives_prompt_truncation_and_padding() -> None:
    class CharacterTokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return list(range(len(messages[-1]["content"])))

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [80, 81, 82]}

    record = TrajectoryRecord(
        qid="q",
        question="Q?",
        dataset="toy",
        action_type="evidence_update",
        agent_role="evidence_updater",
        prompt_text=(
            "Question: Q?\n<state>{\"evidence\":[],\"retrieval_history\":[]}</state>\n"
            '<observation>{"passages":[{"passage_id":0,"text":"' + "x" * 300 + '"}]}</observation>'
        ),
        target_text="out",
    )
    inputs, attention, labels = _tokenize_records([record], CharacterTokenizer(), 180, "system")
    assert inputs and len(inputs[0]) <= 180
    assert labels[0][-4:] == [80, 81, 82, -100]
    batch = MacoRAGSFTCollator(0)(
        [
            {"input_ids": inputs[0], "attention_mask": attention[0], "labels": labels[0]},
            {"input_ids": [1, 80, 99], "attention_mask": [1, 1, 1], "labels": [-100, 80, -100]},
        ]
    )
    assert bool((batch["labels"][batch["attention_mask"] == 0] == -100).all())
