import pytest

from macorag.retrieval_env import InMemoryRetrievalEnv
from macorag.teacher_api import FakeTeacherClient
from macorag.trajectory_builder import _prompt_from_state, build_one_trajectory


def _env(chunks=None):
    return InMemoryRetrievalEnv(
        questions=[
            {
                "qid": "q1",
                "question": "Where was Alice born?",
                "dataset": "hotpotqa",
            }
        ],
        chunks=chunks
        or [
            {
                "chunk_id": "c1",
                "title": "Alice",
                "text": "Alice was born in Paris.",
            }
        ],
        retrieval_budget=3,
    )


def _teacher_response(retrieval_payload: str) -> str:
    return f"""
<plan>{{"sub_query": "Alice birthplace", "rationale": "Find birthplace."}}</plan>
<retrieval>{retrieval_payload}</retrieval>
<update-evidence>{{"accepted_chunk_ids": ["c1"], "rejected_chunk_ids": [], "reason": "The chunk states it."}}</update-evidence>
<answer>{{"answer": "Paris", "supporting_chunk_ids": ["c1"]}}</answer>
"""


def test_build_one_trajectory_uses_teacher_protocol_and_env():
    env = InMemoryRetrievalEnv(
        questions=[
            {
                "qid": "q1",
                "question": "Where was Alice born?",
                "dataset": "hotpotqa",
                "answer": "Paris",
                "supporting_facts": [{"chunk_id": "c1"}],
            }
        ],
        chunks=[
            {
                "chunk_id": "c1",
                "title": "Alice",
                "text": "Alice was born in Paris.",
            },
            {
                "chunk_id": "c2",
                "title": "Bob",
                "text": "Bob wrote a book.",
            },
        ],
        retrieval_budget=3,
    )
    teacher = FakeTeacherClient(
        [
            """
<plan>{"sub_query": "Alice birthplace", "rationale": "Find birthplace."}</plan>
<retrieval>{"query": "Alice birthplace", "top_k": 1}</retrieval>
<update-evidence>{"accepted_chunk_ids": ["c1"], "rejected_chunk_ids": [], "reason": "The chunk states it."}</update-evidence>
<answer>{"answer": "Paris", "supporting_chunk_ids": ["c1"]}</answer>
"""
        ]
    )

    trajectory = build_one_trajectory("q1", env, teacher)

    assert trajectory["qid"] == "q1"
    assert trajectory["dataset"] == "hotpotqa"
    assert set(trajectory) == {"qid", "dataset", "trajectory"}
    assert [step["agent"] for step in trajectory["trajectory"]] == [
        "planner",
        "retriever",
        "evidence_updater",
        "answer_generator",
    ]
    assert trajectory["trajectory"][0]["action"] == {
        "type": "plan_query",
        "sub_query": "Alice birthplace",
        "rationale": "Find birthplace.",
    }
    assert trajectory["trajectory"][1]["action"] == {
        "type": "retrieve",
        "query": "Alice birthplace",
        "top_k": 1,
    }
    assert trajectory["trajectory"][1]["observation"] == {
        "retrieved_chunks": [
            {
                "chunk_id": "c1",
                "title": "Alice",
                "text": "Alice was born in Paris.",
                "score": trajectory["trajectory"][1]["observation"][
                    "retrieved_chunks"
                ][0]["score"],
            }
        ]
    }
    assert trajectory["trajectory"][2]["action"] == {
        "type": "update_evidence",
        "accepted_chunk_ids": ["c1"],
        "rejected_chunk_ids": [],
        "reason": "The chunk states it.",
    }
    assert trajectory["trajectory"][3]["action"] == {
        "type": "final_answer",
        "answer": "Paris",
        "supporting_chunk_ids": ["c1"],
    }

    prompt = teacher.calls[0]
    assert "Where was Alice born?" in prompt
    assert "Retrieval budget: 3" in prompt
    assert "gold answer" in prompt.lower()
    assert "Paris" not in prompt
    assert "supporting_facts" not in prompt


def test_prompt_from_state_uses_visible_state_without_hidden_gold_fields():
    prompt = _prompt_from_state(
        {
            "qid": "q1",
            "dataset": "hotpotqa",
            "question": "Where was Alice born?",
            "retrieval_budget": 2,
            "answer": "Paris",
            "gold_evidence": ["c1"],
            "hidden_evidence": ["Alice was born in Paris."],
        }
    )

    assert "Where was Alice born?" in prompt
    assert "Retrieval budget: 2" in prompt
    assert "<plan>" in prompt
    assert "<retrieval>" in prompt
    assert "<update-evidence>" in prompt
    assert "<answer>" in prompt
    assert "Paris" not in prompt
    assert "gold_evidence" not in prompt
    assert "hidden_evidence" not in prompt


def test_build_one_trajectory_defaults_missing_top_k_to_five():
    env = _env(
        chunks=[
            {
                "chunk_id": f"c{index}",
                "title": "Alice",
                "text": f"Alice birthplace evidence {index}.",
            }
            for index in range(1, 8)
        ]
    )
    teacher = FakeTeacherClient(
        [_teacher_response('{"query": "Alice birthplace"}')]
    )

    trajectory = build_one_trajectory("q1", env, teacher)

    retriever_step = trajectory["trajectory"][1]
    assert retriever_step["action"]["top_k"] == 5
    assert len(retriever_step["observation"]["retrieved_chunks"]) == 5


@pytest.mark.parametrize(
    "top_k_json",
    [
        '"5"',
        "1.5",
        "true",
        '"bad"',
        "0",
        "-1",
        "null",
        "[]",
        "{}",
    ],
)
def test_build_one_trajectory_rejects_invalid_top_k(top_k_json):
    teacher = FakeTeacherClient(
        [_teacher_response(f'{{"query": "Alice birthplace", "top_k": {top_k_json}}}')]
    )

    with pytest.raises(ValueError, match="top_k"):
        build_one_trajectory("q1", _env(), teacher)


def test_build_one_trajectory_copies_update_evidence_lists():
    teacher = FakeTeacherClient(
        [_teacher_response('{"query": "Alice birthplace", "top_k": 1}')]
    )
    trajectory = build_one_trajectory("q1", _env(), teacher)
    evidence_step = trajectory["trajectory"][2]

    evidence_step["action"]["accepted_chunk_ids"].append("mutated")

    assert evidence_step["state_delta"]["added_evidence"] == ["c1"]
