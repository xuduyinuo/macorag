from pytest import approx

from macorag.trajectory_filter import evaluate_trajectory


def _trajectory(
    *,
    accepted_chunk_ids: list[str],
    supporting_chunk_ids: list[str],
    retrieval_budget: int | None = None,
) -> dict:
    value = {
        "qid": "q1",
        "dataset": "hotpotqa",
        "trajectory": [
            {
                "action": {
                    "type": "retrieval",
                    "query": "Paris",
                    "top_k": len(accepted_chunk_ids),
                },
                "observation": {
                    "retrieved_chunks": [
                        {"chunk_id": chunk_id} for chunk_id in accepted_chunk_ids
                    ],
                },
            },
            {
                "action": {
                    "type": "update_evidence",
                    "accepted_chunk_ids": accepted_chunk_ids,
                }
            },
            {
                "action": {
                    "type": "final_answer",
                    "answer": "Paris",
                    "supporting_chunk_ids": supporting_chunk_ids,
                }
            },
        ],
    }
    if retrieval_budget is not None:
        value["metadata"] = {"retrieval_budget": retrieval_budget}
    return value


def test_evaluate_trajectory_accepts_grounded_answer_with_metrics():
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        _trajectory(
            accepted_chunk_ids=["c1"],
            supporting_chunk_ids=["c1"],
            retrieval_budget=2,
        ),
        qrels_by_qid,
        answers_by_qid,
        chunk_meta_by_chunk_id={"c1": {"chunk_id": "c1", "doc_id": "d1"}},
    )

    assert result.accepted is True
    assert result.reasons == []
    assert result.evidence_coverage == approx(1.0)
    assert result.retrieval_efficiency == approx(1.0)
    assert result.gold_evidence_count == 1
    assert result.covered_gold_evidence_count == 1
    assert result.retrieval_count == 1


def test_evaluate_trajectory_rejects_missing_chunk_meta_index():
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"]),
        qrels_by_qid,
        answers_by_qid,
    )

    assert result.accepted is False
    assert "missing_chunk_meta" in result.reasons


def test_evaluate_trajectory_rejects_ungrounded_answer():
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        _trajectory(accepted_chunk_ids=["c2"], supporting_chunk_ids=["c2"]),
        qrels_by_qid,
        answers_by_qid,
    )

    assert result.accepted is False
    assert "no_gold_evidence_overlap" in result.reasons


def test_evaluate_trajectory_rejects_missing_retrieval():
    trajectory = {
        "qid": "q1",
        "dataset": "hotpotqa",
        "trajectory": [
            {
                "action": {
                    "type": "update_evidence",
                    "accepted_chunk_ids": ["c1"],
                }
            },
            {
                "action": {
                    "type": "final_answer",
                    "answer": "Paris",
                    "supporting_chunk_ids": ["c1"],
                }
            },
        ],
    }
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(trajectory, qrels_by_qid, answers_by_qid)

    assert result.accepted is False
    assert "missing_retrieval" in result.reasons


def test_evaluate_trajectory_rejects_retrieval_action_missing_observation():
    trajectory = _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"])
    del trajectory["trajectory"][0]["observation"]
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(trajectory, qrels_by_qid, answers_by_qid)

    assert result.accepted is False
    assert "invalid_retrieval_observation" in result.reasons
    assert result.retrieval_count == 0


def test_evaluate_trajectory_rejects_retrieval_action_empty_observation():
    trajectory = _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"])
    trajectory["trajectory"][0]["observation"] = {"retrieved_chunks": []}
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(trajectory, qrels_by_qid, answers_by_qid)

    assert result.accepted is False
    assert "invalid_retrieval_observation" in result.reasons
    assert result.retrieval_count == 0


def test_evaluate_trajectory_rejects_legacy_retrieval_results_observation():
    trajectory = _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"])
    trajectory["trajectory"][0]["observation"] = {"results": [{"chunk_id": "c1"}]}
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        trajectory,
        qrels_by_qid,
        answers_by_qid,
        chunk_meta_by_chunk_id={"c1": {"chunk_id": "c1", "doc_id": "d1"}},
    )

    assert result.accepted is False
    assert "invalid_retrieval_observation" in result.reasons
    assert result.retrieval_count == 0


def test_evaluate_trajectory_rejects_retrieved_chunk_id_alias():
    trajectory = _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"])
    trajectory["trajectory"][0]["observation"] = {"retrieved_chunks": [{"id": "c1"}]}
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        trajectory,
        qrels_by_qid,
        answers_by_qid,
        chunk_meta_by_chunk_id={"c1": {"chunk_id": "c1", "doc_id": "d1"}},
    )

    assert result.accepted is False
    assert "invalid_retrieval_observation" in result.reasons
    assert result.retrieval_count == 0


def test_evaluate_trajectory_rejects_evidence_not_retrieved():
    trajectory = _trajectory(
        accepted_chunk_ids=["gold-unseen"],
        supporting_chunk_ids=["gold-unseen"],
    )
    trajectory["trajectory"][0]["observation"] = {
        "retrieved_chunks": [{"chunk_id": "retrieved-only"}]
    }
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["gold-unseen"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        trajectory,
        qrels_by_qid,
        answers_by_qid,
        chunk_meta_by_chunk_id={
            "gold-unseen": {"chunk_id": "gold-unseen", "doc_id": "gold-doc"},
            "retrieved-only": {"chunk_id": "retrieved-only", "doc_id": "other-doc"},
        },
    )

    assert result.accepted is False
    assert "evidence_not_retrieved" in result.reasons
    assert "supporting_chunks_not_retrieved" in result.reasons


def test_evaluate_trajectory_rejects_invalid_retrieval_action():
    trajectory = _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"])
    trajectory["trajectory"][0]["action"]["query"] = " "
    trajectory["trajectory"][0]["action"]["top_k"] = 0
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(trajectory, qrels_by_qid, answers_by_qid)

    assert result.accepted is False
    assert "invalid_retrieval_action" in result.reasons
    assert result.retrieval_count == 0


def test_evaluate_trajectory_rejects_float_top_k():
    trajectory = _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"])
    trajectory["trajectory"][0]["action"]["top_k"] = 1.5
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        trajectory,
        qrels_by_qid,
        answers_by_qid,
        chunk_meta_by_chunk_id={"c1": {"chunk_id": "c1", "doc_id": "d1"}},
    )

    assert result.accepted is False
    assert "invalid_retrieval_action" in result.reasons
    assert result.retrieval_count == 0


def test_evaluate_trajectory_rejects_string_top_k():
    trajectory = _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"])
    trajectory["trajectory"][0]["action"]["top_k"] = "5"
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        trajectory,
        qrels_by_qid,
        answers_by_qid,
        chunk_meta_by_chunk_id={"c1": {"chunk_id": "c1", "doc_id": "d1"}},
    )

    assert result.accepted is False
    assert "invalid_retrieval_action" in result.reasons
    assert result.retrieval_count == 0


def test_evaluate_trajectory_rejects_retrieval_budget_exceeded():
    trajectory = _trajectory(
        accepted_chunk_ids=["c1"],
        supporting_chunk_ids=["c1"],
        retrieval_budget=0,
    )
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(trajectory, qrels_by_qid, answers_by_qid)

    assert result.accepted is False
    assert "retrieval_budget_exceeded" in result.reasons


def test_evaluate_trajectory_rejects_supporting_chunks_that_are_not_gold():
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        _trajectory(accepted_chunk_ids=["c1", "c2"], supporting_chunk_ids=["c2"]),
        qrels_by_qid,
        answers_by_qid,
    )

    assert result.accepted is False
    assert "supporting_chunks_not_gold" in result.reasons


def test_evaluate_trajectory_rejects_missing_chunk_meta_mapping():
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"]),
        qrels_by_qid,
        answers_by_qid,
        chunk_meta_by_chunk_id={},
    )

    assert result.accepted is False
    assert "missing_chunk_meta" in result.reasons


def test_evaluate_trajectory_accepts_gold_doc_and_title_matches_from_chunk_meta():
    qrels_by_qid = {
        "q1": {
            "gold_doc_ids": ["doc-paris"],
            "gold_titles": ["Paris"],
            "gold_chunk_ids": [],
        }
    }
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(
        _trajectory(accepted_chunk_ids=["c1"], supporting_chunk_ids=["c1"]),
        qrels_by_qid,
        answers_by_qid,
        chunk_meta_by_chunk_id={
            "c1": {
                "chunk_id": "c1",
                "doc_id": "doc-paris",
                "title": "Paris",
            }
        },
    )

    assert result.accepted is True
    assert result.reasons == []
    assert result.evidence_coverage == approx(1.0)
