from macorag.trajectory_filter import evaluate_trajectory


def test_evaluate_trajectory_accepts_grounded_answer():
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

    assert result.accepted is True
    assert result.reasons == []


def test_evaluate_trajectory_rejects_ungrounded_answer():
    trajectory = {
        "qid": "q1",
        "dataset": "hotpotqa",
        "trajectory": [
            {
                "action": {
                    "type": "update_evidence",
                    "accepted_chunk_ids": ["c2"],
                }
            },
            {
                "action": {
                    "type": "final_answer",
                    "answer": "Paris",
                    "supporting_chunk_ids": ["c2"],
                }
            },
        ],
    }
    qrels_by_qid = {"q1": {"gold_chunk_ids": ["c1"]}}
    answers_by_qid = {"q1": {"answer": "Paris", "aliases": []}}

    result = evaluate_trajectory(trajectory, qrels_by_qid, answers_by_qid)

    assert result.accepted is False
    assert "no_gold_evidence_overlap" in result.reasons
