import pytest
from contextlib import contextmanager
import json

from rl_training import optimization_validation as runner


@pytest.mark.parametrize("off,errors,expected", [(0.71, 0, True), (0.69, 0, False), (0.71, 1, False)])
def test_confirmation_gate(tmp_path, monkeypatch, off, errors, expected):
    names = ("sft", "aligned_control", "no_answer_local")
    for name in names:
        path = tmp_path / "evaluations" / name / "evaluation_contract.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    monkeypatch.setattr(runner, "_assert_evaluation_adapter", lambda *a: None)
    f1 = dict(zip(names, (0.7, 0.65, off)))

    def summary(root, plan, name):
        return {"metrics": {"macro_f1": f1[name]}, "contract": "same",
                "protocol": {"error_count": errors}}, {("2wiki", "q"): f1[name]}

    monkeypatch.setattr(runner, "summarize", summary)
    result = runner.confirmation(tmp_path, {"adapters": {n: n for n in names}})
    assert result["proceed_low_lr"] is expected
    assert (tmp_path / "confirmation.json").exists()


def test_failed_gate_never_launches_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "confirmation", lambda *a: {"proceed_low_lr": False})
    monkeypatch.setattr(runner, "_server", lambda *a: pytest.fail("must not launch server"))
    assert runner.low_lr(tmp_path, {})["status"] == "stopped_at_confirmation_gate"


@pytest.mark.parametrize("fail", [False, True])
def test_replica_startup_always_cleans_owned_contexts(tmp_path, monkeypatch, fail):
    entered, exited = set(), set()

    @contextmanager
    def server(plan, config, adapter, log):
        name = config.stem
        entered.add(name)
        try:
            if fail and name == "server_secondary":
                raise RuntimeError("startup failed")
            yield
        finally:
            exited.add(name)

    monkeypatch.setattr(runner, "_server", server)
    if fail:
        with pytest.raises(RuntimeError, match="startup failed"):
            with runner.evaluation_servers(tmp_path, {}, "adapter", "test"):
                pytest.fail("one failed replica must prevent evaluation")
    else:
        with runner.evaluation_servers(tmp_path, {}, "adapter", "test"):
            assert entered == {"server", "server_secondary"}
    assert entered == exited == {"server", "server_secondary"}


@pytest.mark.parametrize("mutation", [None, "score", "error", "missing_qid", "source"])
def test_explicit_acceptance_keeps_strict_failure_and_checks_evidence(tmp_path, monkeypatch, mutation):
    smoke = tmp_path / "previous"
    smoke.mkdir()
    full = [{"dataset": ds, "qid": f"{ds}-{i}", "question": "question", "answer": "gold"}
            for ds in runner.DATASETS for i in range(4)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in full))
    plan = {"evaluation": {"eval_request_workers": 8}, "manifest_sha256": "manifest",
            "adapter_hashes": {"sft": "adapter"}, "sources": {"src/evaluation/x.py": "source"},
            "metric_sha256": "metric", "prompt_sha256": "prompt",
            "manifest": str(manifest), "sft_adapter": "sft"}
    (smoke / "plan.json").write_text(json.dumps(plan))
    original = json.dumps({"passed": False, "transport_errors": 0})
    (smoke / "smoke.json").write_text(original)
    monkeypatch.setattr(runner, "_assert_evaluation_adapter", lambda *a: None)
    monkeypatch.setattr(runner, "_covered_supporting_fact_count", lambda *a: 2)
    for name in ("smoke_serial", "smoke_parallel"):
        output = smoke / "evaluations" / name
        output.mkdir(parents=True)
        (output / "evaluation_contract.json").write_text("{}")
        for ds in runner.DATASETS:
            rows = [dict(r, gold_answer="gold", pred_answer="gold", trajectory=[name],
                         retrieval_count=2, parse_errors=[]) for r in full if r["dataset"] == ds]
            if name == "smoke_parallel" and ds == "2wiki":
                if mutation == "score":
                    rows[0]["pred_answer"] = "wrong"
                elif mutation == "error":
                    rows[0]["error"] = "timeout"
                elif mutation == "missing_qid":
                    rows.pop()
            (output / ds).mkdir()
            (output / ds / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if mutation == "source":
        plan["sources"]["src/evaluation/x.py"] = "changed"
    if mutation:
        with pytest.raises(ValueError):
            runner.accepted_parallel_evidence(smoke, plan)
    else:
        evidence = runner.accepted_parallel_evidence(smoke, plan)
        assert evidence["original_strict_smoke_passed"] is False
        assert evidence["per_question_metrics_equal"] is True
        assert len(evidence["different_predictions_or_trajectories"]) == 12
        assert len(evidence["artifact_sha256"]) == 10
    assert (smoke / "smoke.json").read_text() == original
