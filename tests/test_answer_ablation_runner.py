from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from rl_training import answer_reward_ablation as runner
from rl_training.config import parse_args


@pytest.fixture(scope="module")
def plan():
    return runner.build_plan(runner.REPO_ROOT / "config/grpo_answer_reward_ablation.yml")


def test_pilot_changes_only_local_answer_weight_and_output_path(plan):
    a, b = [deepcopy(c) for c in plan["variants"].values()]
    assert a.pop("answer_local_reward_weight") == 1.0
    assert b.pop("answer_local_reward_weight") == 0.0
    assert a.pop("output_root") != b.pop("output_root")
    assert a == b
    assert a["sft_adapter_path"] == plan["sft_adapter"]
    assert a["max_steps"] == 3000
    assert a["max_total_samples"] == 3000
    assert a["run_until_step"] == 200
    assert a["seed"] == 42
    assert plan["eval_config"]["eval_request_workers"] == 4
    assert plan["training_prefix_counts"] == {"2wiki": 62, "hotpotqa": 64, "musique": 74}


def test_freeze_configs_are_parseable_and_reject_drift(plan, tmp_path):
    plan = deepcopy(plan)
    plan["run_dir"] = str(tmp_path)
    runner.prepare(plan)
    for name, expected in plan["variants"].items():
        config = tmp_path / "configs" / f"{name}.yml"
        args = parse_args(["--config", str(config)])
        assert args.answer_local_reward_weight == expected["answer_local_reward_weight"]
    runner.prepare(plan)
    changed = deepcopy(plan)
    changed["steps"] += 1
    with pytest.raises(ValueError, match="changed"):
        runner.prepare(changed)
    path = tmp_path / "configs/evaluation.yml"
    payload = yaml.safe_load(path.read_text())
    payload["temperature"] = 0.9
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="Frozen config"):
        runner.prepare(plan)


def test_committed_log_chain_excludes_uncheckpointed_tail(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for p in (first / "checkpoint-2", second / "checkpoint-4"):
        p.mkdir(parents=True)
    (first / "checkpoint-2/checkpoint_manifest.json").write_text('{"global_step": 2}')
    (second / "checkpoint-4/checkpoint_manifest.json").write_text('{"global_step": 4}')
    (first / "resume_meta.json").write_text('{"resume_global_step": 0, "resume_from_checkpoint": null}')
    (second / "resume_meta.json").write_text(json.dumps({
        "resume_global_step": 2, "resume_from_checkpoint": str(first / "checkpoint-2"),
    }))
    (first / "train_metrics.jsonl").write_text("\n".join(json.dumps({"step": n, "kl": 0, "source": "first"}) for n in (1, 2, 3)))
    (second / "train_metrics.jsonl").write_text("\n".join(json.dumps({"step": n, "kl": 0, "source": "second"}) for n in (3, 4)))
    rows = runner._training_rows(second / "checkpoint-4", output_root=tmp_path)
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert [r["source"] for r in rows] == ["first", "first", "second", "second"]


def test_each_fresh_arm_starts_sft_but_interrupted_eval_loads_checkpoint(monkeypatch, tmp_path):
    plan = {"run_dir": str(tmp_path), "sft_adapter": "/sft", "steps": 200,
            "variants": {"control": {}, "off": {}}}
    completed = set()
    checkpoints = {}
    servers = []
    training_commands = []

    @contextmanager
    def server(plan, config, adapter, log):
        servers.append(str(adapter))
        yield

    def train(command, *, env, log):
        name = Path(env["CONFIG_PATH"]).stem
        training_commands.append(command)
        checkpoints[name] = [(200, tmp_path / name / "checkpoint-200")]

    monkeypatch.setattr(runner, "_server", server)
    monkeypatch.setattr(runner, "_command", train)
    monkeypatch.setattr(runner, "_checkpoints", lambda p, n: checkpoints.get(n, []))
    monkeypatch.setattr(runner, "_is_complete_evaluation", lambda p: Path(p).name in completed)
    monkeypatch.setattr(runner, "_evaluate", lambda p, n, a: completed.add(n))
    monkeypatch.setattr(runner, "report", lambda p: {"ok": True})
    assert runner.run(plan) == {"ok": True}
    assert servers == ["/sft", "/sft", "/sft"]
    assert len(training_commands) == 2
    completed.remove("off")
    runner.run(plan)
    assert servers[-1] == str(tmp_path / "off/checkpoint-200")
    assert len(training_commands) == 2


@pytest.mark.parametrize("reuse_address", [False, True])
def test_occupied_port_never_starts_or_kills_a_process(monkeypatch, tmp_path, reuse_address):
    import socket
    with socket.socket() as occupied:
        if reuse_address:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        config = tmp_path / "train.yml"
        config.write_text(yaml.safe_dump({"vllm_port": occupied.getsockname()[1]}))
        monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
        with pytest.raises(RuntimeError, match="occupied"):
            with runner._server({}, config, "/sft", tmp_path / "log"):
                pytest.fail("must not enter")


def test_server_switch_accepts_time_wait_without_launching_gpu_server(monkeypatch, tmp_path):
    import errno
    import socket

    class PreflightPassed(Exception):
        pass

    def reached_launch(*args, **kwargs):
        raise PreflightPassed

    monkeypatch.setattr(runner.subprocess, "Popen", reached_launch)
    address = ("127.0.0.1", 0)
    for _ in range(3):
        # A real HTTP-style listener actively closes its connection, leaving
        # server-side TIME_WAIT. Reuse the same port across multiple stages.
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.settimeout(2)
            listener.bind(address)
            address = listener.getsockname()
            listener.listen(1)
            with socket.create_connection(address, timeout=2) as client:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(2)
                    connection.shutdown(socket.SHUT_WR)
                    assert client.recv(1) == b""
                    client.shutdown(socket.SHUT_WR)
                    assert connection.recv(1) == b""
        with socket.socket() as old_probe:
            with pytest.raises(OSError) as exc:
                old_probe.bind(address)
            assert exc.value.errno == errno.EADDRINUSE
        config = tmp_path / "train.yml"
        config.write_text(yaml.safe_dump({"vllm_port": address[1]}))
        with pytest.raises(PreflightPassed):
            with runner._server({}, config, "/sft", tmp_path / "server.log"):
                pytest.fail("Only preflight is exercised, no real vLLM process")


def test_report_matches_evaluation_loader_whitespace_contract(monkeypatch):
    manifest = {}
    rows = {}
    for ds in runner.DATASETS:
        rows[ds] = []
        for i in range(30):
            qid = str(i)
            manifest[(ds, qid)] = {"answer": " Ada ", "question": " Who? ",
                                   "supporting_facts": [{"title": "Ada"}]}
            rows[ds].append({"dataset": ds, "qid": qid, "gold_answer": "Ada", "pred_answer": "Ada",
                             "question": "Who?", "retrieval_count": 1, "trajectory": []})
    monkeypatch.setattr(runner, "_rows", lambda p: rows[Path(p).parent.name])
    monkeypatch.setattr(runner, "_evaluation_payload", lambda *a, **kw: {
        "metrics": {"datasets": {}}, "protocol": {}, "contract": {"contract_fingerprint": "fixed"},
    })
    stats, scores, contract = runner.summarize_evaluation("/eval", manifest)
    assert stats["macro_f1"] == 1.0 and len(scores) == 90 and contract == "fixed"
    rows["2wiki"][0]["gold_answer"] = "Someone else"
    with pytest.raises(ValueError, match="gold/question"):
        runner.summarize_evaluation("/eval", manifest)
