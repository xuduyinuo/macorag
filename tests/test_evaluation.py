from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from pathlib import Path

import pytest

from rag import AgentRole, RAGState
from evaluation.config import parse_args
from evaluation.data import EvalSample, load_eval_samples
from evaluation.output import make_run_dir
from evaluation.vllm_servers import build_commands, parse_args as parse_vllm_server_args, run_commands
from evaluation.evaluate_rag_model import (
    _adapter_identity,
    _args_to_jsonable,
    _prepare_resume_identity,
    _validate_fixed_manifest,
    _build_retrieval_env,
    _configure_visible_gpus,
    _load_policy,
    VLLMOpenAIPolicy,
    VLLMTrainingServerPolicy,
    format_prediction,
    main,
    run_predictions,
)


from evaluation.local_evaluator import evaluate_predictions


def test_fixed_manifest_selects_per_dataset_deterministically(tmp_path: Path) -> None:
    from evaluation.fixed_manifest import build_fixed_manifest

    source = tmp_path / "source"
    strata = {
        "2wiki": ["compositional", "comparison"],
        "hotpotqa": ["hard/bridge", "hard/comparison"],
        "musique": ["2hop", "3hop1"],
    }
    for dataset, dataset_strata in strata.items():
        rows = []
        for index in range(120):
            stratum = dataset_strata[index % len(dataset_strata)]
            row = {
                "qid": f"{dataset}-{index}",
                "dataset": dataset,
                "question": f"question {index}",
                "answer": "answer",
                "supporting_facts": [],
            }
            if dataset == "2wiki":
                row["question_type"] = stratum
            elif dataset == "hotpotqa":
                level, question_type = stratum.split("/", 1)
                row["metadata"] = {"level": level}
                row["question_type"] = question_type
            else:
                row["qid"] = f"{stratum}__{index}"
            rows.append(row)
        _write_jsonl(source / f"{dataset}.jsonl", rows)

    first = build_fixed_manifest(source, tmp_path / "one", per_dataset=100, seed=7)
    second = build_fixed_manifest(source, tmp_path / "two", per_dataset=100, seed=7)

    assert first["manifest_fingerprint"] == second["manifest_fingerprint"]
    assert first["counts_by_dataset"] == {"2wiki": 100, "hotpotqa": 100, "musique": 100}
    rows = [json.loads(line) for line in (tmp_path / "one" / "manifest.jsonl").read_text().splitlines()]
    assert all(row["sampling_stratum"] for row in rows)

    samples, summary = load_eval_samples(
        data_root=tmp_path / "one",
        data_files=["manifest.jsonl"],
    )
    fingerprint, _ = _validate_fixed_manifest(
        args=SimpleNamespace(manifest_meta_path=str(tmp_path / "one" / "manifest_meta.json")),
        samples=samples,
        sample_summary=summary,
    )
    assert fingerprint == first["manifest_fingerprint"]

    with (tmp_path / "one" / "manifest.jsonl").open("a", encoding="utf-8") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_fixed_manifest(
            args=SimpleNamespace(manifest_meta_path=str(tmp_path / "one" / "manifest_meta.json")),
            samples=samples,
            sample_summary=summary,
        )


def test_fixed_evaluation_resume_identity_rejects_adapter_or_contract_change(tmp_path: Path) -> None:
    contract = {"contract_fingerprint": "contract-a"}
    initial_args = SimpleNamespace(resume=False, adapter_label="sft")
    _prepare_resume_identity(
        output_dir=tmp_path,
        args=initial_args,
        contract=contract,
        adapter_identity={"fingerprint": "adapter-a"},
    )
    dataset_dir = tmp_path / "2wiki"
    dataset_dir.mkdir()
    (dataset_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resume identity"):
        _prepare_resume_identity(
            output_dir=tmp_path,
            args=SimpleNamespace(resume=True, adapter_label="rl-step-300"),
            contract=contract,
            adapter_identity={"fingerprint": "adapter-b"},
        )


def test_run_predictions_resumes_completed_qids(tmp_path: Path, monkeypatch) -> None:
    samples = [
        EvalSample("done", "2wiki", "q1", "a1", [], [], {}),
        EvalSample("pending", "2wiki", "q2", "a2", [], [], {}),
    ]
    completed = {
        "qid": "done",
        "dataset": "2wiki",
        "question": "q1",
        "pred_answer": "a1",
        "gold_answer": "a1",
        "answer_aliases": [],
        "trajectory": [],
        "parse_errors": [],
        "retrieval_count": 0,
    }
    _write_jsonl(tmp_path / "predictions.jsonl", [completed])
    generated = []

    def fake_run_one_prediction(*, index, sample, **kwargs):
        generated.append(sample.qid)
        return index, {**completed, "qid": sample.qid, "question": sample.question, "gold_answer": sample.answer}

    monkeypatch.setattr("evaluation.evaluate_rag_model._run_one_prediction", fake_run_one_prediction)
    args = SimpleNamespace(eval_request_workers=1, disable_tqdm=True, resume=True)

    predictions = run_predictions(args, samples, object(), object(), tmp_path)

    assert generated == ["pending"]
    assert [row["qid"] for row in predictions] == ["done", "pending"]


def test_fixed_eval_config_supports_stable_resumable_output(tmp_path: Path) -> None:
    config = tmp_path / "eval.yml"
    config.write_text(
        "output_dir: outputs/fixed/sft\nresume: true\n"
        "manifest_meta_path: data/fixed/manifest_meta.json\nadapter_label: sft\n"
        "adapter_identity_path: outputs/sft/adapter\n",
        encoding="utf-8",
    )
    args = parse_args(["--config", str(config)])
    assert args.output_dir == "outputs/fixed/sft"
    assert args.resume is True
    assert args.adapter_label == "sft"
    assert args.adapter_identity_path == "outputs/sft/adapter"


def test_fixed_eval_config_supports_training_server_and_four_workers(tmp_path: Path) -> None:
    config = tmp_path / "eval.yml"
    config.write_text(
        "vllm_transport: training_server\neval_request_workers: 4\n",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.vllm_transport == "training_server"
    assert args.eval_request_workers == 4


def test_training_server_policy_uses_thread_local_deterministic_seeds() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is True
            return [1, 2, 3]

        def decode(self, token_ids, *, skip_special_tokens):
            if skip_special_tokens:
                return "generated"
            return "encoded-prompt"

    policy = VLLMTrainingServerPolicy(
        tokenizer=FakeTokenizer(),
        base_urls=["http://127.0.0.1:8000/v1"],
        model="macorag",
        api_key_env="",
        system_prompt=None,
        max_prompt_length=2048,
        max_completion_length=192,
        temperature=0.0,
        top_p=0.95,
        timeout=10,
        retries=1,
        retry_sleep_seconds=0.0,
    )
    payloads = []
    policy._post_generate = lambda payload: payloads.append(payload) or {"completion_ids": [[9]]}
    policy.set_endpoint_index(3)

    first = policy.generate(
        role=AgentRole.QUERY_RETRIEVER,
        question="question",
        state=RAGState(question="question"),
    )
    policy.generate(
        role=AgentRole.QUERY_RETRIEVER,
        question="question",
        state=RAGState(question="question"),
    )

    assert first == "generated"
    assert policy._endpoint() == "http://127.0.0.1:8000/generate/"
    assert [payload["seeds"] for payload in payloads] == [[300], [301]]


def _run_eval_launcher_dry_run(*, config_path: Path, extra_args: list[str]) -> str:
    env = os.environ.copy()
    env["CONFIG_PATH"] = str(config_path)
    env["MACORAG_EVAL_DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", "scripts/eval_macorag.sh", *extra_args],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        env=env,
    )
    return result.stdout


def test_evaluate_shell_script_derives_gpu_visibility_from_yaml() -> None:
    script = Path("scripts/eval_macorag.sh").read_text(encoding="utf-8")

    assert "CONFIG_PATH=" in script
    assert 'ENV_FILE="${REPO_ROOT}/.env"' in script
    assert 'source "${ENV_FILE}"' in script
    assert "parse_args" in script
    assert 'export CUDA_VISIBLE_DEVICES="${EFFECTIVE_GPU_INDICES}"' in script
    assert 'export MACORAG_SILENT_RETRIEVAL="${MACORAG_SILENT_RETRIEVAL:-1}"' in script
    assert "-m evaluation.evaluate_rag_model --config" in script


def test_evaluate_shell_script_uses_cli_config_override_for_cuda_visibility(tmp_path: Path) -> None:
    default_config = tmp_path / "default.yml"
    default_config.write_text('gpu_indices: "1"\n', encoding="utf-8")
    override_config = tmp_path / "override.yml"
    override_config.write_text('gpu_indices: "5"\n', encoding="utf-8")

    output = _run_eval_launcher_dry_run(
        config_path=default_config,
        extra_args=["--config", str(override_config)],
    )

    assert "CUDA_VISIBLE_DEVICES=5" in output


def test_evaluate_shell_script_uses_cli_gpu_indices_override_for_cuda_visibility(tmp_path: Path) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text('gpu_indices: "1"\n', encoding="utf-8")

    output = _run_eval_launcher_dry_run(
        config_path=config,
        extra_args=["--gpu-indices", "7,8"],
    )

    assert "CUDA_VISIBLE_DEVICES=7,8" in output


def test_vllm_server_helper_script_exists() -> None:
    script = Path("scripts/eval_vllm_server.sh")

    text = script.read_text(encoding="utf-8")

    assert "config/eval_vllm_server.yml" in text
    assert "-m evaluation.vllm_servers" in text
    assert "vllm serve" not in text
    assert "argparse" not in text


def test_model_vllm_server_config_file_exists() -> None:
    config = Path("config/eval_vllm_server.yml")

    text = config.read_text(encoding="utf-8")

    assert 'vllm_bin: "/data/conda/envs/macorag/bin/vllm"' in text
    assert 'model_path: "model/NousResearch-Meta-Llama-3-8B-Instruct"' in text
    assert 'adapter_path: "' in text
    assert "vllm_model:" in text
    assert "gpu_indices:" in text
    assert "vllm_base_urls:" in text
    assert "max_model_len:" in text
    assert "max_model_len: null" not in text
    assert "gpu_memory_utilization: 0.85" in text
    assert '--disable-log-requests' in text
    assert "host:" not in text
    assert "trust_remote_code:" not in text
    assert "environment:" not in text


def test_eval_macorag_config_is_vllm_client_only() -> None:
    text = Path("config/eval_macorag.yml").read_text(encoding="utf-8")
    eval_args = parse_args(["--config", "config/eval_macorag.yml"])
    server_args = parse_vllm_server_args(["--config", "config/eval_vllm_server.yml"])

    assert eval_args.data_root == "data/eval_1000_stratified_v2"
    assert eval_args.retrieval_root == "data/eval_1000_stratified_v2_e5_faiss"
    assert "model_path:" in text
    assert "adapter_path:" in text
    assert "inference_backend:" not in text
    assert eval_args.model_path == server_args.model_path
    assert eval_args.adapter_path == server_args.adapter_path
    assert eval_args.adapter_identity_path == server_args.adapter_path
    assert not hasattr(eval_args, "inference_backend")


def test_run_config_snapshot_excludes_config_path_and_keeps_model_provenance() -> None:
    args = SimpleNamespace(
        config="/tmp/eval.yml",
        data_root="data/eval",
        max_rounds=4,
        model_path="model/base",
        adapter_path="outputs/sft/adapter",
    )

    snapshot = _args_to_jsonable(args)

    assert "config" not in snapshot
    assert snapshot == {
        "data_root": "data/eval",
        "max_rounds": 4,
        "model_path": "model/base",
        "adapter_path": "outputs/sft/adapter",
    }


def test_adapter_identity_records_and_validates_corresponding_base_model(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "model/base"}), encoding="utf-8"
    )
    (adapter / "prompt_contract.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    identity = _adapter_identity(
        SimpleNamespace(adapter_identity_path=str(adapter), model_path="model/base")
    )

    assert identity["path"] == str(adapter)
    assert identity["base_model_path"] == "model/base"
    with pytest.raises(ValueError, match="base model mismatch"):
        _adapter_identity(
            SimpleNamespace(adapter_identity_path=str(adapter), model_path="model/other")
        )


def test_vllm_server_module_builds_commands_from_config_and_cli(tmp_path: Path) -> None:
    config = tmp_path / "eval_vllm_server.yml"
    config.write_text(
        "\n".join(
            [
                'vllm_bin: "/opt/vllm/bin/vllm"',
                'model_path: "model/base"',
                'adapter_path: "outputs/adapter"',
                'vllm_model: "adapter-name"',
                'gpu_indices: "2,3"',
                "vllm_base_urls:",
                '  - "http://127.0.0.1:8100/v1"',
                '  - "http://127.0.0.1:8101/v1"',
                'dtype: "float16"',
                "gpu_memory_utilization: 0.8",
                "max_model_len: 2048",
                "extra_args:",
                '  - "--max-num-seqs"',
                '  - "32"',
            ]
        ),
        encoding="utf-8",
    )

    args = parse_vllm_server_args(["--config", str(config)])
    commands = build_commands(args)

    assert commands[0][0] == "2"
    assert commands[0][1]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert commands[0][1]["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"
    assert commands[0][2][:6] == ["/opt/vllm/bin/vllm", "serve", "model/base", "--host", "127.0.0.1", "--port"]
    assert commands[0][2][6] == "8100"
    assert "--enable-lora" in commands[0][2]
    assert f"adapter-name=outputs/adapter" in commands[0][2]
    assert "--max-num-seqs" in commands[0][2]
    assert commands[1][0] == "3"
    assert commands[1][2][6] == "8101"


def test_model_vllm_server_script_dry_run_uses_config_values(tmp_path: Path) -> None:
    adapter = tmp_path / "chosen-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "prompt_contract.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "eval_vllm_server.yml"
    config.write_text(
        "\n".join(
            [
                'vllm_bin: "/opt/vllm/bin/vllm"',
                'model_path: "model/base"',
                f'adapter_path: "{adapter}"',
                'vllm_model: "adapter-name"',
                'gpu_indices: "2,3"',
                "vllm_base_urls:",
                '  - "http://127.0.0.1:8100/v1"',
                '  - "http://127.0.0.1:8101/v1"',
                'host: "0.0.0.0"',
                'dtype: "float16"',
                "gpu_memory_utilization: 0.8",
                "max_model_len: 2048",
                "trust_remote_code: true",
                "environment:",
                '  VLLM_USE_FLASHINFER_SAMPLER: "0"',
                '  VLLM_ATTENTION_BACKEND: "FLASH_ATTN"',
                "extra_args:",
                '  - "--max-num-seqs"',
                '  - "32"',
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["MACORAG_VLLM_DRY_RUN"] = "1"
    env.pop("ADAPTER_PATH", None)

    result = subprocess.run(
        ["bash", "scripts/eval_vllm_server.sh", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        env=env,
    )

    assert "CUDA_VISIBLE_DEVICES=2 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN /opt/vllm/bin/vllm serve model/base --host 0.0.0.0 --port 8100" in result.stdout
    assert "VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN" in result.stdout
    assert f"--enable-lora --lora-modules adapter-name={adapter}" in result.stdout
    assert "--dtype float16 --gpu-memory-utilization 0.8 --max-model-len 2048 --trust-remote-code --max-num-seqs 32" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=3 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN /opt/vllm/bin/vllm serve model/base --host 0.0.0.0 --port 8101" in result.stdout


def test_model_vllm_server_script_cli_overrides_config(tmp_path: Path) -> None:
    adapter = tmp_path / "chosen-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "prompt_contract.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "eval_vllm_server.yml"
    config.write_text(
        "\n".join(
            [
                'vllm_bin: "/opt/vllm/bin/vllm"',
                'model_path: "model/base"',
                f'adapter_path: "{adapter}"',
                'vllm_model: "adapter-name"',
                'gpu_indices: "2"',
                "vllm_base_urls:",
                '  - "http://127.0.0.1:8100/v1"',
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["MACORAG_VLLM_DRY_RUN"] = "1"
    env.pop("ADAPTER_PATH", None)

    result = subprocess.run(
        [
            "bash",
            "scripts/eval_vllm_server.sh",
            "--config",
            str(config),
            "--gpu-indices",
            "4",
            "--vllm-base-urls",
            "http://127.0.0.1:8200/v1",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        env=env,
    )

    assert "CUDA_VISIBLE_DEVICES=4 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN /opt/vllm/bin/vllm serve model/base --host 127.0.0.1 --port 8200" in result.stdout


def test_model_vllm_server_script_rejects_invalid_manual_adapter_path(tmp_path: Path) -> None:
    missing_adapter = tmp_path / "missing-adapter"
    config = tmp_path / "eval_vllm_server.yml"
    config.write_text(
        "\n".join(
            [
                'vllm_bin: "/opt/vllm/bin/vllm"',
                'model_path: "model/base"',
                f'adapter_path: "{missing_adapter}"',
                'vllm_model: "adapter-name"',
                'gpu_indices: "2"',
                "vllm_base_urls:",
                '  - "http://127.0.0.1:8100/v1"',
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["MACORAG_VLLM_DRY_RUN"] = "1"
    env.pop("ADAPTER_PATH", None)

    result = subprocess.run(
        ["bash", "scripts/eval_vllm_server.sh", "--config", str(config)],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        env=env,
    )

    assert result.returncode == 2
    assert "missing adapter_config.json" in result.stderr


def test_vllm_server_run_commands_leaves_process_output_on_console(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeProcess:
        def poll(self) -> int | None:
            return 0

        def wait(self) -> int:
            return 0

    def fake_popen(argv, *, env):
        calls.append({"argv": argv, "env": env})
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    commands = [
        (
            "2",
            {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
            ["/opt/vllm/bin/vllm", "serve", "model/base", "--port", "8100"],
        )
    ]

    with pytest.raises(SystemExit) as exc:
        run_commands(commands)

    assert exc.value.code == 0
    assert calls[0]["argv"] == ["/opt/vllm/bin/vllm", "serve", "model/base", "--port", "8100"]
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert calls[0]["env"]["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"


def test_evaluate_configure_visible_gpus_respects_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    args = SimpleNamespace(gpu_indices="1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    _configure_visible_gpus(args)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_evaluate_build_retrieval_env_uses_eval_retrieval_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_retrieval_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("evaluation.evaluate_rag_model.create_retrieval_env", fake_create_retrieval_env)
    args = SimpleNamespace(
        retrieval_backend="e5_faiss",
        retrieval_root="data/eval_1000_e5_faiss",
        retrieval_embedding_model="intfloat/e5-base-v2",
        retrieval_device="cpu",
        retrieval_max_length=512,
        retrieval_spacy_model="en_core_web_trf",
        retrieval_top_k=5,
        retrieval_max_workers=4,
        retrieval_batch_size=32,
        use_vectorized_retrieval=True,
    )

    _build_retrieval_env(args)

    assert captured["backend"] == "e5_faiss"
    assert captured["retrieval_root"] == "data/eval_1000_e5_faiss"
    assert captured["embedding_model"] == "intfloat/e5-base-v2"
    assert captured["device"] == "cpu"
    assert captured["max_length"] == 512
    assert captured["top_k"] == 5


def test_evaluate_config_loads_e5_faiss_backend_fields(tmp_path: Path) -> None:
    config = tmp_path / "eval.yml"
    config.write_text(
        "\n".join(
            [
                "retrieval_backend: e5_faiss",
                "retrieval_embedding_model: intfloat/e5-base-v2",
                "retrieval_device: cpu",
                "retrieval_max_length: 512",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.retrieval_backend == "e5_faiss"
    assert args.retrieval_embedding_model == "intfloat/e5-base-v2"
    assert args.retrieval_device == "cpu"
    assert args.retrieval_max_length == 512


def test_cached_retrieval_env_shares_dataset_engine_across_threads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from rl_training.retrieval import CachedLinearRAGRetrievalEnv

    created_by_thread: list[str] = []
    queried_by_thread: list[str] = []

    class FakeQueryResult:
        passages = ["passage"]
        scores = [1.0]

    class FakeEngine:
        def __init__(self, owner: str) -> None:
            self.owner = owner

        def query(self, query: str) -> FakeQueryResult:
            queried_by_thread.append(threading.current_thread().name)
            return FakeQueryResult()

    def fake_create_linear_rag_query_engine(**kwargs) -> FakeEngine:
        owner = threading.current_thread().name
        created_by_thread.append(owner)
        return FakeEngine(owner)

    monkeypatch.setattr("rl_training.retrieval.create_linear_rag_query_engine", fake_create_linear_rag_query_engine)
    env = CachedLinearRAGRetrievalEnv(
        retrieval_root=tmp_path,
        embedding_model="embedding",
        spacy_model=None,
        top_k=5,
        max_workers=2,
        batch_size=4,
        use_vectorized_retrieval=True,
    )
    barrier = threading.Barrier(2, timeout=5)

    def query_once() -> None:
        barrier.wait()
        env.query("hotpotqa", "query")

    threads = [threading.Thread(target=query_once, name=f"worker-{index}") for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(created_by_thread) == 1
    assert sorted(queried_by_thread) == ["worker-0", "worker-1"]


def test_cached_retrieval_env_can_prewarm_dataset_engines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from rl_training.retrieval import CachedLinearRAGRetrievalEnv

    created: list[str] = []

    class FakeEngine:
        def query(self, query: str) -> object:
            raise AssertionError("prewarm should not query")

    def fake_create_linear_rag_query_engine(**kwargs) -> FakeEngine:
        created.append(kwargs["dataset"])
        return FakeEngine()

    monkeypatch.setattr("rl_training.retrieval.create_linear_rag_query_engine", fake_create_linear_rag_query_engine)
    env = CachedLinearRAGRetrievalEnv(
        retrieval_root=tmp_path,
        embedding_model="embedding",
        spacy_model=None,
        top_k=5,
        max_workers=2,
        batch_size=4,
        use_vectorized_retrieval=True,
    )

    env.prewarm(["hotpotqa", "2wiki", "hotpotqa"])

    assert created == ["hotpotqa", "2wiki"]


def test_cached_retrieval_env_prewarm_prepares_each_engine_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from rl_training.retrieval import CachedLinearRAGRetrievalEnv

    prepared: list[str] = []

    class FakeEngine:
        def __init__(self, dataset: str) -> None:
            self.dataset = dataset

        def prepare(self) -> None:
            prepared.append(self.dataset)

    monkeypatch.setattr(
        "rl_training.retrieval.create_linear_rag_query_engine",
        lambda **kwargs: FakeEngine(kwargs["dataset"]),
    )
    env = CachedLinearRAGRetrievalEnv(
        retrieval_root=tmp_path,
        embedding_model="embedding",
        spacy_model=None,
        top_k=5,
        max_workers=2,
        batch_size=4,
        use_vectorized_retrieval=True,
    )

    env.prewarm(["hotpotqa", "hotpotqa", "2wiki"])

    assert prepared == ["hotpotqa", "2wiki"]


def test_cached_retrieval_env_batches_queries_with_one_engine_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from data_processing.retrieval import RetrievalResult
    from rl_training.retrieval import CachedLinearRAGRetrievalEnv

    batches: list[list[str]] = []

    class FakeEngine:
        def query_batch(self, queries: list[str]) -> list[RetrievalResult]:
            batches.append(queries)
            return [
                RetrievalResult(
                    dataset="hotpotqa",
                    query=query,
                    passages=[f"passage:{query}"],
                    scores=[float(index)],
                )
                for index, query in enumerate(queries)
            ]

    monkeypatch.setattr(
        "rl_training.retrieval.create_linear_rag_query_engine",
        lambda **kwargs: FakeEngine(),
    )
    env = CachedLinearRAGRetrievalEnv(
        retrieval_root=tmp_path,
        embedding_model="embedding",
        spacy_model=None,
        top_k=5,
        max_workers=2,
        batch_size=4,
        use_vectorized_retrieval=True,
    )

    observations = env.query_batch("hotpotqa", ["q1", "q2"])

    assert batches == [["q1", "q2"]]
    assert [item["query"] for item in observations] == ["q1", "q2"]
    assert [item["passages"][0]["passage_id"] for item in observations] == [0, 0]
    assert [item["passages"][0]["score"] for item in observations] == [0.0, 1.0]
    assert env.query_batch("hotpotqa", []) == []


def test_cached_retrieval_env_deduplicates_batch_misses_and_isolates_copies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from data_processing.retrieval import RetrievalResult
    from rl_training.retrieval import CachedLinearRAGRetrievalEnv

    batches: list[list[str]] = []

    class FakeEngine:
        def query_batch(self, queries):
            batches.append(list(queries))
            return [
                RetrievalResult(
                    dataset="hotpotqa",
                    query=query,
                    passages=[f"passage:{query}"],
                    scores=[1.0],
                )
                for query in queries
            ]

    monkeypatch.setattr(
        "rl_training.retrieval.create_linear_rag_query_engine",
        lambda **kwargs: FakeEngine(),
    )
    env = CachedLinearRAGRetrievalEnv(
        retrieval_root=tmp_path,
        embedding_model="embedding",
        spacy_model=None,
        top_k=5,
        max_workers=2,
        batch_size=4,
        use_vectorized_retrieval=True,
        query_cache_size=8,
    )

    first = env.query_batch("hotpotqa", [" Q1 ", "q1", "q2"])
    first[0]["passages"][0]["text"] = "mutated"
    second = env.query_batch("hotpotqa", ["q1", "q2"])

    assert batches == [[" Q1 ", "q2"]]
    assert [item["query"] for item in first] == [" Q1 ", "q1", "q2"]
    assert second[0]["passages"][0]["text"] == "passage: Q1 "
    assert env.stats()["cache_hits"] == 3
    assert env.stats()["cache_misses"] == 2
    assert env.stats()["time_retrieval_seconds"] >= 0.0


def test_cached_retrieval_env_evicts_lru_and_zero_disables_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from data_processing.retrieval import RetrievalResult
    from rl_training.retrieval import CachedLinearRAGRetrievalEnv

    batches: list[list[str]] = []

    class FakeEngine:
        def query_batch(self, queries):
            batches.append(list(queries))
            return [RetrievalResult("hotpotqa", query, [query], [1.0]) for query in queries]

    monkeypatch.setattr(
        "rl_training.retrieval.create_linear_rag_query_engine",
        lambda **kwargs: FakeEngine(),
    )
    common = dict(
        retrieval_root=tmp_path,
        embedding_model="embedding",
        spacy_model=None,
        top_k=5,
        max_workers=2,
        batch_size=4,
        use_vectorized_retrieval=True,
    )
    cached = CachedLinearRAGRetrievalEnv(**common, query_cache_size=1)
    cached.query_batch("hotpotqa", ["q1"])
    cached.query_batch("hotpotqa", ["q2"])
    cached.query_batch("hotpotqa", ["q1"])

    uncached = CachedLinearRAGRetrievalEnv(**common, query_cache_size=0)
    uncached.query_batch("hotpotqa", ["q"])
    uncached.query_batch("hotpotqa", ["q"])

    assert batches[:3] == [["q1"], ["q2"], ["q1"]]
    assert batches[3:] == [["q"], ["q"]]
    assert cached.stats()["cache_hits"] == 0
    assert cached.stats()["cache_misses"] == 3
    assert uncached.stats()["cache_hits"] == 0
    assert uncached.stats()["cache_misses"] == 2


def test_evaluate_main_writes_local_metrics_for_each_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"prediction_dirs": [], "evaluation_paths": []}

    def fake_evaluate_predictions(predictions_path: Path):
        captured["evaluation_paths"].append(predictions_path)
        return {
            "contain_accuracy": 0.0,
            "exact_match": 0.0,
            "f1": 0.0,
            "num_samples": 0,
        }

    args = SimpleNamespace(
        model_path="model/base",
        adapter_path="outputs/grpo/adapter",
        data_root="data/eval_1000",
        data_files=[],
        retrieval_root="data/eval_1000_retrieval",
        output_root=str(tmp_path / "outputs"),
        max_samples=None,
        max_rounds=3,
        max_prompt_length=128,
        max_completion_length=16,
        temperature=0.0,
        top_p=0.95,
        top_k=5,
        bf16=False,
        fp16=False,
        load_4bit=False,
        gpu_indices="1",
        retrieval_embedding_model="sentence-transformers/all-mpnet-base-v2",
        retrieval_spacy_model="en_core_web_trf",
        retrieval_top_k=5,
        retrieval_max_workers=4,
        retrieval_batch_size=32,
        use_vectorized_retrieval=True,
    )

    monkeypatch.setattr("evaluation.evaluate_rag_model.parse_args", lambda argv=None: args)
    monkeypatch.setattr("evaluation.evaluate_rag_model._configure_visible_gpus", lambda parsed_args: None)
    monkeypatch.setattr("evaluation.evaluate_rag_model._resolved_output_dir", lambda parsed_args: tmp_path / "eval_run")
    samples = [
        EvalSample("h1", "hotpotqa", "Question h?", "Answer h", [], [], {}),
        EvalSample("m1", "musique", "Question m?", "Answer m", [], [], {}),
    ]
    monkeypatch.setattr(
        "evaluation.evaluate_rag_model.load_eval_samples",
        lambda **kwargs: (samples, {"loaded_samples": 2}),
    )
    monkeypatch.setattr("evaluation.evaluate_rag_model._load_policy", lambda parsed_args: object())
    monkeypatch.setattr("evaluation.evaluate_rag_model._build_retrieval_env", lambda parsed_args: object())

    def fake_run_predictions(parsed_args, dataset_samples, policy, retrieval_env, output_dir):
        captured["prediction_dirs"].append((output_dir, [sample.qid for sample in dataset_samples]))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "predictions.jsonl").write_text("{}", encoding="utf-8")
        return []

    monkeypatch.setattr("evaluation.evaluate_rag_model.run_predictions", fake_run_predictions)
    monkeypatch.setattr("evaluation.evaluate_rag_model.evaluate_predictions", fake_evaluate_predictions)

    assert main([]) == 0
    assert captured["prediction_dirs"] == [
        (tmp_path / "eval_run" / "hotpotqa", ["h1"]),
        (tmp_path / "eval_run" / "musique", ["m1"]),
    ]
    assert captured["evaluation_paths"] == [
        tmp_path / "eval_run" / "hotpotqa" / "predictions.jsonl",
        tmp_path / "eval_run" / "musique" / "predictions.jsonl",
    ]
    aggregate = json.loads((tmp_path / "eval_run" / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert aggregate["macro_f1"] == 0.0
    assert set(aggregate["datasets"]) == {"hotpotqa", "musique"}
    assert (tmp_path / "eval_run" / "aggregate_protocol_metrics.json").is_file()
    assert (tmp_path / "eval_run" / "evaluation_contract.json").is_file()


def test_parse_eval_config_loads_yaml_and_cli_overrides(tmp_path: Path) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text(
        "\n".join(
            [
                'data_root: "data/eval_1000"',
                'retrieval_root: "data/eval_1000_retrieval"',
                'output_root: "outputs/eval"',
                "max_samples: 20",
                "max_rounds: 2",
                "retrieval_top_k: 4",
                'gpu_indices: "1"',
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config), "--max-samples", "3"])

    assert args.data_root == "data/eval_1000"
    assert args.retrieval_root == "data/eval_1000_retrieval"
    assert args.output_root == "outputs/eval"
    assert args.max_samples == 3
    assert args.max_rounds == 2
    assert args.retrieval_top_k == 4
    assert args.gpu_indices == "1"


def test_eval_config_has_no_external_judge_fields() -> None:
    args = parse_args(["--config", "config/eval_macorag.yml"])
    text = Path("config/eval_macorag.yml").read_text(encoding="utf-8")

    assert "skip_judge" not in text
    assert "judge_" not in text
    assert not any(name == "skip_judge" or name.startswith("judge_") for name in vars(args))


def test_obsolete_judge_yaml_fields_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "stale.yml"
    config.write_text("judge_model: qwen-plus\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Unknown evaluation config keys.*judge_model"):
        parse_args(["--config", str(config)])


def test_parse_eval_config_loads_vllm_backend_fields(tmp_path: Path) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text(
        "\n".join(
            [
                "vllm_base_urls:",
                '  - "http://127.0.0.1:8000/v1"',
                '  - "http://127.0.0.1:8001/v1"',
                'vllm_model: "macorag-lora"',
                'vllm_api_key_env: ""',
                "vllm_timeout: 30",
                "vllm_retries: 2",
                "vllm_retry_sleep_seconds: 0.1",
                "eval_request_workers: 8",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config), "--eval-request-workers", "4"])

    assert args.vllm_base_urls == ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"]
    assert args.vllm_model == "macorag-lora"
    assert args.vllm_api_key_env == ""
    assert args.vllm_timeout == 30
    assert args.vllm_retries == 2
    assert args.vllm_retry_sleep_seconds == 0.1
    assert args.eval_request_workers == 4


def test_parse_eval_config_rejects_unknown_yaml_keys(tmp_path: Path) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text("unknown_key: 1\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Unknown evaluation config keys"):
        parse_args(["--config", str(config)])


@pytest.mark.parametrize(
    "removed_key",
    ["fixed_output_dir", "gpu_index", "seed", "system_prompt", "disable_tqdm", "top_k"],
)
def test_parse_eval_config_rejects_removed_yaml_keys(tmp_path: Path, removed_key: str) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text(f"{removed_key}: old\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Unknown evaluation config keys"):
        parse_args(["--config", str(config)])


def test_parse_eval_config_rejects_missing_explicit_config_with_equals(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_config.yml"

    with pytest.raises(SystemExit, match="Evaluation config not found"):
        parse_args([f"--config={missing}"])


def test_make_run_dir_creates_timestamped_child(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path / "eval_root", timestamp="2026-07-02_12-34-56")

    assert run_dir == tmp_path / "eval_root" / "2026-07-02_12-34-56"
    assert run_dir.is_dir()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_eval_samples_skips_corpus_and_normalizes_gold_answer(tmp_path: Path) -> None:
    data_root = tmp_path / "eval"
    _write_jsonl(
        data_root / "hotpotqa" / "hotpotqa_dev.jsonl",
        [
            {
                "qid": "q1",
                "dataset": "hotpotqa",
                "question": "Who directed The Tripper?",
                "gold_answer": "David Arquette",
                "answer_aliases": ["Arquette"],
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
                "metadata": {"split": "dev"},
            },
            {
                "qid": "bad",
                "dataset": "hotpotqa",
                "question": "",
                "answer": "missing question",
                "supporting_facts": [],
            },
        ],
    )
    _write_jsonl(data_root / "hotpotqa" / "corpus.jsonl", [{"doc_id": "d1", "text": "not a sample"}])

    samples, summary = load_eval_samples(data_root=data_root, data_files=[], max_samples=None)

    assert len(samples) == 1
    assert samples[0].qid == "q1"
    assert samples[0].dataset == "hotpotqa"
    assert samples[0].answer == "David Arquette"
    assert samples[0].answer_aliases == ["Arquette"]
    assert samples[0].metadata == {"split": "dev"}
    assert summary["loaded_samples"] == 1
    assert summary["skipped_samples"] == 1
    assert summary["source_files"] == [str(data_root / "hotpotqa" / "hotpotqa_dev.jsonl")]


def test_load_eval_samples_uses_gold_answer_when_answer_is_null(tmp_path: Path) -> None:
    data_root = tmp_path / "eval"
    _write_jsonl(
        data_root / "hotpotqa" / "hotpotqa_dev.jsonl",
        [
            {
                "qid": "q2",
                "dataset": "hotpotqa",
                "question": "Who is the director?",
                "answer": None,
                "gold_answer": "David Arquette",
                "answer_aliases": ["Arquette"],
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
            },
        ],
    )

    samples, _ = load_eval_samples(data_root=data_root, data_files=[], max_samples=None)

    assert len(samples) == 1
    assert samples[0].answer == "David Arquette"


def test_load_eval_samples_fails_fast_for_missing_explicit_file_even_after_max_samples(tmp_path: Path) -> None:
    data_root = tmp_path / "eval"
    _write_jsonl(
        data_root / "hotpotqa" / "first.jsonl",
        [
            {
                "qid": "q3",
                "dataset": "hotpotqa",
                "question": "Director?",
                "answer": "David Arquette",
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
            },
        ],
    )

    with pytest.raises(FileNotFoundError, match="missing.jsonl"):
        load_eval_samples(
            data_root=data_root,
            data_files=[
                str(data_root / "hotpotqa" / "first.jsonl"),
                str(data_root / "hotpotqa" / "missing.jsonl"),
            ],
            max_samples=1,
        )


def test_explicit_data_files_skips_corpus_jsonl(tmp_path: Path) -> None:
    data_root = tmp_path / "eval"
    _write_jsonl(
        data_root / "hotpotqa" / "corpus.jsonl",
        [{"doc_id": "d1", "text": "should be skipped"}],
    )
    _write_jsonl(
        data_root / "hotpotqa" / "hotpotqa_dev.jsonl",
        [
            {
                "qid": "q4",
                "dataset": "hotpotqa",
                "question": "Who directed The Tripper?",
                "answer": "David Arquette",
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
            }
        ],
    )

    samples, summary = load_eval_samples(
        data_root=data_root,
        data_files=[
            str(Path("hotpotqa") / "corpus.jsonl"),
            str(Path("hotpotqa") / "hotpotqa_dev.jsonl"),
        ],
    )

    assert len(samples) == 1
    assert summary["loaded_samples"] == 1
    assert summary["source_files"] == [str(data_root / "hotpotqa" / "hotpotqa_dev.jsonl")]


def test_explicit_data_files_accepts_dataset_directory(tmp_path: Path) -> None:
    data_root = tmp_path / "eval"
    _write_jsonl(data_root / "2wiki" / "corpus.jsonl", [{"doc_id": "d1", "text": "should be skipped"}])
    _write_jsonl(
        data_root / "2wiki" / "2wiki_dev.jsonl",
        [
            {
                "qid": "q5",
                "dataset": "2wiki",
                "question": "Where was the director born?",
                "answer": "London",
                "supporting_facts": [{"title": "Director", "text": "Born in London."}],
            }
        ],
    )
    _write_jsonl(
        data_root / "hotpotqa" / "hotpotqa_dev.jsonl",
        [
            {
                "qid": "q6",
                "dataset": "hotpotqa",
                "question": "Who directed The Tripper?",
                "answer": "David Arquette",
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
            }
        ],
    )

    samples, summary = load_eval_samples(data_root=data_root, data_files=["2wiki"], max_samples=None)

    assert [sample.dataset for sample in samples] == ["2wiki"]
    assert summary["source_files"] == [str(data_root / "2wiki" / "2wiki_dev.jsonl")]


def test_shared_answer_metrics_preserve_bidirectional_contain_contract() -> None:
    from answer_metrics import calculate_answer_metrics

    assert calculate_answer_metrics("The David Arquette", "David Arquette") == {
        "exact_match": 1,
        "contain": 1,
        "f1": 1.0,
    }
    assert calculate_answer_metrics("London", "London, England") == {
        "exact_match": 0,
        "contain": 1,
        "f1": pytest.approx(2 / 3),
    }


def test_evaluate_predictions_reads_jsonl_and_writes_summary_without_rewriting_predictions(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    original_text = "\n".join(
        [
            json.dumps(
                {"qid": "q1", "dataset": "hotpotqa", "pred_answer": "David Arquette", "gold_answer": "David Arquette"},
                ensure_ascii=False,
            ),
            json.dumps(
                {"qid": "q2", "dataset": "hotpotqa", "pred_answer": "wrong", "gold_answer": "Right"},
                ensure_ascii=False,
            ),
            json.dumps(
                {"qid": "q3", "dataset": "hotpotqa", "pred_answer": "Right", "gold_answer": "Right"},
                ensure_ascii=False,
            ),
            "",
        ]
    )
    predictions_path.write_text(original_text, encoding="utf-8")
    summary = evaluate_predictions(predictions_path)

    assert summary["contain_accuracy"] == pytest.approx(2 / 3)
    assert summary["exact_match"] == pytest.approx(2 / 3)
    assert summary["f1"] == pytest.approx(2 / 3)
    assert summary["num_samples"] == 3
    assert predictions_path.read_text(encoding="utf-8") == original_text
    evaluation_results = json.loads((tmp_path / "evaluation_results.json").read_text(encoding="utf-8"))
    assert "llm_accuracy" not in evaluation_results
    assert evaluation_results["contain_accuracy"] == pytest.approx(2 / 3)
    assert evaluation_results["exact_match"] == pytest.approx(2 / 3)
    assert evaluation_results["f1"] == pytest.approx(2 / 3)
    assert not (tmp_path / "predictions.json").exists()


def test_evaluate_predictions_preserves_falsy_answers(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps({"qid": "q0", "pred_answer": 0, "gold_answer": 0}, ensure_ascii=False),
                json.dumps({"qid": "q1", "pred_answer": False, "gold_answer": False}, ensure_ascii=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = evaluate_predictions(predictions_path)

    assert summary["contain_accuracy"] == 1.0


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None


def test_vllm_policy_posts_chat_completion_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: int) -> _FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeHTTPResponse({"choices": [{"message": {"content": "<answer>{\"can_answer\": true}</answer>"}}]})

    policy = VLLMOpenAIPolicy(
        base_urls=["http://127.0.0.1:8000/v1"],
        model="macorag-lora",
        api_key_env="",
        system_prompt="sys",
        max_prompt_length=128,
        max_completion_length=32,
        temperature=0.0,
        top_p=0.95,
        timeout=12,
        retries=1,
        retry_sleep_seconds=0.0,
    )
    monkeypatch.setattr(policy._loopback_opener, "open", fake_urlopen)

    response = policy.generate(
        role="answer_generator",
        question="Question?",
        state=RAGState(question="Question?"),
    )

    payload = captured["payload"]
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["timeout"] == 12
    assert payload["model"] == "macorag-lora"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 32
    assert "truncate_prompt_tokens" not in payload
    assert response == '<answer>{"can_answer": true}</answer>'


def test_vllm_policy_round_robins_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def fake_urlopen(request: object, timeout: int) -> _FakeHTTPResponse:
        urls.append(request.full_url)
        return _FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

    policy = VLLMOpenAIPolicy(
        base_urls=["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
        model="macorag-lora",
        api_key_env="",
        system_prompt="sys",
        max_prompt_length=128,
        max_completion_length=16,
        temperature=0.0,
        top_p=0.95,
        timeout=12,
        retries=1,
        retry_sleep_seconds=0.0,
    )
    monkeypatch.setattr(policy._loopback_opener, "open", fake_urlopen)

    policy.set_endpoint_index(0)
    policy.generate(role="answer_generator", question="Question?", state=RAGState(question="Question?"))
    policy.set_endpoint_index(1)
    policy.generate(role="answer_generator", question="Question?", state=RAGState(question="Question?"))

    assert urls == [
        "http://127.0.0.1:8000/v1/chat/completions",
        "http://127.0.0.1:8001/v1/chat/completions",
    ]


def test_format_prediction_matches_linearrag_evaluator_schema() -> None:
    sample = EvalSample(
        qid="q1",
        dataset="hotpotqa",
        question="Who directed The Tripper?",
        answer="David Arquette",
        answer_aliases=["Arquette"],
        supporting_facts=[],
        metadata={"split": "dev"},
    )
    result = SimpleNamespace(
        final_answer="David Arquette",
        trajectory=[{"round": 0}],
        parse_errors=[],
        state=SimpleNamespace(retrieval_count=1),
    )

    prediction = format_prediction(sample, result)

    assert prediction["qid"] == "q1"
    assert prediction["dataset"] == "hotpotqa"
    assert prediction["pred_answer"] == "David Arquette"
    assert prediction["gold_answer"] == "David Arquette"
    assert prediction["answer_aliases"] == ["Arquette"]
    assert prediction["trajectory"] == [{"round": 0}]
    assert prediction["parse_errors"] == []
    assert prediction["retrieval_count"] == 1


class FakePolicy:
    pass


class FakeRetrievalEnv:
    def query(self, dataset: str, query: str) -> dict:
        return {"query": query, "passages": []}


def test_run_predictions_flushes_jsonl_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = EvalSample("q1", "hotpotqa", "Question?", "Answer", [], [], {})
    args = SimpleNamespace(max_rounds=1, disable_tqdm=True)

    class FakeExecutor:
        def __init__(self, *, policy, retrieval_env, max_rounds: int) -> None:
            self.max_rounds = max_rounds

        def run(self, *, question: str, dataset: str):
            return SimpleNamespace(
                final_answer="Answer",
                trajectory=[{"round": 0}],
                parse_errors=[],
                state=SimpleNamespace(retrieval_count=0),
            )

    monkeypatch.setattr("evaluation.evaluate_rag_model.RAGLoopExecutor", FakeExecutor)

    predictions = run_predictions(args, [sample], FakePolicy(), FakeRetrievalEnv(), tmp_path)

    assert predictions[0]["pred_answer"] == "Answer"
    progress_lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(progress_lines) == 1
    assert json.loads(progress_lines[0])["qid"] == "q1"
    assert not (tmp_path / "predictions.json").exists()


def test_run_predictions_truncates_stale_progress_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = EvalSample("q1", "hotpotqa", "Question?", "Answer", [], [], {})
    args = SimpleNamespace(max_rounds=1, disable_tqdm=True)
    progress_path = tmp_path / "predictions.jsonl"
    progress_path.write_text('{"qid": "stale"}\n', encoding="utf-8")

    class FakeExecutor:
        def __init__(self, *, policy, retrieval_env, max_rounds: int) -> None:
            self.max_rounds = max_rounds

        def run(self, *, question: str, dataset: str):
            return SimpleNamespace(
                final_answer="Answer",
                trajectory=[{"round": 0}],
                parse_errors=[],
                state=SimpleNamespace(retrieval_count=0),
            )

    monkeypatch.setattr("evaluation.evaluate_rag_model.RAGLoopExecutor", FakeExecutor)

    run_predictions(args, [sample], FakePolicy(), FakeRetrievalEnv(), tmp_path)

    progress_lines = progress_path.read_text(encoding="utf-8").strip().splitlines()
    assert progress_lines == [
        json.dumps(
            {
                "qid": "q1",
                "dataset": "hotpotqa",
                "question": "Question?",
                "pred_answer": "Answer",
                "gold_answer": "Answer",
                "answer_aliases": [],
                "trajectory": [{"round": 0}],
                "parse_errors": [],
                "retrieval_count": 0,
            },
            ensure_ascii=False,
        )
    ]


def test_run_predictions_re_raises_missing_index_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = EvalSample("q1", "hotpotqa", "Question?", "Answer", [], [], {})
    args = SimpleNamespace(max_rounds=1, disable_tqdm=True)

    class FakeExecutor:
        def __init__(self, *, policy, retrieval_env, max_rounds: int) -> None:
            self.max_rounds = max_rounds

        def run(self, *, question: str, dataset: str):
            raise FileNotFoundError("LinearRAG index not found")

    monkeypatch.setattr("evaluation.evaluate_rag_model.RAGLoopExecutor", FakeExecutor)

    with pytest.raises(FileNotFoundError):
        run_predictions(args, [sample], FakePolicy(), FakeRetrievalEnv(), tmp_path)

    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "predictions.json").exists()


@pytest.mark.parametrize(
    "message",
    [
        "Missing spaCy model en_core_web_trf. Install it before retrieval startup.",
        "Cannot import sentence_transformers. Install sentence-transformers first.",
    ],
)
def test_run_predictions_re_raises_runtime_dependency_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    sample = EvalSample("q1", "hotpotqa", "Question?", "Answer", [], [], {})
    args = SimpleNamespace(max_rounds=1, disable_tqdm=True)

    class FakeExecutor:
        def __init__(self, *, policy, retrieval_env, max_rounds: int) -> None:
            self.max_rounds = max_rounds

        def run(self, *, question: str, dataset: str):
            raise RuntimeError(message)

    monkeypatch.setattr("evaluation.evaluate_rag_model.RAGLoopExecutor", FakeExecutor)

    with pytest.raises(RuntimeError, match=message):
        run_predictions(args, [sample], FakePolicy(), FakeRetrievalEnv(), tmp_path)

    assert not (tmp_path / "predictions.jsonl").exists()


def test_run_predictions_re_raises_vllm_service_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = EvalSample("q1", "hotpotqa", "Question?", "Answer", [], [], {})
    args = SimpleNamespace(max_rounds=1, disable_tqdm=True, inference_backend="vllm_openai", eval_request_workers=1)

    class FakeExecutor:
        def __init__(self, *, policy, retrieval_env, max_rounds: int) -> None:
            self.max_rounds = max_rounds

        def run(self, *, question: str, dataset: str):
            raise RuntimeError("vLLM chat completion failed after 3 attempt(s): connection refused")

    monkeypatch.setattr("evaluation.evaluate_rag_model.RAGLoopExecutor", FakeExecutor)

    with pytest.raises(RuntimeError, match="vLLM chat completion failed"):
        run_predictions(args, [sample], FakePolicy(), FakeRetrievalEnv(), tmp_path)

    assert not (tmp_path / "predictions.jsonl").exists()


def test_load_policy_uses_vllm_without_loading_local_model() -> None:
    args = SimpleNamespace(
        vllm_base_urls=["http://127.0.0.1:8000/v1"],
        vllm_model="macorag-lora",
        vllm_api_key_env="",
        system_prompt="sys",
        max_completion_length=16,
        temperature=0.0,
        top_p=0.95,
        vllm_timeout=30,
        vllm_retries=2,
        vllm_retry_sleep_seconds=0.0,
    )

    policy = _load_policy(args)

    assert isinstance(policy, VLLMOpenAIPolicy)


def test_vllm_policy_bypasses_http_proxy_for_loopback_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    class ProxyHandler(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self) -> None:
            type(self).calls += 1
            self.send_error(502, "proxy must not receive loopback traffic")

        def log_message(self, format: str, *args: object) -> None:
            return None

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    threads = [
        threading.Thread(target=target.serve_forever, daemon=True),
        threading.Thread(target=proxy.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        proxy_url = f"http://127.0.0.1:{proxy.server_port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("http_proxy", proxy_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        policy = VLLMOpenAIPolicy(
            base_urls=[f"http://127.0.0.1:{target.server_port}/v1"],
            model="macorag",
            api_key_env="",
            system_prompt=None,
            max_prompt_length=128,
            max_completion_length=16,
            temperature=0.0,
            top_p=1.0,
            timeout=2,
            retries=1,
            retry_sleep_seconds=0.0,
        )
        response = policy._post_chat_completion({"model": "macorag", "messages": []})
        assert response["choices"][0]["message"]["content"] == "ok"
        assert ProxyHandler.calls == 0
    finally:
        target.shutdown()
        proxy.shutdown()
        target.server_close()
        proxy.server_close()
        for thread in threads:
            thread.join(timeout=2)


def test_load_policy_rejects_vllm_without_base_urls() -> None:
    args = SimpleNamespace(
        vllm_base_urls=[],
        vllm_model="macorag-lora",
        vllm_api_key_env="",
        system_prompt="sys",
        max_completion_length=16,
        temperature=0.0,
        top_p=0.95,
        vllm_timeout=30,
        vllm_retries=2,
        vllm_retry_sleep_seconds=0.0,
    )

    with pytest.raises(SystemExit, match="vllm_base_urls"):
        _load_policy(args)


def test_run_predictions_uses_threads_when_multiple_eval_workers_are_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        EvalSample("q1", "hotpotqa", "Question 1?", "Answer 1", [], [], {}),
        EvalSample("q2", "hotpotqa", "Question 2?", "Answer 2", [], [], {}),
    ]
    args = SimpleNamespace(max_rounds=1, disable_tqdm=True, eval_request_workers=2)
    entered = threading.Barrier(2, timeout=5)
    thread_names: set[str] = set()

    class FakeExecutor:
        def __init__(self, *, policy, retrieval_env, max_rounds: int) -> None:
            self.policy = policy

        def run(self, *, question: str, dataset: str):
            thread_names.add(threading.current_thread().name)
            entered.wait()
            return SimpleNamespace(
                final_answer=question.replace("Question", "Answer").replace("?", ""),
                trajectory=[{"question": question}],
                parse_errors=[],
                state=SimpleNamespace(retrieval_count=0),
            )

    monkeypatch.setattr("evaluation.evaluate_rag_model.RAGLoopExecutor", FakeExecutor)

    predictions = run_predictions(args, samples, FakePolicy(), FakeRetrievalEnv(), tmp_path)

    assert [item["qid"] for item in predictions] == ["q1", "q2"]
    assert len(thread_names) == 2
    assert len((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 2
    assert not (tmp_path / "predictions.json").exists()


def test_run_predictions_drains_successful_futures_before_raising_infrastructure_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        EvalSample("fails", "musique", "Question 1?", "Answer 1", [], [], {}),
        EvalSample("succeeds", "musique", "Question 2?", "Answer 2", [], [], {}),
    ]
    failure_released = threading.Event()

    def fake_run_one_prediction(*, index: int, sample: EvalSample, **kwargs):
        if sample.qid == "fails":
            failure_released.set()
            raise RuntimeError("vLLM chat completion failed after 3 attempt(s): connection refused")
        assert failure_released.wait(timeout=2)
        return index, {
            "qid": sample.qid,
            "dataset": sample.dataset,
            "question": sample.question,
            "pred_answer": sample.answer,
            "gold_answer": sample.answer,
            "answer_aliases": [],
            "trajectory": [],
            "parse_errors": [],
            "retrieval_count": 0,
        }

    monkeypatch.setattr("evaluation.evaluate_rag_model._run_one_prediction", fake_run_one_prediction)
    monkeypatch.setattr("evaluation.evaluate_rag_model.as_completed", lambda futures: iter(futures))
    args = SimpleNamespace(eval_request_workers=2, disable_tqdm=True, resume=False)

    with pytest.raises(RuntimeError, match="vLLM chat completion failed"):
        run_predictions(args, samples, object(), object(), tmp_path)

    rows = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["qid"] for row in rows] == ["succeeds"]


def test_main_validates_retrieval_assets_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    retrieval_root = tmp_path / "retrieval"
    dataset_dir = retrieval_root / "hotpotqa"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for file_name in [
        "passage_embedding.parquet",
        "entity_embedding.parquet",
        "LinearRAG.graphml",
    ]:
        (dataset_dir / file_name).write_text("ok", encoding="utf-8")

    args = SimpleNamespace(
        model_path="model/base",
        adapter_path="outputs/grpo/adapter",
        data_root="data/eval_1000",
        data_files=[],
        retrieval_root=str(retrieval_root),
        output_root=str(tmp_path / "outputs"),
        max_samples=None,
        max_rounds=3,
        max_prompt_length=128,
        max_completion_length=16,
        temperature=0.0,
        top_p=0.95,
        top_k=5,
        bf16=False,
        fp16=False,
        load_4bit=False,
        gpu_indices="1",
        retrieval_embedding_model="sentence-transformers/all-mpnet-base-v2",
        retrieval_spacy_model="en_core_web_trf",
        retrieval_top_k=5,
        retrieval_max_workers=4,
        retrieval_batch_size=32,
        use_vectorized_retrieval=True,
    )
    samples = [
        EvalSample(
            qid="q1",
            dataset="hotpotqa",
            question="Question?",
            answer="Answer",
            answer_aliases=[],
            supporting_facts=[],
            metadata={},
        )
    ]

    monkeypatch.setattr("evaluation.evaluate_rag_model.parse_args", lambda argv=None: args)
    monkeypatch.setattr("evaluation.evaluate_rag_model._configure_visible_gpus", lambda parsed_args: None)
    monkeypatch.setattr("evaluation.evaluate_rag_model._resolved_output_dir", lambda parsed_args: tmp_path / "eval_run")
    monkeypatch.setattr(
        "evaluation.evaluate_rag_model.load_eval_samples",
        lambda **kwargs: (samples, {"loaded_samples": 1}),
    )

    def fail_load_policy(_parsed_args):
        raise AssertionError("_load_policy should not run before retrieval asset preflight")

    monkeypatch.setattr("evaluation.evaluate_rag_model._load_policy", fail_load_policy)

    with pytest.raises(FileNotFoundError, match="sentence_embedding.parquet"):
        main([])


def test_main_uses_timestamped_output_root_before_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "eval_root"
    stale_parent_file = output_root / "predictions.jsonl"
    stale_parent_file.parent.mkdir(parents=True)
    stale_parent_file.write_text('{"qid": "stale"}\n', encoding="utf-8")
    run_dir = output_root / "2026-07-02_12-34-56"
    retrieval_root = tmp_path / "retrieval"
    dataset_dir = retrieval_root / "hotpotqa"
    dataset_dir.mkdir(parents=True)
    for file_name in [
        "passage_embedding.parquet",
        "entity_embedding.parquet",
        "sentence_embedding.parquet",
        "LinearRAG.graphml",
    ]:
        (dataset_dir / file_name).write_text("ok", encoding="utf-8")

    args = SimpleNamespace(
        model_path="model/base",
        adapter_path="outputs/grpo/adapter",
        data_root="data/eval_1000",
        data_files=[],
        retrieval_root=str(retrieval_root),
        output_root=str(output_root),
        max_samples=None,
        max_rounds=3,
        max_prompt_length=128,
        max_completion_length=16,
        temperature=0.0,
        top_p=0.95,
        top_k=5,
        bf16=False,
        fp16=False,
        load_4bit=False,
        gpu_indices="1",
        retrieval_embedding_model="sentence-transformers/all-mpnet-base-v2",
        retrieval_spacy_model="en_core_web_trf",
        retrieval_top_k=5,
        retrieval_max_workers=4,
        retrieval_batch_size=32,
        use_vectorized_retrieval=True,
    )
    samples = [
        EvalSample(
            qid="q1",
            dataset="hotpotqa",
            question="Question?",
            answer="Answer",
            answer_aliases=[],
            supporting_facts=[],
            metadata={},
        )
    ]

    monkeypatch.setattr("evaluation.evaluate_rag_model.parse_args", lambda argv=None: args)
    monkeypatch.setattr("evaluation.evaluate_rag_model._configure_visible_gpus", lambda parsed_args: None)
    monkeypatch.setattr("evaluation.evaluate_rag_model._resolved_output_dir", lambda parsed_args: run_dir)
    monkeypatch.setattr(
        "evaluation.evaluate_rag_model.load_eval_samples",
        lambda **kwargs: (samples, {"loaded_samples": 1}),
    )
    monkeypatch.setattr(
        "evaluation.evaluate_rag_model._load_policy",
        lambda parsed_args: (_ for _ in ()).throw(RuntimeError("model load failed")),
    )

    with pytest.raises(RuntimeError, match="model load failed"):
        main([])

    assert stale_parent_file.exists()
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "data_summary.json").exists()
    recorded = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert "config" not in recorded
    assert recorded["model_path"] == "model/base"
    assert recorded["adapter_path"] == "outputs/grpo/adapter"
