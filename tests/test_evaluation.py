from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from pathlib import Path
import socket

import pytest

from rag import RAGState
from evaluation.config import parse_args
from evaluation.data import EvalSample, load_eval_samples
from evaluation.output import make_run_dir
from evaluation.vllm_servers import build_commands, parse_args as parse_vllm_server_args, run_commands
from evaluation.evaluate_rag_model import (
    _build_retrieval_env,
    _configure_visible_gpus,
    _load_policy,
    VLLMOpenAIPolicy,
    format_prediction,
    main,
    run_predictions,
)


from evaluation.bailian_evaluator import (
    calculate_contain,
    calculate_exact_match,
    calculate_f1,
    calculate_llm_accuracy,
    evaluate_predictions,
)


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
    assert 'model_path: "model/Qwen2.5-7B-Instruct"' in text
    assert "adapter_path:" in text
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

    assert "model_path:" not in text
    assert "adapter_path:" not in text
    assert "inference_backend:" not in text
    assert not hasattr(eval_args, "model_path")
    assert not hasattr(eval_args, "adapter_path")
    assert not hasattr(eval_args, "inference_backend")


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
    assert "--enable-lora --lora-modules adapter-name=outputs/adapter" in result.stdout
    assert "--dtype float16 --gpu-memory-utilization 0.8 --max-model-len 2048 --trust-remote-code --max-num-seqs 32" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=3 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN /opt/vllm/bin/vllm serve model/base --host 0.0.0.0 --port 8101" in result.stdout


def test_model_vllm_server_script_cli_overrides_config(tmp_path: Path) -> None:
    config = tmp_path / "eval_vllm_server.yml"
    config.write_text(
        "\n".join(
            [
                'vllm_bin: "/opt/vllm/bin/vllm"',
                'model_path: "model/base"',
                'adapter_path: "outputs/adapter"',
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


def test_model_vllm_server_script_omits_lora_when_adapter_path_is_empty(tmp_path: Path) -> None:
    config = tmp_path / "eval_vllm_server.yml"
    config.write_text(
        "\n".join(
            [
                'vllm_bin: "/opt/vllm/bin/vllm"',
                'model_path: "model/base"',
                "adapter_path: null",
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

    result = subprocess.run(
        ["bash", "scripts/eval_vllm_server.sh", "--config", str(config)],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        env=env,
    )

    assert "/opt/vllm/bin/vllm serve model/base" in result.stdout
    assert "--served-model-name adapter-name" in result.stdout
    assert "--enable-lora" not in result.stdout
    assert "--lora-modules" not in result.stdout


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

    class FakeRetrievalEnv:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("evaluation.evaluate_rag_model.CachedLinearRAGRetrievalEnv", FakeRetrievalEnv)
    args = SimpleNamespace(
        retrieval_root="data/eval_1000_retrieval",
        retrieval_embedding_model="sentence-transformers/all-mpnet-base-v2",
        retrieval_spacy_model="en_core_web_trf",
        retrieval_top_k=5,
        retrieval_max_workers=4,
        retrieval_batch_size=32,
        use_vectorized_retrieval=True,
    )

    _build_retrieval_env(args)

    assert captured["retrieval_root"] == "data/eval_1000_retrieval"
    assert captured["embedding_model"] == "sentence-transformers/all-mpnet-base-v2"
    assert captured["top_k"] == 5


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


def test_evaluate_main_passes_judge_metadata_to_evaluate_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"prediction_dirs": [], "evaluation_paths": []}

    class FakeJudgeClient:
        def __init__(self, **kwargs) -> None:
            captured["judge_client_kwargs"] = kwargs

    def fake_evaluate_predictions(predictions_path: Path, *, client, max_workers: int, judge_metadata=None):
        captured["evaluation_paths"].append(predictions_path)
        captured["client"] = client
        captured["max_workers"] = max_workers
        captured["judge_metadata"] = judge_metadata
        return {
            "llm_accuracy": 0.0,
            "contain_accuracy": 0.0,
            "num_samples": 0,
            "judge_metadata": judge_metadata,
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
        skip_judge=False,
        judge_model="qwen-plus",
        judge_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        judge_api_key_env="DASHSCOPE_API_KEY",
        judge_temperature=0.0,
        judge_max_tokens=8,
        judge_timeout=120,
        judge_retries=3,
        judge_retry_sleep_seconds=2.0,
        judge_workers=4,
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
    monkeypatch.setattr("evaluation.evaluate_rag_model.BailianJudgeClient", FakeJudgeClient)
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
    assert captured["judge_metadata"] == {
        "judge_model": "qwen-plus",
        "judge_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "judge_api_key_env": "DASHSCOPE_API_KEY",
        "judge_temperature": 0.0,
        "judge_max_tokens": 8,
        "judge_timeout": 120,
        "judge_retries": 3,
        "judge_retry_sleep_seconds": 2.0,
        "judge_workers": 4,
    }
    assert captured["max_workers"] == 4


def test_parse_eval_config_loads_yaml_and_cli_overrides(tmp_path: Path) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text(
        "\n".join(
            [
                'data_root: "data/eval_1000"',
                'retrieval_root: "data/eval_1000_retrieval"',
                'output_root: "outputs/eval"',
                'judge_model: "qwen-plus"',
                'judge_api_key_env: "DASHSCOPE_API_KEY"',
                "max_samples: 20",
                "max_rounds: 2",
                "retrieval_top_k: 4",
                'gpu_indices: "1"',
                "skip_judge: false",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config), "--max-samples", "3", "--skip-judge"])

    assert args.data_root == "data/eval_1000"
    assert args.retrieval_root == "data/eval_1000_retrieval"
    assert args.output_root == "outputs/eval"
    assert args.judge_model == "qwen-plus"
    assert args.judge_api_key_env == "DASHSCOPE_API_KEY"
    assert args.max_samples == 3
    assert args.max_rounds == 2
    assert args.retrieval_top_k == 4
    assert args.gpu_indices == "1"
    assert args.skip_judge is True


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
    ["output_dir", "fixed_output_dir", "gpu_index", "seed", "system_prompt", "disable_tqdm", "top_k"],
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


class FakeJudgeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def infer(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        return self.responses.pop(0)


def test_bailian_evaluator_maps_correct_response_and_contain_accuracy() -> None:
    client = FakeJudgeClient(["correct"])

    llm_acc = calculate_llm_accuracy(client, "David Arquette", "David Arquette")

    assert llm_acc == 1.0
    assert calculate_contain("The answer is David Arquette.", "David Arquette") == 1
    assert calculate_contain("London", "London, England, United Kingdom") == 1
    assert calculate_contain("The answer is Wes Craven.", "David Arquette") == 0
    assert calculate_exact_match("The David Arquette", "David Arquette") == 1
    assert calculate_exact_match("London", "London, England") == 0
    assert calculate_f1("London", "London, England") == pytest.approx(2 / 3)
    assert calculate_f1("David Arquette", "David Arquette") == 1.0
    assert calculate_f1("", "David Arquette") == 0.0
    assert "Respond with ONLY 'correct' or 'incorrect'." in client.messages[0][1]["content"]


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
    client = FakeJudgeClient(["correct", "incorrect", "correct"])

    summary = evaluate_predictions(predictions_path, client=client, max_workers=1)

    assert summary["llm_accuracy"] == pytest.approx(2 / 3)
    assert summary["contain_accuracy"] == pytest.approx(2 / 3)
    assert summary["exact_match"] == pytest.approx(2 / 3)
    assert summary["f1"] == pytest.approx(2 / 3)
    assert summary["num_samples"] == 3
    assert predictions_path.read_text(encoding="utf-8") == original_text
    evaluation_results = json.loads((tmp_path / "evaluation_results.json").read_text(encoding="utf-8"))
    assert evaluation_results["llm_accuracy"] == pytest.approx(2 / 3)
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
    client = FakeJudgeClient(["correct", "correct"])

    summary = evaluate_predictions(predictions_path, client=client, max_workers=1)

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


def test_bailian_judge_client_retries_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int] = {"count": 0}

    def fake_urlopen(_request: object, timeout: int) -> _FakeHTTPResponse:
        called["count"] += 1
        if called["count"] == 1:
            raise socket.timeout("temporary timeout")
        return _FakeHTTPResponse({"choices": [{"message": {"content": "correct"}}]})

    monkeypatch.setenv("TEST_BAILIAN_API_KEY", "k")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    from evaluation.bailian_evaluator import BailianJudgeClient

    client = BailianJudgeClient(
        model="test",
        endpoint="https://example.com/api/v1/chat/completions",
        api_key_env="TEST_BAILIAN_API_KEY",
        temperature=0.0,
        max_tokens=16,
        timeout=1,
        retries=2,
        retry_sleep_seconds=0.0,
    )
    result = client.infer([{"role": "user", "content": "Is this correct?"}])

    assert result == "correct"
    assert called["count"] == 2


def test_vllm_policy_posts_chat_completion_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: int) -> _FakeHTTPResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeHTTPResponse({"choices": [{"message": {"content": "<answer>{\"can_answer\": true}</answer>"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
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
    assert payload["truncate_prompt_tokens"] == 128
    assert response == '<answer>{"can_answer": true}</answer>'


def test_vllm_policy_round_robins_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def fake_urlopen(request: object, timeout: int) -> _FakeHTTPResponse:
        urls.append(request.full_url)
        return _FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
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

    policy.set_endpoint_index(0)
    policy.generate(role="answer_generator", question="Question?", state=RAGState(question="Question?"))
    policy.set_endpoint_index(1)
    policy.generate(role="answer_generator", question="Question?", state=RAGState(question="Question?"))

    assert urls == [
        "http://127.0.0.1:8000/v1/chat/completions",
        "http://127.0.0.1:8001/v1/chat/completions",
    ]


def test_evaluate_predictions_includes_judge_metadata_when_provided(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps([{"qid": "q1", "pred_answer": "David Arquette", "gold_answer": "David Arquette"}], ensure_ascii=False),
        encoding="utf-8",
    )
    client = FakeJudgeClient(["correct"])
    metadata = {"judge_model": "test-model", "temperature": 0.0}

    summary = evaluate_predictions(
        predictions_path,
        client=client,
        max_workers=1,
        judge_metadata=metadata,
    )
    evaluation_results = json.loads((tmp_path / "evaluation_results.json").read_text(encoding="utf-8"))

    assert summary["judge_metadata"] == metadata
    assert evaluation_results["judge_metadata"] == metadata


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
        skip_judge=True,
        judge_model="qwen-plus",
        judge_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        judge_api_key_env="DASHSCOPE_API_KEY",
        judge_temperature=0.0,
        judge_max_tokens=8,
        judge_timeout=120,
        judge_retries=3,
        judge_retry_sleep_seconds=2.0,
        judge_workers=4,
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
        skip_judge=True,
        judge_model="qwen-plus",
        judge_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        judge_api_key_env="DASHSCOPE_API_KEY",
        judge_temperature=0.0,
        judge_max_tokens=8,
        judge_timeout=120,
        judge_retries=3,
        judge_retry_sleep_seconds=2.0,
        judge_workers=4,
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
