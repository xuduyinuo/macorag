from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import sys
import types

import pytest
import torch

from rag import AgentRole, RAGState
from prompt_config import load_system_prompt
from rl_training.config import parse_args
from rl_training.data import load_rl_samples
from rl_training.data import RLSample
import rl_training.data as rl_data_module
from rl_training.policy import HFSharedPolicy
from rl_training.policy import sequence_logprobs
import rl_training.policy as policy_module
import rl_training.rewards as reward_module
import rl_training.trainer as trainer_module
from rl_training.rewards import compute_answer_f1, compute_rl_rewards
from rl_training.train_grpo_macorag import _parse_gpu_indices
from rl_training.train_grpo_macorag import _build_policy
from rl_training.train_grpo_macorag import _extract_vllm_server_model_paths
from rl_training.train_grpo_macorag import _validate_local_vllm_server_model
from rl_training.train_grpo_macorag import _train_on_rollouts
from rl_training.train_grpo_macorag import _validate_vllm_gpu_placement
from rl_training.train_grpo_macorag import _build_train_metrics_payload
from rl_training.train_grpo_macorag import _dataset_rollout_path
from rl_training.trainer import compute_grpo_loss
from rl_training.vllm_client import collect_trainable_named_parameters


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample(qid: str, dataset: str) -> RLSample:
    return RLSample(
        qid=qid,
        dataset=dataset,
        question=f"question {qid}",
        answer=f"answer {qid}",
        answer_aliases=[],
        supporting_facts=[],
        context_doc_ids=[],
        metadata={},
    )


def test_select_balanced_samples_limits_total_deterministically() -> None:
    samples = [
        _sample(f"{dataset}-{index}", dataset)
        for dataset in ("2wiki", "hotpotqa", "musique")
        for index in range(10)
    ]

    selected = rl_data_module.select_balanced_samples(
        samples,
        max_total_samples=20,
        seed=42,
    )
    repeated = rl_data_module.select_balanced_samples(
        samples,
        max_total_samples=20,
        seed=42,
    )

    counts = {
        dataset: sum(item.dataset == dataset for item in selected)
        for dataset in ("2wiki", "hotpotqa", "musique")
    }
    assert counts == {"2wiki": 7, "hotpotqa": 7, "musique": 6}
    assert [item.qid for item in selected] == [item.qid for item in repeated]
    assert len({item.qid for item in selected}) == 20


def test_epoch_sample_order_is_deterministic_and_changes_by_epoch() -> None:
    samples = [_sample(f"q-{index}", "hotpotqa") for index in range(12)]

    epoch_one = rl_data_module.epoch_sample_order(samples, seed=7, epoch=1)
    repeated = rl_data_module.epoch_sample_order(samples, seed=7, epoch=1)
    epoch_two = rl_data_module.epoch_sample_order(samples, seed=7, epoch=2)

    assert [item.qid for item in epoch_one] == [item.qid for item in repeated]
    assert [item.qid for item in epoch_one] != [item.qid for item in epoch_two]
    assert {item.qid for item in epoch_one} == {item.qid for item in samples}


def test_parse_args_loads_train_grpo_yaml(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                'model_path: "model/base"',
                'sft_adapter_path: "outputs/sft/adapter"',
                'rl_data_root: "data/rl/train"',
                'retrieval_root: "data/trajectory_train_retrieval"',
                'output_root: "outputs/grpo"',
                "max_samples: 8",
                "max_total_samples: 20",
                "max_rounds: 2",
                "group_size: 4",
                "kl_beta: 0.03",
                "clip_epsilon: 0.15",
                "learning_rate: 0.00001",
                "per_device_train_batch_size: 1",
                "reference_per_device_batch_size: 4",
                "gradient_accumulation_steps: 2",
                "skip_zero_advantage_updates: true",
                "gpu_indices: \"0,1\"",
                "disable_tqdm: false",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config), "--max-samples", "3"])

    assert args.model_path == "model/base"
    assert args.sft_adapter_path == "outputs/sft/adapter"
    assert args.rl_data_root == "data/rl/train"
    assert args.retrieval_root == "data/trajectory_train_retrieval"
    assert args.output_root == "outputs/grpo"
    assert args.max_samples == 3
    assert args.max_total_samples == 20
    assert args.max_rounds == 2
    assert args.group_size == 4
    assert args.kl_beta == 0.03
    assert args.clip_epsilon == 0.15
    assert args.gradient_accumulation_steps == 2
    assert args.reference_per_device_batch_size == 4
    assert args.skip_zero_advantage_updates is True
    assert args.gpu_indices == "0,1"
    assert args.disable_tqdm is False


def test_parse_args_loads_role_credit_weights(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                "query_local_credit_weight: 0.8",
                "evidence_local_credit_weight: 0.6",
                "answer_local_credit_weight: 0.2",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.query_local_credit_weight == 0.8
    assert args.evidence_local_credit_weight == 0.6
    assert args.answer_local_credit_weight == 0.2


def test_rl_default_system_prompt_comes_from_shared_prompt_file() -> None:
    args = parse_args(["--config", "config/train_grpo.yml", "--max-samples", "1"])

    assert args.system_prompt == load_system_prompt()


def test_parse_args_supports_output_root(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text('output_root: "outputs/grpo_runs"\n', encoding="utf-8")

    args = parse_args(["--config", str(config)])

    assert args.output_root == "outputs/grpo_runs"
    assert not hasattr(args, "output_dir")


def test_parse_args_rejects_removed_output_and_log_path_keys(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                'output_dir: "outputs/legacy_grpo"',
                'log_jsonl_path: "outputs/metrics.jsonl"',
                'rollout_jsonl_path: "outputs/rollouts.jsonl"',
            ]
        ),
        encoding="utf-8",
    )

    try:
        parse_args(["--config", str(config)])
    except SystemExit as exc:
        message = str(exc)
        assert "output_dir" in message
        assert "log_jsonl_path" in message
        assert "rollout_jsonl_path" in message
    else:
        raise AssertionError("expected removed output/log path keys to fail")


def test_parse_args_defaults_vllm_lora_adapter_path_to_sft_adapter_path(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                'sft_adapter_path: "outputs/sft/adapter"',
                'vllm_sync_mode: "lora"',
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.vllm_lora_adapter_path == "outputs/sft/adapter"


def test_make_timestamped_run_dir_uses_linearrag_style_child_directory() -> None:
    from rl_training.logging_utils import make_timestamped_run_dir

    assert make_timestamped_run_dir("outputs/grpo_runs", timestamp="2026-07-01_12-34-56") == Path(
        "outputs/grpo_runs/2026-07-01_12-34-56"
    )


def test_rl_runtime_helpers_are_extracted_and_reexported() -> None:
    import rl_training.runtime as runtime
    import rl_training.train_grpo_macorag as entrypoint

    assert entrypoint._parse_gpu_indices is runtime.parse_gpu_indices
    assert entrypoint._validate_vllm_gpu_placement is runtime.validate_vllm_gpu_placement
    assert entrypoint._validate_local_vllm_server_model is runtime.validate_local_vllm_server_model


def test_rl_logging_helpers_are_extracted_and_reexported() -> None:
    import rl_training.logging_utils as logging_utils
    import rl_training.train_grpo_macorag as entrypoint

    assert entrypoint._write_json is logging_utils.write_json
    assert entrypoint._append_jsonl is logging_utils.append_jsonl


def test_train_grpo_yaml_has_documented_sections() -> None:
    text = Path("config/train_grpo.yml").read_text(encoding="utf-8")

    for heading in [
        "# 基础路径",
        "# rollout 与采样",
        "# GRPO 优化",
        "# vLLM 生成与权重同步",
        "# 检索环境",
        "# 日志与检查点",
        "# 运行环境",
    ]:
        assert heading in text


def test_single_gpu_script_forces_hf_offline_mode() -> None:
    script = Path("src/rl_single/run_train_grpo_single.sh").read_text(encoding="utf-8")

    assert "HF_HUB_OFFLINE" in script
    assert "TRANSFORMERS_OFFLINE" in script


def test_linear_rag_query_engine_loads_sentence_transformer_offline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    class FakeLinearRAGConfig:
        def __init__(self, **kwargs: object) -> None:
            captured["config"] = kwargs

    class FakeLinearRAG:
        def __init__(self, config: object) -> None:
            self.config = config

    import data_processing.retrieval as retrieval

    monkeypatch.setattr(retrieval, "_load_linearrag_modules", lambda: (FakeLinearRAGConfig, FakeLinearRAG))
    monkeypatch.setattr(retrieval, "_resolve_spacy_model", lambda model: model or "en_core_web_sm")
    index_dir = tmp_path / "hotpotqa"
    index_dir.mkdir(parents=True)
    for filename in ("passage_embedding.parquet", "entity_embedding.parquet", "sentence_embedding.parquet"):
        (index_dir / filename).write_text("placeholder", encoding="utf-8")

    retrieval.LinearRAGQueryEngine(
        retrieval_root=tmp_path,
        dataset="hotpotqa",
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        spacy_model=None,
    )

    assert captured["model_name"] == "sentence-transformers/all-mpnet-base-v2"
    assert captured["kwargs"]["local_files_only"] is True


def test_linearrag_prepares_query_state_only_once(monkeypatch) -> None:
    import data_processing.retrieval as retrieval

    _, linear_rag_cls = retrieval._load_linearrag_modules()
    engine = object.__new__(linear_rag_cls)
    engine._retrieval_state_prepared = False
    engine.config = Namespace(use_vectorized_retrieval=True)
    engine.device = torch.device("cpu")
    engine.entity_embedding_store = Namespace(hash_id_to_text={"e": "entity"}, embeddings=[[1.0]])
    engine.passage_embedding_store = Namespace(hash_id_to_text={"p": "passage"}, embeddings=[[2.0]])
    engine.sentence_embedding_store = Namespace(hash_id_to_text={"s": "sentence"}, embeddings=[[3.0]])
    engine.graph_loaded = False
    engine.ner_mappings_loaded = True
    engine.entity_hash_id_to_sentence_hash_ids = {"e": {"s"}}
    engine.sentence_hash_id_to_entity_hash_ids = {"s": {"e"}}
    monkeypatch.setattr(engine, "_ensure_graph_ready_for_query", lambda: False)
    sparse_calls: list[str] = []

    def prepare_sparse() -> None:
        sparse_calls.append("prepared")
        fake_sparse = Namespace(shape=(1, 1), _nnz=lambda: 1)
        engine.entity_to_sentence_sparse = fake_sparse
        engine.sentence_to_entity_sparse = fake_sparse

    monkeypatch.setattr(engine, "_precompute_sparse_matrices", prepare_sparse)

    engine._prepare_retrieval_state()
    first_entity_embeddings = engine.entity_embeddings
    engine._prepare_retrieval_state()

    assert sparse_calls == ["prepared"]
    assert engine._retrieval_state_prepared is True
    assert engine.entity_embeddings is first_entity_embeddings
    assert engine.passage_node_indices == []


def test_linearrag_encodes_query_batch_in_one_model_call(monkeypatch) -> None:
    import numpy as np
    import data_processing.retrieval as retrieval

    _, linear_rag_cls = retrieval._load_linearrag_modules()
    engine = object.__new__(linear_rag_cls)
    engine._retrieval_state_prepared = True
    engine._graph_ready_for_query = False
    engine.passage_embedding_store = Namespace(texts=["passage"])

    class FakeEmbeddingModel:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def encode(self, value, **kwargs):
            self.calls.append(value)
            return np.asarray([[1.0], [2.0]], dtype=np.float32)

    embedding_model = FakeEmbeddingModel()
    engine.config = Namespace(
        embedding_model=embedding_model,
        batch_size=8,
        retrieval_top_k=1,
    )
    monkeypatch.setattr(engine, "get_seed_entities", lambda question: ([], [], [], []))
    monkeypatch.setattr(
        engine,
        "dense_passage_retrieval",
        lambda embedding: ([0], [float(np.asarray(embedding).reshape(-1)[0])]),
    )

    rows = engine.retrieve([{"question": "q1"}, {"question": "q2"}])

    assert embedding_model.calls == [["q1", "q2"]]
    assert [row["question"] for row in rows] == ["q1", "q2"]
    assert [row["sorted_passage_scores"] for row in rows] == [[1.0], [2.0]]


def test_linear_rag_query_engine_batches_queries_in_order() -> None:
    import data_processing.retrieval as retrieval

    calls: list[list[dict[str, str]]] = []

    class FakeEngine:
        def retrieve(self, questions: list[dict[str, str]]) -> list[dict[str, object]]:
            calls.append(questions)
            return [
                {
                    "sorted_passage": [f"passage:{item['question']}"],
                    "sorted_passage_scores": [float(index)],
                }
                for index, item in enumerate(questions)
            ]

    query_engine = object.__new__(retrieval.LinearRAGQueryEngine)
    query_engine.dataset = "hotpotqa"
    query_engine.engine = FakeEngine()

    results = query_engine.query_batch(["q1", "q2"])

    assert calls == [[{"question": "q1"}, {"question": "q2"}]]
    assert [item.query for item in results] == ["q1", "q2"]
    assert [item.passages for item in results] == [["passage:q1"], ["passage:q2"]]
    assert [item.scores for item in results] == [[0.0], [1.0]]
    assert query_engine.query_batch([]) == []


def test_linear_rag_query_engine_rejects_mismatched_batch_size() -> None:
    import data_processing.retrieval as retrieval

    query_engine = object.__new__(retrieval.LinearRAGQueryEngine)
    query_engine.dataset = "hotpotqa"
    query_engine.engine = Namespace(retrieve=lambda questions: [])

    with pytest.raises(RuntimeError, match="batch size"):
        query_engine.query_batch(["q1"])


def test_train_grpo_yaml_keeps_tuning_keys_and_removes_low_frequency_defaults() -> None:
    import yaml

    config = yaml.safe_load(Path("config/train_grpo.yml").read_text(encoding="utf-8"))

    for key in [
        "output_root",
        "max_samples",
        "max_rounds",
        "group_size",
        "num_train_epochs",
        "max_steps",
        "learning_rate",
        "gradient_accumulation_steps",
        "kl_beta",
        "clip_epsilon",
        "max_prompt_length",
        "max_completion_length",
        "temperature",
        "top_p",
        "top_k",
        "retrieval_top_k",
        "save_steps",
        "logging_steps",
    ]:
        assert key in config

    for key in [
        "output_dir",
        "vllm_lora_adapter_path",
        "vllm_host",
        "vllm_port",
        "vllm_sync_after_step",
        "vllm_sync_trainable_only",
        "vllm_timeout_seconds",
        "log_jsonl_path",
        "rollout_jsonl_path",
        "gpu_index",
        "check_only",
        "disable_tqdm",
    ]:
        assert key not in config


def test_dataset_rollout_path_groups_samples_by_dataset(tmp_path: Path) -> None:
    assert _dataset_rollout_path(tmp_path, "hotpotqa") == tmp_path / "rollout_samples" / "hotpotqa.jsonl"
    assert _dataset_rollout_path(tmp_path, "2wiki/train") == tmp_path / "rollout_samples" / "2wiki_train.jsonl"


def test_build_train_metrics_payload_includes_gold_answer_and_nested_timing() -> None:
    sample = Namespace(qid="q1", dataset="hotpotqa", answer="Gold answer")
    best_rollout = {
        "final_answer": "Predicted answer",
        "trajectory": [{"answer": {"answer": "Predicted answer"}}],
        "parse_errors": [],
        "rewards": {
            "query_reward": 0.5,
            "evidence_reward": 1.0,
            "answer_f1": 0.75,
        },
    }
    rollouts = [
        {"advantage": -1.0, "rewards": {"total": 1.0}, "actions": [Namespace(advantage=-0.5)]},
        {"advantage": 1.0, "rewards": {"total": 3.0}, "actions": [Namespace(advantage=1.0)]},
    ]
    metrics = {
        "loss": 0.2,
        "policy_loss": 0.1,
        "kl": 0.01,
        "time_policy_forward_seconds": 2.0,
        "time_reference_forward_seconds": 1.5,
        "time_backward_seconds": 4.0,
        "time_optimizer_step_seconds": 0.2,
    }
    rollout_timing = {
        "time_rollout_seconds": 10.0,
        "time_vllm_generate_seconds": 6.0,
        "time_behavior_rescore_seconds": 0.25,
        "time_reward_seconds": 0.3,
    }

    payload = _build_train_metrics_payload(
        epoch=1,
        sample_index=0,
        sample_total=5,
        sample=sample,
        global_step=2,
        metrics=metrics,
        rollouts=rollouts,
        best_rollout=best_rollout,
        learning_rate=0.00001,
        rollout_timing=rollout_timing,
        time_weight_sync_seconds=0.4,
        time_total_seconds=15.0,
    )

    assert payload["gold_answer"] == "Gold answer"
    assert payload["generated_answer"] == "Predicted answer"
    assert payload["action_advantage_mean"] == 0.25
    assert payload["timing"] == {
        "rollout_seconds": 10.0,
        "vllm_generate_seconds": 6.0,
        "behavior_rescore_seconds": 0.25,
        "reward_seconds": 0.3,
        "policy_forward_seconds": 2.0,
        "reference_forward_seconds": 1.5,
        "backward_seconds": 4.0,
        "optimizer_step_seconds": 0.2,
        "weight_sync_seconds": 0.4,
        "total_seconds": 15.0,
    }
    assert all(not key.startswith("time_") for key in payload)


def test_parse_args_supports_disabling_rl_progress_bar(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text("disable_tqdm: true\n", encoding="utf-8")

    args = parse_args(["--config", str(config)])

    assert args.disable_tqdm is True


def test_parse_args_loads_vllm_generation_config(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                "use_vllm_generation: true",
                'vllm_host: "127.0.0.1"',
                "vllm_port: 8123",
                'vllm_gpu_indices: "0"',
                "vllm_tensor_parallel_size: 1",
                "vllm_data_parallel_size: 2",
                "vllm_gpu_memory_utilization: 0.70",
                "vllm_max_model_len: 4608",
                'vllm_dtype: "auto"',
                "vllm_sync_after_step: true",
                "vllm_sync_every_steps: 4",
                "vllm_sync_trainable_only: true",
                "vllm_timeout_seconds: 90",
                'gpu_indices: "1"',
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.use_vllm_generation is True
    assert args.vllm_host == "127.0.0.1"
    assert args.vllm_port == 8123
    assert args.vllm_gpu_indices == "0"
    assert args.vllm_tensor_parallel_size == 1
    assert args.vllm_data_parallel_size == 2
    assert args.vllm_gpu_memory_utilization == 0.70
    assert args.vllm_max_model_len == 4608
    assert args.vllm_dtype == "auto"
    assert args.vllm_sync_after_step is True
    assert args.vllm_sync_every_steps == 4
    assert args.vllm_sync_trainable_only is True
    assert args.vllm_timeout_seconds == 90
    assert args.gpu_indices == "1"


def test_parse_args_loads_vllm_lora_sync_config(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                "use_vllm_generation: true",
                'vllm_sync_mode: "lora"',
                'vllm_lora_name: "macorag_train"',
                "vllm_lora_int_id: 7",
                'vllm_lora_adapter_path: "outputs/adapter"',
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.vllm_sync_mode == "lora"
    assert args.vllm_lora_name == "macorag_train"
    assert args.vllm_lora_int_id == 7
    assert args.vllm_lora_adapter_path == "outputs/adapter"


def test_parse_gpu_indices_normalizes_comma_lists() -> None:
    assert _parse_gpu_indices("0, 1") == {"0", "1"}
    assert _parse_gpu_indices(2) == {"2"}
    assert _parse_gpu_indices("") == set()
    assert _parse_gpu_indices(None) == set()


def test_validate_vllm_gpu_placement_rejects_overlap() -> None:
    args = Namespace(use_vllm_generation=True, gpu_indices="0,1", gpu_index=1, vllm_gpu_indices="0")

    try:
        _validate_vllm_gpu_placement(args)
    except SystemExit as exc:
        assert "vLLM GPU overlap" in str(exc)
    else:
        raise AssertionError("expected GPU overlap validation to fail")


def test_validate_vllm_gpu_placement_allows_separate_gpus() -> None:
    args = Namespace(use_vllm_generation=True, gpu_indices="1", gpu_index=1, vllm_gpu_indices="0")

    _validate_vllm_gpu_placement(args)


def test_extract_vllm_server_model_paths_from_process_cmdlines() -> None:
    cmdlines = [
        ["python", "/data/conda/envs/macorag/bin/trl", "vllm-serve", "--model", "model/Qwen2.5-7B-Instruct"],
        ["python", "other.py"],
        ["trl", "vllm-serve", "--host", "127.0.0.1", "--model=model/Qwen2.5-3B-Instruct"],
    ]

    assert _extract_vllm_server_model_paths(cmdlines) == [
        "model/Qwen2.5-7B-Instruct",
        "model/Qwen2.5-3B-Instruct",
    ]


def test_validate_local_vllm_server_model_rejects_stale_server() -> None:
    args = Namespace(
        use_vllm_generation=True,
        vllm_host="127.0.0.1",
        model_path="model/Qwen2.5-7B-Instruct",
    )

    try:
        _validate_local_vllm_server_model(
            args,
            cmdlines=[["trl", "vllm-serve", "--model", "model/Qwen2.5-3B-Instruct"]],
        )
    except SystemExit as exc:
        assert "vLLM server model mismatch" in str(exc)
        assert "Qwen2.5-3B-Instruct" in str(exc)
        assert "Qwen2.5-7B-Instruct" in str(exc)
    else:
        raise AssertionError("expected stale vLLM server model validation to fail")


def test_validate_local_vllm_server_model_allows_matching_server() -> None:
    args = Namespace(
        use_vllm_generation=True,
        vllm_host="127.0.0.1",
        model_path="model/Qwen2.5-7B-Instruct",
    )

    _validate_local_vllm_server_model(
        args,
        cmdlines=[["trl", "vllm-serve", "--model", "model/Qwen2.5-7B-Instruct"]],
    )


class _TinyParamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.lora_a = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=True)
        self.lora_b = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=True)


class _TinyPeftModel(torch.nn.Module):
    prefix = "lora_"

    def __init__(self) -> None:
        super().__init__()
        self.merged = False
        self.unmerged = False
        self._named_params = [
            (
                "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
                torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
                torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float16), requires_grad=True),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
                torch.nn.Parameter(torch.tensor([3.0]), requires_grad=True),
            ),
            (
                "base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight",
                torch.nn.Parameter(torch.tensor([4.0], dtype=torch.float16), requires_grad=False),
            ),
        ]

    def merge_adapter(self) -> None:
        self.merged = True

    def unmerge_adapter(self) -> None:
        self.unmerged = True

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        yield from self._named_params


class Params4bit:
    def __init__(self) -> None:
        self.packed = torch.zeros(1)
        self.dense = torch.ones(2, 3)
        self.quant_state = object()

    def detach(self) -> torch.Tensor:
        return self.packed


def test_normalize_peft_lora_name_maps_qwen_modules() -> None:
    from rl_training.vllm_lora_mapping import normalize_peft_lora_name

    assert normalize_peft_lora_name(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    ) == "model.layers.0.self_attn.q_proj.lora_A.weight"
    assert normalize_peft_lora_name(
        "base_model.model.model.layers.31.mlp.down_proj.lora_B.default.weight"
    ) == "model.layers.31.mlp.down_proj.lora_B.weight"


def test_normalize_peft_lora_name_ignores_non_lora_weights() -> None:
    from rl_training.vllm_lora_mapping import normalize_peft_lora_name

    assert normalize_peft_lora_name("base_model.model.model.embed_tokens.weight") is None
    assert (
        normalize_peft_lora_name("base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight")
        is None
    )


def test_collect_lora_named_tensors_maps_only_lora_params() -> None:
    from rl_training.vllm_lora_mapping import collect_lora_named_tensors

    model = _TinyPeftModel()

    tensors = collect_lora_named_tensors(model)

    assert sorted(tensors) == [
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    assert all(tensor.device.type == "cpu" for tensor in tensors.values())
    assert tensors["model.layers.0.self_attn.q_proj.lora_A.weight"].dtype is torch.float16


def test_collect_lora_named_tensors_filters_frozen_lora_params() -> None:
    from rl_training.vllm_lora_mapping import collect_lora_named_tensors

    model = _TinyPeftModel()

    tensors = collect_lora_named_tensors(model)

    assert "model.layers.0.self_attn.k_proj.lora_A.weight" not in tensors


def test_collect_trainable_named_parameters_returns_only_trainable_cpu_tensors() -> None:
    model = _TinyParamModel()

    params = collect_trainable_named_parameters(model)

    assert sorted(params) == ["lora_a", "lora_b"]
    assert all(not tensor.requires_grad for tensor in params.values())
    assert all(tensor.device.type == "cpu" for tensor in params.values())
    assert params["lora_a"].item() == 2.0
    assert params["lora_b"].item() == 3.0


class _FakeTRLClient:
    def __init__(self, *, sync_device: torch.device | None = None) -> None:
        self.updated: list[tuple[str, torch.Tensor]] = []
        self.health_checked = False
        self.communicator_initialized = False
        self.sync_device = sync_device

    def check_server(self) -> None:
        self.health_checked = True

    def init_communicator(self) -> None:
        self.communicator_initialized = True
        if self.sync_device is not None and not hasattr(self, "pynccl_comm"):
            self.pynccl_comm = type("FakeCommunicator", (), {"device": self.sync_device})()

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        assert self.communicator_initialized is True
        if self.sync_device is not None:
            assert weights.device == self.sync_device
        self.updated.append((name, weights))


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, text: str = "ok", payload: dict | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(
        self,
        *,
        health_payload: dict | None = None,
        update_status_code: int = 200,
        update_payload: dict | None = None,
        update_states: list[dict] | None = None,
    ) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []
        self.health_payload = health_payload or {}
        self.update_status_code = update_status_code
        self.update_payload = update_payload or {}
        self.update_states = update_states or []

    def get(self, url: str) -> _FakeResponse:
        self.gets.append(url)
        if "/lora_update_status/" in url:
            payload = self.update_states.pop(0) if self.update_states else {"state": "ok", "error": None}
            return _FakeResponse(payload=payload)
        return _FakeResponse(payload=self.health_payload)

    def post(self, url: str, json: dict) -> _FakeResponse:
        self.posts.append((url, json))
        return _FakeResponse(
            status_code=self.update_status_code,
            text="unsupported" if self.update_status_code != 200 else "ok",
            payload=self.update_payload,
        )


def test_vllm_generation_client_batches_prompts_with_aligned_logprobs() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    backend.session = _FakeSession(
        update_payload={
            "completion_ids": [[10, 11], [20]],
            "logprobs": [[-0.1, -0.2], [-0.3]],
        }
    )
    backend.base_url = "http://127.0.0.1:8000"
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)

    outputs = client.generate_batch(
        ["first", "second"],
        max_tokens=8,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
    )

    assert [item.completion_ids for item in outputs] == [[10, 11], [20]]
    assert [item.logprobs for item in outputs] == [[-0.1, -0.2], [-0.3]]
    assert backend.session.posts == [
        (
            "http://127.0.0.1:8000/generate/",
            {
                "prompts": ["first", "second"],
                "n": 1,
                "repetition_penalty": 1.0,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 5,
                "max_tokens": 8,
            },
        )
    ]


def test_vllm_generation_client_rejects_misaligned_logprobs() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    backend.session = _FakeSession(
        update_payload={"completion_ids": [[10, 11]], "logprobs": [[-0.1]]}
    )
    backend.base_url = "http://127.0.0.1:8000"
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)

    with pytest.raises(RuntimeError, match="logprob length"):
        client.generate_batch(
            ["prompt"],
            max_tokens=8,
            temperature=0.7,
            top_p=0.9,
            top_k=5,
        )


def test_vllm_generation_client_empty_batch_avoids_backend_call() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)

    assert client.generate_batch([], max_tokens=8, temperature=0.7, top_p=0.9, top_k=5) == []


def test_vllm_generation_client_syncs_trainable_parameters() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    model = _TinyParamModel()

    elapsed = client.sync_trainable_parameters(model)

    assert elapsed >= 0.0
    assert backend.communicator_initialized is True
    assert [name for name, _ in backend.updated] == ["lora_a", "lora_b"]
    assert all(tensor.device.type == "cpu" for _, tensor in backend.updated)


def test_vllm_generation_client_syncs_parameters_on_communicator_device() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient(sync_device=torch.device("meta"))
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    model = _TinyParamModel()

    client.sync_trainable_parameters(model)

    assert [name for name, _ in backend.updated] == ["lora_a", "lora_b"]
    assert all(tensor.device.type == "meta" for _, tensor in backend.updated)


def test_vllm_generation_client_syncs_lora_parameters_only() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient(sync_device=torch.device("meta"))
    backend.session = _FakeSession(update_payload={"update_id": "sync-1"})
    backend.base_url = "http://127.0.0.1:8000"
    backend.rank = 1
    backend.pynccl_comm = type(
        "FakeCommunicator",
        (),
        {
            "device": torch.device("meta"),
            "broadcast": lambda self, tensor, src: backend.updated.append(("broadcast", tensor)),
            "group": type("Group", (), {"barrier": lambda self: None})(),
        },
    )()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    model = _TinyPeftModel()

    elapsed = client.sync_lora_parameters(model)

    assert elapsed >= 0.0
    assert backend.communicator_initialized is True
    assert len(backend.session.posts) == 1
    assert backend.session.posts[0][0] == "http://127.0.0.1:8000/update_lora_params/"
    assert [item["name"] for item in backend.session.posts[0][1]["tensors"]] == [
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    assert [name for name, _ in backend.updated] == ["broadcast", "broadcast"]
    assert all(tensor.device.type == "meta" for _, tensor in backend.updated)
    assert backend.session.gets == [
        "http://127.0.0.1:8000/lora_update_status/sync-1",
    ]


def test_vllm_generation_client_syncs_lora_parameters_in_one_batch_request() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient(sync_device=torch.device("meta"))
    backend.session = _FakeSession(update_payload={"update_id": "batch-1"})
    backend.base_url = "http://127.0.0.1:8000"
    backend.rank = 1
    backend.pynccl_comm = type(
        "FakeCommunicator",
        (),
        {
            "device": torch.device("meta"),
            "broadcast": lambda self, tensor, src: backend.updated.append(("broadcast", tensor)),
            "group": type("Group", (), {"barrier": lambda self: backend.updated.append(("barrier", torch.empty(0)))})(),
        },
    )()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)

    client.sync_lora_parameters(_TinyPeftModel())

    assert len(backend.session.posts) == 1
    url, payload = backend.session.posts[0]
    assert url == "http://127.0.0.1:8000/update_lora_params/"
    assert [item["name"] for item in payload["tensors"]] == [
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    assert [name for name, _ in backend.updated] == ["broadcast", "broadcast", "barrier"]
    assert backend.session.gets == ["http://127.0.0.1:8000/lora_update_status/batch-1"]


def test_vllm_generation_client_sync_lora_parameters_raises_on_update_error_status() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    backend.session = _FakeSession(
        update_payload={"update_id": "sync-err"},
        update_states=[{"state": "error", "error": "worker failed"}],
    )
    backend.base_url = "http://127.0.0.1:8000"
    backend.rank = 1
    backend.pynccl_comm = type(
        "FakeCommunicator",
        (),
        {
            "broadcast": lambda self, tensor, src: backend.updated.append(("broadcast", tensor)),
            "group": type("Group", (), {"barrier": lambda self: None})(),
        },
    )()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=0.1, backend=backend)

    try:
        client.sync_lora_parameters(_TinyPeftModel())
    except RuntimeError as exc:
        assert "vLLM LoRA update failed" in str(exc)
        assert "worker failed" in str(exc)
    else:
        raise AssertionError("expected LoRA update error status to fail")


def test_vllm_generation_client_sync_lora_parameters_does_not_broadcast_after_preflight_error() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    backend.session = _FakeSession(update_status_code=400, update_payload={"detail": "shape mismatch"})
    backend.base_url = "http://127.0.0.1:8000"
    backend.rank = 1
    backend.pynccl_comm = type(
        "FakeCommunicator",
        (),
        {
            "broadcast": lambda self, tensor, src: backend.updated.append(("broadcast", tensor)),
            "group": type("Group", (), {"barrier": lambda self: None})(),
        },
    )()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=0.1, backend=backend)

    try:
        client.sync_lora_parameters(_TinyPeftModel())
    except RuntimeError as exc:
        assert "Request failed: 400" in str(exc)
    else:
        raise AssertionError("expected LoRA preflight rejection to fail")

    assert backend.updated == []
    assert backend.session.gets == []


def test_vllm_generation_client_sync_lora_parameters_requires_update_id() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    backend.session = _FakeSession(update_payload={})
    backend.base_url = "http://127.0.0.1:8000"
    backend.rank = 1
    backend.pynccl_comm = type(
        "FakeCommunicator",
        (),
        {
            "broadcast": lambda self, tensor, src: backend.updated.append(("broadcast", tensor)),
            "group": type("Group", (), {"barrier": lambda self: None})(),
        },
    )()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=0.1, backend=backend)

    try:
        client.sync_lora_parameters(_TinyPeftModel())
    except RuntimeError as exc:
        assert "missing update_id" in str(exc)
    else:
        raise AssertionError("expected missing LoRA update id to fail")

    assert backend.updated == []
    assert backend.session.gets == []


def test_vllm_generation_client_validate_lora_server_rejects_identity_mismatch() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    backend.session = _FakeSession(
        health_payload={
            "status": "ok",
            "sync_mode": "lora",
            "lora_name": "other",
            "lora_int_id": 1,
            "model": "model/Qwen2.5-7B-Instruct",
            "lora_adapter_path": "outputs/adapter",
            "supports_lora_param_update": True,
        }
    )
    backend.base_url = "http://127.0.0.1:8000"
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    args = Namespace(
        model_path="model/Qwen2.5-7B-Instruct",
        vllm_lora_name="macorag_train",
        vllm_lora_int_id=1,
        vllm_lora_adapter_path="outputs/adapter",
    )

    try:
        client.validate_lora_server(args)
    except SystemExit as exc:
        assert "LoRA server identity mismatch" in str(exc)
        assert "lora_name" in str(exc)
    else:
        raise AssertionError("expected LoRA server identity validation to fail")


def test_vllm_generation_client_validate_lora_server_rejects_unsupported_update_endpoint() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    backend.session = _FakeSession(
        health_payload={
            "status": "ok",
            "sync_mode": "lora",
            "lora_name": "macorag_train",
            "lora_int_id": 1,
            "model": "model/Qwen2.5-7B-Instruct",
            "lora_adapter_path": "outputs/adapter",
            "supports_lora_param_update": False,
        },
    )
    backend.base_url = "http://127.0.0.1:8000"
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    args = Namespace(
        model_path="model/Qwen2.5-7B-Instruct",
        vllm_lora_name="macorag_train",
        vllm_lora_int_id=1,
        vllm_lora_adapter_path="outputs/adapter",
    )

    try:
        client.validate_lora_server(args)
    except SystemExit as exc:
        assert "LoRA hot sync is unsupported" in str(exc)
    else:
        raise AssertionError("expected unsupported LoRA update validation to fail")


def test_vllm_generation_client_validate_lora_server_accepts_health_capability_without_probe_post() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    backend.session = _FakeSession(
        health_payload={
            "status": "ok",
            "sync_mode": "lora",
            "lora_name": "macorag_train",
            "lora_int_id": 1,
            "model": "model/Qwen2.5-7B-Instruct",
            "lora_adapter_path": "outputs/adapter",
            "supports_lora_param_update": True,
        }
    )
    backend.base_url = "http://127.0.0.1:8000"
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    args = Namespace(
        model_path="model/Qwen2.5-7B-Instruct",
        vllm_lora_name="macorag_train",
        vllm_lora_int_id=1,
        vllm_lora_adapter_path="outputs/adapter",
    )

    client.validate_lora_server(args)

    assert backend.session.posts == []


def test_build_policy_keeps_dense_mode_on_existing_health_check(monkeypatch) -> None:
    import rl_training.train_grpo_macorag as train_grpo

    calls: list[str] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            calls.append("init")

        def check_server(self) -> None:
            calls.append("check_server")

        def validate_lora_server(self, args) -> None:
            calls.append("validate_lora_server")

    monkeypatch.setattr(train_grpo, "_validate_local_vllm_server_model", lambda args: calls.append("model_check"))
    monkeypatch.setattr(train_grpo, "VLLMGenerationClient", FakeClient)
    args = Namespace(
        use_vllm_generation=True,
        vllm_sync_mode="dense",
        vllm_host="127.0.0.1",
        vllm_port=8000,
        vllm_timeout_seconds=5,
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=2,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
    )

    policy = _build_policy(args, _LogprobModel(), _FakeTokenizer())

    assert policy.vllm_client.__class__ is FakeClient
    assert calls == ["model_check", "init", "check_server"]


def test_build_policy_validates_lora_server_before_generic_health(monkeypatch) -> None:
    import rl_training.train_grpo_macorag as train_grpo

    calls: list[str] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            calls.append("init")

        def check_server(self) -> None:
            calls.append("check_server")

        def validate_lora_server(self, args) -> None:
            calls.append("validate_lora_server")
            raise SystemExit("LoRA hot sync is unsupported")

    monkeypatch.setattr(train_grpo, "_validate_local_vllm_server_model", lambda args: calls.append("model_check"))
    monkeypatch.setattr(train_grpo, "VLLMGenerationClient", FakeClient)
    args = Namespace(
        use_vllm_generation=True,
        vllm_sync_mode="lora",
        vllm_host="127.0.0.1",
        vllm_port=8000,
        vllm_timeout_seconds=5,
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=2,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
    )

    try:
        _build_policy(args, _LogprobModel(), _FakeTokenizer())
    except SystemExit as exc:
        assert "LoRA hot sync is unsupported" in str(exc)
    else:
        raise AssertionError("expected LoRA server validation to fail")

    assert calls == ["model_check", "init", "validate_lora_server"]


class _FakePolicyWithLoraClient:
    def __init__(self) -> None:
        self.vllm_client = type(
            "Client",
            (),
            {
                "dense_called": False,
                "lora_called": False,
                "sync_trainable_parameters": lambda self_client, model: setattr(self_client, "dense_called", True)
                or 1.0,
                "sync_lora_parameters": lambda self_client, model: setattr(self_client, "lora_called", True)
                or 2.0,
            },
        )()


def test_sync_vllm_after_optimizer_step_uses_lora_mode() -> None:
    from rl_training.train_grpo_macorag import _sync_vllm_after_optimizer_step

    policy = _FakePolicyWithLoraClient()
    args = Namespace(use_vllm_generation=True, vllm_sync_after_step=True, vllm_sync_mode="lora", vllm_sync_every_steps=1)

    elapsed = _sync_vllm_after_optimizer_step(policy, object(), args)

    assert elapsed == 2.0
    assert policy.vllm_client.lora_called is True
    assert policy.vllm_client.dense_called is False


def test_sync_vllm_after_optimizer_step_respects_sync_interval() -> None:
    from rl_training.train_grpo_macorag import _sync_vllm_after_optimizer_step

    policy = _FakePolicyWithLoraClient()
    args = Namespace(use_vllm_generation=True, vllm_sync_after_step=True, vllm_sync_mode="dense", vllm_sync_every_steps=4)

    elapsed = _sync_vllm_after_optimizer_step(policy, object(), args, completed_step=3)

    assert elapsed == 0.0
    assert policy.vllm_client.dense_called is False
    assert policy.vllm_client.lora_called is False


def test_sync_vllm_before_first_rollout_ignores_post_step_cadence() -> None:
    import rl_training.train_grpo_macorag as train_module

    policy = _FakePolicyWithLoraClient()
    args = Namespace(
        use_vllm_generation=True,
        vllm_sync_after_step=False,
        vllm_sync_mode="lora",
        vllm_sync_every_steps=100,
    )

    elapsed = train_module._sync_vllm_before_first_rollout(policy, object(), args)

    assert elapsed == 2.0
    assert policy.vllm_client.lora_called is True
    assert policy.vllm_client.dense_called is False


def test_sync_vllm_before_first_rollout_skips_disabled_or_non_main(monkeypatch) -> None:
    import rl_training.train_grpo_macorag as train_module

    policy = _FakePolicyWithLoraClient()
    args = Namespace(use_vllm_generation=False, vllm_sync_mode="lora")
    assert train_module._sync_vllm_before_first_rollout(policy, object(), args) == 0.0

    args.use_vllm_generation = True
    monkeypatch.setattr(train_module, "_is_main_process", lambda: False)
    assert train_module._sync_vllm_before_first_rollout(policy, object(), args) == 0.0
    assert policy.vllm_client.lora_called is False


def test_vllm_generation_client_dequantizes_4bit_weights_before_sync(monkeypatch) -> None:
    import rl_training.vllm_client as vllm_client
    from rl_training.vllm_client import _move_tensor_for_sync

    parameter = Params4bit()

    called = {}

    def fake_dequantize(weight, state=None):
        called["weight"] = weight
        called["state"] = state
        return parameter.dense

    monkeypatch.setattr(vllm_client, "_dequantize_bnb_weight", fake_dequantize)
    tensor = _move_tensor_for_sync(parameter, device=None)

    assert called == {"weight": parameter, "state": parameter.quant_state}
    assert tensor.shape == (2, 3)
    assert tensor.device.type == "cpu"


def test_vllm_generation_client_merges_peft_adapter_before_sync() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient(sync_device=torch.device("meta"))
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    model = _TinyPeftModel()

    client.sync_trainable_parameters(model)

    assert model.merged is True
    assert model.unmerged is True
    assert [name for name, _ in backend.updated] == ["model.layers.0.self_attn.q_proj.weight"]
    assert all(tensor.device.type == "meta" for _, tensor in backend.updated)


class _FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool):
        assert add_generation_prompt is True
        assert tokenize is True
        joined = "\n".join(item["content"] for item in messages)
        return [min(98, ord(char) % 100) for char in joined][-32:]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "decoded response"


class _LogprobModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
        vocab_size = 128
        logits = self.weight * torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab_size, device=input_ids.device)
        return type("Output", (), {"logits": logits})


class _FakeVLLMClient:
    def __init__(self) -> None:
        self.prompts: list[list[int]] = []

    def generate(self, prompt_token_ids, *, max_tokens, temperature, top_p, top_k):
        assert isinstance(prompt_token_ids, str)
        self.prompts.append([ord(char) for char in prompt_token_ids[:4]])
        return [10, 11], "decoded response"


def test_vllm_shared_policy_generates_and_records_trace() -> None:
    from rl_training.policy import VLLMSharedPolicy

    client = _FakeVLLMClient()
    model = _LogprobModel()
    tokenizer = _FakeTokenizer()
    policy = VLLMSharedPolicy(
        model=model,
        tokenizer=tokenizer,
        vllm_client=client,
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=2,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
    )

    response = policy.generate(
        role=AgentRole.QUERY_RETRIEVER,
        question="Who?",
        state=RAGState(question="Who?"),
    )
    policy.generate(
        role=AgentRole.QUERY_RETRIEVER,
        question="Who?",
        state=RAGState(question="Who?"),
    )

    assert response == "decoded response"
    assert len(client.prompts) == 2
    assert len(policy.trace.actions) == 2
    action = policy.trace.actions[0]
    assert action.role == AgentRole.QUERY_RETRIEVER
    assert [item.round_index for item in policy.trace.actions] == [0, 1]
    assert action.local_reward == 0.0
    assert action.terminal_reward == 0.0
    assert action.advantage == 0.0
    assert action.completion_ids == [10, 11]
    assert action.response == "decoded response"
    assert action.old_logprobs.shape == (2,)
    assert policy.timing["time_vllm_generate_seconds"] >= 0.0


def test_vllm_shared_policy_batches_requests_and_uses_server_logprobs(monkeypatch) -> None:
    from rl_training.policy import PolicyGenerationRequest, RolloutTrace, VLLMSharedPolicy
    from rl_training.vllm_client import VLLMGenerationOutput

    class FakeBatchClient:
        def __init__(self) -> None:
            self.prompt_batches: list[list[str]] = []

        def generate_batch(self, prompts, **kwargs):
            self.prompt_batches.append(prompts)
            return [
                VLLMGenerationOutput(completion_ids=[10, 11], logprobs=[-0.1, -0.2]),
                VLLMGenerationOutput(completion_ids=[20], logprobs=[-0.3]),
            ]

    monkeypatch.setattr(
        "rl_training.policy.sequence_logprobs",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("HF rescoring should not run")),
    )
    client = FakeBatchClient()
    policy = VLLMSharedPolicy(
        model=_LogprobModel(),
        tokenizer=_FakeTokenizer(),
        vllm_client=client,
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=2,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
    )
    traces = [RolloutTrace(), RolloutTrace()]
    requests = [
        PolicyGenerationRequest(
            role=AgentRole.QUERY_RETRIEVER,
            question="first",
            state=RAGState(question="first"),
        ),
        PolicyGenerationRequest(
            role=AgentRole.QUERY_RETRIEVER,
            question="second",
            state=RAGState(question="second"),
        ),
    ]

    responses = policy.generate_batch(requests, traces=traces)

    assert responses == ["decoded response", "decoded response"]
    assert len(client.prompt_batches) == 1
    assert [trace.actions[0].completion_ids for trace in traces] == [[10, 11], [20]]
    assert torch.equal(traces[0].actions[0].old_logprobs, torch.tensor([-0.1, -0.2]))
    assert torch.equal(traces[1].actions[0].old_logprobs, torch.tensor([-0.3]))
    assert [trace.actions[0].round_index for trace in traces] == [0, 0]
    assert policy.timing["time_behavior_rescore_seconds"] == 0.0


def test_vllm_shared_policy_falls_back_to_hf_rescore_without_server_logprobs(monkeypatch) -> None:
    from rl_training.policy import PolicyGenerationRequest, RolloutTrace, VLLMSharedPolicy
    from rl_training.vllm_client import VLLMGenerationOutput

    class FakeBatchClient:
        def generate_batch(self, prompts, **kwargs):
            return [VLLMGenerationOutput(completion_ids=[10, 11], logprobs=None)]

    rescored: list[list[int]] = []

    def fake_sequence_logprobs(**kwargs):
        rescored.append(kwargs["completion_ids"])
        return torch.tensor([-1.0, -2.0])

    monkeypatch.setattr("rl_training.policy.sequence_logprobs", fake_sequence_logprobs)
    policy = VLLMSharedPolicy(
        model=_LogprobModel(),
        tokenizer=_FakeTokenizer(),
        vllm_client=FakeBatchClient(),
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=2,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
    )
    trace = RolloutTrace()

    policy.generate_batch(
        [
            PolicyGenerationRequest(
                role=AgentRole.QUERY_RETRIEVER,
                question="question",
                state=RAGState(question="question"),
            )
        ],
        traces=[trace],
    )

    assert rescored == [[10, 11]]
    assert torch.equal(trace.actions[0].old_logprobs, torch.tensor([-1.0, -2.0]))
    assert policy.timing["time_behavior_rescore_seconds"] >= 0.0


def test_batched_rollout_one_round_batches_each_role_and_retrieval() -> None:
    from rl_training.batched_rollout import run_batched_rollouts
    from rl_training.policy import GeneratedAction

    class FakeBatchPolicy:
        def __init__(self) -> None:
            self.role_batches: list[list[AgentRole]] = []

        def generate_batch(self, requests, *, traces):
            self.role_batches.append([request.role for request in requests])
            responses = []
            for request, trace in zip(requests, traces):
                if request.role == AgentRole.QUERY_RETRIEVER:
                    response = (
                        '<query-retriever>{"sub_goal":"find director",'
                        '"query":"The Tripper director"}</query-retriever>'
                    )
                elif request.role == AgentRole.EVIDENCE_UPDATER:
                    response = (
                        '<update-evidence>{"selected_passage_ids":[0],'
                        '"rationale":"supports"}</update-evidence>'
                    )
                else:
                    response = '<answer>{"can_answer":true,"answer":"David Arquette"}</answer>'
                trace.actions.append(
                    GeneratedAction(
                        role=request.role,
                        prompt="prompt",
                        response=response,
                        prompt_ids=[1],
                        completion_ids=[2],
                        old_logprobs=torch.zeros(1),
                        round_index=sum(1 for item in trace.actions if item.role == request.role),
                    )
                )
                responses.append(response)
            return responses

    class FakeRetrievalEnv:
        def __init__(self) -> None:
            self.query_batches: list[list[str]] = []

        def query_batch(self, dataset: str, queries: list[str]):
            assert dataset == "hotpotqa"
            self.query_batches.append(queries)
            return [
                {
                    "query": query,
                    "passages": [
                        {
                            "passage_id": 0,
                            "text": "The Tripper was directed by David Arquette.",
                            "score": 1.0,
                        }
                    ],
                }
                for query in queries
            ]

    policy = FakeBatchPolicy()
    retrieval_env = FakeRetrievalEnv()

    rollouts = run_batched_rollouts(
        question="Who directed The Tripper?",
        dataset="hotpotqa",
        group_size=4,
        max_rounds=2,
        policy=policy,
        retrieval_env=retrieval_env,
    )

    assert policy.role_batches == [
        [AgentRole.QUERY_RETRIEVER] * 4,
        [AgentRole.EVIDENCE_UPDATER] * 4,
        [AgentRole.ANSWER_GENERATOR] * 4,
    ]
    assert retrieval_env.query_batches == [["The Tripper director"] * 4]
    assert [item.result.final_answer for item in rollouts] == ["David Arquette"] * 4
    assert [len(item.trace.actions) for item in rollouts] == [3, 3, 3, 3]
    assert [item.result.trajectory[0]["generated_roles"] for item in rollouts] == [
        ["query_retriever", "evidence_updater", "answer_generator"]
    ] * 4


def test_batched_rollout_masks_finished_and_parse_failed_candidates() -> None:
    from rl_training.batched_rollout import run_batched_rollouts
    from rl_training.policy import GeneratedAction

    class DivergentPolicy:
        def __init__(self) -> None:
            self.trace_ids: dict[int, int] = {}
            self.batch_sizes: list[tuple[AgentRole, int]] = []

        def generate_batch(self, requests, *, traces):
            self.batch_sizes.append((requests[0].role, len(requests)))
            responses = []
            for trace_index, (request, trace) in enumerate(zip(requests, traces)):
                candidate_id = self.trace_ids.setdefault(id(trace), len(self.trace_ids))
                role_round = sum(1 for item in trace.actions if item.role == request.role)
                if request.role == AgentRole.QUERY_RETRIEVER:
                    response = (
                        "invalid query"
                        if candidate_id == 2 and role_round == 0
                        else '<query-retriever>{"sub_goal":"find",'
                        f'"query":"query-{candidate_id}-{role_round}"}}</query-retriever>'
                    )
                elif request.role == AgentRole.EVIDENCE_UPDATER:
                    response = (
                        "invalid evidence"
                        if candidate_id == 3 and role_round == 0
                        else '<update-evidence>{"selected_passage_ids":[0]}</update-evidence>'
                    )
                else:
                    can_answer = candidate_id != 1 or role_round == 1
                    encoded_can_answer = (
                        f'"{str(can_answer).lower()}"'
                        if candidate_id == 1
                        else str(can_answer).lower()
                    )
                    response = (
                        '<answer>{"can_answer":'
                        f'{encoded_can_answer},"answer":"answer-{candidate_id}"}}</answer>'
                    )
                trace.actions.append(
                    GeneratedAction(
                        role=request.role,
                        prompt="prompt",
                        response=response,
                        prompt_ids=[1],
                        completion_ids=[2],
                        old_logprobs=torch.zeros(1),
                        round_index=role_round,
                    )
                )
                responses.append(response)
            return responses

    class BatchRetrieval:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def query_batch(self, dataset, queries):
            self.batches.append(queries)
            return [
                {"query": query, "passages": [{"passage_id": 0, "text": query, "score": 1.0}]}
                for query in queries
            ]

    policy = DivergentPolicy()
    retrieval = BatchRetrieval()

    rollouts = run_batched_rollouts(
        question="question",
        dataset="hotpotqa",
        group_size=4,
        max_rounds=2,
        policy=policy,
        retrieval_env=retrieval,
    )

    assert policy.batch_sizes == [
        (AgentRole.QUERY_RETRIEVER, 4),
        (AgentRole.EVIDENCE_UPDATER, 3),
        (AgentRole.ANSWER_GENERATOR, 2),
        (AgentRole.QUERY_RETRIEVER, 1),
        (AgentRole.EVIDENCE_UPDATER, 1),
        (AgentRole.ANSWER_GENERATOR, 1),
    ]
    assert retrieval.batches == [
        ["query-0-0", "query-1-0", "query-3-0"],
        ["query-1-1"],
    ]
    assert [item.result.final_answer for item in rollouts] == [
        "answer-0",
        "answer-1",
        None,
        None,
    ]
    assert rollouts[2].result.trajectory[0]["parse_error_role"] == "query_retriever"
    assert rollouts[2].result.trajectory[0]["generated_roles"] == ["query_retriever"]
    assert rollouts[3].result.trajectory[0]["parse_error_role"] == "evidence_updater"
    assert rollouts[3].result.trajectory[0]["generated_roles"] == [
        "query_retriever",
        "evidence_updater",
    ]
    assert len(rollouts[1].result.trajectory) == 2


def test_rollout_group_uses_one_batched_executor_call(monkeypatch) -> None:
    from rag import RAGLoopResult
    from rl_training.batched_rollout import BatchedRolloutResult
    from rl_training.data import RLSample
    from rl_training.policy import RolloutTrace
    from rl_training.train_grpo_macorag import _rollout_group

    calls: list[dict[str, object]] = []

    def fake_run_batched_rollouts(**kwargs):
        calls.append(kwargs)
        return [
            BatchedRolloutResult(
                result=RAGLoopResult(
                    question=kwargs["question"],
                    dataset=kwargs["dataset"],
                    trajectory=[],
                    state=RAGState(question=kwargs["question"]),
                    final_answer="gold",
                    parse_errors=[],
                ),
                trace=RolloutTrace(),
            )
            for _ in range(kwargs["group_size"])
        ]

    monkeypatch.setattr(
        "rl_training.train_grpo_macorag.run_batched_rollouts",
        fake_run_batched_rollouts,
    )

    class FakePolicy:
        def __init__(self) -> None:
            self.timing = {}

        def reset_trace(self) -> None:
            self.timing = {
                "time_vllm_generate_seconds": 1.5,
                "time_behavior_rescore_seconds": 0.0,
            }

        def generate_batch(self, requests, *, traces):
            raise AssertionError("fake batched executor owns generation")

    sample = RLSample(
        qid="q1",
        dataset="hotpotqa",
        question="question",
        answer="gold",
        answer_aliases=[],
        supporting_facts=[],
        context_doc_ids=[],
        metadata={},
    )
    args = Namespace(
        group_size=4,
        max_rounds=2,
        query_local_credit_weight=0.75,
        evidence_local_credit_weight=0.70,
        answer_local_credit_weight=0.30,
    )

    rollouts, timing = _rollout_group(
        args=args,
        sample=sample,
        policy=FakePolicy(),
        retrieval_env=object(),
    )

    assert len(calls) == 1
    assert calls[0]["group_size"] == 4
    assert [item["group_index"] for item in rollouts] == [0, 1, 2, 3]
    assert timing["time_vllm_generate_seconds"] == 1.5
    assert timing["time_behavior_rescore_seconds"] == 0.0


def test_build_policy_uses_hf_policy_when_vllm_disabled() -> None:
    args = Namespace(
        use_vllm_generation=False,
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=2,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
    )

    policy = _build_policy(args, _LogprobModel(), _FakeTokenizer())

    assert isinstance(policy, HFSharedPolicy)


def test_train_on_rollouts_reports_optimizer_step_flag() -> None:
    model = _LogprobModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = type(
        "Args",
        (),
        {"gradient_accumulation_steps": 1, "clip_epsilon": 0.2, "kl_beta": 0.0},
    )()
    action = type(
        "Action",
        (),
        {
            "prompt_ids": [1, 2],
            "completion_ids": [3],
            "old_logprobs": torch.zeros(1),
            "advantage": 1.0,
        },
    )()
    rollouts = [{"advantage": 1.0, "actions": [action]}]

    metrics = _train_on_rollouts(
        rollouts=rollouts,
        train_model=model,
        raw_policy_model=model,
        ref_model=model,
        optimizer=optimizer,
        args=args,
        torch=torch,
        device=torch.device("cpu"),
        should_step=True,
    )

    assert metrics["did_optimizer_step"] is True
    assert "time_optimizer_step_seconds" in metrics


def test_load_rl_samples_reads_existing_extracted_files(tmp_path: Path) -> None:
    data_root = tmp_path / "rl"
    _write_jsonl(
        data_root / "hotpotqa_rl.jsonl",
        [
            {
                "qid": "q1",
                "dataset": "hotpotqa",
                "question": "Who directed The Tripper?",
                "answer": "David Arquette",
                "answer_aliases": ["Arquette"],
                "supporting_facts": [
                    {
                        "doc_id": "d1",
                        "title": "The Tripper",
                        "text": "The Tripper was directed by David Arquette.",
                    }
                ],
            },
            {
                "qid": "q2",
                "dataset": "hotpotqa",
                "question": "Bad row has no answer",
                "supporting_facts": [],
            },
        ],
    )

    samples, summary = load_rl_samples(data_root=data_root, data_files=[], max_samples=1)

    assert len(samples) == 1
    assert samples[0].qid == "q1"
    assert samples[0].answer == "David Arquette"
    assert samples[0].answer_aliases == ["Arquette"]
    assert samples[0].supporting_facts[0]["title"] == "The Tripper"
    assert summary["loaded_samples"] == 1
    assert summary["skipped_samples"] == 1
    assert summary["counts_by_dataset"] == {"hotpotqa": 1}


def test_load_rl_samples_limits_max_samples_per_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "rl"
    _write_jsonl(
        data_root / "hotpotqa_train.jsonl",
        [
            {
                "qid": "h1",
                "dataset": "hotpotqa",
                "question": "Question h1?",
                "answer": "Answer h1",
                "supporting_facts": [{"title": "h", "text": "Answer h1"}],
            },
            {
                "qid": "h2",
                "dataset": "hotpotqa",
                "question": "Question h2?",
                "answer": "Answer h2",
                "supporting_facts": [{"title": "h", "text": "Answer h2"}],
            },
        ],
    )
    _write_jsonl(
        data_root / "musique_train.jsonl",
        [
            {
                "qid": "m1",
                "dataset": "musique",
                "question": "Question m1?",
                "answer": "Answer m1",
                "supporting_facts": [{"title": "m", "text": "Answer m1"}],
            },
            {
                "qid": "m2",
                "dataset": "musique",
                "question": "Question m2?",
                "answer": "Answer m2",
                "supporting_facts": [{"title": "m", "text": "Answer m2"}],
            },
        ],
    )

    samples, summary = load_rl_samples(data_root=data_root, data_files=[], max_samples=1)

    assert [sample.qid for sample in samples] == ["h1", "m1"]
    assert summary["loaded_samples"] == 2
    assert summary["counts_by_dataset"] == {"hotpotqa": 1, "musique": 1}
    assert summary["max_samples"] == 1


def test_load_rl_samples_default_files_ignore_corpus_jsonl(tmp_path: Path) -> None:
    data_root = tmp_path / "rl"
    _write_jsonl(
        data_root / "hotpotqa" / "hotpotqa_train.jsonl",
        [
            {
                "qid": "q1",
                "dataset": "hotpotqa",
                "question": "Who directed The Tripper?",
                "answer": "David Arquette",
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
            }
        ],
    )
    _write_jsonl(
        data_root / "hotpotqa" / "corpus.jsonl",
        [{"doc_id": "d1", "title": "The Tripper", "text": "Corpus rows are not RL samples."}],
    )

    samples, summary = load_rl_samples(data_root=data_root, data_files=[])

    assert len(samples) == 1
    assert summary["skipped_samples"] == 0
    assert summary["source_files"] == [str(data_root / "hotpotqa" / "hotpotqa_train.jsonl")]


def test_run_train_grpo_script_derives_gpu_visibility_from_yaml() -> None:
    script = Path("scripts/run_train_grpo.sh").read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES:-0,1" not in script
    assert "CONFIG_PATH=" in script
    assert "yaml.safe_load" in script
    assert "NPROC_PER_NODE" in script
    assert 'export CUDA_VISIBLE_DEVICES="${YAML_GPU_INDICES}"' in script
    assert 'export MACORAG_SILENT_RETRIEVAL="${MACORAG_SILENT_RETRIEVAL:-1}"' in script
    assert "--nproc_per_node=${NPROC_PER_NODE}" in script


def test_run_grpo_vllm_server_script_uses_vllm_gpu_and_trl_server() -> None:
    script = Path("scripts/run_grpo_vllm_server.sh").read_text(encoding="utf-8")

    assert "CONFIG_PATH=" in script
    assert "vllm_sync_mode" in script
    assert "vllm_gpu_indices" in script
    assert 'export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"' in script
    assert "trl vllm-serve" in script
    assert "rl_training.vllm_lora_server" in script
    assert "vllm_lora_name" in script
    assert "vllm_lora_int_id" in script
    assert "vllm_lora_adapter_path" in script
    assert "--data-parallel-size" in script
    assert "--model" in script
    assert "--host" in script
    assert "--port" in script
    assert "--tensor-parallel-size" in script
    assert "--gpu-memory-utilization" in script


def test_run_grpo_vllm_lora_server_script_removed_after_merging_into_main_launcher() -> None:
    assert not Path("scripts/run_grpo_vllm_lora_server.sh").exists()


def test_compute_answer_f1_uses_normalized_token_overlap_and_aliases() -> None:
    assert compute_answer_f1("the david  arquette!", "David Arquette", []) == 1.0
    assert compute_answer_f1("Arquette", "David Arquette", ["Arquette"]) == 1.0
    assert compute_answer_f1("David", "David Arquette", []) == 2 / 3
    assert compute_answer_f1("", "David Arquette", []) == 0.0


def test_compute_action_rewards_assigns_distinct_role_round_credit() -> None:
    support_d1 = {
        "passage_id": 0,
        "doc_id": "d1",
        "title": "Bullitt",
        "text": "Bullitt was directed by Peter Yates.",
    }
    support_d2 = {
        "passage_id": 1,
        "doc_id": "d2",
        "title": "Peter Yates",
        "text": "Peter Yates was born in Aldershot, Hampshire.",
    }
    rollout = {
        "trajectory": [
            {
                "round": 0,
                "query_retriever": {"sub_goal": "find director", "query": "Bullitt film director"},
                "observation": {"passages": [support_d1, support_d2]},
                "update_evidence": {"selected_passage_ids": [0]},
                "answer": {"can_answer": False, "answer": None},
            },
            {
                "round": 1,
                "query_retriever": {"sub_goal": "repeat director", "query": "Bullitt film director"},
                "observation": {"passages": [support_d1]},
                "update_evidence": {"selected_passage_ids": [0]},
                "answer": {"can_answer": True, "answer": "Aldershot"},
            },
        ],
        "parse_errors": [],
        "final_answer": "Aldershot",
    }
    sample = {
        "answer": "Aldershot",
        "answer_aliases": [],
        "supporting_facts": [support_d1, support_d2],
    }

    rewards = reward_module.compute_action_rewards(rollout=rollout, sample=sample)
    by_key = {
        (item["role"], item["round_index"]): item["local_reward"]
        for item in rewards["action_rewards"]
    }

    assert set(by_key) == {
        ("query_retriever", 0),
        ("evidence_updater", 0),
        ("answer_generator", 0),
        ("query_retriever", 1),
        ("evidence_updater", 1),
        ("answer_generator", 1),
    }
    assert by_key[("query_retriever", 0)] > by_key[("query_retriever", 1)]
    assert by_key[("evidence_updater", 0)] > by_key[("evidence_updater", 1)]
    assert by_key[("answer_generator", 0)] != by_key[("answer_generator", 1)]
    assert rewards["terminal_reward"] == 2.25


def test_compute_action_rewards_does_not_infer_sufficiency_without_support_labels() -> None:
    rollout = {
        "trajectory": [
            {
                "round": 0,
                "query_retriever": {"sub_goal": "find person", "query": "Ada Lovelace biography"},
                "observation": {"passages": []},
                "update_evidence": {"selected_passage_ids": []},
                "answer": {"can_answer": False, "answer": None},
            }
        ],
        "parse_errors": [],
        "final_answer": None,
    }
    sample = {"answer": "Ada Lovelace", "answer_aliases": [], "supporting_facts": []}

    rewards = reward_module.compute_action_rewards(rollout=rollout, sample=sample)
    answer_reward = next(
        item["local_reward"]
        for item in rewards["action_rewards"]
        if item["role"] == "answer_generator"
    )

    assert answer_reward == 0.0


def test_compute_action_rewards_assigns_parse_failure_to_failed_role() -> None:
    support = {
        "passage_id": 0,
        "doc_id": "d1",
        "title": "Bullitt",
        "text": "Bullitt was directed by Peter Yates.",
    }
    rollout = {
        "trajectory": [
            {
                "round": 0,
                "generated_roles": ["query_retriever", "evidence_updater"],
                "parse_error_role": "evidence_updater",
                "query_retriever": {"sub_goal": "find director", "query": "Bullitt film director"},
                "observation": {"passages": [support]},
                "update_evidence": {},
                "answer": {},
            }
        ],
        "parse_errors": ["Missing required tag: update-evidence"],
        "final_answer": None,
    }
    sample = {"answer": "Peter Yates", "answer_aliases": [], "supporting_facts": [support]}

    rewards = reward_module.compute_action_rewards(rollout=rollout, sample=sample)
    by_role = {item["role"]: item["local_reward"] for item in rewards["action_rewards"]}

    assert set(by_role) == {"query_retriever", "evidence_updater"}
    assert by_role["query_retriever"] > 0.0
    assert by_role["evidence_updater"] < -1.0


def test_compute_action_rewards_ignores_answer_text_when_can_answer_is_false() -> None:
    rollout = {
        "trajectory": [
            {
                "round": 0,
                "query_retriever": {"sub_goal": "find person", "query": "Ada Lovelace biography"},
                "observation": {"passages": []},
                "update_evidence": {"selected_passage_ids": []},
                "answer": {"can_answer": False, "answer": "Ada Lovelace"},
            }
        ],
        "parse_errors": [],
        "final_answer": None,
    }
    sample = {"answer": "Ada Lovelace", "answer_aliases": [], "supporting_facts": []}

    rewards = reward_module.compute_action_rewards(rollout=rollout, sample=sample)
    aggregate_rewards = compute_rl_rewards(rollout=rollout, sample=sample)

    assert rewards["terminal_reward"] == 0.0
    assert aggregate_rewards["answer_f1"] == 0.0


def test_compute_action_rewards_requires_all_labeled_support_facts() -> None:
    supports = [
        {
            "passage_id": index,
            "doc_id": f"d{index}",
            "title": f"Doc {index}",
            "text": f"Supporting fact {index}.",
        }
        for index in range(3)
    ]
    rollout = {
        "trajectory": [
            {
                "round": 0,
                "query_retriever": {"sub_goal": "collect facts", "query": "collect supporting facts"},
                "observation": {"passages": supports},
                "update_evidence": {"selected_passage_ids": [0, 1]},
                "answer": {"can_answer": False, "answer": None},
            }
        ],
        "parse_errors": [],
        "final_answer": None,
    }
    sample = {"answer": "result", "answer_aliases": [], "supporting_facts": supports}

    rewards = reward_module.compute_action_rewards(rollout=rollout, sample=sample)
    answer_reward = next(
        item for item in rewards["action_rewards"] if item["role"] == "answer_generator"
    )

    assert answer_reward["components"]["correct_wait"] == 0.25
    assert answer_reward["components"]["false_abstention"] == 0.0
    assert rewards["terminal_reward"] == 2 / 3


def test_compute_rl_rewards_scores_query_evidence_and_final_answer() -> None:
    rollout = {
        "trajectory": [
            {
                "query_retriever": {
                    "sub_goal": "find director",
                    "query": "The Tripper director",
                },
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "doc_id": "d1",
                            "title": "The Tripper",
                            "text": "The Tripper was directed by David Arquette.",
                        },
                        {"passage_id": 1, "doc_id": "d2", "title": "Noise", "text": "Irrelevant."},
                    ]
                },
                "update_evidence": {"selected_passage_ids": [0]},
                "answer": {"can_answer": True, "answer": "David Arquette"},
            }
        ],
        "parse_errors": [],
        "final_answer": "David Arquette",
    }
    sample = {
        "answer": "David Arquette",
        "answer_aliases": [],
        "supporting_facts": [
            {
                "doc_id": "d1",
                "title": "The Tripper",
                "text": "The Tripper was directed by David Arquette.",
            }
        ],
    }

    rewards = compute_rl_rewards(rollout=rollout, sample=sample)

    assert rewards["query_reward"] > 0.0
    assert rewards["evidence_reward"] > 0.0
    assert rewards["answer_f1"] == 1.0
    assert rewards["answer_reward"] == 2.0
    assert rewards["support_coverage"] == 1.0
    assert rewards["retrieval_hit_reward"] == 0.5
    assert rewards["total"] > 3.0


def test_compute_rl_rewards_caps_duplicate_support_evidence_and_penalizes_repeated_queries() -> None:
    rollout = {
        "trajectory": [
            {
                "query_retriever": {"sub_goal": "find director", "query": "Bullitt film director"},
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "doc_id": "d1",
                            "title": "Bullitt",
                            "text": "Bullitt was directed by Peter Yates.",
                        }
                    ]
                },
                "update_evidence": {"selected_passage_ids": [0]},
                "answer": {"can_answer": False, "answer": None},
            },
            {
                "query_retriever": {"sub_goal": "find director again", "query": "Bullitt film director"},
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "doc_id": "d1",
                            "title": "Bullitt",
                            "text": "Bullitt was directed by Peter Yates.",
                        }
                    ]
                },
                "update_evidence": {"selected_passage_ids": [0]},
                "answer": {"can_answer": True, "answer": "Peter Yates"},
            },
        ],
        "parse_errors": [],
        "final_answer": "Peter Yates",
    }
    sample = {
        "answer": "Peter Yates",
        "answer_aliases": [],
        "supporting_facts": [
            {"doc_id": "d1", "title": "Bullitt", "text": "Bullitt was directed by Peter Yates."}
        ],
    }

    rewards = compute_rl_rewards(rollout=rollout, sample=sample)

    assert rewards["support_facts_covered"] == 1.0
    assert rewards["support_coverage"] == 1.0
    assert rewards["evidence_reward"] == 1.0
    assert rewards["repeated_query_penalty"] == -0.2
    assert rewards["retrieval_cost"] == -0.2


def test_compute_rl_rewards_penalizes_wrong_premature_multihop_answer() -> None:
    rollout = {
        "trajectory": [
            {
                "query_retriever": {"sub_goal": "find director", "query": "Bullitt film director"},
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "doc_id": "d1",
                            "title": "Bullitt",
                            "text": "Bullitt was directed by Peter Yates.",
                        }
                    ]
                },
                "update_evidence": {"selected_passage_ids": [0]},
                "answer": {"can_answer": True, "answer": "London"},
            }
        ],
        "parse_errors": [],
        "final_answer": "London",
    }
    sample = {
        "answer": "Aldershot",
        "answer_aliases": [],
        "supporting_facts": [
            {"doc_id": "d1", "title": "Bullitt", "text": "Bullitt was directed by Peter Yates."},
            {
                "doc_id": "d2",
                "title": "Peter Yates",
                "text": "Peter Yates was born in Aldershot, Hampshire.",
            },
        ],
    }

    rewards = compute_rl_rewards(rollout=rollout, sample=sample)

    assert rewards["answer_f1"] == 0.0
    assert rewards["support_facts_required"] == 2.0
    assert rewards["support_facts_covered"] == 1.0
    assert rewards["support_coverage"] == 0.5
    assert rewards["premature_answer_penalty"] == -1.0


def test_compute_rl_rewards_discounts_correct_answer_with_incomplete_support() -> None:
    rollout = {
        "trajectory": [
            {
                "query_retriever": {"sub_goal": "find director", "query": "Bullitt film director"},
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "doc_id": "d1",
                            "title": "Bullitt",
                            "text": "Bullitt was directed by Peter Yates.",
                        }
                    ]
                },
                "update_evidence": {"selected_passage_ids": [0]},
                "answer": {"can_answer": True, "answer": "Aldershot"},
            }
        ],
        "parse_errors": [],
        "final_answer": "Aldershot",
    }
    sample = {
        "answer": "Aldershot",
        "answer_aliases": [],
        "supporting_facts": [
            {"doc_id": "d1", "title": "Bullitt", "text": "Bullitt was directed by Peter Yates."},
            {
                "doc_id": "d2",
                "title": "Peter Yates",
                "text": "Peter Yates was born in Aldershot, Hampshire.",
            },
        ],
    }

    rewards = compute_rl_rewards(rollout=rollout, sample=sample)

    assert rewards["answer_f1"] == 1.0
    assert rewards["support_coverage"] == 0.5
    assert rewards["premature_answer_penalty"] == -0.25
    assert rewards["total"] < 3.5


def test_compute_rl_rewards_does_not_penalize_correct_or_sufficient_multihop_answer() -> None:
    sample = {
        "answer": "Aldershot",
        "answer_aliases": [],
        "supporting_facts": [
            {"doc_id": "d1", "title": "Bullitt", "text": "Bullitt was directed by Peter Yates."},
            {
                "doc_id": "d2",
                "title": "Peter Yates",
                "text": "Peter Yates was born in Aldershot, Hampshire.",
            },
        ],
    }
    rollout = {
        "trajectory": [
            {
                "query_retriever": {"sub_goal": "find director", "query": "Bullitt film director"},
                "observation": {
                    "passages": [
                        {
                            "passage_id": 0,
                            "doc_id": "d1",
                            "title": "Bullitt",
                            "text": "Bullitt was directed by Peter Yates.",
                        },
                        {
                            "passage_id": 1,
                            "doc_id": "d2",
                            "title": "Peter Yates",
                            "text": "Peter Yates was born in Aldershot, Hampshire.",
                        },
                    ]
                },
                "update_evidence": {"selected_passage_ids": [0, 1]},
                "answer": {"can_answer": True, "answer": "Aldershot"},
            }
        ],
        "parse_errors": [],
        "final_answer": "Aldershot",
    }

    rewards = compute_rl_rewards(rollout=rollout, sample=sample)

    assert rewards["support_facts_required"] == 2.0
    assert rewards["support_facts_covered"] == 2.0
    assert rewards["premature_answer_penalty"] == 0.0


def test_compute_grpo_loss_uses_advantages_clipping_and_kl() -> None:
    current = torch.log(torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float32))
    old = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32))
    reference = torch.log(torch.tensor([[0.55, 0.45], [0.4, 0.6]], dtype=torch.float32))
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)

    loss, metrics = compute_grpo_loss(
        current_logprobs=current,
        old_logprobs=old,
        ref_logprobs=reference,
        action_mask=mask,
        advantages=advantages,
        clip_epsilon=0.2,
        kl_beta=0.1,
    )

    assert loss.requires_grad is False
    assert torch.isfinite(loss)
    assert metrics["policy_loss"] != 0.0
    assert metrics["kl"] >= 0.0
    assert metrics["loss"] == float(loss.item())


def test_assign_action_advantages_normalizes_by_role_and_round() -> None:
    class Action:
        def __init__(self, role: AgentRole, round_index: int) -> None:
            self.role = role
            self.round_index = round_index
            self.local_reward = 0.0
            self.terminal_reward = 0.0
            self.advantage = 0.0

    first_q0 = Action(AgentRole.QUERY_RETRIEVER, 0)
    first_q1 = Action(AgentRole.QUERY_RETRIEVER, 1)
    second_q0 = Action(AgentRole.QUERY_RETRIEVER, 0)
    rollouts = [
        {
            "actions": [first_q0, first_q1],
            "terminal_reward": 4.0,
            "action_rewards": [
                {"role": "query_retriever", "round_index": 0, "local_reward": 1.0},
                {"role": "query_retriever", "round_index": 1, "local_reward": 5.0},
            ],
        },
        {
            "actions": [second_q0],
            "terminal_reward": 2.0,
            "action_rewards": [
                {"role": "query_retriever", "round_index": 0, "local_reward": 3.0},
            ],
        },
    ]

    trainer_module.assign_action_advantages(
        rollouts,
        local_weights={"query_retriever": 0.75},
    )

    assert first_q0.local_reward == 1.0
    assert first_q0.terminal_reward == 4.0
    assert first_q0.advantage == -0.5
    assert second_q0.advantage == 0.5
    assert first_q1.advantage == 0.0


def test_policy_generate_disables_cache_for_gradient_checkpointing(monkeypatch) -> None:
    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is True
            return [1, 2, 3]

        def decode(self, token_ids, skip_special_tokens=True):
            return "<answer>{\"can_answer\":true,\"answer\":\"Ada\"}</answer>"

    class DummyModel:
        def __init__(self) -> None:
            self.generate_kwargs = None
            self.parameter = torch.nn.Parameter(torch.tensor(1.0))

        def parameters(self):
            return iter([self.parameter])

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return torch.tensor([[1, 2, 3, 4, 9]])

    def fake_logprobs(**kwargs):
        return torch.zeros(len(kwargs["completion_ids"]))

    model = DummyModel()
    monkeypatch.setattr("rl_training.policy.sequence_logprobs", fake_logprobs)
    policy = HFSharedPolicy(
        model=model,
        tokenizer=DummyTokenizer(),
        system_prompt="system",
        max_prompt_length=16,
        max_completion_length=8,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )

    policy.generate(
        role=AgentRole.ANSWER_GENERATOR,
        question="Who?",
        state=RAGState(question="Who?"),
    )

    assert model.generate_kwargs["use_cache"] is False


def test_sequence_logprobs_keeps_only_completion_logits() -> None:
    class DummyOutput:
        def __init__(self, logits):
            self.logits = logits

    class DummyModel:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            input_ids = kwargs["input_ids"]
            logits_to_keep = kwargs["logits_to_keep"]
            vocab_size = 16
            logits = torch.full((1, logits_to_keep, vocab_size), -20.0)
            labels = torch.nn.functional.pad(input_ids[:, 1:], (0, 1), value=-100)
            kept_labels = labels[:, -logits_to_keep:]
            for index, token_id in enumerate(kept_labels[0].tolist()):
                if token_id >= 0:
                    logits[0, index, token_id] = 20.0
            return DummyOutput(logits)

    model = DummyModel()
    prompt_ids = [1, 2, 3, 4, 5]
    completion_ids = [6, 7, 8]

    logprobs = sequence_logprobs(
        model=model,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        device=torch.device("cpu"),
    )

    assert model.kwargs["logits_to_keep"] == len(completion_ids) + 1
    assert logprobs.shape == (len(completion_ids),)
    assert torch.all(logprobs > -1e-4)


def test_batched_sequence_logprobs_matches_scalar_for_variable_lengths() -> None:
    class DummyOutput:
        def __init__(self, logits):
            self.logits = logits

    class DummyModel:
        def __init__(self) -> None:
            self.logits_to_keep = None

        def __call__(self, *, input_ids, attention_mask, **kwargs):
            del attention_mask
            vocab_size = 16
            logits = torch.full((*input_ids.shape, vocab_size), -20.0)
            labels = torch.nn.functional.pad(input_ids[:, 1:], (0, 1), value=0)
            logits.scatter_(2, labels.unsqueeze(-1), 20.0)
            self.logits_to_keep = kwargs.get("logits_to_keep")
            if self.logits_to_keep is not None:
                logits = logits[:, -self.logits_to_keep :, :]
            return DummyOutput(logits)

    model = DummyModel()
    prompt_batches = [[1, 2], [3, 4, 5]]
    completion_batches = [[6, 7], [8]]
    scalar = [
        sequence_logprobs(
            model=model,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            device=torch.device("cpu"),
        )
        for prompt_ids, completion_ids in zip(prompt_batches, completion_batches)
    ]

    batched, mask = policy_module.batched_sequence_logprobs(
        model=model,
        prompt_id_batches=prompt_batches,
        completion_id_batches=completion_batches,
        device=torch.device("cpu"),
        pad_token_id=0,
    )

    assert mask.tolist() == [[True, True], [True, False]]
    assert torch.allclose(batched[0, :2], scalar[0])
    assert torch.allclose(batched[1, :1], scalar[1])
    assert batched[1, 1].item() == 0.0
    assert model.logits_to_keep == max(map(len, completion_batches)) + 1


def test_train_on_rollouts_uses_configured_action_microbatches(monkeypatch) -> None:
    class DummyAction:
        prompt_ids = [1, 2]
        completion_ids = [3]
        old_logprobs = torch.zeros(1)
        advantage = 1.0

    class Args:
        per_device_train_batch_size = 2
        reference_per_device_batch_size = 4
        gradient_accumulation_steps = 1
        clip_epsilon = 0.2
        kl_beta = 0.02

    forward_batch_sizes: list[tuple[str, int]] = []
    backward_calls: list[float] = []

    def fake_batched_sequence_logprobs(*, model, completion_id_batches, **kwargs):
        del kwargs
        forward_batch_sizes.append((model, len(completion_id_batches)))
        value = 1.0 if model == "train" else 0.0
        logprobs = torch.full(
            (len(completion_id_batches), 1),
            value,
            dtype=torch.float32,
            requires_grad=model == "train",
        )
        return logprobs, torch.ones_like(logprobs, dtype=torch.bool)

    original_backward = torch.Tensor.backward

    def counting_backward(self, *args, **kwargs):
        backward_calls.append(float(self.detach().item()))
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(
        "rl_training.train_grpo_macorag.batched_sequence_logprobs",
        fake_batched_sequence_logprobs,
        raising=False,
    )
    monkeypatch.setattr(torch.Tensor, "backward", counting_backward)

    metrics = _train_on_rollouts(
        rollouts=[{"actions": [DummyAction() for _ in range(5)]}],
        train_model="train",
        raw_policy_model=object(),
        ref_model="ref",
        optimizer=object(),
        args=Args(),
        torch=torch,
        device=torch.device("cpu"),
        should_step=False,
    )

    assert forward_batch_sizes == [
        ("ref", 4),
        ("ref", 1),
        ("train", 2),
        ("train", 2),
        ("train", 1),
    ]
    assert len(backward_calls) == 3
    assert "time_policy_forward_seconds" in metrics
    assert "time_reference_forward_seconds" in metrics


def test_train_on_rollouts_skips_zero_advantage_without_model_work(monkeypatch) -> None:
    class DummyAction:
        prompt_ids = [1, 2]
        completion_ids = [3]
        old_logprobs = torch.zeros(1)
        advantage = 0.0

    class Args:
        per_device_train_batch_size = 1
        reference_per_device_batch_size = 4
        gradient_accumulation_steps = 1
        skip_zero_advantage_updates = True
        clip_epsilon = 0.2
        kl_beta = 0.02

    def unexpected_forward(**kwargs):
        raise AssertionError("zero-advantage samples must not run model forwards")

    class UnexpectedOptimizer:
        def step(self):
            raise AssertionError("zero-advantage samples must not step the optimizer")

        def zero_grad(self, **kwargs):
            raise AssertionError("zero-advantage samples must not clear gradients")

    monkeypatch.setattr(
        "rl_training.train_grpo_macorag.batched_sequence_logprobs",
        unexpected_forward,
    )

    metrics = _train_on_rollouts(
        rollouts=[{"actions": [DummyAction()]}],
        train_model=object(),
        raw_policy_model=object(),
        ref_model=object(),
        optimizer=UnexpectedOptimizer(),
        args=Args(),
        torch=torch,
        device=torch.device("cpu"),
        should_step=True,
    )

    assert metrics["skipped_update_reason"] == "zero_advantage"
    assert metrics["did_optimizer_step"] is False
    assert metrics["time_policy_forward_seconds"] == 0.0
    assert metrics["time_reference_forward_seconds"] == 0.0


def test_train_on_rollouts_backprops_each_action_to_release_graphs(monkeypatch) -> None:
    class DummyAction:
        def __init__(self, value: float) -> None:
            self.prompt_ids = [1, 2]
            self.completion_ids = [3]
            self.old_logprobs = torch.tensor([value], dtype=torch.float32)
            self.advantage = 1.0

    class DummyOptimizer:
        def __init__(self) -> None:
            self.steps = 0
            self.zero_grad_calls = 0

        def step(self) -> None:
            self.steps += 1

        def zero_grad(self, set_to_none: bool = False) -> None:
            assert set_to_none is True
            self.zero_grad_calls += 1

    class Args:
        gradient_accumulation_steps = 1
        clip_epsilon = 0.2
        kl_beta = 0.02

    backward_calls = []

    def fake_batched_sequence_logprobs(**kwargs):
        batch_size = len(kwargs["completion_id_batches"])
        values = torch.full((batch_size, 1), 3.0, requires_grad=True)
        return values, torch.ones_like(values, dtype=torch.bool)

    original_backward = torch.Tensor.backward

    def counting_backward(self, *args, **kwargs):
        backward_calls.append(float(self.detach().item()))
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(
        "rl_training.train_grpo_macorag.batched_sequence_logprobs",
        fake_batched_sequence_logprobs,
    )
    monkeypatch.setattr(torch.Tensor, "backward", counting_backward)

    metrics = _train_on_rollouts(
        rollouts=[
            {
                "advantage": 1.0,
                "actions": [DummyAction(3.0), DummyAction(4.0)],
            }
        ],
        train_model=object(),
        raw_policy_model=object(),
        ref_model=object(),
        optimizer=DummyOptimizer(),
        args=Args(),
        torch=torch,
        device=torch.device("cpu"),
        should_step=True,
    )

    assert len(backward_calls) == 2
    assert metrics["loss"] != 0.0
    assert "time_backward_seconds" in metrics
    assert "time_optimizer_step_seconds" in metrics
    assert "time_train_seconds" not in metrics


def test_train_on_rollouts_uses_each_actions_own_advantage(monkeypatch) -> None:
    class DummyAction:
        def __init__(self, advantage: float) -> None:
            self.prompt_ids = [1, 2]
            self.completion_ids = [3]
            self.old_logprobs = torch.zeros(1)
            self.advantage = advantage

    class Args:
        gradient_accumulation_steps = 1
        clip_epsilon = 0.2
        kl_beta = 0.02

    captured_advantages: list[float] = []

    def fake_batched_sequence_logprobs(**kwargs):
        batch_size = len(kwargs["completion_id_batches"])
        values = torch.zeros((batch_size, 1), requires_grad=True)
        return values, torch.ones_like(values, dtype=torch.bool)

    def fake_grpo_loss(*, current_logprobs, advantages, **kwargs):
        captured_advantages.append(float(advantages.item()))
        loss = current_logprobs.sum() * 0.0
        return loss, {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "clip_fraction": 0.0}

    monkeypatch.setattr(
        "rl_training.train_grpo_macorag.batched_sequence_logprobs",
        fake_batched_sequence_logprobs,
    )
    monkeypatch.setattr("rl_training.train_grpo_macorag.compute_grpo_loss", fake_grpo_loss)

    _train_on_rollouts(
        rollouts=[
            {
                "advantage": 9.0,
                "actions": [DummyAction(-0.75), DummyAction(0.5)],
            }
        ],
        train_model=object(),
        raw_policy_model=object(),
        ref_model=object(),
        optimizer=object(),
        args=Args(),
        torch=torch,
        device=torch.device("cpu"),
        should_step=False,
    )

    assert captured_advantages == [-0.75, 0.5]
