from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch

from rag import AgentRole, RAGState
from rl_training.config import parse_args
from rl_training.data import load_rl_samples
from rl_training.policy import HFSharedPolicy
from rl_training.policy import sequence_logprobs
from rl_training.rewards import compute_answer_f1, compute_rl_rewards
from rl_training.train_grpo_macorag import _parse_gpu_indices
from rl_training.train_grpo_macorag import _build_policy
from rl_training.train_grpo_macorag import _extract_vllm_server_model_paths
from rl_training.train_grpo_macorag import _validate_local_vllm_server_model
from rl_training.train_grpo_macorag import _train_on_rollouts
from rl_training.train_grpo_macorag import _validate_vllm_gpu_placement
from rl_training.train_grpo_macorag import _write_train_event
from rl_training.trainer import compute_grpo_loss
from rl_training.vllm_client import collect_trainable_named_parameters


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_parse_args_loads_train_grpo_yaml(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                'model_path: "model/base"',
                'sft_adapter_path: "outputs/sft/adapter"',
                'rl_data_root: "data/rl/train"',
                'retrieval_root: "data/trajectory_train_retrieval"',
                'output_dir: "outputs/grpo"',
                "max_samples: 8",
                "max_rounds: 2",
                "group_size: 4",
                "kl_beta: 0.03",
                "clip_epsilon: 0.15",
                "learning_rate: 0.00001",
                "per_device_train_batch_size: 1",
                "gradient_accumulation_steps: 2",
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
    assert args.output_dir == "outputs/grpo"
    assert args.max_samples == 3
    assert args.max_rounds == 2
    assert args.group_size == 4
    assert args.kl_beta == 0.03
    assert args.clip_epsilon == 0.15
    assert args.gradient_accumulation_steps == 2
    assert args.gpu_indices == "0,1"
    assert args.disable_tqdm is False


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

    assert response == "decoded response"
    assert len(client.prompts) == 1
    assert len(policy.trace.actions) == 1
    action = policy.trace.actions[0]
    assert action.role == AgentRole.QUERY_RETRIEVER
    assert action.completion_ids == [10, 11]
    assert action.response == "decoded response"
    assert action.old_logprobs.shape == (2,)
    assert policy.timing["time_vllm_generate_seconds"] >= 0.0


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
    assert "vllm_gpu_indices" in script
    assert 'export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"' in script
    assert "trl vllm-serve" in script
    assert "--model" in script
    assert "--host" in script
    assert "--port" in script
    assert "--tensor-parallel-size" in script
    assert "--gpu-memory-utilization" in script


def test_run_grpo_vllm_lora_server_script_uses_custom_server_and_lora_config() -> None:
    script = Path("scripts/run_grpo_vllm_lora_server.sh").read_text(encoding="utf-8")

    assert "rl_training.vllm_lora_server" in script
    assert "vllm_lora_name" in script
    assert "vllm_lora_int_id" in script
    assert "vllm_lora_adapter_path" in script
    assert 'export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"' in script
    assert "--data-parallel-size" in script


def test_write_train_event_records_sample_progress(tmp_path: Path) -> None:
    event_path = tmp_path / "train_events.jsonl"

    _write_train_event(
        event_path,
        event="sample_start",
        epoch=1,
        sample_index=4,
        sample_total=12,
        sample_qid="q5",
        sample_dataset="2wiki",
        step=3,
        group_index=0,
    )

    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "event": "sample_start",
            "epoch": 1,
            "sample": 5,
            "sample_total": 12,
            "qid": "q5",
            "dataset": "2wiki",
            "step": 3,
            "group_index": 0,
        }
    ]


def test_compute_answer_f1_uses_normalized_token_overlap_and_aliases() -> None:
    assert compute_answer_f1("the david  arquette!", "David Arquette", []) == 1.0
    assert compute_answer_f1("Arquette", "David Arquette", ["Arquette"]) == 1.0
    assert compute_answer_f1("David", "David Arquette", []) == 2 / 3
    assert compute_answer_f1("", "David Arquette", []) == 0.0


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
    assert rewards["answer_reward"] == 1.0
    assert rewards["total"] > 2.0


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
    assert rewards["premature_answer_penalty"] == -1.0


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


def test_train_on_rollouts_backprops_each_action_to_release_graphs(monkeypatch) -> None:
    class DummyAction:
        def __init__(self, value: float) -> None:
            self.prompt_ids = [1, 2]
            self.completion_ids = [3]
            self.old_logprobs = torch.tensor([value], dtype=torch.float32)

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

    def fake_sequence_logprobs(**kwargs):
        value = float(kwargs["completion_ids"][0])
        return torch.tensor([value], dtype=torch.float32, requires_grad=True)

    original_backward = torch.Tensor.backward

    def counting_backward(self, *args, **kwargs):
        backward_calls.append(float(self.detach().item()))
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr("rl_training.train_grpo_macorag.sequence_logprobs", fake_sequence_logprobs)
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
