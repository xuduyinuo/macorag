from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from rl_training.vllm_lora_server import parse_server_args


def test_parse_server_args_requires_lora_identity() -> None:
    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )

    assert args.model == "model/Qwen2.5-7B-Instruct"
    assert args.lora_name == "macorag_train"
    assert args.lora_int_id == 1
    assert args.lora_adapter_path == "outputs/adapter"


def test_build_lora_request_uses_configured_identity() -> None:
    from rl_training.vllm_lora_server import build_lora_request, parse_server_args

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )

    request = build_lora_request(args)

    assert request.lora_name == "macorag_train"
    assert request.lora_int_id == 1
    assert request.lora_path == "outputs/adapter"


def test_register_lora_adapter_on_workers_dispatches_startup_registration() -> None:
    from rl_training.vllm_lora_server import parse_server_args, register_lora_adapter_on_workers

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    llm = _FakeLLM()

    register_lora_adapter_on_workers(llm, args)

    assert llm.collective_rpc_calls == [
        {
            "method": "register_lora_adapter",
            "args": ("macorag_train", 1, "outputs/adapter"),
            "kwargs": {},
        }
    ]


class _FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeLLM:
    def __init__(self) -> None:
        self.generate_calls = []
        self.collective_rpc_calls = []
        self.reset_prefix_cache_calls = 0

    def generate(self, prompts, *, sampling_params, lora_request):
        self.generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "lora_request": lora_request,
            }
        )
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=(11, 12, 13))])]

    def collective_rpc(self, *, method, args=(), kwargs=None):
        self.collective_rpc_calls.append({"method": method, "args": args, "kwargs": kwargs or {}})
        return [None]

    def reset_prefix_cache(self):
        self.reset_prefix_cache_calls += 1
        return True


class _FakeLoRALayer:
    def __init__(self) -> None:
        self.rank = 2
        self.lora_alpha = 4
        self.lora_a = torch.zeros(3, 2, dtype=torch.float16)
        self.lora_b = torch.zeros(2, 4, dtype=torch.float16)


class _FakeLoRAModel:
    def __init__(self) -> None:
        self.loras = {"model.layers.0.self_attn.q_proj": _FakeLoRALayer()}


class _FakeAdapterManager:
    def __init__(self) -> None:
        self._registered_adapters = {1: _FakeLoRAModel()}
        self._active_adapters = {1: None}
        self.deactivated: list[int] = []
        self.activated: list[int] = []

    def _deactivate_adapter(self, lora_id: int) -> None:
        self.deactivated.append(lora_id)
        self._active_adapters.pop(lora_id, None)

    def activate_adapter(self, lora_id: int) -> bool:
        self.activated.append(lora_id)
        self._active_adapters[lora_id] = None
        return True


class _FakePackedLoRALayer:
    def __init__(self) -> None:
        self.rank = 2
        self.lora_alphas = [4, 6, 8]
        self.lora_a = [
            torch.zeros(3, 2, dtype=torch.float16),
            torch.zeros(5, 2, dtype=torch.float16),
            torch.zeros(7, 2, dtype=torch.float16),
        ]
        self.lora_b = [
            torch.zeros(2, 11, dtype=torch.float16),
            torch.zeros(2, 13, dtype=torch.float16),
            torch.zeros(2, 17, dtype=torch.float16),
        ]


class _FakePackedAdapterManager(_FakeAdapterManager):
    def __init__(self) -> None:
        super().__init__()
        self.packed_modules = {
            "model.layers.0.self_attn.qkv_proj": [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
            ],
            "model.layers.0.mlp.gate_up_proj": [
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.mlp.up_proj",
            ],
        }
        self._registered_adapters[1].loras = {
            "model.layers.0.self_attn.qkv_proj": _FakePackedLoRALayer(),
            "model.layers.0.mlp.gate_up_proj": _FakePackedLoRALayer(),
        }


def test_parse_vllm_lora_tensor_name_maps_module_and_side() -> None:
    from rl_training.vllm_lora_server import parse_vllm_lora_tensor_name

    assert parse_vllm_lora_tensor_name(
        "model.layers.0.self_attn.q_proj.lora_A.weight"
    ) == ("model.layers.0.self_attn.q_proj", "lora_A")
    assert parse_vllm_lora_tensor_name(
        "model.layers.31.mlp.down_proj.lora_B.weight"
    ) == ("model.layers.31.mlp.down_proj", "lora_B")


def test_parse_vllm_lora_tensor_name_rejects_unsupported_name() -> None:
    from rl_training.vllm_lora_server import parse_vllm_lora_tensor_name

    try:
        parse_vllm_lora_tensor_name("model.embed_tokens.weight")
    except ValueError as exc:
        assert "Unsupported LoRA tensor name" in str(exc)
    else:
        raise AssertionError("expected unsupported LoRA tensor name to fail")


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.self_attn.rotary_emb.lora_A.weight",
        "model.layers.0.foo.bar.lora_B.weight",
    ],
)
def test_parse_vllm_lora_tensor_name_rejects_unsupported_layer_module(name: str) -> None:
    from rl_training.vllm_lora_server import parse_vllm_lora_tensor_name

    with pytest.raises(ValueError, match="Unsupported LoRA tensor name"):
        parse_vllm_lora_tensor_name(name)


def test_update_registered_lora_tensor_transposes_peft_a_weight() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakeAdapterManager()
    incoming = torch.arange(6, dtype=torch.float16).reshape(2, 3)

    shape = update_registered_lora_tensor(
        manager,
        1,
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        incoming,
    )

    layer = manager._registered_adapters[1].loras["model.layers.0.self_attn.q_proj"]
    assert shape == (3, 2)
    assert torch.equal(layer.lora_a, incoming.T)


def test_update_registered_lora_tensor_transposes_peft_b_weight() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakeAdapterManager()
    incoming = torch.arange(8, dtype=torch.float16).reshape(4, 2)

    shape = update_registered_lora_tensor(
        manager,
        1,
        "model.layers.0.self_attn.q_proj.lora_B.weight",
        incoming,
    )

    layer = manager._registered_adapters[1].loras["model.layers.0.self_attn.q_proj"]
    assert shape == (2, 4)
    assert torch.equal(layer.lora_b, incoming.T * 2)


def test_update_registered_lora_tensor_maps_q_proj_into_packed_qkv_index() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakePackedAdapterManager()
    incoming = torch.arange(6, dtype=torch.float16).reshape(2, 3)

    shape = update_registered_lora_tensor(
        manager,
        1,
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        incoming,
    )

    layer = manager._registered_adapters[1].loras["model.layers.0.self_attn.qkv_proj"]
    assert shape == (3, 2)
    assert torch.equal(layer.lora_a[0], incoming.T)
    assert torch.equal(layer.lora_a[1], torch.zeros_like(layer.lora_a[1]))


def test_update_registered_lora_tensor_maps_gate_proj_into_packed_gate_up_index_and_scales_b() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakePackedAdapterManager()
    incoming = torch.arange(22, dtype=torch.float16).reshape(11, 2)

    shape = update_registered_lora_tensor(
        manager,
        1,
        "model.layers.0.mlp.gate_proj.lora_B.weight",
        incoming,
    )

    layer = manager._registered_adapters[1].loras["model.layers.0.mlp.gate_up_proj"]
    assert shape == (2, 11)
    assert torch.equal(layer.lora_b[0], incoming.T * 2)
    assert torch.equal(layer.lora_b[1], torch.zeros_like(layer.lora_b[1]))


def test_update_registered_lora_tensor_rejects_shape_mismatch() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakeAdapterManager()
    incoming = torch.zeros(5, 5, dtype=torch.float16)

    try:
        update_registered_lora_tensor(
            manager,
            1,
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            incoming,
        )
    except ValueError as exc:
        assert "LoRA tensor shape mismatch" in str(exc)
    else:
        raise AssertionError("expected shape mismatch to fail")


def test_update_registered_lora_tensor_rejects_missing_module() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakeAdapterManager()
    incoming = torch.zeros(2, 3, dtype=torch.float16)

    try:
        update_registered_lora_tensor(
            manager,
            1,
            "model.layers.0.self_attn.k_proj.lora_A.weight",
            incoming,
        )
    except KeyError as exc:
        assert "Registered LoRA module not found" in str(exc)
    else:
        raise AssertionError("expected missing module to fail")


def test_refresh_active_lora_reactivates_active_adapter() -> None:
    from rl_training.vllm_lora_server import refresh_active_lora

    manager = _FakeAdapterManager()

    refreshed = refresh_active_lora(manager, 1)

    assert refreshed is True
    assert manager.deactivated == [1]
    assert manager.activated == [1]


def test_refresh_active_lora_pops_active_adapter_before_reactivation() -> None:
    from rl_training.vllm_lora_server import refresh_active_lora

    manager = _FakeAdapterManager()

    def _sticky_deactivate(lora_id: int) -> None:
        manager.deactivated.append(lora_id)

    manager._deactivate_adapter = _sticky_deactivate

    refreshed = refresh_active_lora(manager, 1)

    assert refreshed is True
    assert manager.deactivated == [1]
    assert manager.activated == [1]


def test_refresh_active_lora_raises_when_reactivation_fails() -> None:
    from rl_training.vllm_lora_server import refresh_active_lora

    manager = _FakeAdapterManager()

    def _fail_activate(lora_id: int) -> bool:
        manager.activated.append(lora_id)
        return False

    manager.activate_adapter = _fail_activate

    with pytest.raises(RuntimeError, match="Failed to reactivate LoRA adapter"):
        refresh_active_lora(manager, 1)
    assert manager.deactivated == [1]
    assert manager.activated == [1]


def test_refresh_active_lora_returns_false_when_adapter_was_not_active() -> None:
    from rl_training.vllm_lora_server import refresh_active_lora

    manager = _FakeAdapterManager()
    manager._active_adapters.clear()

    refreshed = refresh_active_lora(manager, 1)

    assert refreshed is False
    assert manager.deactivated == []
    assert manager.activated == []


def test_generate_endpoint_passes_fixed_lora_request() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    llm = _FakeLLM()
    app = create_app(args, llm=llm, sampling_params_cls=_FakeSamplingParams)

    response = TestClient(app).post(
        "/generate/",
        json={"prompts": ["hello"], "temperature": 0.7, "top_p": 0.9, "top_k": 20, "max_tokens": 4},
    )

    assert response.status_code == 200
    assert response.json() == {"completion_ids": [[11, 12, 13]]}
    assert llm.generate_calls[0]["prompts"] == ["hello"]
    assert llm.generate_calls[0]["sampling_params"].kwargs["temperature"] == 0.7
    assert llm.generate_calls[0]["lora_request"].lora_name == "macorag_train"
    assert llm.generate_calls[0]["lora_request"].lora_int_id == 1
    assert llm.generate_calls[0]["lora_request"].lora_path == "outputs/adapter"


def test_health_endpoint_exposes_lora_identity_and_capability_status() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    app = create_app(args, llm=_FakeLLM(), sampling_params_cls=_FakeSamplingParams)

    response = TestClient(app).get("/health/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "sync_mode": "lora",
        "model": "model/Qwen2.5-7B-Instruct",
        "lora_name": "macorag_train",
        "lora_int_id": 1,
        "lora_adapter_path": "outputs/adapter",
        "supports_lora_param_update": True,
    }


def test_update_lora_param_endpoint_dispatches_collective_rpc_with_lora_id() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    llm = _FakeLLM()
    app = create_app(args, llm=llm, sampling_params_cls=_FakeSamplingParams)

    response = TestClient(app).post(
        "/update_lora_param/",
        json={
            "name": "model.layers.0.self_attn.q_proj.lora_A.weight",
            "dtype": "torch.float32",
            "shape": [2, 3],
        },
    )

    assert response.status_code == 200
    update_id = response.json()["update_id"]
    for _ in range(20):
        status = TestClient(app).get(f"/lora_update_status/{update_id}").json()
        if status["state"] == "ok":
            break
        time.sleep(0.01)

    assert status == {"state": "ok", "error": None}
    assert llm.collective_rpc_calls[-2:] == [
        {
            "method": "validate_lora_params",
            "args": ([("model.layers.0.self_attn.q_proj.lora_A.weight", torch.float32, (2, 3))], 1),
            "kwargs": {},
        },
        {
            "method": "update_lora_param",
            "args": ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.float32, (2, 3), 1),
            "kwargs": {},
        },
    ]


def test_update_lora_params_endpoint_dispatches_single_batch_collective_rpc() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    llm = _FakeLLM()
    client = TestClient(create_app(args, llm=llm, sampling_params_cls=_FakeSamplingParams))

    response = client.post(
        "/update_lora_params/",
        json={
            "tensors": [
                {
                    "name": "model.layers.0.self_attn.q_proj.lora_A.weight",
                    "dtype": "torch.float32",
                    "shape": [2, 3],
                },
                {
                    "name": "model.layers.0.self_attn.q_proj.lora_B.weight",
                    "dtype": "torch.float16",
                    "shape": [4, 2],
                },
            ]
        },
    )

    assert response.status_code == 200
    update_id = response.json()["update_id"]
    for _ in range(20):
        status = client.get(f"/lora_update_status/{update_id}").json()
        if status["state"] == "ok":
            break
        time.sleep(0.01)

    assert status == {"state": "ok", "error": None}
    expected_specs = [
        ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.float32, (2, 3)),
        ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.float16, (4, 2)),
    ]
    assert llm.collective_rpc_calls[-2:] == [
        {"method": "validate_lora_params", "args": (expected_specs, 1), "kwargs": {}},
        {"method": "update_lora_params", "args": (expected_specs, 1), "kwargs": {}},
    ]


def test_update_lora_params_endpoint_rejects_failed_preflight_without_update_id() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    class RejectingLLM(_FakeLLM):
        def collective_rpc(self, *, method, args=(), kwargs=None):
            self.collective_rpc_calls.append({"method": method, "args": args, "kwargs": kwargs or {}})
            if method == "validate_lora_params":
                raise ValueError("shape mismatch")
            return [None]

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    llm = RejectingLLM()
    client = TestClient(create_app(args, llm=llm, sampling_params_cls=_FakeSamplingParams), raise_server_exceptions=False)

    response = client.post(
        "/update_lora_params/",
        json={
            "tensors": [
                {
                    "name": "model.layers.0.self_attn.q_proj.lora_A.weight",
                    "dtype": "torch.float16",
                    "shape": [999, 3],
                }
            ]
        },
    )

    assert response.status_code == 400
    assert "shape mismatch" in response.text
    assert [call["method"] for call in llm.collective_rpc_calls] == ["validate_lora_params"]


def test_update_lora_param_endpoint_returns_before_collective_rpc_completes_and_exposes_status() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    class BlockingLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.release = threading.Event()

        def collective_rpc(self, *, method, args=(), kwargs=None):
            self.collective_rpc_calls.append({"method": method, "args": args, "kwargs": kwargs or {}})
            if method != "validate_lora_params":
                self.release.wait(timeout=2)
            return [None]

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    llm = BlockingLLM()
    client = TestClient(create_app(args, llm=llm, sampling_params_cls=_FakeSamplingParams))

    start = time.perf_counter()
    response = client.post(
        "/update_lora_param/",
        json={
            "name": "model.layers.0.self_attn.q_proj.lora_A.weight",
            "dtype": "torch.float32",
            "shape": [2, 3],
        },
    )
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert elapsed < 0.5
    update_id = response.json()["update_id"]
    assert client.get(f"/lora_update_status/{update_id}").json() == {"state": "pending", "error": None}

    llm.release.set()
    for _ in range(20):
        status = client.get(f"/lora_update_status/{update_id}").json()
        if status["state"] == "ok":
            break
        time.sleep(0.01)

    assert status == {"state": "ok", "error": None}


def test_init_communicator_endpoint_returns_before_collective_rpc_completes() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    class BlockingLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.release = threading.Event()

        def collective_rpc(self, *, method, args=(), kwargs=None):
            self.collective_rpc_calls.append({"method": method, "args": args, "kwargs": kwargs or {}})
            self.release.wait(timeout=2)
            return [None]

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    llm = BlockingLLM()
    client = TestClient(create_app(args, llm=llm, sampling_params_cls=_FakeSamplingParams))

    start = time.perf_counter()
    response = client.post("/init_communicator/", json={"host": "0.0.0.0", "port": 12345, "world_size": 2})
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert elapsed < 0.5
    for _ in range(20):
        if llm.collective_rpc_calls:
            break
        time.sleep(0.01)
    assert llm.collective_rpc_calls == [
        {"method": "init_communicator", "args": ("0.0.0.0", 12345, 2), "kwargs": {}}
    ]
    llm.release.set()


def test_lora_update_status_endpoint_returns_404_for_unknown_update() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    client = TestClient(create_app(args, llm=_FakeLLM(), sampling_params_cls=_FakeSamplingParams))

    response = client.get("/lora_update_status/missing")

    assert response.status_code == 404


def test_update_lora_param_endpoint_rejects_bad_dtype() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    app = create_app(args, llm=_FakeLLM(), sampling_params_cls=_FakeSamplingParams)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/update_lora_param/",
        json={"name": "model.layers.0.self_attn.q_proj.lora_A.weight", "dtype": "bad", "shape": [2, 3]},
    )

    assert response.status_code == 400


def test_weight_sync_lora_worker_extension_updates_registered_adapter() -> None:
    from rl_training.vllm_lora_server import WeightSyncLoRAWorkerExtension

    manager = _FakeAdapterManager()
    extension = WeightSyncLoRAWorkerExtension()
    extension.device = torch.device("cpu")
    extension.client_rank = 1
    extension.model_runner = type(
        "Runner",
        (),
        {"lora_manager": type("WorkerManager", (), {"_adapter_manager": manager})()},
    )()

    class FakeComm:
        def __init__(self) -> None:
            self.group = type("Group", (), {"barrier": lambda self_group: None})()

        def broadcast(self, tensor, src):
            assert src == 1
            tensor.copy_(torch.arange(6, dtype=tensor.dtype).reshape(2, 3))

    extension.pynccl_comm = FakeComm()

    extension.update_lora_param(
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        torch.float16,
        (2, 3),
        1,
    )

    layer = manager._registered_adapters[1].loras["model.layers.0.self_attn.q_proj"]
    assert torch.equal(layer.lora_a, torch.arange(6, dtype=torch.float16).reshape(2, 3).T)
    assert manager.deactivated == [1]
    assert manager.activated == [1]


def test_weight_sync_lora_worker_extension_updates_registered_adapter_batch() -> None:
    from rl_training.vllm_lora_server import WeightSyncLoRAWorkerExtension

    manager = _FakeAdapterManager()
    extension = WeightSyncLoRAWorkerExtension()
    extension.device = torch.device("cpu")
    extension.client_rank = 1
    extension.model_runner = type(
        "Runner",
        (),
        {"lora_manager": type("WorkerManager", (), {"_adapter_manager": manager})()},
    )()

    payloads = [
        torch.arange(6, dtype=torch.float16).reshape(2, 3),
        torch.arange(8, dtype=torch.float16).reshape(4, 2),
    ]

    class FakeComm:
        def __init__(self) -> None:
            self.index = 0
            self.group = type("Group", (), {"barrier": lambda self_group: None})()

        def broadcast(self, tensor, src):
            assert src == 1
            tensor.copy_(payloads[self.index])
            self.index += 1

    extension.pynccl_comm = FakeComm()

    extension.update_lora_params(
        [
            ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.float16, (2, 3)),
            ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.float16, (4, 2)),
        ],
        1,
    )

    layer = manager._registered_adapters[1].loras["model.layers.0.self_attn.q_proj"]
    assert torch.equal(layer.lora_a, payloads[0].T)
    assert torch.equal(layer.lora_b, payloads[1].T * 2)
    assert manager.deactivated == [1]
    assert manager.activated == [1]


def test_weight_sync_lora_worker_extension_batch_validates_before_broadcast() -> None:
    from rl_training.vllm_lora_server import WeightSyncLoRAWorkerExtension

    manager = _FakeAdapterManager()
    extension = WeightSyncLoRAWorkerExtension()
    extension.device = torch.device("cpu")
    extension.client_rank = 1
    extension.model_runner = type(
        "Runner",
        (),
        {"lora_manager": type("WorkerManager", (), {"_adapter_manager": manager})()},
    )()

    class FakeComm:
        def __init__(self) -> None:
            self.broadcast_calls = 0
            self.group = type("Group", (), {"barrier": lambda self_group: None})()

        def broadcast(self, tensor, src):
            self.broadcast_calls += 1

    communicator = FakeComm()
    extension.pynccl_comm = communicator

    with pytest.raises(ValueError, match="shape mismatch"):
        extension.update_lora_params(
            [
                ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.float16, (2, 3)),
                ("model.layers.0.self_attn.q_proj.lora_B.weight", torch.float16, (999, 2)),
            ],
            1,
        )

    assert communicator.broadcast_calls == 0


def test_weight_sync_lora_worker_extension_registers_and_pins_adapter() -> None:
    from vllm.lora.request import LoRARequest

    from rl_training.vllm_lora_server import WeightSyncLoRAWorkerExtension

    class FakeWorkerLoRAManager:
        def __init__(self) -> None:
            self.added: list[LoRARequest] = []
            self.pinned: list[int] = []

        def add_adapter(self, request: LoRARequest) -> bool:
            self.added.append(request)
            return True

        def pin_adapter(self, lora_int_id: int) -> bool:
            self.pinned.append(lora_int_id)
            return True

    worker_lora_manager = FakeWorkerLoRAManager()
    extension = WeightSyncLoRAWorkerExtension()
    extension.model_runner = type("Runner", (), {"lora_manager": worker_lora_manager})()

    extension.register_lora_adapter("macorag_train", 1, "outputs/adapter")

    assert [(request.lora_name, request.lora_int_id, request.lora_path) for request in worker_lora_manager.added] == [
        ("macorag_train", 1, "outputs/adapter")
    ]
    assert worker_lora_manager.pinned == [1]


def test_reset_prefix_cache_endpoint_calls_llm_reset() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )
    llm = _FakeLLM()
    app = create_app(args, llm=llm, sampling_params_cls=_FakeSamplingParams)

    response = TestClient(app).post("/reset_prefix_cache/")

    assert response.status_code == 200
    assert llm.reset_prefix_cache_calls == 1
