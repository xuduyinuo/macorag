from __future__ import annotations

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
    assert torch.equal(layer.lora_b, incoming.T)


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


def test_refresh_active_lora_returns_false_when_reactivation_fails() -> None:
    from rl_training.vllm_lora_server import refresh_active_lora

    manager = _FakeAdapterManager()

    def _fail_activate(lora_id: int) -> bool:
        manager.activated.append(lora_id)
        return False

    manager.activate_adapter = _fail_activate

    refreshed = refresh_active_lora(manager, 1)

    assert refreshed is False
    assert manager.deactivated == [1]
    assert manager.activated == [1]


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
        "supports_lora_param_update": False,
    }


def test_update_lora_param_endpoint_fails_explicitly() -> None:
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
        json={"name": "model.layers.0.self_attn.q_proj.lora_A.weight", "dtype": "torch.float32", "shape": [64, 3584]},
    )

    assert response.status_code == 501
    assert "LoRA in-memory tensor replacement is not implemented" in response.text


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
