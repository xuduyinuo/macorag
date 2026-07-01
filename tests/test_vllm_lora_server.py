from __future__ import annotations

from types import SimpleNamespace

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
