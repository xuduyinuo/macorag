from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests
import torch

from rag import AgentRole, RAGState
from rl_training.policy import PolicyGenerationRequest, RolloutTrace, VLLMSharedPolicy
from rl_training.vllm_client import VLLMGenerationClient, VLLMGenerationOutput


class _Response:
    status_code = 200
    text = "ok"

    def json(self):
        return {
            "completion_ids": [[10], [20]],
            "logprobs": [[-0.1], [-0.2]],
        }


class _Session:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def post(self, url: str, *, json: dict[str, object]):
        self.payloads.append(dict(json))
        return _Response()


def test_vllm_client_sends_one_seed_per_prompt() -> None:
    session = _Session()
    client = VLLMGenerationClient(
        host="127.0.0.1",
        port=8000,
        timeout_seconds=5,
        backend=SimpleNamespace(session=session, base_url="http://127.0.0.1:8000"),
    )

    outputs = client.generate_batch(
        ["first", "second"],
        max_tokens=8,
        temperature=0.8,
        top_p=0.95,
        top_k=5,
        seeds=[101, 102],
    )

    assert [item.completion_ids for item in outputs] == [[10], [20]]
    assert session.payloads[0]["seeds"] == [101, 102]


def test_vllm_client_rejects_seed_count_mismatch() -> None:
    client = VLLMGenerationClient(
        host="127.0.0.1",
        port=8000,
        timeout_seconds=5,
        backend=SimpleNamespace(session=_Session(), base_url="http://127.0.0.1:8000"),
    )

    with pytest.raises(ValueError, match="seed"):
        client.generate_batch(
            ["first", "second"],
            max_tokens=8,
            temperature=0.8,
            top_p=0.95,
            top_k=5,
            seeds=[101],
        )


def test_vllm_client_transport_retry_reuses_identical_seed_payload() -> None:
    payloads: list[dict[str, object]] = []

    class FailingSession:
        def post(self, url: str, *, json: dict[str, object]):
            payloads.append(dict(json))
            raise requests.ConnectionError("disconnect")

        def close(self) -> None:
            pass

    class SuccessfulResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {"completion_ids": [[10]], "logprobs": [[-0.1]]}

    class SuccessfulSession:
        def post(self, url: str, *, json: dict[str, object]):
            payloads.append(dict(json))
            return SuccessfulResponse()

    backend = SimpleNamespace(
        session=FailingSession(),
        base_url="http://127.0.0.1:8000",
    )
    client = VLLMGenerationClient(
        host="127.0.0.1",
        port=8000,
        timeout_seconds=5,
        backend=backend,
        sleep_fn=lambda seconds: None,
        session_factory=SuccessfulSession,
    )

    client.generate_batch(
        ["prompt"],
        max_tokens=8,
        temperature=0.8,
        top_p=0.95,
        top_k=5,
        seeds=[707],
    )

    assert [payload["seeds"] for payload in payloads] == [[707], [707]]


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool):
        return [1, 2, 3]

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "decoded"


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _SeedRecordingClient:
    def __init__(self) -> None:
        self.seed_batches: list[list[int]] = []

    def generate_batch(self, prompts, *, seeds, **kwargs):
        self.seed_batches.append(list(seeds))
        return [
            VLLMGenerationOutput(completion_ids=[10], logprobs=[-0.1])
            for _ in prompts
        ]


def _request(question: str) -> PolicyGenerationRequest:
    return PolicyGenerationRequest(
        role=AgentRole.QUERY_RETRIEVER,
        question=question,
        state=RAGState(question=question),
    )


def test_vllm_policy_generation_counter_produces_distinct_resumable_seeds() -> None:
    client = _SeedRecordingClient()
    policy = VLLMSharedPolicy(
        model=_Model(),
        tokenizer=_Tokenizer(),
        vllm_client=client,
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=8,
        temperature=0.8,
        top_p=0.95,
        top_k=5,
        generation_seed=100,
        generation_counter=10,
    )

    policy.generate_batch(
        [_request("first"), _request("second")],
        traces=[RolloutTrace(), RolloutTrace()],
    )

    assert client.seed_batches == [[110, 111]]
    assert policy.generation_counter == 12

    resumed_client = _SeedRecordingClient()
    resumed = VLLMSharedPolicy(
        model=_Model(),
        tokenizer=_Tokenizer(),
        vllm_client=resumed_client,
        system_prompt="system",
        max_prompt_length=32,
        max_completion_length=8,
        temperature=0.8,
        top_p=0.95,
        top_k=5,
        generation_seed=100,
        generation_counter=policy.generation_counter,
    )
    resumed.generate_batch([_request("third")], traces=[RolloutTrace()])

    assert resumed_client.seed_batches == [[112]]


def test_vllm_server_builds_one_sampling_params_per_seed() -> None:
    from rl_training.vllm_lora_server import _build_sampling_params

    class SamplingParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    result = _build_sampling_params(
        SamplingParams,
        {"temperature": 0.8, "max_tokens": 8},
        seeds=[101, 102],
    )

    assert [item.kwargs["seed"] for item in result] == [101, 102]


def test_lora_server_validation_requires_prompt_seed_capability() -> None:
    class HealthResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "status": "ok",
                "sync_mode": "lora",
                "model": "base",
                "lora_name": "policy",
                "lora_int_id": 1,
                "lora_adapter_path": "adapter",
                "supports_lora_param_update": True,
            }

    class HealthSession:
        def get(self, url: str):
            return HealthResponse()

    client = VLLMGenerationClient(
        host="127.0.0.1",
        port=8000,
        timeout_seconds=5,
        backend=SimpleNamespace(
            session=HealthSession(),
            base_url="http://127.0.0.1:8000",
        ),
    )
    args = SimpleNamespace(
        model_path="base",
        vllm_lora_name="policy",
        vllm_lora_int_id=1,
        vllm_lora_adapter_path="adapter",
    )

    with pytest.raises(SystemExit, match="prompt seed"):
        client.validate_lora_server(args)
