from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from .config import MAPPOConfig, parse_args as parse_mappo_args
from .mappo_types import AgentRole, RLSample
from .models import FINAL_ANSWER_GUIDED_REGEX
from .protocol import OUTPUT_CONTRACT_MARKER, SYSTEM_PROMPTS
from .rollout import RolloutCollector
from .vllm_client import VLLMClient


def load_aligned_mappo_config(path: str | Path) -> MAPPOConfig:
    """Load the authoritative training config used for inference semantics."""

    return parse_mappo_args(["--config", str(path)])


def align_evaluation_args(args: Any, config: MAPPOConfig) -> None:
    """Make external evaluation use the same rollout contract as validation."""

    configured_model = str(getattr(args, "model_path", "") or "").strip()
    if configured_model and configured_model != config.model_path:
        raise ValueError(
            "MAPPO evaluation base model differs from training: "
            f"{configured_model!r} != {config.model_path!r}"
        )
    args.model_path = config.model_path
    args.max_rounds = config.max_rounds
    args.max_prompt_length = config.max_prompt_length
    args.max_completion_length = config.max_completion_length
    args.temperature = config.validation_temperature
    args.top_p = config.top_p
    args.retrieval_backend = config.retrieval_backend
    args.retrieval_embedding_model = config.retrieval_embedding_model
    args.retrieval_device = config.retrieval_device
    args.retrieval_max_length = config.retrieval_max_length
    args.retrieval_batch_size = config.retrieval_batch_size
    args.retrieval_top_k = config.retrieval_top_k
    args._mappo_config = config


class _ZeroCritic:
    """Evaluation-only critic preserving the exact RolloutCollector path."""

    def __init__(self) -> None:
        import torch

        self.torch = torch

    def __call__(self, states: list[dict[str, Any]]) -> Any:
        return self.torch.zeros(len(states), dtype=self.torch.float32)


class MAPPOInferenceActor:
    """Inference-only counterpart of VLLMRoleConditionedActor."""

    def __init__(self, *, tokenizer: Any, base_urls: list[str], model: str,
                 config: MAPPOConfig, timeout: float, attempts: int,
                 backoff: float) -> None:
        if not base_urls:
            raise ValueError("MAPPO evaluation requires at least one vLLM endpoint")
        self.tokenizer = tokenizer
        self.base_urls = [value.rstrip("/") for value in base_urls]
        self.model = model
        self.config = config
        self.timeout = timeout
        self.attempts = attempts
        self.backoff = backoff
        self.max_prompt_length = config.max_prompt_length
        self.max_completion_length = config.max_completion_length
        self.temperature = config.validation_temperature
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.force_final_answer_decoding = config.force_final_answer_decoding
        self._thread_local = threading.local()

    def set_endpoint_index(self, index: int) -> None:
        self._thread_local.endpoint_index = int(index)
        self._thread_local.generation_counter = int(index) * 100

    def _client(self) -> VLLMClient:
        index = int(getattr(self._thread_local, "endpoint_index", 0))
        selected = self.base_urls[index % len(self.base_urls)]
        if getattr(self._thread_local, "client_url", None) != selected:
            parsed = urlparse(selected)
            if not parsed.hostname or parsed.port is None:
                raise ValueError(f"Invalid MAPPO evaluation endpoint: {selected}")
            self._thread_local.client = VLLMClient(
                host=parsed.hostname, port=parsed.port,
                model_name=self.model, lora_name=self.model,
                timeout=self.timeout, attempts=self.attempts,
                backoff=self.backoff,
            )
            self._thread_local.client_url = selected
        return self._thread_local.client

    def encode_prompt(self, role: AgentRole, prompt: str) -> list[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[role]},
            {"role": "user", "content": prompt},
        ]
        ids = list(self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
        ))
        if len(ids) <= self.max_prompt_length:
            return ids
        system_ids = list(self.tokenizer.apply_chat_template(
            [messages[0]], add_generation_prompt=False, tokenize=True,
        ))
        head_length = min(len(system_ids), self.max_prompt_length)
        if OUTPUT_CONTRACT_MARKER in prompt and head_length < self.max_prompt_length:
            tail_length = self.max_prompt_length - head_length
            return [*ids[:head_length], *ids[-tail_length:]]
        head_length = min(max(1, self.max_prompt_length // 4), head_length)
        return [*ids[:head_length], *ids[-(self.max_prompt_length - head_length):]]

    def generate(self, role: AgentRole, prompt: str, *,
                 guided_regex: str | None = None,
                 max_tokens: int | None = None) -> tuple[str, list[int], list[int], Any]:
        import torch

        prompt_ids = self.encode_prompt(role, prompt)
        counter = int(getattr(self._thread_local, "generation_counter", 0))
        self._thread_local.generation_counter = counter + 1
        output = self._client().generate(
            prompt_ids,
            max_tokens=max_tokens or self.max_completion_length,
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            seed=self.config.seed + counter,
            guided_regex=(
                guided_regex
                or (
                    FINAL_ANSWER_GUIDED_REGEX
                    if self.force_final_answer_decoding
                    and role is AgentRole.ANSWER
                    and "This is the final round." in prompt
                    else None
                )
            ),
        )
        response = output.text or self.tokenizer.decode(
            output.token_ids, skip_special_tokens=True,
        )
        return (
            response, prompt_ids, output.token_ids,
            torch.tensor(output.token_logprobs, dtype=torch.float32),
        )


class MAPPOAlignedEvaluationPolicy:
    """Run external test samples through the training validation algorithm."""

    def __init__(self, *, args: Any, config: MAPPOConfig) -> None:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise SystemExit("transformers is required for MAPPO evaluation") from exc
        tokenizer_root = str(getattr(args, "adapter_path", "") or config.model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, trust_remote_code=True)
        self.actor = MAPPOInferenceActor(
            tokenizer=tokenizer,
            base_urls=list(getattr(args, "vllm_base_urls", []) or []),
            model=str(getattr(args, "vllm_model", "") or ""),
            config=config,
            timeout=float(getattr(args, "vllm_timeout", 120)),
            attempts=int(getattr(args, "vllm_retries", 3)),
            backoff=float(getattr(args, "vllm_retry_sleep_seconds", 1.0)),
        )
        self.config = config
        self.critic = _ZeroCritic()

    def set_endpoint_index(self, index: int) -> None:
        self.actor.set_endpoint_index(index)

    def run_evaluation(self, sample: Any, retrieval: Any) -> Any:
        episode = RolloutCollector(
            actor=self.actor, critic=self.critic,
            retrieval=retrieval, config=self.config,
        ).collect(RLSample(
            qid=sample.qid, dataset=sample.dataset,
            question=sample.question, answer=sample.answer,
            answer_aliases=tuple(sample.answer_aliases),
            supporting_facts=tuple(sample.supporting_facts),
            metadata=dict(sample.metadata),
        ))
        return SimpleNamespace(
            final_answer=episode.final_answer,
            trajectory=episode.trajectory,
            parse_errors=episode.parse_errors,
            state=SimpleNamespace(retrieval_count=len(episode.trajectory)),
            mappo_protocol={
                "parse_failed": bool(episode.parse_errors),
                "missing_answer_tag": any(
                    "Missing required tag: answer" in error
                    for error in episode.parse_errors
                ),
                "final_compliant": (
                    not episode.parse_errors
                    and isinstance(episode.final_answer, str)
                    and bool(episode.final_answer.strip())
                ),
            },
        )
