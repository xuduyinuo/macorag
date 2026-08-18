from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag import (
    AgentRole,
    RAGState,
    build_answer_generator_prompt,
    build_evidence_updater_prompt,
    build_query_retriever_prompt,
)

from .vllm_client import VLLMGenerationOutput


@dataclass
class GeneratedAction:
    role: AgentRole
    prompt: str
    response: str
    prompt_ids: list[int]
    completion_ids: list[int]
    old_logprobs: Any
    round_index: int = 0
    local_reward: float = 0.0
    terminal_reward: float = 0.0
    advantage: float = 0.0


@dataclass
class RolloutTrace:
    actions: list[GeneratedAction] = field(default_factory=list)


@dataclass(frozen=True)
class PolicyGenerationRequest:
    role: AgentRole
    question: str
    state: RAGState
    observation: dict[str, Any] | None = None


class HFSharedPolicy:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        system_prompt: str,
        max_prompt_length: int,
        max_completion_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.trace = RolloutTrace()

    def reset_trace(self) -> None:
        self.trace = RolloutTrace()

    def _next_round_index(self, role: AgentRole) -> int:
        return sum(1 for action in self.trace.actions if action.role == role)

    def _prompt_for(
        self,
        *,
        role: AgentRole,
        question: str,
        state: RAGState,
        observation: dict[str, Any] | None,
    ) -> str:
        if role == AgentRole.QUERY_RETRIEVER:
            return build_query_retriever_prompt(question=question, state=state)
        if role == AgentRole.EVIDENCE_UPDATER:
            return build_evidence_updater_prompt(
                question=question,
                state=state,
                observation=observation or {"passages": []},
            )
        return build_answer_generator_prompt(question=question, state=state)

    def _encode_prompt(self, prompt: str) -> list[int]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        return list(prompt_ids)[-self.max_prompt_length :]

    def generate(
        self,
        *,
        role: AgentRole,
        question: str,
        state: RAGState,
        observation: dict[str, Any] | None = None,
    ) -> str:
        import torch

        prompt = self._prompt_for(role=role, question=question, state=state, observation=observation)
        prompt_ids = self._encode_prompt(prompt)
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        do_sample = self.temperature > 0.0
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_completion_length,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                top_k=self.top_k if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )
        completion = generated[0, input_ids.shape[1] :].tolist()
        if self.tokenizer.eos_token_id in completion:
            completion = completion[: completion.index(self.tokenizer.eos_token_id) + 1]
        response = self.tokenizer.decode(completion, skip_special_tokens=True)
        old_logprobs = sequence_logprobs(
            model=self.model,
            prompt_ids=prompt_ids,
            completion_ids=completion,
            device=device,
        ).detach().cpu()
        self.trace.actions.append(
            GeneratedAction(
                role=role,
                prompt=prompt,
                response=response,
                prompt_ids=prompt_ids,
                completion_ids=completion,
                old_logprobs=old_logprobs,
                round_index=self._next_round_index(role),
            )
        )
        return response


class VLLMSharedPolicy(HFSharedPolicy):
    def __init__(self, *, vllm_client: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.vllm_client = vllm_client
        self.timing: dict[str, float] = {
            "time_vllm_generate_seconds": 0.0,
            "time_behavior_rescore_seconds": 0.0,
        }

    def reset_trace(self) -> None:
        super().reset_trace()
        self.timing = {
            "time_vllm_generate_seconds": 0.0,
            "time_behavior_rescore_seconds": 0.0,
        }

    def generate(
        self,
        *,
        role: AgentRole,
        question: str,
        state: RAGState,
        observation: dict[str, Any] | None = None,
    ) -> str:
        return self.generate_batch(
            [
                PolicyGenerationRequest(
                    role=role,
                    question=question,
                    state=state,
                    observation=observation,
                )
            ],
            traces=[self.trace],
        )[0]

    def generate_batch(
        self,
        requests: list[PolicyGenerationRequest],
        *,
        traces: list[RolloutTrace] | None = None,
    ) -> list[str]:
        import time
        import torch

        if not requests:
            return []
        if traces is None:
            traces = [self.trace for _ in requests]
        if len(traces) != len(requests):
            raise ValueError(
                f"Expected one rollout trace per generation request, got {len(traces)} for {len(requests)}."
            )

        prompts = [
            self._prompt_for(
                role=request.role,
                question=request.question,
                state=request.state,
                observation=request.observation,
            )
            for request in requests
        ]
        prompt_id_batches = [self._encode_prompt(prompt) for prompt in prompts]
        decoded_prompts = [
            self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            for prompt_ids in prompt_id_batches
        ]
        generate_start = time.perf_counter()
        batch_generator = getattr(self.vllm_client, "generate_batch", None)
        if callable(batch_generator):
            outputs = batch_generator(
                decoded_prompts,
                max_tokens=self.max_completion_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            outputs = []
            for decoded_prompt in decoded_prompts:
                completion_ids, text = self.vllm_client.generate(
                    decoded_prompt,
                    max_tokens=self.max_completion_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
                outputs.append(VLLMGenerationOutput(completion_ids=completion_ids, text=text))
        self.timing["time_vllm_generate_seconds"] += time.perf_counter() - generate_start
        if len(outputs) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(requests)} policy requests."
            )

        device = next(self.model.parameters()).device
        responses: list[str] = []
        for request, trace, prompt, prompt_ids, output in zip(
            requests,
            traces,
            prompts,
            prompt_id_batches,
            outputs,
        ):
            completion_ids = list(output.completion_ids)
            output_logprobs = list(output.logprobs) if output.logprobs is not None else None
            if self.tokenizer.eos_token_id in completion_ids:
                end = completion_ids.index(self.tokenizer.eos_token_id) + 1
                completion_ids = completion_ids[:end]
                if output_logprobs is not None:
                    output_logprobs = output_logprobs[:end]
            response = output.text or self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            if output_logprobs is None:
                rescore_start = time.perf_counter()
                with torch.no_grad():
                    old_logprobs = sequence_logprobs(
                        model=self.model,
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                        device=device,
                    ).detach().cpu()
                self.timing["time_behavior_rescore_seconds"] += time.perf_counter() - rescore_start
            else:
                old_logprobs = torch.tensor(output_logprobs, dtype=torch.float32)
            round_index = sum(1 for action in trace.actions if action.role == request.role)
            trace.actions.append(
                GeneratedAction(
                    role=request.role,
                    prompt=prompt,
                    response=response,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    old_logprobs=old_logprobs,
                    round_index=round_index,
                )
            )
            responses.append(response)
        return responses


def sequence_logprobs(
    *,
    model: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: Any,
) -> Any:
    import torch
    import torch.nn.functional as functional

    if not completion_ids:
        return torch.empty(0, dtype=torch.float32, device=device)
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    logits_to_keep = len(completion_ids) + 1
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
        )
    except TypeError:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    labels = functional.pad(input_ids[:, 1:], (0, 1), value=-100)
    logits = outputs.logits
    kept_labels = labels[:, -logits.shape[1] :]
    if logits.shape[1] > logits_to_keep:
        logits = logits[:, -logits_to_keep:, :]
        kept_labels = kept_labels[:, -logits_to_keep:]

    target_logits = logits[:, :-1, :]
    target_labels = kept_labels[:, :-1]
    logprobs = (
        functional.log_softmax(target_logits, dim=-1)
        .gather(-1, target_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    return logprobs[0, -len(completion_ids) :]


def batched_sequence_logprobs(
    *,
    model: Any,
    prompt_id_batches: list[list[int]],
    completion_id_batches: list[list[int]],
    device: Any,
    pad_token_id: int,
) -> tuple[Any, Any]:
    """Return completion-token logprobs from one padded causal-LM forward."""
    import torch
    import torch.nn.functional as functional

    if len(prompt_id_batches) != len(completion_id_batches):
        raise ValueError(
            "prompt_id_batches and completion_id_batches must have identical batch sizes."
        )
    batch_size = len(prompt_id_batches)
    if batch_size == 0:
        empty = torch.empty((0, 0), dtype=torch.float32, device=device)
        return empty, torch.empty((0, 0), dtype=torch.bool, device=device)
    if any(
        completion_ids and not prompt_ids
        for prompt_ids, completion_ids in zip(prompt_id_batches, completion_id_batches)
    ):
        raise ValueError("A non-empty completion requires at least one prompt token.")

    sequence_lengths = [
        len(prompt_ids) + len(completion_ids)
        for prompt_ids, completion_ids in zip(prompt_id_batches, completion_id_batches)
    ]
    max_sequence_length = max(sequence_lengths, default=0)
    max_completion_length = max(map(len, completion_id_batches), default=0)
    if max_completion_length == 0:
        empty = torch.empty((batch_size, 0), dtype=torch.float32, device=device)
        return empty, torch.empty((batch_size, 0), dtype=torch.bool, device=device)

    input_ids = torch.full(
        (batch_size, max_sequence_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    completion_mask = torch.zeros(
        (batch_size, max_completion_length),
        dtype=torch.bool,
        device=device,
    )
    predictor_positions = torch.zeros(
        (batch_size, max_completion_length),
        dtype=torch.long,
        device=device,
    )
    completion_tokens = torch.zeros_like(predictor_positions)

    for batch_index, (prompt_ids, completion_ids) in enumerate(
        zip(prompt_id_batches, completion_id_batches)
    ):
        sequence = prompt_ids + completion_ids
        left_padding = max_sequence_length - len(sequence)
        if sequence:
            input_ids[batch_index, left_padding:] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )
            attention_mask[batch_index, left_padding:] = 1
        completion_length = len(completion_ids)
        if completion_length:
            completion_mask[batch_index, :completion_length] = True
            predictor_positions[batch_index, :completion_length] = torch.arange(
                left_padding + len(prompt_ids) - 1,
                left_padding + len(prompt_ids) + completion_length - 1,
                device=device,
            )
            completion_tokens[batch_index, :completion_length] = torch.tensor(
                completion_ids,
                dtype=torch.long,
                device=device,
            )

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(
        predictor_positions
    )
    selected_logits = outputs.logits[batch_indices, predictor_positions]
    logprobs = functional.log_softmax(selected_logits, dim=-1).gather(
        -1,
        completion_tokens.unsqueeze(-1),
    ).squeeze(-1)
    return logprobs.masked_fill(~completion_mask, 0.0), completion_mask
