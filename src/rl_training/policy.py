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
        self.timing: dict[str, float] = {"time_vllm_generate_seconds": 0.0}

    def reset_trace(self) -> None:
        super().reset_trace()
        self.timing = {"time_vllm_generate_seconds": 0.0}

    def generate(
        self,
        *,
        role: AgentRole,
        question: str,
        state: RAGState,
        observation: dict[str, Any] | None = None,
    ) -> str:
        import time
        import torch

        prompt = self._prompt_for(role=role, question=question, state=state, observation=observation)
        prompt_ids = self._encode_prompt(prompt)
        generate_start = time.perf_counter()
        completion, response = self.vllm_client.generate(
            self.tokenizer.decode(prompt_ids, skip_special_tokens=False),
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        self.timing["time_vllm_generate_seconds"] += time.perf_counter() - generate_start
        if self.tokenizer.eos_token_id in completion:
            completion = completion[: completion.index(self.tokenizer.eos_token_id) + 1]
        if not response:
            response = self.tokenizer.decode(completion, skip_special_tokens=True)
        device = next(self.model.parameters()).device
        with torch.no_grad():
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
