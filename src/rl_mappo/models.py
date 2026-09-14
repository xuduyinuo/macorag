from __future__ import annotations

import hashlib
import inspect
import re
from pathlib import Path
from typing import Any

from .protocol import OUTPUT_CONTRACT_MARKER, SYSTEM_PROMPTS
from .mappo_types import AgentRole


FINAL_ANSWER_GUIDED_REGEX = (
    r'<answer>\{"can_answer":true,"answer":"([^"\\]|\\.)+",'
    r'"rationale":"([^"\\]|\\.)*"\}</answer>'
)


class CentralizedCritic:
    """Small trainable critic over the joint RAG state.

    Stable feature hashing keeps the critic independent of the actor tokenizer
    and avoids loading a second 7B backbone. It sees evidence, observations,
    retrieval history, round progress and acting role; the actor sees only its
    role-specific prompt.
    """

    def __init__(self, *, hash_buckets: int, embedding_dim: int, hidden_dim: int, text_max_tokens: int, device: Any) -> None:
        import torch
        self.torch = torch
        self.hash_buckets = hash_buckets
        self.text_max_tokens = text_max_tokens
        self.module = torch.nn.ModuleDict({
            "embedding": torch.nn.EmbeddingBag(hash_buckets, embedding_dim, mode="mean"),
            "network": torch.nn.Sequential(
                torch.nn.Linear(embedding_dim + 8, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, 1),
            ),
        }).to(device)
        self.device = device

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state)

    @staticmethod
    def _text(state: dict[str, Any]) -> str:
        pieces = [str(state.get("question") or ""), str(state.get("sub_goal") or "")]
        for field in ("evidence", "retrieval_history"):
            for item in state.get(field, []) or []:
                pieces.extend([str(item.get("title") or ""), str(item.get("text") or ""), str(item.get("query") or "")])
        for item in (state.get("observation") or {}).get("passages", []) or []:
            pieces.extend([str(item.get("title") or ""), str(item.get("text") or "")])
        return " ".join(pieces)

    def _hash(self, token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big") % self.hash_buckets

    def _features(self, states: list[dict[str, Any]]) -> tuple[Any, Any, Any]:
        token_ids: list[int] = []
        offsets: list[int] = []
        numeric: list[list[float]] = []
        role_order = ("query_retriever", "evidence_updater", "answer_generator")
        for state in states:
            offsets.append(len(token_ids))
            tokens = re.findall(r"[\w]+", self._text(state).casefold())[:self.text_max_tokens]
            token_ids.extend(self._hash(token) for token in (tokens or ["<empty>"]))
            role = str(state.get("role") or "")
            observation = (state.get("observation") or {}).get("passages", []) or []
            evidence = state.get("evidence", []) or []
            history = state.get("retrieval_history", []) or []
            round_index = int(state.get("round_index", 0))
            max_rounds = max(1, int(state.get("max_rounds", 1)))
            numeric.append([
                float(role == role_order[0]), float(role == role_order[1]), float(role == role_order[2]),
                round_index / max_rounds, len(evidence) / 20.0, len(history) / max_rounds,
                len(observation) / 20.0, float(round_index + 1 == max_rounds),
            ])
        torch = self.torch
        return (
            torch.tensor(token_ids, dtype=torch.long, device=self.device),
            torch.tensor(offsets, dtype=torch.long, device=self.device),
            torch.tensor(numeric, dtype=torch.float32, device=self.device),
        )

    def __call__(self, states: list[dict[str, Any]]) -> Any:
        ids, offsets, numeric = self._features(states)
        embedded = self.module["embedding"](ids, offsets)
        return self.module["network"](self.torch.cat([embedded, numeric], dim=-1)).squeeze(-1)


class RoleConditionedActor:
    def __init__(self, model: Any, tokenizer: Any, *, max_prompt_length: int, max_completion_length: int, temperature: float, top_p: float, top_k: int, device: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.device = device
        self.policy_adapter_name: str | None = None
        self.reference_adapter_name: str | None = None
        forward_parameters = inspect.signature(self.model.forward).parameters
        self._supports_logits_to_keep = (
            "logits_to_keep" in forward_parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in forward_parameters.values()
            )
        )

    def encode_prompt(self, role: AgentRole, prompt: str) -> list[int]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[role]},
            {"role": "user", "content": prompt},
        ]
        ids = list(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
        if len(ids) <= self.max_prompt_length:
            return ids

        # A plain tail slice deletes the system role first. Preserve the exact
        # system-message prefix, then spend all remaining tokens on the tail,
        # where build_prompt deliberately places few-shots and the contract.
        system_ids = list(self.tokenizer.apply_chat_template(
            [messages[0]], add_generation_prompt=False, tokenize=True,
        ))
        head_length = min(len(system_ids), self.max_prompt_length)
        if OUTPUT_CONTRACT_MARKER in prompt and head_length < self.max_prompt_length:
            tail_length = self.max_prompt_length - head_length
            return [*ids[:head_length], *ids[-tail_length:]]
        # Defensive fallback for third-party/custom prompts without our marker.
        head_length = min(max(1, self.max_prompt_length // 4), head_length)
        return [*ids[:head_length], *ids[-(self.max_prompt_length - head_length):]]

    def generate(self, role: AgentRole, prompt: str) -> tuple[str, list[int], list[int], Any]:
        import torch
        prompt_ids = self.encode_prompt(role, prompt)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(inputs)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=inputs,
                    attention_mask=attention,
                    max_new_tokens=self.max_completion_length,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=self.top_p if self.temperature > 0 else None,
                    top_k=self.top_k if self.temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            action_ids = output[0, len(prompt_ids):].tolist()
            eos = self.tokenizer.eos_token_id
            if eos in action_ids:
                action_ids = action_ids[:action_ids.index(eos) + 1]
            old_logprobs, _ = self.score_batch([(prompt_ids, action_ids)])
        finally:
            self.model.train(was_training)
        response = self.tokenizer.decode(action_ids, skip_special_tokens=True)
        return response, prompt_ids, action_ids, old_logprobs[0].detach().cpu()

    def score_batch(self, sequences: list[tuple[list[int], list[int]]]) -> tuple[Any, Any]:
        """Return completion token log-probabilities and token entropies."""
        import torch
        if not sequences:
            raise ValueError("score_batch requires at least one sequence")
        pad_id = int(self.tokenizer.pad_token_id)
        max_action = max(len(action) for _, action in sequences)
        rows_logp: list[Any] = []
        rows_entropy: list[Any] = []
        # Score separately to keep the 7B actor's full-vocabulary logits within
        # the same memory envelope as generation. MAPPO minibatches default to
        # one; larger values trade speed for activation memory.
        for prompt, action in sequences:
            full = prompt + action
            input_ids = torch.tensor([full], dtype=torch.long, device=self.device)
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "use_cache": False,
            }
            if self._supports_logits_to_keep:
                # We need prediction positions [prompt_len - 1, full_len - 2].
                # Keeping one extra tail position supplies exactly that slice
                # without materializing prompt-length vocabulary logits.
                forward_kwargs["logits_to_keep"] = len(action) + 1
            outputs = self.model(**forward_kwargs)
            start = 0 if self._supports_logits_to_keep else len(prompt) - 1
            completion_logits = outputs.logits[0, start:start + len(action)].float()
            targets = torch.tensor(action, dtype=torch.long, device=self.device)
            log_partition = torch.logsumexp(completion_logits, dim=-1)
            chosen = completion_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            token_logp = chosen - log_partition
            probabilities = torch.softmax(completion_logits, dim=-1)
            token_entropy = log_partition - (probabilities * completion_logits).sum(-1)
            padding = max_action - len(action)
            rows_logp.append(torch.nn.functional.pad(token_logp, (0, padding)))
            rows_entropy.append(torch.nn.functional.pad(token_entropy, (0, padding)))
        return torch.stack(rows_logp), torch.stack(rows_entropy)

    def score_batch_no_grad(self, sequences: list[tuple[list[int], list[int]]]) -> tuple[Any, Any]:
        """Score a fixed policy snapshot deterministically without retaining activations."""
        import torch
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logprobs, entropy = self.score_batch(sequences)
        finally:
            self.model.train(was_training)
        return logprobs.detach(), entropy.detach()

    def score_reference_batch_no_grad(
        self, sequences: list[tuple[list[int], list[int]]]
    ) -> tuple[Any, Any]:
        """Score fixed actions with the frozen initial SFT adapter."""
        if not self.policy_adapter_name or not self.reference_adapter_name:
            raise RuntimeError("SFT reference adapter is not configured")
        import torch
        was_training = self.model.training
        self.model.eval()
        try:
            self.model.set_adapter(self.reference_adapter_name)
            with torch.no_grad():
                logprobs, entropy = self.score_batch(sequences)
        finally:
            self.model.set_adapter(self.policy_adapter_name)
            # PEFT set_adapter() toggles trainability. Keep the fixed reference
            # frozen even after switching back to the policy adapter.
            for name, parameter in self.model.named_parameters():
                if f".{self.reference_adapter_name}." in name:
                    parameter.requires_grad_(False)
            self.model.train(was_training)
        return logprobs.detach(), entropy.detach()


class VLLMRoleConditionedActor(RoleConditionedActor):
    """Local trainable actor whose rollout actions come from a vLLM service.

    The server log-probabilities are retained on the transition initially for
    diagnostics. MAPPOTrainer replaces them with local-actor scores immediately
    before an update when ppo_old_logprob_source=local_actor. This avoids mixing
    BF16 vLLM log-probabilities with 4-bit learner log-probabilities in PPO ratios.
    """

    def __init__(self, *args: Any, vllm_client: Any, generation_seed: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vllm_client = vllm_client
        self.generation_seed = int(generation_seed)
        self.generation_counter = 0

    def generate(self, role: AgentRole, prompt: str) -> tuple[str, list[int], list[int], Any]:
        import torch
        prompt_ids = self.encode_prompt(role, prompt)
        seed = self.generation_seed + self.generation_counter
        self.generation_counter += 1
        output = self.vllm_client.generate(
            prompt_ids,
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=seed,
            guided_regex=(
                FINAL_ANSWER_GUIDED_REGEX
                if getattr(self, "force_final_answer_decoding", False)
                and role is AgentRole.ANSWER
                and "This is the final round." in prompt
                else None
            ),
        )
        response = output.text or self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        return (
            response,
            prompt_ids,
            output.token_ids,
            torch.tensor(output.token_logprobs, dtype=torch.float32),
        )

    def sync_generation(self, *, sync_root: Path, step: int, keep: int) -> Path:
        return self.vllm_client.sync_lora(
            self.model, self.tokenizer, sync_root=sync_root, step=step, keep=keep,
        )


def load_actor(config: Any, device: Any, *, vllm_client: Any | None = None) -> RoleConditionedActor:
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing MAPPO dependency: {exc.name}") from exc
    adapter = Path(config.resume_from_checkpoint or config.sft_adapter_path)
    adapter_dir = adapter / "actor" if (adapter / "actor").is_dir() else adapter
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else torch.float32
    kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": config.attn_implementation}
    if config.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        quantized_device = (
            int(device.index or 0)
            if getattr(device, "type", None) == "cuda"
            else str(device)
        )
        kwargs["device_map"] = {"": quantized_device}
    base = AutoModelForCausalLM.from_pretrained(config.model_path, **kwargs)
    base.config.use_cache = False
    if config.load_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=config.gradient_checkpointing)
    if (adapter_dir / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    else:
        lora = LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[x.strip() for x in config.lora_target_modules.split(",") if x.strip()],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base, lora)
    if not config.load_4bit:
        model.to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    actor_class = VLLMRoleConditionedActor if vllm_client is not None else RoleConditionedActor
    extra = (
        {"vllm_client": vllm_client, "generation_seed": config.seed}
        if vllm_client is not None else {}
    )
    reference_adapter_name = None
    if float(getattr(config, "reference_kl_beta", 0.0)) > 0:
        reference_dir = Path(config.sft_adapter_path)
        reference_dir = (
            reference_dir / "actor"
            if (reference_dir / "actor" / "adapter_config.json").is_file()
            else reference_dir
        )
        if not (reference_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"SFT reference adapter_config.json not found: {reference_dir}"
            )
        reference_adapter_name = "sft_reference"
        model.load_adapter(
            reference_dir,
            adapter_name=reference_adapter_name,
            is_trainable=False,
        )
        model.set_adapter("default")
        setattr(model, "_mappo_policy_adapter_name", "default")

    actor = actor_class(
        model, tokenizer, max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
        device=device, **extra,
    )
    if reference_adapter_name is not None:
        actor.policy_adapter_name = "default"
        actor.reference_adapter_name = reference_adapter_name
    actor.force_final_answer_decoding = bool(getattr(
        config, "force_final_answer_decoding", False,
    ))
    return actor
