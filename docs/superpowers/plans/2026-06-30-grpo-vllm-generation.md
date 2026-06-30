# GRPO vLLM Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move MACORAG GRPO rollout generation to a vLLM server while preserving the existing custom GRPO trainer, RAG loop, reward code, logs, checkpoints, and LoRA-only training.

**Architecture:** Keep the current trainer as the source of truth for policy optimization and logprob computation. Add a vLLM-backed `SharedPolicy` that sends prompt completions to a TRL/vLLM server, computes `old_logprobs` locally, and synchronizes trainable LoRA parameters after real optimizer steps. Keep the existing HF generation path behind `use_vllm_generation: false` for baseline and compatibility.

**Tech Stack:** Python 3.9, PyTorch, Transformers, PEFT/LoRA, TRL vLLM training client/server, vLLM, PyYAML, pytest, existing MACORAG `rag` and `rl_training` packages.

## Global Constraints

- Do not migrate MACORAG to TRL `GRPOTrainer`.
- Do not rewrite the RAG loop, LinearRAG retrieval, reward functions, or output artifact layout.
- Do not switch from LoRA training to full-model fine-tuning.
- Do not silently fall back to Hugging Face generation when vLLM is explicitly enabled.
- Default vLLM server GPU is GPU0 via `vllm_gpu_indices: "0"`.
- Trainer GPU placement remains controlled by `gpu_indices`.
- Synchronize only parameters where `requires_grad=True`; in this workflow that means LoRA adapter parameters.
- If `use_vllm_generation: true`, fail fast when the server is unreachable, versions are incompatible, GPU placement overlaps, or LoRA sync cannot be performed.
- Preserve `use_vllm_generation: false` as the existing HF generation baseline.
- Keep `scripts/` shell-only; Python logic belongs in `src/`.

---

## File Structure

- Modify `src/rl_training/config.py`: add vLLM config defaults, parser arguments, and strict YAML key support.
- Create `src/rl_training/vllm_client.py`: isolate optional TRL/vLLM imports, generation calls, health checks, trainable-parameter collection, and LoRA parameter sync.
- Modify `src/rl_training/policy.py`: add a vLLM-backed policy class while preserving `HFSharedPolicy` and `sequence_logprobs`.
- Modify `src/rl_training/train_grpo_macorag.py`: choose policy backend, validate GPU placement, accumulate generation/retrieval/sync timing, and sync LoRA after optimizer steps.
- Modify `config/train_grpo.yml`: enable and configure vLLM generation defaults.
- Create `scripts/run_grpo_vllm_server.sh`: start the TRL/vLLM server on `vllm_gpu_indices`.
- Modify `scripts/run_train_grpo.sh`: keep trainer launch on `gpu_indices` and leave vLLM server lifecycle separate.
- Modify `tests/test_rl_training.py`: cover config, GPU validation, vLLM policy behavior, LoRA parameter collection, sync timing, and launcher content.

---

### Task 1: Add vLLM Configuration and GPU Validation

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: config attributes `use_vllm_generation: bool`, `vllm_host: str`, `vllm_port: int`, `vllm_gpu_indices: str`, `vllm_tensor_parallel_size: int`, `vllm_gpu_memory_utilization: float`, `vllm_max_model_len: int`, `vllm_dtype: str`, `vllm_sync_after_step: bool`, `vllm_sync_trainable_only: bool`, `vllm_timeout_seconds: float`.
- Produces: `_parse_gpu_indices(value: str | int | None) -> set[str]`.
- Produces: `_validate_vllm_gpu_placement(args: Any) -> None`.
- Consumes: existing `parse_args()`.

- [ ] **Step 1: Write failing config parse test**

Append this test to `tests/test_rl_training.py`:

```python
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
                "vllm_gpu_memory_utilization: 0.70",
                "vllm_max_model_len: 4608",
                'vllm_dtype: "auto"',
                "vllm_sync_after_step: true",
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
    assert args.vllm_gpu_memory_utilization == 0.70
    assert args.vllm_max_model_len == 4608
    assert args.vllm_dtype == "auto"
    assert args.vllm_sync_after_step is True
    assert args.vllm_sync_trainable_only is True
    assert args.vllm_timeout_seconds == 90
    assert args.gpu_indices == "1"
```

- [ ] **Step 2: Write failing GPU overlap tests**

Add imports and tests to `tests/test_rl_training.py`:

```python
from argparse import Namespace

from rl_training.train_grpo_macorag import _parse_gpu_indices
from rl_training.train_grpo_macorag import _validate_vllm_gpu_placement
```

```python
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
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
pytest -q tests/test_rl_training.py::test_parse_args_loads_vllm_generation_config tests/test_rl_training.py::test_parse_gpu_indices_normalizes_comma_lists tests/test_rl_training.py::test_validate_vllm_gpu_placement_rejects_overlap tests/test_rl_training.py::test_validate_vllm_gpu_placement_allows_separate_gpus
```

Expected: FAIL because the new config keys and GPU validation helpers do not exist yet.

- [ ] **Step 4: Add config defaults and parser arguments**

In `src/rl_training/config.py`, add these keys to `DEFAULT_ARG_VALUES`:

```python
    "use_vllm_generation": False,
    "vllm_host": "127.0.0.1",
    "vllm_port": 8000,
    "vllm_gpu_indices": "0",
    "vllm_tensor_parallel_size": 1,
    "vllm_gpu_memory_utilization": 0.75,
    "vllm_max_model_len": 4608,
    "vllm_dtype": "auto",
    "vllm_sync_after_step": True,
    "vllm_sync_trainable_only": True,
    "vllm_timeout_seconds": 120.0,
```

In `_build_parser()`, add:

```python
    parser.add_argument("--use-vllm-generation", action=BooleanOptionalAction, default=defaults["use_vllm_generation"])
    parser.add_argument("--vllm-host", default=defaults["vllm_host"])
    parser.add_argument("--vllm-port", type=int, default=defaults["vllm_port"])
    parser.add_argument("--vllm-gpu-indices", default=defaults["vllm_gpu_indices"])
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=defaults["vllm_tensor_parallel_size"])
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=defaults["vllm_gpu_memory_utilization"])
    parser.add_argument("--vllm-max-model-len", type=int, default=defaults["vllm_max_model_len"])
    parser.add_argument("--vllm-dtype", default=defaults["vllm_dtype"])
    parser.add_argument("--vllm-sync-after-step", action=BooleanOptionalAction, default=defaults["vllm_sync_after_step"])
    parser.add_argument("--vllm-sync-trainable-only", action=BooleanOptionalAction, default=defaults["vllm_sync_trainable_only"])
    parser.add_argument("--vllm-timeout-seconds", type=float, default=defaults["vllm_timeout_seconds"])
```

- [ ] **Step 5: Add GPU validation helpers**

In `src/rl_training/train_grpo_macorag.py`, add near `_configure_visible_gpus()`:

```python
def _parse_gpu_indices(value: str | int | None) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def _validate_vllm_gpu_placement(args: Any) -> None:
    if not getattr(args, "use_vllm_generation", False):
        return
    trainer_gpus = _parse_gpu_indices(getattr(args, "gpu_indices", None))
    if not trainer_gpus:
        trainer_gpus = _parse_gpu_indices(getattr(args, "gpu_index", None))
    vllm_gpus = _parse_gpu_indices(getattr(args, "vllm_gpu_indices", None))
    overlap = trainer_gpus & vllm_gpus
    if overlap:
        raise SystemExit(
            "vLLM GPU overlap detected: trainer gpu_indices="
            f"{sorted(trainer_gpus)} and vllm_gpu_indices={sorted(vllm_gpus)} share {sorted(overlap)}. "
            "Use separate GPUs for trainer and vLLM generation."
        )
```

Call it in `main()` immediately after `args = parse_args()` and before `_configure_visible_gpus(args)`:

```python
    _validate_vllm_gpu_placement(args)
```

- [ ] **Step 6: Run task tests**

Run:

```bash
pytest -q tests/test_rl_training.py::test_parse_args_loads_vllm_generation_config tests/test_rl_training.py::test_parse_gpu_indices_normalizes_comma_lists tests/test_rl_training.py::test_validate_vllm_gpu_placement_rejects_overlap tests/test_rl_training.py::test_validate_vllm_gpu_placement_allows_separate_gpus
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rl_training/config.py src/rl_training/train_grpo_macorag.py tests/test_rl_training.py
git commit -m "feat: add grpo vllm config validation"
```

---

### Task 2: Add vLLM Client and LoRA Parameter Collection

**Files:**
- Create: `src/rl_training/vllm_client.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `VLLMGenerationClient`.
- Produces: `collect_trainable_named_parameters(model: Any) -> dict[str, Any]`.
- Produces: `VLLMGenerationClient.generate(prompt_token_ids: list[int], *, max_tokens: int, temperature: float, top_p: float, top_k: int) -> tuple[list[int], str]`.
- Produces: `VLLMGenerationClient.sync_trainable_parameters(model: Any) -> float`.
- Consumes: trainable LoRA parameters from the PEFT policy model.

- [ ] **Step 1: Write failing trainable parameter collection test**

Append to `tests/test_rl_training.py`:

```python
from rl_training.vllm_client import collect_trainable_named_parameters
```

```python
class _TinyParamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.lora_a = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=True)
        self.lora_b = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=True)


def test_collect_trainable_named_parameters_returns_only_trainable_cpu_tensors() -> None:
    model = _TinyParamModel()

    params = collect_trainable_named_parameters(model)

    assert sorted(params) == ["lora_a", "lora_b"]
    assert all(not tensor.requires_grad for tensor in params.values())
    assert all(tensor.device.type == "cpu" for tensor in params.values())
    assert params["lora_a"].item() == 2.0
    assert params["lora_b"].item() == 3.0
```

- [ ] **Step 2: Write failing client sync test using a fake backend**

Append:

```python
class _FakeTRLClient:
    def __init__(self) -> None:
        self.updated: list[tuple[str, torch.Tensor]] = []
        self.health_checked = False

    def check_server(self) -> None:
        self.health_checked = True

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        self.updated.append((name, weights))


def test_vllm_generation_client_syncs_trainable_parameters() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    model = _TinyParamModel()

    elapsed = client.sync_trainable_parameters(model)

    assert elapsed >= 0.0
    assert [name for name, _ in backend.updated] == ["lora_a", "lora_b"]
    assert all(tensor.device.type == "cpu" for _, tensor in backend.updated)
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
pytest -q tests/test_rl_training.py::test_collect_trainable_named_parameters_returns_only_trainable_cpu_tensors tests/test_rl_training.py::test_vllm_generation_client_syncs_trainable_parameters
```

Expected: FAIL because `rl_training.vllm_client` does not exist.

- [ ] **Step 4: Create vLLM client module**

Create `src/rl_training/vllm_client.py` with:

```python
from __future__ import annotations

import time
from typing import Any


def collect_trainable_named_parameters(model: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if getattr(parameter, "requires_grad", False):
            params[name] = parameter.detach().float().cpu()
    return params


class VLLMGenerationClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_seconds: float,
        backend: Any | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self._backend = backend

    @property
    def backend(self) -> Any:
        if self._backend is None:
            try:
                from trl.extras.vllm_client import VLLMClient
            except ModuleNotFoundError as exc:
                raise SystemExit(
                    "TRL is required for vLLM GRPO generation. Install a TRL version that provides "
                    "trl.extras.vllm_client.VLLMClient in the macorag environment."
                ) from exc
            except ImportError as exc:
                raise SystemExit(
                    "Installed TRL does not expose trl.extras.vllm_client.VLLMClient. "
                    "Install a TRL/vLLM combination with vLLM training support."
                ) from exc
            self._backend = VLLMClient(host=self.host, port=self.port, connection_timeout=self.timeout_seconds)
        return self._backend

    def check_server(self) -> None:
        checker = getattr(self.backend, "check_server", None)
        if checker is None:
            raise SystemExit("TRL VLLMClient is missing check_server(); installed TRL is incompatible.")
        checker()

    def generate(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> tuple[list[int], str]:
        generator = getattr(self.backend, "generate", None)
        if generator is None:
            raise SystemExit("TRL VLLMClient is missing generate(); installed TRL is incompatible.")
        outputs = generator(
            prompts=[prompt_token_ids],
            n=1,
            repetition_penalty=1.0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        first = outputs[0]
        token_ids = list(getattr(first, "token_ids", None) or first.get("token_ids", []))
        text = str(getattr(first, "text", None) or first.get("text", ""))
        return token_ids, text

    def sync_trainable_parameters(self, model: Any) -> float:
        updater = getattr(self.backend, "update_named_param", None)
        if updater is None:
            raise SystemExit("TRL VLLMClient is missing update_named_param(); hot LoRA sync is unavailable.")
        start = time.perf_counter()
        for name, tensor in collect_trainable_named_parameters(model).items():
            updater(name, tensor)
        return time.perf_counter() - start
```

- [ ] **Step 5: Run task tests**

Run:

```bash
pytest -q tests/test_rl_training.py::test_collect_trainable_named_parameters_returns_only_trainable_cpu_tensors tests/test_rl_training.py::test_vllm_generation_client_syncs_trainable_parameters
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rl_training/vllm_client.py tests/test_rl_training.py
git commit -m "feat: add grpo vllm client wrapper"
```

---

### Task 3: Add vLLM-backed SharedPolicy

**Files:**
- Modify: `src/rl_training/policy.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `VLLMSharedPolicy`.
- Consumes: `VLLMGenerationClient.generate(...) -> tuple[list[int], str]`.
- Preserves: `GeneratedAction`, `RolloutTrace`, `sequence_logprobs`.

- [ ] **Step 1: Write failing vLLM policy trace test**

Append to `tests/test_rl_training.py`:

```python
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
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab_size, device=input_ids.device)
        return type("Output", (), {"logits": logits})


class _FakeVLLMClient:
    def __init__(self) -> None:
        self.prompts: list[list[int]] = []

    def generate(self, prompt_token_ids, *, max_tokens, temperature, top_p, top_k):
        self.prompts.append(list(prompt_token_ids))
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
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest -q tests/test_rl_training.py::test_vllm_shared_policy_generates_and_records_trace
```

Expected: FAIL because `VLLMSharedPolicy` does not exist.

- [ ] **Step 3: Implement `VLLMSharedPolicy`**

In `src/rl_training/policy.py`, add after `HFSharedPolicy`:

```python
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
            prompt_ids,
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
            )
        )
        return response
```

- [ ] **Step 4: Run task test**

Run:

```bash
pytest -q tests/test_rl_training.py::test_vllm_shared_policy_generates_and_records_trace
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rl_training/policy.py tests/test_rl_training.py
git commit -m "feat: add vllm shared policy"
```

---

### Task 4: Wire vLLM Generation into the GRPO Trainer Loop

**Files:**
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `_build_policy(args: Any, raw_policy_model: Any, tokenizer: Any) -> HFSharedPolicy | VLLMSharedPolicy`.
- Produces: `_sync_vllm_after_optimizer_step(policy: Any, raw_policy_model: Any, args: Any) -> float`.
- Modifies: `_rollout_group(...)` timing output to include `time_vllm_generate_seconds`.
- Modifies: `_train_on_rollouts(...)` to return `did_optimizer_step: bool`.

- [ ] **Step 1: Write failing policy builder test**

Append:

```python
def test_build_policy_uses_hf_policy_when_vllm_disabled() -> None:
    from argparse import Namespace
    from rl_training.train_grpo_macorag import _build_policy

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
```

- [ ] **Step 2: Write failing optimizer-step metric test**

Modify `test_sequence_logprobs_keeps_only_completion_logits` only if needed to use existing fixtures unchanged. Add a focused fake rollout test:

```python
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
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
pytest -q tests/test_rl_training.py::test_build_policy_uses_hf_policy_when_vllm_disabled tests/test_rl_training.py::test_train_on_rollouts_reports_optimizer_step_flag
```

Expected: FAIL because `_build_policy` and `did_optimizer_step` do not exist.

- [ ] **Step 4: Add policy builder and sync helper**

In `src/rl_training/train_grpo_macorag.py`, update imports:

```python
from .policy import HFSharedPolicy, VLLMSharedPolicy, sequence_logprobs
from .vllm_client import VLLMGenerationClient
```

Add:

```python
def _build_policy(args: Any, raw_policy_model: Any, tokenizer: Any) -> HFSharedPolicy:
    common = {
        "model": raw_policy_model,
        "tokenizer": tokenizer,
        "system_prompt": args.system_prompt,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if not args.use_vllm_generation:
        return HFSharedPolicy(**common)
    client = VLLMGenerationClient(
        host=args.vllm_host,
        port=args.vllm_port,
        timeout_seconds=args.vllm_timeout_seconds,
    )
    client.check_server()
    return VLLMSharedPolicy(vllm_client=client, **common)


def _sync_vllm_after_optimizer_step(policy: Any, raw_policy_model: Any, args: Any) -> float:
    if not getattr(args, "use_vllm_generation", False):
        return 0.0
    if not getattr(args, "vllm_sync_after_step", True):
        return 0.0
    client = getattr(policy, "vllm_client", None)
    if client is None:
        raise SystemExit("vLLM generation is enabled but policy has no vLLM client.")
    return float(client.sync_trainable_parameters(raw_policy_model))
```

- [ ] **Step 5: Report optimizer-step flag**

In `_train_on_rollouts()`, add `"did_optimizer_step": False` to the no-action return dict.

After `optimizer.step()` branch, set a local flag:

```python
    did_optimizer_step = False
    if should_step:
        optimizer_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        did_optimizer_step = True
        time_optimizer_step_seconds += time.perf_counter() - optimizer_start
```

Add to the return dict:

```python
        "did_optimizer_step": did_optimizer_step,
```

- [ ] **Step 6: Add vLLM generation timing to rollout timing**

In `_rollout_group()`, after each `executor.run(...)`, add:

```python
        time_vllm_generate_seconds = float(getattr(policy, "timing", {}).get("time_vllm_generate_seconds", 0.0))
```

Initialize before the loop:

```python
    time_vllm_generate_seconds = 0.0
```

Return it in the timing dict:

```python
        "time_vllm_generate_seconds": time_vllm_generate_seconds,
```

- [ ] **Step 7: Use `_build_policy` in `main()`**

Replace the direct `HFSharedPolicy(...)` construction with:

```python
    policy = _build_policy(args, raw_policy_model, tokenizer)
```

- [ ] **Step 8: Sync after real optimizer steps and log timing**

After `_train_on_rollouts(...)` in `main()`, add:

```python
                time_weight_sync_seconds = 0.0
                if metrics.get("did_optimizer_step"):
                    time_weight_sync_seconds = _sync_vllm_after_optimizer_step(policy, raw_policy_model, args)
```

Add to the metrics payload:

```python
                        "time_vllm_generate_seconds": rollout_timing.get("time_vllm_generate_seconds", 0.0),
                        "time_weight_sync_seconds": time_weight_sync_seconds,
```

Add to the sample complete event:

```python
                        time_vllm_generate_seconds=payload["time_vllm_generate_seconds"],
                        time_weight_sync_seconds=payload["time_weight_sync_seconds"],
```

- [ ] **Step 9: Add vLLM metadata**

In `train_meta.json`, add:

```python
                "use_vllm_generation": args.use_vllm_generation,
                "vllm_host": args.vllm_host,
                "vllm_port": args.vllm_port,
                "vllm_gpu_indices": args.vllm_gpu_indices,
                "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
                "vllm_max_model_len": args.vllm_max_model_len,
```

- [ ] **Step 10: Run task tests**

Run:

```bash
pytest -q tests/test_rl_training.py::test_build_policy_uses_hf_policy_when_vllm_disabled tests/test_rl_training.py::test_train_on_rollouts_reports_optimizer_step_flag tests/test_rl_training.py::test_vllm_shared_policy_generates_and_records_trace
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/rl_training/train_grpo_macorag.py tests/test_rl_training.py
git commit -m "feat: wire vllm generation into grpo trainer"
```

---

### Task 5: Add vLLM Config Defaults and Server Launcher

**Files:**
- Modify: `config/train_grpo.yml`
- Create: `scripts/run_grpo_vllm_server.sh`
- Modify: `tests/test_rl_training.py`

**Interfaces:**
- Consumes: `config/train_grpo.yml`.
- Produces: shell launcher `scripts/run_grpo_vllm_server.sh`.

- [ ] **Step 1: Write failing launcher test**

Append:

```python
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
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest -q tests/test_rl_training.py::test_run_grpo_vllm_server_script_uses_vllm_gpu_and_trl_server
```

Expected: FAIL because `scripts/run_grpo_vllm_server.sh` does not exist.

- [ ] **Step 3: Update GRPO config**

Add these keys to `config/train_grpo.yml`:

```yaml
use_vllm_generation: true
vllm_host: "127.0.0.1"
vllm_port: 8000
vllm_gpu_indices: "0"
vllm_tensor_parallel_size: 1
vllm_gpu_memory_utilization: 0.75
vllm_max_model_len: 4608
vllm_dtype: "auto"
vllm_sync_after_step: true
vllm_sync_trainable_only: true
vllm_timeout_seconds: 120
```

Ensure `gpu_indices` remains separate from `vllm_gpu_indices`; for the requested layout, keep trainer `gpu_indices: "1"` and vLLM `vllm_gpu_indices: "0"`.

- [ ] **Step 4: Create vLLM server launcher**

Create `scripts/run_grpo_vllm_server.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/train_grpo.yml}"

read -r YAML_MODEL_PATH YAML_ADAPTER_PATH YAML_HOST YAML_PORT YAML_VLLM_GPU_INDICES YAML_TP YAML_GPU_UTIL YAML_MAX_LEN YAML_DTYPE < <(
  "${PYTHON:-python}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(
    config.get("model_path", "model/Qwen2.5-3B-Instruct"),
    config.get("sft_adapter_path", ""),
    config.get("vllm_host", "127.0.0.1"),
    int(config.get("vllm_port", 8000)),
    str(config.get("vllm_gpu_indices", "0")),
    int(config.get("vllm_tensor_parallel_size", 1)),
    float(config.get("vllm_gpu_memory_utilization", 0.75)),
    int(config.get("vllm_max_model_len", 4608)),
    config.get("vllm_dtype", "auto"),
)
PY
)

export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"

exec trl vllm-serve \
  --model "${YAML_MODEL_PATH}" \
  --host "${YAML_HOST}" \
  --port "${YAML_PORT}" \
  --tensor-parallel-size "${YAML_TP}" \
  --gpu-memory-utilization "${YAML_GPU_UTIL}" \
  --max-model-len "${YAML_MAX_LEN}" \
  --dtype "${YAML_DTYPE}" \
  --enable-lora \
  --lora-modules "macorag=${YAML_ADAPTER_PATH}" \
  "$@"
```

- [ ] **Step 5: Make launcher executable**

Run:

```bash
chmod +x scripts/run_grpo_vllm_server.sh
```

Expected: no output.

- [ ] **Step 6: Run launcher test**

Run:

```bash
pytest -q tests/test_rl_training.py::test_run_grpo_vllm_server_script_uses_vllm_gpu_and_trl_server
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add config/train_grpo.yml scripts/run_grpo_vllm_server.sh tests/test_rl_training.py
git commit -m "feat: add grpo vllm server launcher"
```

---

### Task 6: Install and Verify TRL/vLLM Compatibility in `macorag`

**Files:**
- Modify only if needed: `docs/superpowers/plans/2026-06-30-grpo-vllm-generation.md` with actual version evidence if package compatibility forces a plan adjustment.

**Interfaces:**
- Consumes: local `/data/conda/envs/macorag` environment.
- Produces: installed TRL package exposing `trl.extras.vllm_client.VLLMClient` and CLI `trl vllm-serve`.

- [ ] **Step 1: Capture current versions**

Run:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH python -c "import importlib.util, torch, transformers, peft, vllm; print('torch', torch.__version__, torch.version.cuda); print('transformers', transformers.__version__); print('peft', peft.__version__); print('vllm', vllm.__version__); print('trl', 'OK' if importlib.util.find_spec('trl') else 'MISSING')"
```

Expected before install: `trl MISSING` or an installed TRL version that still needs API verification.

- [ ] **Step 2: Install TRL without forcing vLLM upgrade first**

Run:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH pip install "trl"
```

Expected: install completes. If pip attempts to replace `torch`, `transformers`, or `vllm` with incompatible versions, stop and inspect the resolver output before proceeding.

- [ ] **Step 3: Verify TRL training API**

Run:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH python -c "from trl.extras.vllm_client import VLLMClient; import inspect; print(VLLMClient); print(hasattr(VLLMClient, 'update_named_param') or 'update_named_param' in dir(VLLMClient)); print('generate' in dir(VLLMClient));"
```

Expected: import succeeds and both update/generate capabilities are visible. If import fails, identify the installed TRL version with `pip show trl` and choose the nearest compatible TRL/vLLM pair before changing code.

- [ ] **Step 4: Verify CLI**

Run:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH trl vllm-serve --help | head -80
```

Expected: help text includes `vllm-serve`, `--model`, `--host`, and `--port`.

- [ ] **Step 5: Run non-GPU unit tests after install**

Run:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH pytest -q tests/test_rl_training.py
```

Expected: PASS.

- [ ] **Step 6: Commit only if files changed**

If no repository files changed, do not commit. If compatibility required a code or plan adjustment, commit the changed files:

```bash
git add <changed-files>
git commit -m "chore: align grpo vllm dependency compatibility"
```

---

### Task 7: Add Smoke Verification Commands and Run Tests

**Files:**
- Modify if needed: `tests/test_rl_training.py`
- Runtime artifacts only: `outputs/grpo_qwen2.5-3b_*`

**Interfaces:**
- Consumes: implemented vLLM server launcher and trainer.
- Produces: smoke run evidence in `train_metrics.jsonl`, `train_events.jsonl`, `train_meta.json`.

- [ ] **Step 1: Run complete RL unit suite**

Run:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH pytest -q tests/test_rl_training.py
```

Expected: PASS.

- [ ] **Step 2: Run broader affected tests**

Run:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH pytest -q tests/test_rl_training.py tests/test_rag.py tests/test_sft_training.py tests/test_retrieval_env.py
```

Expected: PASS. If a listed test file is absent, run `rg --files tests` and replace the command with the existing affected test files.

- [ ] **Step 3: Start vLLM server on GPU0**

Run in a terminal or tmux pane:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH CONFIG_PATH=config/train_grpo.yml bash scripts/run_grpo_vllm_server.sh
```

Expected: server starts on GPU0 and listens on `127.0.0.1:8000`. If GPU0 is occupied, lower `vllm_gpu_memory_utilization` or stop the conflicting process before retrying.

- [ ] **Step 4: Run one-step vLLM GRPO smoke**

Run in another terminal:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH CONFIG_PATH=config/train_grpo.yml bash scripts/run_train_grpo.sh --max-samples 1 --group-size 1 --max-steps 1 --disable-tqdm true
```

Expected: command completes one sample, one optimizer step, and one vLLM sync. Output directory is printed as `outputs/grpo_qwen2.5-3b_<timestamp>`.

- [ ] **Step 5: Verify smoke artifacts**

Replace `<RUN_DIR>` with the printed output directory and run:

```bash
python -c "import json, pathlib; run=pathlib.Path('<RUN_DIR>'); metrics=[json.loads(x) for x in (run/'train_metrics.jsonl').read_text().splitlines()]; meta=json.loads((run/'train_meta.json').read_text()); print(metrics[-1]['time_vllm_generate_seconds']); print(metrics[-1]['time_weight_sync_seconds']); print(meta['use_vllm_generation']);"
```

Expected:

- First printed value is a non-negative float.
- Second printed value is a non-negative float.
- Third printed value is `True`.

- [ ] **Step 6: Run HF baseline smoke for timing comparison**

Run:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH CONFIG_PATH=config/train_grpo.yml bash scripts/run_train_grpo.sh --use-vllm-generation false --max-samples 1 --group-size 1 --max-steps 1 --disable-tqdm true
```

Expected: command completes through the HF path. Compare `time_rollout_seconds` from this run with the vLLM run. Report both values and separately report `time_vllm_generate_seconds` for the vLLM run.

- [ ] **Step 7: Final commit if smoke-related fixes were needed**

If no files changed, do not commit. If fixes were required:

```bash
git add <changed-files>
git commit -m "fix: stabilize grpo vllm smoke path"
```

---

## Self-Review Checklist

- Spec coverage: Tasks cover config, vLLM client, vLLM policy, trainer wiring, LoRA sync, server launcher, dependency install, tests, and smoke validation.
- Placeholder scan: Checked for banned placeholder patterns; none remain.
- Type consistency: `VLLMGenerationClient`, `VLLMSharedPolicy`, `_build_policy`, `_sync_vllm_after_optimizer_step`, `_parse_gpu_indices`, and `_validate_vllm_gpu_placement` are introduced before later tasks consume them.
- Compatibility: HF generation remains available through `use_vllm_generation: false`.
- Performance proof: Smoke validation requires separate generation and total rollout timing.
