# GRPO vLLM Generation Design

Date: 2026-06-30

## Goal

Use vLLM for rollout generation during MACORAG GRPO training while preserving the current custom trainer, RAG loop, reward functions, logging, and LoRA-only training workflow.

The intended runtime split is:

- Trainer process: trains the PEFT/LoRA policy with `forward -> logprob -> GRPO loss -> backward -> optimizer.step()`.
- vLLM server: generates query, evidence-update, and answer completions for rollouts.
- After each real optimizer step, the trainer sends updated LoRA parameters to the vLLM server so the next rollout uses the latest policy adapter.

The design follows the vLLM/TRL training pattern described in:

- https://vllm.hyper.ai/docs/training/trl/
- https://huggingface.co/docs/trl/main/en/vllm_integration

## Non-Goals

- Do not migrate MACORAG to TRL `GRPOTrainer`.
- Do not rewrite the RAG loop, LinearRAG retrieval, reward functions, or output artifact layout.
- Do not switch from LoRA training to full-model fine-tuning.
- Do not silently fall back to Hugging Face generation when vLLM is explicitly enabled.

## Current State

The current GRPO entrypoint is `src/rl_training/train_grpo_macorag.py`.

The current rollout path uses `HFSharedPolicy` in `src/rl_training/policy.py`:

1. Build a role-specific prompt for `query_retriever`, `evidence_updater`, or `answer_generator`.
2. Generate with the local HF/PEFT training model.
3. Compute `old_logprobs` with the same local policy model.
4. Store the action in the rollout trace.

Training then computes current policy logprobs, reference logprobs, GRPO loss, backward, and optimizer step in the trainer process.

This design keeps that training path intact and replaces only the generation backend used during rollout.

## Architecture

Add a vLLM-backed policy implementation that satisfies the same interface used by `RAGLoopExecutor`.

At runtime:

1. Start a vLLM server on GPU0 with the base model and the initial SFT LoRA adapter.
2. Start the GRPO trainer on the configured training GPU or GPUs.
3. The trainer builds role-specific prompts locally and sends generation requests to the vLLM server.
4. The trainer keeps LinearRAG retrieval local, so retrieval, parsing, and reward behavior stay unchanged.
5. The trainer computes `old_logprobs` locally against the HF/PEFT policy model for the exact generated completion.
6. The trainer computes current/ref logprobs and GRPO loss exactly as it does today.
7. After each optimizer step, the trainer synchronizes trainable LoRA parameters to the vLLM server.

This preserves MACORAG's multi-step RAG rollout shape while moving the expensive completion generation work to vLLM.

## Components

### `src/rl_training/policy.py`

Keep `HFSharedPolicy` for compatibility.

Add a vLLM-backed policy class with the same public behavior:

- Reuse the existing role-specific prompt builders.
- Reuse the local tokenizer chat template so prompt tokenization is consistent.
- Generate completions through a vLLM client instead of `model.generate`.
- Decode/store completions in `GeneratedAction`.
- Compute and store `old_logprobs` with the local trainer policy model.
- Record timing for pure vLLM generation.

The old HF path remains available through configuration for baseline comparisons and recovery.

### `src/rl_training/vllm_client.py`

Add a small adapter around TRL's vLLM training client.

Responsibilities:

- Connect to the configured vLLM host and port.
- Perform a health check before training begins.
- Send generation requests with temperature, top-p, top-k, and max completion length.
- Synchronize trainable LoRA parameters after optimizer steps.
- Fail clearly if the installed TRL/vLLM versions do not support the required parameter update API.

The synchronization path must collect only parameters where `requires_grad=True`. In this project that means LoRA adapter parameters, not the frozen base model.

### `src/rl_training/train_grpo_macorag.py`

Extend the main loop without changing the GRPO loss semantics:

- Build either `HFSharedPolicy` or the vLLM-backed policy based on config.
- Validate that trainer GPUs and vLLM GPUs do not overlap when vLLM generation is enabled.
- Call LoRA synchronization only when `optimizer.step()` actually runs.
- Record timing for generation, retrieval, reward, backward, optimizer, sync, and total sample time.
- Include vLLM settings in `train_meta.json`.

### `config/train_grpo.yml`

Add vLLM generation controls:

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

The trainer continues to use `gpu_indices` for training placement. For the current requested layout, trainer GPU remains controlled by `gpu_indices`, and vLLM defaults to GPU0.

### Scripts

Keep `scripts/run_train_grpo.sh` focused on the trainer.

Add a separate launcher such as `scripts/run_grpo_vllm_server.sh`:

- Exports `CUDA_VISIBLE_DEVICES` from `vllm_gpu_indices`.
- Starts the TRL/vLLM server with the configured model, LoRA adapter, host, port, tensor parallel size, max model length, dtype, and GPU memory utilization.
- Uses the repo-local `macorag` environment.

Keeping the server as a separate process makes failures explicit and avoids mixing vLLM server lifecycle with `torchrun`.

## Data Flow

1. vLLM server starts on GPU0 and loads the base model plus initial LoRA adapter.
2. Trainer starts on the configured training GPU and loads the same base model, the same initial LoRA adapter, and the frozen reference model.
3. For each sample and group rollout:
   - The trainer builds the query prompt and asks vLLM to generate the query action.
   - The trainer runs LinearRAG retrieval locally.
   - The trainer builds the evidence-update prompt and asks vLLM to generate the update action.
   - The trainer builds the answer prompt and asks vLLM to generate the answer action.
   - The existing parser, reward code, and advantage normalization run unchanged.
4. The trainer computes old/current/reference logprobs locally for the generated token ids.
5. The trainer executes GRPO loss, backward, and optimizer step.
6. If an optimizer step occurred, the trainer sends updated trainable LoRA parameters to vLLM.
7. Logs and checkpoints are written in the existing output directory layout.

## Performance Requirement

The change is expected to improve generation efficiency during GRPO because vLLM is used for completion generation instead of HF `model.generate`.

The implementation must make this measurable rather than implicit:

- Add per-sample `time_vllm_generate_seconds`.
- Keep `time_rollout_seconds`.
- Add or preserve separate timing for retrieval, reward, backward, optimizer step, and weight sync.
- Provide a smoke comparison path using `use_vllm_generation: false` versus `true` on the same small sample set.

Acceptance should be based on the generation portion. If retrieval dominates a sample, total sample time may not drop as much as pure generation time. The logs must make that distinction visible.

## Error Handling

When `use_vllm_generation: true`:

- If the vLLM server is unreachable, fail before the first training sample.
- If TRL/vLLM does not expose the required hot-update API, fail with a dependency/version message.
- If trainer GPUs overlap with vLLM GPUs, fail before model loading.
- If LoRA parameter names cannot be matched for synchronization, fail after reporting the missing names.
- If a vLLM generation call times out, log the current sample identity and fail instead of silently switching to HF generation.

When `use_vllm_generation: false`, the current HF generation path remains the baseline and compatibility path.

## Dependency Strategy

The current `macorag` environment already has `vllm==0.8.3`; `trl` still needs to be installed.

Implementation should:

1. Inspect installed `torch`, `transformers`, `peft`, and `vllm` versions.
2. Install a TRL version compatible with the current vLLM stack if possible.
3. Avoid upgrading vLLM unless the required TRL client API cannot work with `vllm==0.8.3`.
4. If a vLLM upgrade is required, report the exact reason and preserve existing MACORAG behavior with a smoke test after installation.

This avoids breaking existing model serving or evaluation workflows unnecessarily.

## Testing

Add focused tests for:

- Config parsing for the new vLLM keys.
- Trainer/vLLM GPU overlap validation.
- vLLM policy trace creation without using local `model.generate`.
- Collection of trainable LoRA parameters only.
- Failure behavior when vLLM is enabled but the server/client is unavailable.
- Timing fields in emitted metrics.

Run existing RL tests after implementation, especially `tests/test_rl_training.py`.

Add a smoke run:

- `max_samples=1`
- `group_size=1`
- `max_steps=1`
- vLLM server on GPU0
- trainer on the configured training GPU

The smoke run must complete one rollout, one optimizer step, one LoRA sync, and write `train_metrics.jsonl`, `train_events.jsonl`, and `train_meta.json`.

## Acceptance Criteria

- Existing HF generation mode still works when `use_vllm_generation: false`.
- vLLM generation mode fails fast if the server is missing or incompatible.
- vLLM generation mode completes a one-sample GRPO smoke run.
- The trainer synchronizes only LoRA trainable parameters after real optimizer steps.
- The next rollout after a sync uses the updated vLLM-side adapter.
- Metrics include generation, retrieval, reward, backward, optimizer, sync, and total timing.
- A small A/B timing comparison can show whether vLLM reduced the generation portion of rollout time.

