# RL BF16 and FlashAttention 2 Implementation Plan

> **For AI agent workers:** Required sub-skill: use superpowers:executing-plans to implement this plan inline. Track every step with the checkboxes below.

**Goal:** Enable explicit BF16 compute and Transformers FlashAttention 2 for the GRPO training-side policy/reference model while leaving the vLLM rollout backend and LoRA hot-sync protocol unchanged.

**Architecture:** Add one shared `attn_implementation` configuration field to the RL parser and active YAML, validate BF16/CUDA/FlashAttention compatibility before loading the 7B model, and pass the selected backend into `AutoModelForCausalLM.from_pretrained`. Reuse the proven SFT validation behavior so failure is explicit rather than silently falling back to SDPA.

**Technical stack:** Python 3.9, PyTorch 2.6, Transformers 4.57, PEFT/QLoRA, FlashAttention 2.8, pytest, RTX 4090.

---

### Task 1: Specify the RL acceleration contract

**Files:**
- Modify: `tests/test_rl_training.py`
- Modify: `config/train_grpo.yml`
- Modify: `src/rl_training/config.py`

- [x] Add failing tests that require the active RL config to resolve to `bf16=True`, `fp16=False`, and `attn_implementation="flash_attention_2"`, and require `_model_kwargs()` to forward the attention backend.
- [x] Run the focused tests and confirm they fail because RL does not expose `attn_implementation` yet.
- [x] Add the parser choice `eager|sdpa|flash_attention_2`, defaults, and explicit active YAML values.
- [x] Pass `attn_implementation` through `_model_kwargs()` and rerun the focused tests.

### Task 2: Add fail-fast runtime validation

**Files:**
- Modify: `tests/test_rl_training.py`
- Modify: `src/rl_training/train_grpo_macorag.py`

- [x] Add failing tests for conflicting BF16/FP16, absent or ABI-broken `flash_attn`, missing CUDA, unsupported BF16, and local-rank device selection.
- [x] Add `_validate_acceleration_runtime()` using the same dependency and device contract as the SFT path.
- [x] Invoke validation after distributed device setup and before loading the full policy/reference model.
- [x] Run the focused acceleration tests until green.

### Task 3: Preserve resume metadata and verify the real environment

**Files:**
- Modify: `src/rl_training/checkpointing.py`
- Test: `tests/test_rl_checkpointing.py`
- Test: `tests/test_rl_training.py`

- [x] Add `attn_implementation` to checkpoint identity so a resume cannot silently change attention kernels.
- [x] Run the complete RL unit suite, compile check, launcher syntax checks, and `git diff --check`.
- [x] Run `--check-only` with the active configuration in `/data/conda/envs/macorag`.
- [x] Run a bounded CUDA smoke that loads the real Qwen2.5-7B base with 4-bit BF16 FlashAttention 2, performs one forward/backward pass through a temporary LoRA adapter, and exits without starting full GRPO training.

### Non-goals

- Do not change vLLM's backend selection or force `VLLM_ATTENTION_BACKEND`.
- Do not disable gradient checkpointing.
- Do not start, stop, resume, or overwrite a full GRPO run.
- Do not change GRPO rewards, rollout generation, retrieval, or LoRA synchronization.
