# GRPO Shared Base Dual Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load one training-side Qwen base with a trainable policy LoRA and a frozen KL-reference LoRA while preserving existing training, vLLM, and output contracts.

**Architecture:** `train_grpo_macorag.py` owns adapter lifecycle helpers and a reference-scoring context manager. The training loop switches the shared PEFT model to `reference` only for no-gradient KL scoring and restores `default` before every trainable or externally visible operation.

**Tech Stack:** Python 3.11, PyTorch, Transformers, PEFT 0.17.1, pytest

## Global Constraints

- Preserve the user's uncommitted `config/train_grpo.yml` exactly.
- Keep the policy adapter name `default` because vLLM LoRA name mapping accepts `.default.` parameters.
- Keep `reference` frozen and omit it from every saved checkpoint.
- Keep gradient checkpointing enabled by default while making its false setting effective.
- Do not start a long training run as part of unit verification.

---

### Task 1: Shared model loading and adapter lifecycle

**Files:**
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `_activate_policy_adapter(model) -> None`
- Produces: `_reference_adapter_context(reference_model, policy_model)` context manager
- Changes: `_load_policy_and_reference(...)` returns the same model object for policy and reference

- [ ] **Step 1: Write failing loader and lifecycle tests**

Add fake PEFT/base objects and tests asserting one base load, `default` trainable load, frozen `reference` load, same returned model identity, adapter/mode restoration, and exception-safe restoration.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py -k 'shared_base or reference_adapter or gradient_checkpointing_toggle'`

Expected: failures because the second base is still loaded and adapter lifecycle helpers do not exist.

- [ ] **Step 3: Implement the minimal lifecycle**

Add constants for `default` and `reference`, adapter-parameter freezing, policy activation, an exception-safe reference context, single-base dual-adapter loading, and pass `use_gradient_checkpointing=args.gradient_checkpointing` to k-bit preparation.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command.

Expected: all selected tests pass.

### Task 2: Shared reference scoring and policy-only external operations

**Files:**
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`
- Test: `tests/test_vllm_lora_server.py`

**Interfaces:**
- Consumes: `_reference_adapter_context(reference_model, policy_model)`
- Changes: `_save_checkpoint(...)` and final saving persist `selected_adapters=["default"]`

- [ ] **Step 1: Write failing scoring, sync, and save tests**

Test that shared reference forward observes `reference/eval/frozen`, policy forward observes `default/train`, restoration occurs after a scoring error, sync begins with `default`, and save calls select only `default`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py -k 'train_on_rollouts_shared or sync_vllm_policy_adapter or save_checkpoint_policy_only'`

Expected: failures because scoring does not switch adapters and saving does not select an adapter.

- [ ] **Step 3: Implement minimal scoring/sync/save integration**

Wrap reference batches in the context manager, activate policy before vLLM sync, activate policy before optimizer construction, and save only `default` for intermediate and final outputs.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command, then `PYTHONPATH=src pytest -q tests/test_vllm_lora_server.py`.

Expected: all tests pass.

### Task 3: Regression verification

**Files:**
- Modify only if a regression directly caused by Tasks 1-2 is found.

**Interfaces:**
- Verifies the complete feature contract.

- [ ] **Step 1: Run the RL training suite**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py`

Expected: all tests pass.

- [ ] **Step 2: Run the project suite**

Run: `PYTHONPATH=src pytest -q -k 'not test_single_gpu_script_forces_hf_offline_mode'`

Expected: all selected tests pass; the excluded fixture is the known unrelated missing single-GPU launcher.

- [ ] **Step 3: Run static checks**

Run: `python -m py_compile src/rl_training/train_grpo_macorag.py && git diff --check && git status --short`

Expected: compilation and whitespace checks pass; `config/train_grpo.yml` remains the user's only unrelated modification.
