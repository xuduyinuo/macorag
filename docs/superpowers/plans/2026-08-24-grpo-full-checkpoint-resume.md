# GRPO Full-State Checkpoint and Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax to track progress.

**Goal:** Make new MACORAG GRPO checkpoints atomically save and restore all state required to continue from the latest completed optimizer boundary.

**Architecture:** Add a focused checkpoint module for full-state serialization, validation, restoration, and retention. Integrate it at optimizer boundaries in the existing trainer, while preserving legacy policy-only warm starts. Add recoverable per-prompt seeds to the vLLM generation path.

**Tech Stack:** Python 3.9, PyTorch, NumPy, PEFT, vLLM, FastAPI/Pydantic, pytest.

---

### Task 1: Full checkpoint serialization and retention

**Files:**
- Create: `src/rl_training/checkpointing.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

- [ ] Write failing tests that save a toy model/optimizer checkpoint and assert `optimizer.pt`, `trainer_state.pt`, rank RNG files, manifest, and `COMPLETE` exist only in the final directory.
- [ ] Run the focused tests and confirm failure because the full-state API is absent.
- [ ] Implement `save_full_checkpoint()` using `.checkpoint-<step>.tmp`, rank barriers, final validation, atomic rename, and a save callback for the policy/tokenizer.
- [ ] Add failing retention tests for newest-three plus 1000-step milestones, then implement pruning scoped to the current run directory.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Strict full resume and legacy compatibility

**Files:**
- Modify: `src/rl_training/checkpointing.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Modify: `src/rl_training/config.py`
- Modify: `config/train_grpo.yml`
- Test: `tests/test_rl_training.py`

- [ ] Write failing tests for manifest-based automatic epoch/sample/global-step resolution, optimizer/RNG restoration, incomplete-file rejection, fingerprint mismatch rejection, and policy-only legacy warm starts.
- [ ] Run the tests and confirm the expected failures.
- [ ] Implement checkpoint inspection and full-state restoration. Restore RNG after model, optimizer, retriever, policy, and vLLM synchronization initialization and before the first resumed rollout.
- [ ] Add `save_milestone_steps` with default/config value 1000 and use the existing `save_total_limit` default/config value 3.
- [ ] Update run metadata so full resumes record `optimizer_state_restored: true`, `rng_state_restored: true`, and the source checkpoint schema.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Recoverable vLLM prompt seeds

**Files:**
- Modify: `src/rl_training/policy.py`
- Modify: `src/rl_training/vllm_client.py`
- Modify: `src/rl_training/vllm_lora_server.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`
- Test: `tests/test_vllm_lora_server.py`

- [ ] Write failing tests asserting one stable seed per prompt, unchanged seeds across transport retry, distinct seeds for grouped trajectories, and server construction of per-prompt sampling parameters.
- [ ] Run the focused tests and confirm failure because seeds are not carried through the API.
- [ ] Add a generation counter to `VLLMSharedPolicy`; derive seeds from the base training seed and counter, send `seeds` through the client, and construct a `SamplingParams` list in the server.
- [ ] Save and restore the generation counter through trainer state.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Training-loop integration and end-to-end recovery test

**Files:**
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

- [ ] Write a failing test that compares uninterrupted toy optimization with an interrupted save/resume sequence and asserts identical policy parameters, optimizer tensors, next sample, global step, and next generation seed.
- [ ] Run the test and confirm the existing policy-only checkpoint path cannot satisfy it.
- [ ] Track rank-local `samples_consumed_in_epoch`, save only when `did_optimizer_step` is true, and defer periodic saves that fall inside an accumulation window.
- [ ] Restore full state before the first rollout and preserve first-rollout LoRA synchronization.
- [ ] Run the end-to-end recovery test and the complete RL/vLLM server suites.

### Task 5: Verification

**Files:**
- Verify all modified files.

- [ ] Run `/data/conda/envs/macorag/bin/python -m compileall -q src/rl_training`.
- [ ] Run `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py tests/test_vllm_lora_server.py`.
- [ ] Run `git diff --check` and Bash syntax checks for the GRPO launchers.
- [ ] Create a toy full checkpoint, inspect its manifest/files, resume it, and verify optimizer/RNG restoration flags without starting the 7B model.
