# SFT Throughput and Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SFT validation epoch-scoped, padding-efficient, measurable, shuffled for training, and explicitly resumable.

**Architecture:** Keep Hugging Face Trainer as the execution engine. Add small sampler and callback helpers around it, while configuration controls evaluation cadence and resume state.

**Tech Stack:** Python 3.9, PyTorch 2.6, Transformers 4.57, PEFT 0.17, pytest, YAML.

## Global Constraints

- Preserve current teacher-data, prompt-contract, QLoRA, and target-only-loss behavior.
- Do not stop the active SFT process.
- Preserve all pre-existing uncommitted changes.

---

### Task 1: Epoch validation and resume contract

**Files:**
- Modify: `config/train_sft.yml`
- Modify: `src/sft_training/config.py`
- Modify: `src/sft_training/train_sft_lora_macorag.py`
- Test: `tests/test_sft_training.py`

**Interfaces:**
- Consumes: YAML/CLI values `eval_strategy` and `resume_from_checkpoint`.
- Produces: validated `TrainingArguments` and `trainer.train(resume_from_checkpoint=...)` call.

- [x] Write tests asserting epoch strategy parsing, no step interval requirement, and explicit resume propagation.
- [x] Run the focused tests and confirm they fail for the missing behavior.
- [x] Implement the minimal parser, validation, TrainingArguments, and Trainer call changes.
- [x] Run the focused tests and confirm they pass.

### Task 2: Train shuffle and length-grouped evaluation

**Files:**
- Modify: `src/sft_training/trainer.py`
- Test: `tests/test_sft_training.py`

**Interfaces:**
- Consumes: datasets whose items contain variable-length `input_ids`.
- Produces: random/distributed train samplers and a deterministic evaluation batch sampler grouping nearby lengths.

- [x] Write tests proving train order is not sequential and eval batches cover each index once with bounded padding.
- [x] Run the focused tests and confirm they fail for current ordered sampling.
- [x] Implement Trainer-compatible samplers without changing evaluation membership.
- [x] Run the focused tests and confirm they pass.

### Task 3: Phase timing and token throughput

**Files:**
- Modify: `src/sft_training/callbacks.py`
- Modify: `src/sft_training/train_sft_lora_macorag.py`
- Test: `tests/test_sft_training.py`

**Interfaces:**
- Consumes: Trainer callback lifecycle events and dataset token counts.
- Produces: append-only `phase_metrics.jsonl` records for train, evaluate, and save phases.

- [x] Write a deterministic callback test using an injected monotonic clock.
- [x] Run the focused test and confirm it fails because the callback is absent.
- [x] Implement the callback and wire distributed token reduction plus main-process writes.
- [x] Run the focused test and confirm it passes.

### Task 4: Regression and launch verification

**Files:**
- Verify: all changed SFT files.

**Interfaces:**
- Consumes: the completed behavior from Tasks 1-3.
- Produces: evidence that the config parses and SFT regressions remain green.

- [x] Run `pytest -q tests/test_sft_training.py` in the macorag environment.
- [x] Run the broader SFT-related tests selected by imports and launcher contracts.
- [x] Run `compileall`, `bash -n scripts/run_train_sft.sh`, check-only, and `git diff --check`.
- [x] Inspect the final diff and audit remaining performance risks without making unrelated changes.
