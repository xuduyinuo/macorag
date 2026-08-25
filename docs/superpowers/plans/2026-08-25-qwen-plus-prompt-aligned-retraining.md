# Qwen-Plus Prompt-Aligned Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Regenerate three 1,000-trajectory Qwen-Plus teacher datasets and make teacher generation, SFT, GRPO, and evaluation consume one versioned `macorag-rag-v2` prompt/round/retrieval contract.

**Architecture:** `config/prompts.yml` is the single declarative prompt source, `src/prompt_config.py` validates and fingerprints it, and `src/rag/prompts.py` is the only student prompt renderer. Teacher generation derives semantically equivalent JSON-only instructions from the same contract. Every downstream stage records the contract fingerprint and fails fast on incompatible data, adapters, checkpoints, or retrieval indexes.

**Tech Stack:** Python 3.10+, PyYAML, Transformers/PEFT, PyTorch, FAISS, `intfloat/e5-base-v2`, Qwen-Plus OpenAI-compatible API, pytest, Bash.

## Global Constraints

- Preserve the approved design in `docs/superpowers/specs/2026-08-25-qwen-plus-prompt-aligned-retraining-design.md`.
- Preserve existing user edits in the dirty checkout. Do not stage or commit pre-existing changes from overlapping files.
- Keep all old teacher data, SFT runs, GRPO runs, and evaluation outputs intact; use v2 output namespaces.
- Freeze `max_rounds=4`, `retrieval_top_k=5`, `intfloat/e5-base-v2`, CPU FAISS `IndexFlatIP`, and SFT base `model/Qwen2.5-7B-Instruct`.
- Do not expose gold answers to Qwen-Plus. Gold/alias data is validation-only after a trajectory is generated.
- Do not use constrained decoding in this cycle.
- Use test-first changes and run the named focused test after each task. Because overlapping source files are already dirty, use verification checkpoints instead of implementation commits unless an exact isolated patch can be staged without user changes.

## Task 1: Introduce and validate the v2 prompt contract

**Files:**

- Modify: `config/prompts.yml`
- Modify: `src/prompt_config.py`
- Create: `tests/test_prompt_config.py`
- Modify: `tests/test_rag.py`

**Interfaces:**

- `PromptContract(version, system_prompts, instructions, fingerprint)`
- `load_prompt_contract(path=None) -> PromptContract`
- `system_prompt_for(role, contract=None) -> str`
- Keep `load_system_prompt()` as a compatibility wrapper only while callers migrate.

**Steps:**

- [ ] Add failing tests for required contract version, all three non-empty role system prompts, deterministic SHA-256 fingerprinting, rejection of unknown/malformed contracts, and compatibility loading.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_prompt_config.py tests/test_rag.py -x` and confirm the new tests fail.
- [ ] Replace the single permissive system prompt with `prompt_contract_version: macorag-rag-v2`, role-specific system prompts, and normal/final answer rules containing only valid JSON examples.
- [ ] Implement immutable contract loading and canonical fingerprinting. Include the source path and raw semantic payload in the object so run metadata can preserve provenance.
- [ ] Re-run the focused tests and `python -m compileall -q src/prompt_config.py`.

## Task 2: Make runtime prompts round-aware and byte-stable

**Files:**

- Modify: `src/rag/prompts.py`
- Modify: `src/rag/schema.py`
- Modify: `src/rag/executor.py`
- Modify: `src/rag/__init__.py`
- Modify: `tests/test_rag.py`

**Interfaces:**

- `AnswerPromptContext(round_index: int, max_rounds: int)` with validated `is_final_round` and `remaining_rounds`
- `build_answer_generator_prompt(question, state, context)`; remove caller-owned `force_final_answer`
- `advance_rag_state(state, query_action, observation, update_action) -> RAGState`
- Raw response fields in every attempted role turn: `raw_responses`, `parse_error_role`, and `generated_roles`

**Steps:**

- [ ] Add failing tests for normal versus final answer prompts, valid literal JSON examples, exact `fallback_guess:` marker, invalid round indexes, and shared state transitions.
- [ ] Add executor tests proving the final mode is derived from `(round_index, max_rounds)`, terminal false/null is rejected, and the raw failed answer response remains in the trajectory.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_rag.py -x` and confirm failure.
- [ ] Implement `AnswerPromptContext`, remove ellipses from JSON examples, and route each prompt through the role-specific system contract.
- [ ] Implement one state transition helper that increments retrieval count and records query, sub-goal, passage IDs, and scores.
- [ ] Update executor call sites and trajectory logging without changing parser strictness.
- [ ] Re-run `PYTHONPATH=src pytest -q tests/test_rag.py -x`.

## Task 3: Replace blind token slicing with deterministic prompt compaction

**Files:**

- Create: `src/rag/prompt_budget.py`
- Modify: `src/rl_training/policy.py`
- Modify: `src/evaluation/evaluate_rag_model.py`
- Modify: `src/sft_training/dataset.py`
- Create: `tests/test_prompt_budget.py`
- Modify: `tests/test_rl_training.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_sft_training.py`

**Interfaces:**

- `compact_prompt_inputs(question, state, observation, tokenizer, system_prompt, builder, max_tokens) -> CompactedPrompt`
- Deterministic priority: keep protocol instructions/question/current sub-goal; retain selected evidence newest-first; compact retrieval history before evidence; compact observation last; fail with an explicit reason if the fixed portion alone exceeds budget.

**Steps:**

- [ ] Add failing tests showing the opening protocol and required output tag survive compaction and that SFT/RL/eval render identical messages for identical structured input.
- [ ] Run the focused prompt-budget, SFT, RL, and evaluation tests and confirm failure.
- [ ] Implement structure-aware compaction and diagnostic metadata (`original_tokens`, `final_tokens`, removed item counts).
- [ ] Remove `list(prompt_ids)[-max_prompt_length:]` from RL and request-side truncation as the primary evaluation policy.
- [ ] Make SFT use the same compaction rule before target concatenation; retain explicit skipped-record diagnostics only when no valid compacted prompt can fit.
- [ ] Re-run focused tests.

## Task 4: Make SFT consume runtime prompt builders and enforce trajectory provenance

**Files:**

- Modify: `src/sft_training/data.py`
- Modify: `src/sft_training/config.py`
- Modify: `src/sft_training/train_sft_lora_macorag.py`
- Modify: `src/sft_training/trainer.py`
- Modify: `tests/test_sft_training.py`

**Interfaces:**

- `trajectory_to_sft_records()` calls only public builders from `src/rag/prompts.py`.
- Each record carries `round_index`, `max_rounds`, and `prompt_contract_fingerprint`.
- `validate_teacher_dataset_contract(data_root, expected_contract, max_rounds, top_k)` fails before model load.
- SFT writes `train_meta.json` with prompt, teacher run, retrieval index, and data fingerprints.

**Steps:**

- [ ] Add failing byte-equality tests for all three role prompts and normal/final answer modes, including the fourth-round final prompt.
- [ ] Add failing tests for mismatched/missing teacher run provenance and for persisted SFT metadata.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_sft_training.py -x` and confirm failure.
- [ ] Delete private `_build_*_prompt` implementations and use shared `RAGState`, `AnswerPromptContext`, and state transitions.
- [ ] Extend records/config/training metadata and add fail-fast validation before tokenizer/model construction.
- [ ] Re-run the focused tests.

## Task 5: Convert teacher generation to E5/FAISS and enforce final-round semantics

**Files:**

- Modify: `src/data_processing/generate_teacher_sft.py`
- Create: `tests/test_generate_teacher_sft_v2.py`
- Modify: `tests/test_data_processing_package_boundary.py`

**Interfaces:**

- Teacher `SFTConfig` gains `retrieval_backend`, `retrieval_device`, `retrieval_max_length`, and `prompt_config_path`; `force_final_answer` is removed.
- Teacher messages are produced from the same contract semantics but remain strict JSON-object transport.
- Finalization rejects terminal `can_answer=false`, null/empty terminal answers, and fallback rationales without exact `fallback_guess:` prefix.
- Teacher output rows and `run_config.json` include contract/index fingerprints; `summary.json` includes protocol rates and exact per-dataset valid counts.

**Steps:**

- [ ] Add failing tests for E5 config/routing, semantic normal/final teacher prompts, terminal validation, no gold leakage in outbound messages, aliases accepted only by post-generation validation, and provenance output.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_generate_teacher_sft_v2.py -x` and confirm failure.
- [ ] Replace LinearRAG-only retrieval construction with the repository retrieval factory/E5 query engine.
- [ ] Remove no-op force handling and derive finality from round context.
- [ ] Record every raw teacher response and all rejected trajectory diagnostics; preserve resumability by qid.
- [ ] Add protocol gate calculations and fail the command if any dataset does not reach exactly 1,000 valid rows in a full run.
- [ ] Re-run focused tests plus a two-sample-per-dataset dry run to a temporary output directory.

## Task 6: Add full trajectory E5 index configuration and preflight validation

**Files:**

- Create: `config/build_retrieval_trajectory_train_e5.yml`
- Modify: `src/data_processing/e5_faiss.py`
- Modify: `src/data_processing/retrieval_cli.py`
- Modify: `scripts/build_retrieval.sh`
- Modify: `tests/test_retrieval_env.py`

**Interfaces:**

- Build root `data/trajectory_train_e5_faiss` from `data/trajectory_train`.
- Index metadata/fingerprint includes dataset corpus count/hash, model, dimension, max length, normalization, metric, and FAISS type.
- `validate_e5_index_contract(...)` checks expected corpus counts and settings before teacher, RL, or evaluation uses an index.

**Steps:**

- [ ] Add failing metadata/fingerprint/preflight tests including detection of the smaller RL-2000 index used against trajectory-train.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_retrieval_env.py -x` and confirm failure.
- [ ] Implement metadata fingerprinting and validation without breaking existing E5 index loading.
- [ ] Add the dedicated config and allow `CONFIG_PATH=... scripts/build_retrieval.sh` so launchers select configs consistently.
- [ ] Re-run focused tests and a config-only/preflight command that does not build the full index.

## Task 7: Enforce the same contract in GRPO, checkpoints, and evaluation

**Files:**

- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/batched_rollout.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Modify: `src/rl_training/trainer.py`
- Modify: `src/rl_training/checkpointing.py`
- Modify: `src/evaluation/config.py`
- Modify: `src/evaluation/evaluate_rag_model.py`
- Modify: `tests/test_rl_training.py`
- Modify: `tests/test_rl_checkpointing.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**

- GRPO/eval load role system prompts from the contract, never a hardcoded system string.
- Batched rollout passes `round_index/max_rounds`, not a force boolean.
- RL `train_meta.json` and checkpoint manifest preserve SFT adapter contract fingerprint and refuse mismatches.
- Evaluation `run_config.json` preserves prompt/index/model identity and fails fast on adapter/contract mismatch when metadata exists.
- Protocol monitor computes parse failure, missing-answer-tag, and final-compliance rates; checkpoint eligibility requires `<=1%`, `<=0.2%`, and `>=99%` respectively; two consecutive 100-step windows above 2% parse failures stop training.

**Steps:**

- [ ] Add failing tests for prompt loading, round context, adapter/checkpoint mismatch rejection, evaluation metadata, raw response persistence, rolling protocol gates, and checkpoint ineligibility.
- [ ] Run focused RL/checkpoint/evaluation tests and confirm failure.
- [ ] Implement provenance validation before model/vLLM startup.
- [ ] Implement protocol windows and checkpoint eligibility while preserving existing full-state resume semantics.
- [ ] Replace evaluation hardcoded system prompt and record the effective contract.
- [ ] Re-run focused tests.

## Task 8: Freeze v2 configs and make startup scripts safe and composable

**Files:**

- Modify: `config/generate_teacher_sft.yml`
- Modify: `config/train_sft.yml`
- Modify: `config/train_grpo.yml`
- Modify: `config/eval_macorag.yml`
- Modify: `scripts/generate_teacher_sft.sh`
- Modify: `scripts/run_train_sft.sh`
- Modify: `scripts/run_train_grpo.sh`
- Modify: `scripts/eval_macorag.sh`
- Create: `scripts/validate_retraining_v2.sh`
- Create: `tests/test_retraining_launchers.py`

**Steps:**

- [ ] Add failing tests that load all four YAML files and assert prompt version, max rounds, top-k, E5 model/root, v2 output roots, 7B SFT base, and new-SFT GRPO initialization.
- [ ] Add Bash dry-run tests showing every launcher honors `CONFIG_PATH`, prints the resolved stage identity, and performs preflight without launching expensive work.
- [ ] Run `PYTHONPATH=src pytest -q tests/test_retraining_launchers.py -x` and confirm failure.
- [ ] Freeze teacher output to `data/sft/teacher_qwen_plus_trajectory_train_v2`, full teacher index root to `data/trajectory_train_e5_faiss`, SFT output to a fresh 7B namespace, and GRPO adapter path as an explicit placeholder that must resolve to a completed v2 SFT run.
- [ ] Make all launchers accept `CONFIG_PATH`, support a non-mutating dry-run/preflight mode, load `.env` consistently where API access is needed, and fail before expensive initialization on missing artifacts.
- [ ] Implement `scripts/validate_retraining_v2.sh` as a read-only cross-stage compatibility check.
- [ ] Re-run launcher tests and `bash -n` on all modified/new scripts.

## Task 9: End-to-end verification and operator handoff

**Files:**

- Verify all files above; do not start paid full teacher generation or long GPU training without an explicit user launch command.

**Steps:**

- [ ] Run `PYTHONPATH=src pytest -q tests/test_prompt_config.py tests/test_prompt_budget.py tests/test_generate_teacher_sft_v2.py tests/test_rag.py tests/test_sft_training.py tests/test_retrieval_env.py tests/test_rl_training.py tests/test_rl_checkpointing.py tests/test_evaluation.py tests/test_retraining_launchers.py`.
- [ ] Run `python -m compileall -q src`.
- [ ] Run `bash -n scripts/build_retrieval.sh scripts/generate_teacher_sft.sh scripts/run_train_sft.sh scripts/run_train_grpo.sh scripts/eval_macorag.sh scripts/validate_retraining_v2.sh`.
- [ ] Run teacher dry-run with 2 candidates/2 valid rows per dataset into `/tmp/macorag_teacher_v2_dry_run` and inspect all JSONL rows plus `summary.json`/`run_config.json`.
- [ ] Run each launcher in its dry-run/preflight mode and capture resolved configs without starting Qwen-Plus billing, index build, vLLM, SFT, GRPO, or full evaluation.
- [ ] Run `git diff --check` and inspect `git status --short` to verify old artifacts and unrelated edits remain untouched.
- [ ] Hand off the exact ordered commands: environment/API check; full trajectory E5 index build; teacher API smoke; 32-per-dataset pilot and gates; full 1,000-per-dataset generation; SFT check-only and training; SFT adapter serving/evaluation; GRPO training; final evaluation. State clearly which commands are expensive and which output path must be substituted after timestamped SFT completion.
