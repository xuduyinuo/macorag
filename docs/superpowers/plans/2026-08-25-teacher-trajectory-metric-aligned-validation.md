# Teacher Trajectory Metric-Aligned Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove confirmed teacher-trajectory false positives, align answer admission with existing evaluation metrics, and persist auditable filtered candidates.

**Architecture:** A neutral `answer_metrics` module becomes the single implementation used by evaluation and teacher validation without changing metric semantics. Teacher validation separates metric-correct final answers from deterministic evidence support, while generation writes every rejected candidate to a dedicated JSONL diagnostic stream.

**Tech Stack:** Python 3, JSONL, pytest, existing Qwen-Plus/E5 teacher pipeline.

## Global Constraints

- Keep contain bidirectional exactly as `gold in prediction or prediction in gold` after normalization.
- Admit a teacher final answer only when normalized exact match against the primary gold is 1.
- Use aliases only for deterministic evidence support, not primary-gold answer admission.
- Do not modify evaluation output schema, RL rewards, retrieval, or final-round prompt semantics.
- Preserve unrelated dirty-worktree changes and do not write formal teacher output during tests.

---

### Task 1: Shared evaluation answer metrics

**Files:**
- Create: `src/answer_metrics.py`
- Modify: `src/evaluation/bailian_evaluator.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `normalize_answer(text)`, `calculate_contain(prediction, gold)`, `calculate_exact_match(prediction, gold)`, `calculate_f1(prediction, gold)`, and `calculate_answer_metrics(prediction, gold)`.
- Preserves: imports of the four existing functions from `evaluation.bailian_evaluator`.

- [x] **Step 1: Write a failing test** that imports `calculate_answer_metrics` from `answer_metrics` and asserts `The David Arquette` versus `David Arquette` yields exact 1, bidirectional contain 1, and F1 1.
- [x] **Step 2: Run the focused test** and confirm failure because `answer_metrics` does not exist.
- [x] **Step 3: Move the existing metric implementations unchanged** into `src/answer_metrics.py`; make `bailian_evaluator.py` import and re-export them.
- [x] **Step 4: Run evaluation metric tests** and confirm all existing contain/EM/F1 assertions remain unchanged.

### Task 2: Metric-aligned trajectory validation

**Files:**
- Modify: `src/data_processing/generate_teacher_sft.py`
- Modify: `tests/test_sft_data_generation.py`

**Interfaces:**
- Changes: `query_has_unseen_intermediate_terms(..., answer_forms=())` checks only unseen primary-gold/alias leakage.
- Changes: `answer_supported_by_evidence(..., accepted_answers=())` accepts deterministic gold/alias evidence forms.
- Produces: `trajectory_answer_metrics(sample)` and authored-field-only forbidden-term validation.

- [x] **Step 1: Write failing tests** for `Gold Coast` observation acceptance, authored `gold answer` rationale rejection, generic `Birth Place` query acceptance, unseen gold leakage rejection, semantic-but-metric-unequal answer rejection, and alias-backed evidence acceptance.
- [x] **Step 2: Run the focused tests** and verify each fails for the expected old behavior.
- [x] **Step 3: Replace capitalization filtering** with normalized answer-form leakage checks using question and prior selected evidence as the allowed provenance.
- [x] **Step 4: Restrict forbidden-term scanning** to query/sub-goal, update rationale, answer text, and answer rationale.
- [x] **Step 5: Require shared normalized exact match 1** against primary gold and pass gold/aliases only into evidence-support checks.
- [x] **Step 6: Run `tests/test_sft_data_generation.py`** and update old capitalization-heuristic expectations to the approved contract.

### Task 3: Persist filtered candidate diagnostics

**Files:**
- Modify: `src/data_processing/generate_teacher_sft.py`
- Modify: `tests/test_sft_data_generation.py`

**Interfaces:**
- Produces: `<output_dir>/teacher_filtered.jsonl`.
- Each record contains: dataset/qid/question/stage, validation errors, filter reasons, metric components, evidence support, candidate trajectory or partial turn, prompt contract identity, and retrieval index fingerprint.

- [x] **Step 1: Write a failing generation test** that forces a validation rejection and asserts one complete diagnostic JSONL record with raw responses and exact metric fields.
- [x] **Step 2: Run the focused test** and confirm failure because `teacher_filtered.jsonl` is absent.
- [x] **Step 3: Return structured diagnostics** from both early query leakage and final validation rejection paths.
- [x] **Step 4: Append diagnostics atomically per completed future** in the existing main-thread result loop; keep runtime failures in `teacher_errors.jsonl`.
- [x] **Step 5: Add `filtered_output`, `filtered_records`, and `filter_reason_occurrences` to summary** while preserving existing counters.
- [x] **Step 6: Run generation tests** and confirm candidate counts remain distinct from reason-occurrence counts.

### Task 4: Regression and real-smoke audit

**Files:**
- Modify if required: `docs/superpowers/specs/2026-08-25-teacher-trajectory-metric-aligned-validation-design.md`

**Interfaces:**
- Verifies: existing smoke artifacts remain valid and evaluation contain remains bidirectional.

- [x] **Step 1: Run focused tests** for answer metrics and teacher generation.
- [x] **Step 2: Revalidate the three previously written smoke trajectories** with the updated validator.
- [x] **Step 3: Run the broader affected pytest suite and `git diff --check`**.
- [x] **Step 4: Run a fresh, isolated Qwen-Plus smoke only if local tests pass**; report valid yield and exact filter diagnostics without touching formal output.
