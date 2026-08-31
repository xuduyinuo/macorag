# Resumable Teacher E5 Index Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated teacher-index launcher plus visible, shared-model, batch-resumable E5/FAISS index construction.

**Architecture:** `E5Encoder` exposes passage batches; `build_e5_faiss_index` persists each batch into a contract-bound NumPy memmap and atomically advances a JSON state file. The retrieval CLI constructs one encoder for all datasets and reports dataset lifecycle events, while a new shell launcher selects the teacher config by default.

**Tech Stack:** Python 3, NumPy memmap, PyTorch/Transformers, FAISS, tqdm, pytest, Bash.

## Global Constraints

- Preserve the existing `scripts/build_retrieval.sh` default.
- Do not launch the full teacher corpus build in tests.
- Resume only when corpus hash, model, max length, count, dimension, and dataset match.
- Keep final index and metadata writes atomic.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Dedicated teacher launcher

**Files:**
- Create: `scripts/build_teacher_retrieval.sh`
- Modify: `tests/test_retraining_launchers.py`

**Interfaces:**
- Consumes: `CONFIG_PATH`, `PYTHON`, `MACORAG_LAUNCH_DRY_RUN` environment variables.
- Produces: a launcher whose default config is `config/build_retrieval_trajectory_train_e5.yml`.

- [x] **Step 1: Write a failing launcher test** asserting syntax validity, dry-run output, and the teacher config default.
- [x] **Step 2: Run `pytest -q tests/test_retraining_launchers.py`** and confirm failure because the launcher is absent.
- [x] **Step 3: Add the launcher** following `build_retrieval.sh` environment conventions and printing the resolved config before execution.
- [x] **Step 4: Run `pytest -q tests/test_retraining_launchers.py`** and confirm it passes.

### Task 2: Batch-resumable E5 encoding

**Files:**
- Modify: `src/data_processing/e5_faiss.py`
- Modify: `tests/test_retrieval_env.py`

**Interfaces:**
- Produces: `E5Encoder.encode_passage_batches(texts, *, start=0, show_progress=True, progress_desc=None) -> Iterable[tuple[int, np.ndarray]]`.
- Produces: resume artifacts `.e5_embeddings.npy` and `.e5_build_state.json` owned by `build_e5_faiss_index`.

- [x] **Step 1: Write failing tests** for interruption persistence, restart from encoded row count, resume-contract mismatch rejection, successful cleanup, and completed-index skipping.
- [x] **Step 2: Run each new test** and verify it fails for the missing behavior.
- [x] **Step 3: Implement batch iteration** without changing query encoding behavior.
- [x] **Step 4: Implement atomic state writes and NumPy memmap resume** with strict contract validation.
- [x] **Step 5: Validate complete assets before skipping** and retain existing atomic final index/metadata writes.
- [x] **Step 6: Run `pytest -q tests/test_retrieval_env.py`** and confirm it passes.

### Task 3: Shared encoder and dataset progress

**Files:**
- Modify: `src/data_processing/retrieval_cli.py`
- Modify: `tests/test_retrieval_env.py`

**Interfaces:**
- Consumes: `E5Encoder(model_name, device, max_length, batch_size)`.
- Produces: one shared encoder per CLI build and stderr lifecycle messages for every dataset.

- [x] **Step 1: Extend the CLI test** to assert all dataset builds receive the same encoder object and progress messages are emitted.
- [x] **Step 2: Run the focused CLI test** and verify failure because no encoder is passed.
- [x] **Step 3: Replace the comprehension with an explicit loop** that creates one encoder, prints start/complete messages, and collects the same JSON summary.
- [x] **Step 4: Run the focused CLI test** and confirm it passes.

### Task 4: Verification and usage contract

**Files:**
- Modify if needed: `docs/superpowers/specs/2026-08-25-resumable-teacher-e5-index-build-design.md`

**Interfaces:**
- Produces: verified command `bash scripts/build_teacher_retrieval.sh` and optional override `CONFIG_PATH=... bash scripts/build_teacher_retrieval.sh`.

- [x] **Step 1: Run `bash -n scripts/build_teacher_retrieval.sh scripts/build_retrieval.sh`**.
- [x] **Step 2: Run `MACORAG_LAUNCH_DRY_RUN=1 bash scripts/build_teacher_retrieval.sh`** and inspect the resolved teacher config.
- [x] **Step 3: Run `pytest -q tests/test_retrieval_env.py tests/test_retraining_launchers.py`**.
- [x] **Step 4: Run the broader affected test suite** and report exact pass/failure counts without starting a full index build.
