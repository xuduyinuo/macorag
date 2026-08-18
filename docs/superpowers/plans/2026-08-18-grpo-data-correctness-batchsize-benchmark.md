# GRPO Data Correctness and Batch-Size Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair generated-action and first-rollout correctness, add reproducible sampling and actionable timing/group diagnostics, apply bounded training/retrieval optimizations, and select an action microbatch size from deterministic 20-sample runs.

**Architecture:** Canonicalize model output at the parser boundary so scalar/batched executors and rewards consume the same types. Build deterministic balanced sample ordering before rank partitioning, add shared bounded retrieval caching and counters, precompute fixed-reference logprobs in independent batches, and expose group/token/timing metadata. A fail-fast shell benchmark runs batch sizes 1, 2, and 4 against identical QIDs and records GPU memory.

**Tech Stack:** Python 3.9, PyTorch, PEFT/QLoRA, vLLM/TRL, LinearRAG, pytest, Bash, NVIDIA SMI.

## Global Constraints

- Work directly on the current `master` branch as explicitly authorized by the user.
- Preserve the user's uncommitted `config/train_grpo.yml` change from `max_samples: 200` to `max_samples: 50`.
- Preserve shared-policy role/round action-credit and reward formulas.
- Do not overwrite or resume `outputs/grpo_qwen2.5-7b/2026-08-18_17-01-03`.
- Use a new output root for every benchmark candidate.
- Use the same deterministic balanced 20 QIDs for batch sizes 1, 2, and 4.
- Keep gradient checkpointing, sampling, KL, learning rate, group size, and maximum rounds fixed during batch-size comparison.
- Write production code only after observing the corresponding test fail.

---

### Task 1: Canonical Generated-Action Validation

**Files:**
- Modify: `src/rag/parser.py`
- Modify: `src/rag/executor.py`
- Modify: `src/rl_training/batched_rollout.py`
- Modify: `src/rl_training/rewards.py`
- Test: `tests/test_rag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `parse_can_answer(value: Any) -> bool` in `rag.parser`.
- Preserves: `parse_action_text(text, role) -> ParsedAction`.
- Guarantees: parsed answers contain a real `bool`; selected passage IDs contain only non-boolean integers.

- [ ] **Step 1: Write failing parser tests**

Add tests proving `"false"` and `"TRUE"` normalize to booleans, while `"yes"`, `1`, a numeric answer, a non-string rationale, boolean passage IDs, and string passage IDs raise `ValueError` with the relevant field name.

```python
@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("TRUE", True)])
def test_parse_answer_normalizes_boolean_strings(raw, expected):
    action = parse_action_text(
        f'<answer>{{"can_answer":"{raw}","answer":"x","rationale":"r"}}</answer>',
        AgentRole.ANSWER_GENERATOR,
    )
    assert action.answer["can_answer"] is expected
```

- [ ] **Step 2: Run parser tests and verify RED**

Run: `conda run --no-capture-output -n macorag python -m pytest -q tests/test_rag.py -k 'normalizes_boolean or rejects_invalid_answer or rejects_invalid_passage'`

Expected: failures show unnormalized strings and accepted invalid field types.

- [ ] **Step 3: Implement canonical validation**

Implement `parse_can_answer()` and field validators in `src/rag/parser.py`. Use the canonical boolean in both executors:

```python
if answer["can_answer"] is True:
    candidate.final_answer = answer["answer"]
```

Import `parse_can_answer` in rewards for historical-artifact fallback; remove the duplicate truth conversion.

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
conda run --no-capture-output -n macorag python -m pytest -q tests/test_rag.py tests/test_rl_training.py -k 'parse_action or executor or batched_rollout or reward'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit canonical validation**

```bash
git add src/rag/parser.py src/rag/executor.py src/rl_training/batched_rollout.py src/rl_training/rewards.py tests/test_rag.py tests/test_rl_training.py
git commit -m "fix: canonicalize grpo generated actions"
```

### Task 2: Balanced Deterministic Sample Selection

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/data.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `select_balanced_samples(samples, *, max_total_samples, seed) -> list[RLSample]`.
- Produces: `epoch_sample_order(samples, *, seed, epoch) -> list[RLSample]`.
- Adds CLI/YAML key: `max_total_samples: int | None`.

- [ ] **Step 1: Write failing selection/order tests**

Create 10 samples for each of three datasets. Assert a total limit of 20 returns counts `7/7/6`, identical seed gives identical QIDs, different epochs give different order, and all selected QIDs appear exactly once. Assert `_rank_samples()` partitions the same global epoch order without overlap.

- [ ] **Step 2: Run selection tests and verify RED**

Run: `conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'balanced_samples or epoch_sample_order'`

Expected: missing helper/config failures.

- [ ] **Step 3: Implement deterministic selection**

Use local `random.Random` instances only. Apply balanced selection immediately after `load_rl_samples()`. Build rank samples inside each epoch from `epoch_sample_order()` rather than reusing one file-ordered list.

- [ ] **Step 4: Verify config and ordering**

Run:

```bash
conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'parse_args or balanced_samples or epoch_sample_order or rank_samples'
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m rl_training.train_grpo_macorag --config config/train_grpo.yml --check-only --max-total-samples 20
```

Expected: check-only reports exactly 20 selected samples with balanced counts.

- [ ] **Step 5: Commit deterministic sampling**

```bash
git add src/rl_training/config.py src/rl_training/data.py src/rl_training/train_grpo_macorag.py tests/test_rl_training.py
git commit -m "feat: interleave grpo datasets deterministically"
```

### Task 3: Initial vLLM Policy Synchronization

**Files:**
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `_sync_vllm_before_first_rollout(policy, raw_policy_model, args) -> float`.
- Preserves: `_sync_vllm_after_optimizer_step(...)` cadence.

- [ ] **Step 1: Write failing initial-sync tests**

Assert LoRA synchronization is called once before `_rollout_group`, ignores `vllm_sync_every_steps`, returns zero when vLLM is disabled, and performs no call on non-main ranks.

- [ ] **Step 2: Run initial-sync tests and verify RED**

Run: `conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'sync_vllm_before_first_rollout'`

Expected: helper missing.

- [ ] **Step 3: Implement and wire initial synchronization**

Reuse the validated sync mode and client methods without applying post-step cadence checks. Call after policy construction and before output-loop rollout. Store the duration as `time_initial_weight_sync_seconds`.

- [ ] **Step 4: Run sync regressions**

Run: `conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'sync_vllm or build_policy'`

Expected: all selected tests pass.

- [ ] **Step 5: Commit initial synchronization**

```bash
git add src/rl_training/train_grpo_macorag.py tests/test_rl_training.py
git commit -m "fix: sync policy before first vllm rollout"
```

### Task 4: Bounded Retrieval Cache and Timing

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/retrieval.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_evaluation.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Adds config: `retrieval_query_cache_size: int = 4096`.
- Produces: `CachedLinearRAGRetrievalEnv.stats() -> dict[str, float | int]`.
- Cache key: `(dataset, " ".join(query.casefold().split()))`.

- [ ] **Step 1: Write failing cache tests**

Assert batch duplicate misses are sent once to the engine, output order matches input order, cached observations are copy-isolated, LRU eviction occurs at the configured bound, size zero disables caching, and counters/timing increase correctly.

- [ ] **Step 2: Run cache tests and verify RED**

Run: `conda run --no-capture-output -n macorag python -m pytest -q tests/test_evaluation.py tests/test_rl_training.py -k 'retrieval and (cache or timing or batch)'`

Expected: constructor/config/stat failures.

- [ ] **Step 3: Implement shared scalar/batch LRU cache**

Use `OrderedDict`, the existing dataset lock, `copy.deepcopy`, and `time.perf_counter`. Batch only unique misses and reconstruct every original query position.

- [ ] **Step 4: Wire per-rollout counter deltas**

Snapshot stats before and after `_rollout_group()` and add `time_retrieval_seconds`, `retrieval_cache_hits`, and `retrieval_cache_misses` to rollout timing.

- [ ] **Step 5: Run retrieval/RL regressions and commit**

Run:

```bash
conda run --no-capture-output -n macorag python -m pytest -q tests/test_evaluation.py tests/test_rl_training.py -k 'retrieval or rollout_timing'
```

Then:

```bash
git add src/rl_training/config.py src/rl_training/retrieval.py src/rl_training/train_grpo_macorag.py tests/test_evaluation.py tests/test_rl_training.py
git commit -m "perf: cache grpo retrieval queries"
```

### Task 5: Independent Reference Batching and No-Signal Skip

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Adds config: `reference_per_device_batch_size: int = 4`.
- Adds config: `skip_zero_advantage_updates: bool = true`.
- Extends `_train_on_rollouts()` metrics with action/token counts, clip fraction, forward batch counts, and `skipped_update_reason`.

- [ ] **Step 1: Write failing reference/skip tests**

For five actions, action batch 2, and reference batch 4, assert policy forwards are `2/2/1`, reference forwards are `4/1`, and loss matches reference batch 1. For all-zero advantages with accumulation 1, assert no model forward, backward, optimizer, or post-step sync eligibility. Assert accumulation greater than one disables the skip and preserves existing accumulation semantics.

- [ ] **Step 2: Run trainer tests and verify RED**

Run: `conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'reference_batch or zero_advantage'`

Expected: missing config and current per-policy-batch reference calls.

- [ ] **Step 3: Implement reference precomputation**

Flatten actions once. Under `torch.no_grad()`, compute reference logprobs in reference-sized chunks, detach them, and associate them with stable action indices. Policy microbatches consume aligned slices and preserve valid-token weighting.

- [ ] **Step 4: Implement zero-signal skip and metric aggregation**

Check canonical action advantages before reference/policy work. Aggregate `clip_fraction` with valid-token weights and return counts/timing for empty, skipped, and trained paths.

- [ ] **Step 5: Run trainer regressions and commit**

Run:

```bash
conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'train_on_rollouts or sequence_logprobs or grpo_loss'
```

Then:

```bash
git add src/rl_training/config.py src/rl_training/train_grpo_macorag.py tests/test_rl_training.py
git commit -m "perf: batch reference grpo scoring"
```

### Task 6: Group Diagnostics and Reproducible Metadata

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Adds config: `log_all_group_rollouts: bool = true`.
- Produces: `_rollout_log_payload(...) -> dict[str, Any]`.
- Extends metrics timing without removing existing keys.

- [ ] **Step 1: Write failing payload tests**

Assert metric payload contains group reward min/max/mean/std, rollout/action advantage std, clip fraction, action/token counts, retrieval counters, and distinct initial/post-step synchronization. Assert rollout payload retains existing best fields and contains four compact group members when enabled.

- [ ] **Step 2: Run logging tests and verify RED**

Run: `conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'metrics_payload or rollout_log_payload or train_meta'`

Expected: missing fields/helper failures.

- [ ] **Step 3: Implement payload builders and resolved metadata snapshot**

Use `statistics.pstdev` with zero for singleton groups. Serialize resolved `vars(args)` values that are JSON-compatible plus initial sync time and selected QIDs/counts.

- [ ] **Step 4: Run logging regressions and commit**

Run:

```bash
conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'payload or logging or parse_args'
```

Then:

```bash
git add src/rl_training/config.py src/rl_training/train_grpo_macorag.py tests/test_rl_training.py
git commit -m "feat: log grpo group diagnostics"
```

### Task 7: Batch-Size Benchmark Runner

**Files:**
- Create: `scripts/benchmark_grpo_batch_sizes.sh`
- Create: `scripts/summarize_grpo_batch_benchmark.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Shell inputs: `CONFIG`, `SAMPLE_COUNT`, `BATCH_SIZES`, `TRAIN_GPU`, `BENCHMARK_ROOT`.
- Summary output: `<BENCHMARK_ROOT>/benchmark_summary.json`.

- [ ] **Step 1: Write failing dry-run and summary tests**

Assert `DRY_RUN=1` prints three commands with `--max-total-samples 20`, unique output roots, and batch sizes 1/2/4 without starting training. Feed synthetic metrics/memory files to the summary script and assert the 5% valid-completion-tokens-per-training-second selection rule prefers the correct successful candidate.

- [ ] **Step 2: Run benchmark-runner tests and verify RED**

Run: `conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'batch_benchmark'`

Expected: missing script failures.

- [ ] **Step 3: Implement fail-fast sequential runner**

Use `/data/conda/envs/macorag/bin/python`, `PYTHONPATH=src`, explicit `--max-total-samples`, and `--output-root`. Poll physical GPU memory once per second while each child is alive. Retain logs on failure and stop larger candidates after CUDA OOM.

- [ ] **Step 4: Implement summary and dry-run verification**

Run:

```bash
bash -n scripts/benchmark_grpo_batch_sizes.sh
DRY_RUN=1 SAMPLE_COUNT=20 BATCH_SIZES='1 2 4' bash scripts/benchmark_grpo_batch_sizes.sh
conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py -k 'batch_benchmark'
```

- [ ] **Step 5: Commit benchmark runner**

```bash
git add scripts/benchmark_grpo_batch_sizes.sh scripts/summarize_grpo_batch_benchmark.py tests/test_rl_training.py
git commit -m "feat: benchmark grpo action batch sizes"
```

### Task 8: Pre-Benchmark Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run relevant regression suite**

```bash
conda run --no-capture-output -n macorag python -m pytest -q tests/test_rl_training.py tests/test_rag.py tests/test_evaluation.py tests/test_vllm_lora_server.py -k 'not single_gpu_script_forces_hf_offline_mode'
```

- [ ] **Step 2: Run compile and configuration checks**

```bash
conda run --no-capture-output -n macorag python -m compileall -q src/rl_training src/rag src/data_processing LinearRAG/src scripts/summarize_grpo_batch_benchmark.py
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m rl_training.train_grpo_macorag --config config/train_grpo.yml --check-only --max-total-samples 20
git diff --check
```

Expected: all commands exit zero; check-only selects exactly 20 balanced samples.

### Task 9: Execute 20-Sample Batch-Size Benchmark

**Files:**
- Runtime artifacts only under `outputs/grpo_batchsize_benchmark/`.

- [ ] **Step 1: Verify exclusive training GPU and vLLM health**

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits
curl -fsS http://127.0.0.1:8000/health/
pgrep -af 'rl_training.train_grpo_macorag' | grep -v 'pgrep -af' || true
```

Expected: no trainer process, GPU 1 available, health reports the configured LoRA identity.

- [ ] **Step 2: Run candidates sequentially**

```bash
SAMPLE_COUNT=20 BATCH_SIZES='1 2 4' TRAIN_GPU=1 \
BENCHMARK_ROOT=outputs/grpo_batchsize_benchmark/2026-08-18_correctness_v1 \
bash scripts/benchmark_grpo_batch_sizes.sh
```

Expected: each successful candidate processes the identical 20 QIDs; OOM stops larger candidates.

- [ ] **Step 3: Validate artifacts and selection**

```bash
/data/conda/envs/macorag/bin/python scripts/summarize_grpo_batch_benchmark.py \
  outputs/grpo_batchsize_benchmark/2026-08-18_correctness_v1
```

Inspect every candidate's metrics for finite loss/KL/reward, parser errors, selected QID order, peak memory, and timing. Report the selected action batch size and whether reference batch 4 remained viable.

### Task 10: Final Verification and Review

**Files:**
- Verify only.

- [ ] **Step 1: Run fresh regression and repository checks**

Repeat Task 8 commands after benchmark execution and run `git status --short` to distinguish the preserved user config edit from implementation changes.

- [ ] **Step 2: Review the complete commit range**

Inspect `934f570^..HEAD` for correctness, compatibility, accidental output artifacts, and unresolved important issues. Do not commit benchmark outputs.

- [ ] **Step 3: Hand off evidence**

Report commits, exact test counts, output directories, per-candidate mean/median/P90/total times, valid completion tokens per training-side second, peak memory, invalid output counts, skipped zero-signal steps, and the selected batch size. State any unrelated pre-existing test failures separately.
