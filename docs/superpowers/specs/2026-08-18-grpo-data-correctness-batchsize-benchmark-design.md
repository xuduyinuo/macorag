# GRPO Data Correctness and Batch-Size Benchmark Design

## Goal

Repair the correctness defects found in `outputs/grpo_qwen2.5-7b/2026-08-18_17-01-03`, improve the diagnostic data needed to distinguish rollout, retrieval, and optimizer bottlenecks, then benchmark action microbatch sizes 1, 2, and 4 on the same deterministic balanced set of 20 RL samples. The benchmark must select the fastest configuration that does not OOM and does not introduce non-finite training metrics.

## Evidence Driving the Design

The inspected run completed 150 steps in 4552.9 seconds. Rollout consumed 46.0% of wall time, while policy forward, reference forward, and backward together consumed 50.9%. Behavior rescoring was eliminated and weight synchronization was only 0.9%, so those are not current priorities.

The run also exposed correctness defects:

- 28 final answer actions used non-boolean `can_answer` values; 21 were the string `"false"`. The executor treats every non-empty string as true and terminates the trajectory.
- Reward code has a separate truth-value conversion, so execution and reward semantics disagree.
- Samples are processed as three contiguous dataset blocks instead of a deterministic interleaving.
- A persistent vLLM server may retain the previous run's hot-synced LoRA weights. The trainer validates server identity but does not synchronize the newly loaded policy before the first rollout.
- Only the best trajectory is persisted, hiding group reward variance, invalid output rates, and padding/token statistics.

## Considered Approaches

### Approach A: Configuration-only tuning

Increase `per_device_train_batch_size`, temperature, and KL strength without changing validation or logging. This is fast to attempt but cannot repair corrupted termination semantics and would make benchmark results depend on stale vLLM state. Rejected.

### Approach B: Correctness-first incremental optimization

Normalize and validate generated actions, synchronize vLLM before rollout, deterministically interleave datasets, extend observability, then apply bounded performance changes and benchmark one variable at a time. This is the selected approach because it produces comparable runs and keeps reward definitions unchanged.

### Approach C: Replace the rollout protocol with schema-constrained generation

Adopt role-specific vLLM JSON-schema decoding and redesign all tagged responses. This could eliminate malformed output at generation time, but it expands the server/client protocol and makes the batch-size benchmark depend on a second major behavioral change. Deferred until after the correctness-first benchmark.

## P0 Correctness Design

### Canonical Generated Actions

`src/rag/parser.py` will be the single normalization boundary.

- `answer.can_answer` accepts JSON booleans and the case-insensitive strings `"true"` and `"false"`, returning a real Python `bool`.
- Values such as `"yes"`, `1`, lists, and null are rejected with a role-specific `ValueError`.
- `answer.answer` and `answer.rationale`, when present, must be strings. `answer.answer` remains required.
- Query `sub_goal` and `query` must be strings; `sub_goal` remains non-empty.
- `selected_passage_ids` must be a list containing only integer IDs. Booleans are rejected even though `bool` subclasses `int` in Python.
- Once parsed, executors use `answer["can_answer"] is True`; they never apply generic Python truthiness.

The reward module will import the same boolean normalizer for defensive handling of historical artifacts. New rollouts will only contain canonical values.

### Initial vLLM Synchronization

After the training policy, optimizer, retrieval environment, and vLLM client are initialized—but before the first rollout—the trainer will synchronize the loaded policy adapter to the vLLM server. This applies even when `vllm_sync_every_steps` is greater than one. The initial synchronization duration will be stored separately from post-step synchronization.

This makes independent training and benchmark processes start from identical SFT adapter weights even when they reuse a long-running vLLM server.

### Deterministic Balanced Ordering

The loader keeps its existing per-dataset cap. A new optional `max_total_samples` limits the final training set after loading. Selection and ordering use `seed` and are deterministic:

1. Group samples by dataset.
2. Shuffle each dataset bucket with a local RNG derived from `seed`.
3. Round-robin across sorted dataset names until the total limit is reached.
4. At each epoch, deterministically reshuffle the selected samples using `seed + epoch` before distributed rank partitioning.

For the benchmark, `max_total_samples=20` yields a balanced 7/7/6 sample allocation. Every candidate batch size receives the same ordered QIDs.

## Diagnostic Data Design

### Group-Level Logging

Each training metric record will add:

- group reward minimum, maximum, mean, and population standard deviation;
- rollout and action advantage population standard deviation;
- count of trainable actions and valid completion tokens;
- policy clip fraction;
- skipped-update reason when no policy signal exists;
- retrieval time and retrieval-cache hits/misses;
- initial and post-step vLLM synchronization times as distinct fields.

The dataset-scoped rollout JSONL keeps the existing best-rollout fields for compatibility and adds a compact `group_rollouts` list. Each group member records reward components, parse errors, final answer, generated action count, trajectory, and action credit. This is enabled with `log_all_group_rollouts`; the benchmark enables it.

`train_meta.json` will snapshot all resolved argument values needed to reproduce the run, including action and reference batch sizes, quantization, gradient checkpointing, sample selection, shuffle seed, retrieval cache size, and initial vLLM synchronization.

### Retrieval Timing and Cache

`CachedLinearRAGRetrievalEnv` will maintain bounded per-dataset LRU caches keyed by normalized query text. Scalar and batch APIs share the cache. Batch retrieval sends only cache misses to LinearRAG, restores original query order, and returns independent observation copies so downstream mutation cannot corrupt cached values.

The cache exposes cumulative hit, miss, and retrieval-duration counters. `_rollout_group()` records per-sample deltas. A cache size of zero disables this feature.

## P1 Training-Side Performance Design

### Independent Reference Batching

`reference_per_device_batch_size` controls reference-model forward batches independently from policy action microbatches. The fixed reference logprobs are computed under `torch.no_grad()` in larger chunks and stored as small per-token tensors for the duration of one sample update. Policy forward and backward remain controlled by `per_device_train_batch_size`.

The default reference batch size is 4. If it OOMs, the benchmark runner reports the failure and retries with the policy batch size. Loss, masks, stable action order, role/round advantages, and gradient-accumulation semantics remain unchanged.

### No-Signal Groups

Before any policy/reference forward, the trainer checks whether every trainable action advantage is numerically zero. When `skip_zero_advantage_updates=true`, it skips policy, reference, backward, optimizer, and vLLM post-step synchronization for that sample, records `zero_advantage`, and preserves the global sample-step counter.

The first implementation does not automatically resample low-reward collapsed groups. The new group statistics provide the evidence required to design that behavior without conflating it with batch-size selection.

### Gradient Checkpointing

The benchmark does not change gradient checkpointing. Batch size is the only benchmarked variable. After selecting a batch size, a separate A/B run may compare checkpointing on and off; combining both variables would prevent attribution.

## Batch-Size Benchmark Protocol

### Inputs

- Base model and SFT adapter: unchanged from `config/train_grpo.yml`.
- Dataset subset: the same deterministic balanced 20 samples for every candidate.
- Candidates: action microbatch sizes 1, 2, and 4 in ascending order.
- Reference batch size: 4 for every candidate unless it OOMs, in which case it falls back to the action batch size and records the fallback.
- All sampling, reward, KL, learning-rate, and retrieval settings remain fixed.
- Each candidate starts in a new explicit output namespace and performs initial vLLM synchronization.

### Execution

The benchmark runs candidates sequentially on training GPU 1 while the existing vLLM server remains on GPU 0. A monitor samples GPU 1 memory usage during each run. Candidate 4 is not attempted if candidate 2 OOMs. Any failed candidate retains its logs and failure reason.

### Selection Rule

Choose the largest candidate satisfying all of the following:

1. No CUDA OOM or process failure.
2. Finite loss, KL, rewards, and gradients for all 20 samples.
3. Exact same selected QID sequence as the other successful candidates.
4. No increase in parse-error or invalid-action counts attributable to execution differences.
5. At least a 5% reduction in median non-skipped sample wall time versus the next smaller successful candidate; otherwise prefer the smaller batch to preserve memory headroom.

Report mean, median, P90, total wall time, training-side time, rollout time, peak GPU memory, actions/tokens processed, skipped no-signal steps, and output directory for every candidate.

## Testing Strategy

All implementation follows red-green TDD.

- Parser tests reproduce string `"false"`, string `"true"`, invalid `"yes"`, numeric answers, and invalid selected passage IDs.
- Executor tests prove canonical false continues and canonical true stops in scalar and batched rollout paths.
- Ordering tests prove balanced 20-sample selection, deterministic epoch shuffling, no loss/duplication, and consistent distributed partitioning.
- Initial-sync tests prove synchronization happens before generation and only on the main process.
- Retrieval tests cover LRU hits, batch miss deduplication, order preservation, eviction, disabled cache, and copy isolation.
- Trainer tests compare independent reference batching against the existing batch-size-one loss and verify zero-advantage skips perform no model forwards or optimizer step.
- Logging tests verify group statistics, retrieval timing, complete resolved metadata, and backwards-compatible best-rollout fields.
- Final verification runs the relevant RL/RAG/retrieval/vLLM tests, compileall, `--check-only --max-total-samples 20`, and `git diff --check` before starting GPU benchmarks.

## Safety and Artifact Boundaries

- Preserve the user's uncommitted `max_samples: 50` edit in `config/train_grpo.yml`.
- Do not overwrite or resume `2026-08-18_17-01-03`.
- Every benchmark candidate uses a new output root and timestamped run directory.
- Do not restart or terminate the user's vLLM server; initial synchronization makes reuse deterministic.
- Do not select the final adapter based on training best-of-four F1. The benchmark selects throughput configuration only.
