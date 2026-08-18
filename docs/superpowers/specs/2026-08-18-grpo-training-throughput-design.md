# GRPO Training Throughput Optimization Design

## Goal

Reduce MACORAG online RAG-GRPO wall-clock time by removing repeated retrieval preparation, exposing real vLLM batching, avoiding redundant behavior-policy forward passes, and training generated actions in padded microbatches. Preserve the existing shared policy, role/round action-credit assignment, reward definitions, generated trajectories, and single-item APIs.

## Evidence and Scope

The existing 7B run recorded 1,480 samples and 35.6 hours of measured work. Rollout consumed 67.2% of total time, while training forward and backward consumed about 32.5%. Reward computation and LoRA synchronization together consumed less than 0.4%.

This change covers four coupled bottlenecks:

1. LinearRAG query-time state is rebuilt for every query.
2. A GRPO group is generated one trajectory at a time even though the vLLM server accepts prompt lists.
3. vLLM generations are rescored one action at a time by the HF policy to obtain behavior logprobs.
4. Policy/reference logprobs and backward passes are computed one action at a time; `per_device_train_batch_size` is currently unused.

It does not change reward weights, action-advantage normalization, retrieval ranking, GRPO equations, sampling parameters, maximum rounds, group size, or model/adapters.

## Approach Selection

### Selected: incremental batch interfaces with compatibility wrappers

Add batch-native primitives and retain existing single-item methods as wrappers. This gives tests small boundaries, avoids a second training implementation, and permits rollout batching without breaking evaluation or existing callers.

### Rejected: thread the existing single-item executor

Concurrent threads could submit overlapping vLLM requests, but shared policy traces are mutable and the retrieval environment serializes each dataset with a lock. This approach would complicate ordering and reproducibility without producing true prompt batching.

### Rejected: replace the trainer with TRL GRPOTrainer

That would be a larger algorithm and data-contract migration. It risks losing the current multi-role state machine and role/round credit assignment, so it is outside this optimization.

## Retrieval Preparation

`LinearRAG` will have an idempotent `_prepare_retrieval_state()` method. It materializes immutable hash-id lists, NumPy embedding matrices, graph mappings, passage node indices, and optional sparse entity/sentence matrices once per engine. `retrieve()` calls this guard but never rebuilds prepared data.

`LinearRAGQueryEngine.query_batch(queries)` will pass all questions to one `retrieve()` call and map results back in input order. `query(query)` remains a one-item wrapper. `CachedLinearRAGRetrievalEnv.query_batch(dataset, queries)` will acquire the dataset lock once and return normalized observations in order; `query()` remains compatible.

Preparation is allowed to be expensive once during prewarm. Query results before and after the change must match for the same engine, index, and input.

## vLLM Generation and Behavior Logprobs

Introduce a result value carrying `completion_ids`, decoded text, and optional chosen-token logprobs. `VLLMGenerationClient.generate_batch(prompts, ...)` submits the complete prompt list in one backend request and preserves output order. Existing `generate(prompt, ...)` delegates to it and returns the legacy tuple for compatibility.

The LoRA server requests one chosen-token logprob per generated token and returns both completion IDs and aligned logprobs. The client validates that each completion and logprob sequence has identical length. For compatibility with an older or fake backend that lacks logprobs, the policy may fall back to HF rescoring; the real MACORAG LoRA server must return logprobs.

Behavior logprobs must be produced by the exact vLLM policy that sampled each completion. This remains correct if policy synchronization is later made less frequent.

## Batched Rollout State Machine

Add a batch executor for one dataset and one question replicated `group_size` times. It maintains independent state, trajectory, parse errors, final answer, and generated action trace for each candidate.

For each round it performs three masked phases:

1. Batch-generate Query actions for active candidates, parse them, then batch-retrieve nonempty queries.
2. Batch-generate Evidence actions for candidates that passed Query parsing, parse selections, and update their candidate state.
3. Batch-generate Answer actions for candidates that passed Evidence parsing, then finalize or keep each candidate active.

A parse failure only terminates the affected candidate and preserves its previously generated roles, matching current partial-turn semantics. Candidate ordering is stable from group index through reward assignment. The existing `RAGLoopExecutor.run()` remains unchanged for evaluation and non-training callers.

The shared policy receives explicit trace slots for a batch; it must not use one mutable trace concurrently. Each generated action keeps the current `role`, `round_index`, prompt/completion IDs, response, behavior logprobs, and later credit fields.

## Batched Training

Introduce a batch logprob function that accepts variable-length prompt/completion sequences, pads complete sequences, builds attention and completion masks, and returns chosen-token logprobs plus the completion mask. It must preserve the current truncation and next-token alignment.

`_train_on_rollouts()` will flatten trainable actions in stable order and split them into microbatches controlled by `per_device_train_batch_size`. For each microbatch it performs one policy forward, one no-grad reference forward, one masked GRPO loss, and one backward. Padding tokens contribute neither policy loss nor KL.

Per-action advantages are broadcast across that action's completion tokens. Metrics remain token-masked and are aggregated with token-count weighting so padding and variable action lengths do not bias the result. Gradient accumulation retains its existing sample-step meaning.

The reference model remains separate in this change. Adapter sharing or KL removal is not part of the throughput optimization.

## Configuration and Observability

`per_device_train_batch_size` becomes an active RL training control and will be exposed in `config/train_grpo.yml`, initially set to `1` for behavior compatibility. Users may raise it after measuring available VRAM.

Timing logs will separate:

- retrieval preparation;
- retrieval query execution;
- vLLM generation;
- behavior-logprob fallback rescoring;
- policy/reference forward;
- backward;
- optimizer and weight synchronization.

No default sampling or training-quality parameter changes are included.

## Error Handling

- Batch methods reject mismatched input/output counts.
- Server/client reject completion/logprob length mismatches.
- Empty prompt batches return empty result lists without a server call.
- Empty retrieval queries retain the existing empty-observation behavior.
- A failed candidate does not corrupt other candidate states or traces.
- If vLLM logprobs are unavailable, fallback rescoring is explicit and timed.

## Testing

Tests will be added before implementation for:

1. Retrieval preparation happens once across repeated and batched queries.
2. Batch retrieval preserves ordering and single-query compatibility.
3. vLLM client/server preserve prompt order and aligned chosen-token logprobs.
4. Single generation remains backward compatible.
5. Batched rollout matches current state transitions, early stopping, and partial parse failure behavior.
6. Batched logprobs match the existing scalar implementation on deterministic toy models.
7. Padded completion tokens do not affect loss or metrics.
8. Microbatch size controls the number of model forwards/backwards without changing loss at batch size one.
9. Existing action-credit, RAG, vLLM LoRA server, and RL training tests remain green.

## Acceptance Criteria

- Existing single-item interfaces and role/round credit tests remain compatible.
- A group of four active candidates produces one vLLM request per role/round phase rather than four requests.
- Real vLLM generations no longer require HF behavior-policy rescoring.
- Retrieval arrays and sparse matrices are materialized at most once per cached engine.
- RL action training honors `per_device_train_batch_size` with correct masks.
- A local deterministic benchmark demonstrates fewer retrieval preparations, vLLM calls, and model forward/backward invocations; a full GPU speedup claim requires a separate hardware run.
