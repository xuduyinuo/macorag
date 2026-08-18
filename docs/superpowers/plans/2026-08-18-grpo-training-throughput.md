# GRPO Training Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MACORAG online RAG-GRPO use one-time retrieval preparation, batched vLLM rollout generation with behavior logprobs, and padded action microbatches while preserving shared-policy role/round credit semantics.

**Architecture:** Add batch-native retrieval, generation, rollout, and logprob primitives while keeping all current single-item methods as wrappers. The training rollout path uses a masked batch state machine across the GRPO group; the optimizer path flattens actions into padded microbatches and applies the existing masked GRPO loss.

**Tech Stack:** Python 3, PyTorch, PEFT, vLLM, FastAPI, LinearRAG, pytest.

## Global Constraints

- Preserve one shared Query/Evidence/Answer policy and current `(role, round_index)` action advantages.
- Do not change reward definitions, sampling parameters, group size, maximum rounds, model paths, or adapter paths.
- Preserve current single-query retrieval, single-prompt generation, and `RAGLoopExecutor.run()` interfaces.
- Use vLLM chosen-token logprobs when available and explicitly fall back to HF rescoring otherwise.
- Do not claim hardware speedup without a separate real GPU run.
- Write each behavior test first and observe the expected failure before production edits.

---

### Task 1: Cache LinearRAG Query State and Add Batch Retrieval

**Files:**
- Modify: `LinearRAG/src/LinearRAG.py:19-248`
- Modify: `src/data_processing/retrieval.py:60-137`
- Modify: `src/rl_training/retrieval.py:10-81`
- Test: `tests/test_rl_training.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `LinearRAG._prepare_retrieval_state() -> None`
- Produces: `LinearRAGQueryEngine.query_batch(queries: list[str]) -> list[RetrievalResult]`
- Produces: `CachedLinearRAGRetrievalEnv.query_batch(dataset: str, queries: list[str]) -> list[dict[str, Any]]`
- Preserves: both existing `query(dataset, query)` wrappers.

- [ ] **Step 1: Write a failing test for idempotent LinearRAG preparation**

Create a lightweight `LinearRAG` instance with fake embedding stores and monkeypatch `_precompute_sparse_matrices` to count calls. Call `retrieve()` twice and assert that embedding materialization and sparse preparation occur once.

```python
def test_linearrag_prepares_query_state_only_once(monkeypatch):
    engine = object.__new__(LinearRAG)
    engine._retrieval_state_prepared = False
    engine.config = SimpleNamespace(use_vectorized_retrieval=True)
    engine.entity_embedding_store = SimpleNamespace(hash_id_to_text={"e": "entity"}, embeddings=[[1.0]])
    engine.passage_embedding_store = SimpleNamespace(hash_id_to_text={"p": "passage"}, embeddings=[[1.0]])
    engine.sentence_embedding_store = SimpleNamespace(hash_id_to_text={"s": "sentence"}, embeddings=[[1.0]])
    engine.graph_loaded = False
    engine.ner_mappings_loaded = True
    engine.entity_hash_id_to_sentence_hash_ids = {"e": {"s"}}
    engine.sentence_hash_id_to_entity_hash_ids = {"s": {"e"}}
    monkeypatch.setattr(engine, "_ensure_graph_ready_for_query", lambda: False)
    calls = []
    monkeypatch.setattr(engine, "_precompute_sparse_matrices", lambda: calls.append("sparse"))
    engine._prepare_retrieval_state()
    engine._prepare_retrieval_state()
    assert calls == ["sparse"]
    assert engine._retrieval_state_prepared is True
```

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run: `pytest -q tests/test_rl_training.py -k 'linearrag_prepares_query_state_only_once'`

Expected: FAIL because `_prepare_retrieval_state` and its guard do not exist and current `retrieve()` rebuilds sparse state twice.

- [ ] **Step 3: Move immutable query preparation behind an idempotent guard**

In `LinearRAG`, initialize `_retrieval_state_prepared = False`, move lines that materialize hash IDs/NumPy arrays, graph mappings, passage node indices, and sparse matrices into `_prepare_retrieval_state()`, then set the guard only after successful preparation. Make `retrieve()` call it once before its question loop.

```python
def _prepare_retrieval_state(self):
    if getattr(self, "_retrieval_state_prepared", False):
        return
    self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text)
    self.entity_embeddings = np.asarray(self.entity_embedding_store.embeddings)
    # passage/sentence arrays and graph/vectorized preparation
    self._retrieval_state_prepared = True
```

- [ ] **Step 4: Run the focused test and confirm it passes**

Run: `pytest -q tests/test_rl_training.py -k 'linearrag_prepares_query_state_only_once'`

Expected: `1 passed`.

- [ ] **Step 5: Write failing tests for ordered batch retrieval and wrappers**

Test that one `LinearRAGQueryEngine.query_batch(["q1", "q2"])` call invokes `engine.retrieve()` once, returns two ordered `RetrievalResult` objects, and that `query("q1")` delegates consistently. Test that `CachedLinearRAGRetrievalEnv.query_batch()` acquires one engine and returns normalized passage IDs/scores in order; empty input returns `[]`.

- [ ] **Step 6: Run the new batch retrieval tests and verify failure**

Run: `pytest -q tests/test_rl_training.py tests/test_evaluation.py -k 'batch_retrieval or query_batch'`

Expected: FAIL because neither batch method exists.

- [ ] **Step 7: Implement query batch methods and single-item wrappers**

Implement `query_batch()` in both layers. Validate output count against input count and raise `RuntimeError` on mismatch. Keep the dataset lock around one complete batch engine call.

```python
def query(self, query: str) -> RetrievalResult:
    return self.query_batch([query])[0]

def query_batch(self, queries: list[str]) -> list[RetrievalResult]:
    if not queries:
        return []
    rows = self.engine.retrieve([{"question": item} for item in queries])
    if len(rows) != len(queries):
        raise RuntimeError("LinearRAG returned a mismatched batch size.")
    return [self._to_result(query, row) for query, row in zip(queries, rows)]
```

- [ ] **Step 8: Run retrieval regression tests**

Run: `pytest -q tests/test_rl_training.py tests/test_evaluation.py -k 'retrieval or linearrag or query_batch'`

Expected: all selected tests pass.

- [ ] **Step 9: Commit the retrieval optimization**

```bash
git add LinearRAG/src/LinearRAG.py src/data_processing/retrieval.py src/rl_training/retrieval.py tests/test_rl_training.py tests/test_evaluation.py
git commit -m "perf: cache and batch linearrag retrieval"
```

---

### Task 2: Return Batched vLLM Completions with Behavior Logprobs

**Files:**
- Modify: `src/rl_training/vllm_lora_server.py:292-372`
- Modify: `src/rl_training/vllm_client.py:56-175`
- Modify: `src/rl_training/policy.py:15-201`
- Test: `tests/test_vllm_lora_server.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `VLLMGenerationOutput(completion_ids: list[int], logprobs: list[float] | None, text: str = "")`
- Produces: `VLLMGenerationClient.generate_batch(prompts: list[str], ...) -> list[VLLMGenerationOutput]`
- Produces: `VLLMSharedPolicy.generate_batch(requests, traces) -> list[str]`
- Preserves: `VLLMGenerationClient.generate(prompt, ...) -> tuple[list[int], str]` and policy `generate(...) -> str`.

- [ ] **Step 1: Write failing server tests for completion/logprob alignment**

Extend fake vLLM outputs so each completion exposes `token_ids` and per-token logprob dictionaries. POST two prompts to `/generate/` and assert:

```python
assert response.json() == {
    "completion_ids": [[10, 11], [20]],
    "logprobs": [[-0.1, -0.2], [-0.3]],
}
assert sampling_params.kwargs["logprobs"] == 1
```

Also test a malformed/missing chosen-token entry produces HTTP 500 with a useful alignment error.

- [ ] **Step 2: Run the server tests and observe failure**

Run: `pytest -q tests/test_vllm_lora_server.py -k 'generate and logprob'`

Expected: FAIL because the endpoint currently returns only `completion_ids` and does not request logprobs.

- [ ] **Step 3: Implement chosen-token logprob extraction on the server**

Set `logprobs=1` in `SamplingParams`. For each output token, find its chosen-token entry and extract `.logprob` or a numeric value. Return aligned nested arrays and fail loudly on count mismatch.

- [ ] **Step 4: Run the focused server tests and confirm green**

Run: `pytest -q tests/test_vllm_lora_server.py -k 'generate and logprob'`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing client batch tests**

Test an empty prompt batch avoids the backend call, two prompts are submitted in one request, output order is preserved, and a completion/logprob length mismatch raises `RuntimeError`. Test the legacy `generate()` tuple remains unchanged.

- [ ] **Step 6: Run client tests and observe failure**

Run: `pytest -q tests/test_rl_training.py -k 'vllm_generation_client and (batch or logprob or legacy)'`

Expected: FAIL because `generate_batch` and `VLLMGenerationOutput` do not exist.

- [ ] **Step 7: Implement batch client parsing with dense-server fallback**

Use the backend session `/generate/` response when it provides nested `completion_ids` and `logprobs`. Otherwise normalize the existing TRL `generate()` return and set `logprobs=None`. Validate prompt/output count and per-output token/logprob alignment.

- [ ] **Step 8: Write failing policy tests proving no HF rescoring when logprobs exist**

Monkeypatch `sequence_logprobs` to raise, return vLLM logprobs from the fake client, generate two requests, and assert both traces contain those exact CPU tensors. Add a fallback test where `logprobs=None` and scalar rescoring is called.

- [ ] **Step 9: Implement batch policy requests and trace routing**

Add a small generation-request data class carrying role/question/state/observation. Encode prompts in input order, submit one client batch, decode outputs, and append each `GeneratedAction` to its explicit `RolloutTrace`. Derive round indices independently per trace. The scalar method delegates through one request and `self.trace`.

- [ ] **Step 10: Run client/policy/server regressions**

Run: `pytest -q tests/test_rl_training.py tests/test_vllm_lora_server.py -k 'vllm or policy_generate'`

Expected: all selected tests pass.

- [ ] **Step 11: Commit vLLM batching and behavior logprobs**

```bash
git add src/rl_training/vllm_lora_server.py src/rl_training/vllm_client.py src/rl_training/policy.py tests/test_vllm_lora_server.py tests/test_rl_training.py
git commit -m "perf: batch vllm rollout generation logprobs"
```

---

### Task 3: Execute Each GRPO Group as a Masked Batched Rollout

**Files:**
- Create: `src/rl_training/batched_rollout.py`
- Modify: `src/rl_training/train_grpo_macorag.py:243-296`
- Test: `tests/test_rl_training.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: policy `generate_batch(requests, traces)` from Task 2.
- Consumes: retrieval `query_batch(dataset, queries)` from Task 1.
- Produces: `run_batched_rollouts(question, dataset, group_size, max_rounds, policy, retrieval_env) -> list[BatchedRolloutResult]`
- Each result exposes the same `RAGLoopResult` fields and one `RolloutTrace`.

- [ ] **Step 1: Write a failing one-round batch test**

Use a fake batch policy that records request roles and a fake retrieval environment that records query lists. For group size four and immediate correct answers, assert exactly three policy batch calls with four requests each, one retrieval batch call with four queries, four independent traces, and four completed results.

- [ ] **Step 2: Run the focused test and observe failure**

Run: `pytest -q tests/test_rl_training.py -k 'batched_rollout_one_round'`

Expected: FAIL because the batch rollout module does not exist.

- [ ] **Step 3: Implement the minimal candidate state machine**

Create an internal candidate data class holding state, trajectory, parse errors, final answer, active flag, pending turn data, and trace. Implement Query, Evidence, and Answer phases using the same state-copy and trajectory dictionaries as `RAGLoopExecutor`.

- [ ] **Step 4: Run the one-round test and confirm green**

Run: `pytest -q tests/test_rl_training.py -k 'batched_rollout_one_round'`

Expected: `1 passed`.

- [ ] **Step 5: Write failing tests for divergent candidates**

Cover:

- one candidate answers in round zero while another continues;
- Query parse failure affects only that candidate;
- Evidence parse failure preserves its Query action and `generated_roles`;
- Answer parse failure preserves Query/Evidence actions and records `parse_error_role`;
- empty query skips only its retrieval item while maintaining candidate ordering.

- [ ] **Step 6: Run divergent-candidate tests and observe expected failures**

Run: `pytest -q tests/test_rl_training.py -k 'batched_rollout and (divergent or parse or empty_query)'`

Expected: FAIL on missing masked/partial-turn behavior.

- [ ] **Step 7: Complete masked phase transitions and partial-turn handling**

Mirror `RAGLoopExecutor` dictionaries exactly. Maintain a mapping from compact active request indices back to candidate indices at every phase. Append a trajectory turn at completion or parse failure and never mutate an inactive candidate.

- [ ] **Step 8: Integrate `_rollout_group()` with batch results**

Replace the group loop with one `run_batched_rollouts()` call. Build the existing rollout dictionaries in group-index order, compute the same rollout/action rewards, normalize the same role/round buckets, and keep timing keys backward compatible while adding retrieval and fallback-rescore detail.

- [ ] **Step 9: Run rollout, reward, and action-credit tests**

Run: `pytest -q tests/test_rl_training.py tests/test_rag.py -k 'rollout or action_reward or action_advantage or rag_executor'`

Expected: all selected tests pass.

- [ ] **Step 10: Commit batched group rollout**

```bash
git add src/rl_training/batched_rollout.py src/rl_training/train_grpo_macorag.py tests/test_rl_training.py tests/test_rag.py
git commit -m "perf: batch grpo group rollouts"
```

---

### Task 4: Train Actions in Padded Microbatches

**Files:**
- Modify: `src/rl_training/policy.py:204-242`
- Modify: `src/rl_training/train_grpo_macorag.py:299-378`
- Modify: `config/train_grpo.yml:22-31`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `batched_sequence_logprobs(model, prompt_id_batches, completion_id_batches, device, pad_token_id) -> tuple[Tensor, Tensor]`
- Consumes: `args.per_device_train_batch_size` as the number of actions per microbatch.
- Preserves: scalar `sequence_logprobs()` and `compute_grpo_loss()`.

- [ ] **Step 1: Write a failing scalar-versus-batch logprob equivalence test**

Use a deterministic toy causal LM with two different prompt and completion lengths. Compare each valid row/token returned by `batched_sequence_logprobs()` against separate `sequence_logprobs()` calls and assert the padded mask is zero.

- [ ] **Step 2: Run the logprob test and observe failure**

Run: `pytest -q tests/test_rl_training.py -k 'batched_sequence_logprobs'`

Expected: FAIL because the batch function does not exist.

- [ ] **Step 3: Implement left-padded batch logprob extraction**

Construct left-padded `input_ids` and attention masks. Run one model forward. Build explicit batch/token indices for each completion token's predicting logit, gather selected vocabulary logits, apply `log_softmax`, gather chosen token IDs, and return `[batch, max_completion]` logprobs plus a boolean completion mask.

- [ ] **Step 4: Run scalar/batch equivalence test and confirm green**

Run: `pytest -q tests/test_rl_training.py -k 'batched_sequence_logprobs'`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing microbatch trainer tests**

For five actions and `per_device_train_batch_size=2`, assert three policy forwards, three reference forwards, and three backward calls. Assert batch size one retains the current per-action loss/advantage behavior. Assert changing padded token values does not change loss.

- [ ] **Step 6: Run microbatch tests and observe failure**

Run: `pytest -q tests/test_rl_training.py -k 'train_on_rollouts and (microbatch or padding or batch_size)'`

Expected: FAIL because `_train_on_rollouts()` ignores `per_device_train_batch_size` and loops per action.

- [ ] **Step 7: Implement stable action microbatches**

Flatten actions in rollout/action order. Slice by `max(1, args.per_device_train_batch_size)`, pad stored old logprobs to the returned completion width, broadcast action advantages, call policy/reference batch logprobs once each, and pass the completion mask to `compute_grpo_loss()`.

Scale each microbatch loss by its valid-token count divided by total valid tokens and by `gradient_accumulation_steps`. Aggregate metrics with the same valid-token weights.

- [ ] **Step 8: Add forward timing and active config**

Record policy/reference forward seconds separately and expose `per_device_train_batch_size: 1` in `config/train_grpo.yml`. Do not alter other defaults.

- [ ] **Step 9: Run all RL unit tests**

Run: `pytest -q tests/test_rl_training.py -k 'not single_gpu_script_forces_hf_offline_mode'`

Expected: all selected tests pass with the known unrelated single-GPU fixture deselected.

- [ ] **Step 10: Commit action microbatch training**

```bash
git add src/rl_training/policy.py src/rl_training/train_grpo_macorag.py config/train_grpo.yml tests/test_rl_training.py
git commit -m "perf: train grpo actions in microbatches"
```

---

### Task 5: Integration Verification and Deterministic Throughput Regression

**Files:**
- Modify: `tests/test_rl_training.py`
- Modify: `tests/test_vllm_lora_server.py`
- Modify if needed: implementation files from Tasks 1-4

**Interfaces:**
- Verifies all acceptance criteria from `docs/superpowers/specs/2026-08-18-grpo-training-throughput-design.md`.

- [ ] **Step 1: Add a deterministic call-count regression test**

Run a group size four, one-round toy rollout and one training step. Assert one retrieval preparation, one retrieval batch, three generation batches, no HF behavior rescore when server logprobs exist, and microbatch-sized policy/reference forwards.

- [ ] **Step 2: Run the deterministic throughput test**

Run: `pytest -q tests/test_rl_training.py -k 'throughput_call_counts'`

Expected: `1 passed`.

- [ ] **Step 3: Run the complete relevant test suite**

Run: `pytest -q tests/test_rl_training.py tests/test_rag.py tests/test_evaluation.py tests/test_vllm_lora_server.py -k 'not single_gpu_script_forces_hf_offline_mode'`

Expected: zero failures; only the explicitly excluded pre-existing fixture is deselected.

- [ ] **Step 4: Run syntax and whitespace checks**

Run: `python -m compileall -q src/rl_training src/rag src/data_processing LinearRAG/src`

Expected: exit code 0.

Run: `git diff --check`

Expected: exit code 0 with no output.

- [ ] **Step 5: Run check-only configuration validation**

Run: `PYTHONPATH=src python -m rl_training.train_grpo_macorag --config config/train_grpo.yml --check-only --max-samples 1`

Expected: dataset summary followed by `Check-only complete. No model training started.`

- [ ] **Step 6: Review acceptance criteria and document hardware boundary**

Confirm the tests prove reduced expensive call counts and semantic compatibility. State explicitly that GPU wall-clock acceleration remains unmeasured unless a vLLM server and free trainer GPU are available for a controlled smoke benchmark.

- [ ] **Step 7: Commit integration tests or final fixes**

```bash
git add tests/test_rl_training.py tests/test_vllm_lora_server.py src/rl_training src/data_processing LinearRAG/src config/train_grpo.yml
git commit -m "test: verify grpo throughput optimizations"
```
