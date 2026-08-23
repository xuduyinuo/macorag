# E5-FAISS Retrieval, Final-Round Fallback, and Agent Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make E5-base-v2 plus FAISS the default MACORAG retriever, force one best-effort final-round answer, and implement paper-aligned per-agent cross-round GRPO credit assignment.

**Architecture:** Add a self-contained E5 encoder/index/query module and a backend factory that preserves LinearRAG as an explicit option. Carry a final-round flag through synchronous and batched policy requests. Replace role-round advantage buckets with role-only buckets over combined decision returns while retaining the token-level clipped GRPO and KL loss.

**Tech Stack:** Python 3.9, PyTorch 2.6, Transformers 4.57, NumPy 1.26, FAISS CPU, argparse/PyYAML, pytest, Bash.

## Global Constraints

- Match R-Search retrieval: `intfloat/e5-base-v2`, `query:` / `passage:` prefixes, masked mean pooling, L2 normalization, max length 512, FAISS `IndexFlatIP`, top-k 5.
- Build independent indexes for `data/eval_1000` and `data/rl_train_2000`; never import or call R-Search at runtime.
- Default query encoding and FAISS search to CPU so GPU 0 remains for vLLM and GPU 1 for GRPO training.
- Preserve `linear_rag` as an explicit non-default backend and preserve all existing LinearRAG assets.
- Final round has one Answer Agent generation only; no retry action.
- Use `lambda_Q=1/3`, `lambda_E=3/7`, `lambda_A=7/3`, population standard deviation, and `advantage_epsilon=1e-8`.
- Preserve all pre-existing dirty changes in overlapping GRPO files.
- Do not claim completion unless complete indexes are built, validated, and relevant tests pass.

---

### Task 1: E5 encoder, deterministic assets, and FAISS query engine

**Files:**
- Create: `src/data_processing/e5_faiss.py`
- Modify: `requirements.txt`
- Modify: `environment.yml`
- Test: `tests/test_retrieval_env.py`

**Interfaces:**
- Produces: `E5Encoder`, `build_e5_faiss_index()`, `E5FaissQueryEngine`, `validate_e5_faiss_assets()`, and `E5FaissResult`.
- Consumes: `<data_root>/<dataset>/corpus.jsonl` and locally loadable E5 weights.

- [ ] **Step 1: Write failing prefix, pooling, metadata, build/load, and validation tests**

Use injected fake tokenizer/model/FAISS objects so RED does not require a model download. The central encoder test is:

```python
encoder = E5Encoder(
    model_name="intfloat/e5-base-v2",
    device="cpu",
    max_length=512,
    tokenizer=FakeTokenizer(),
    model=FakeModel(),
)
query_vectors = encoder.encode_queries(["Who directed Bullitt?"])
passage_vectors = encoder.encode_passages(["Bullitt was directed by Peter Yates."])
assert encoder.tokenizer.inputs == [
    ["query: Who directed Bullitt?"],
    ["passage: Bullitt was directed by Peter Yates."],
]
np.testing.assert_allclose(np.linalg.norm(query_vectors, axis=1), [1.0])
np.testing.assert_allclose(np.linalg.norm(passage_vectors, axis=1), [1.0])
```

The tiny-corpus test builds two passages, reloads the index, queries it, and checks stable ID/title/text and descending score. Separate tests mutate corpus bytes, metadata dimension, and index `ntotal` and expect explicit errors.

- [ ] **Step 2: Run Task 1 tests and confirm RED**

Run: `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_retrieval_env.py -k 'e5 or faiss'`

Expected: import failure because `data_processing.e5_faiss` does not exist.

- [ ] **Step 3: Implement the minimal E5/FAISS module**

Implement these public interfaces exactly:

- `E5FaissResult(dataset: str, query: str, passages: list[dict[str, Any]], scores: list[float])` as a frozen dataclass.
- `E5Encoder.__init__(model_name, device, max_length=512, batch_size=32, tokenizer=None, model=None)`.
- `E5Encoder.masked_mean_pool(last_hidden_state, attention_mask)`.
- `E5Encoder.encode_queries(texts: list[str]) -> np.ndarray`.
- `E5Encoder.encode_passages(texts: list[str]) -> np.ndarray`.
- `build_e5_faiss_index(data_root, output_root, dataset, model_name, device, max_length=512, batch_size=64, encoder=None, faiss_module=None) -> dict[str, Any]`.
- `validate_e5_faiss_assets(retrieval_root, dataset, expected_model=None, faiss_module=None) -> dict[str, Any]`.
- `E5FaissQueryEngine.__init__(retrieval_root, dataset, model_name, device, top_k=5, max_length=512, batch_size=32, encoder=None, faiss_module=None)`.
- `E5FaissQueryEngine.query(query: str) -> E5FaissResult`.
- `E5FaissQueryEngine.query_batch(queries: list[str]) -> list[E5FaissResult]`.

Use lazy imports. Prefix texts, apply attention-mask mean pooling, normalize, and emit contiguous float32 vectors. Hash exact emitted `corpus.jsonl` bytes. Save metadata atomically after index/corpus completion. Exclude FAISS `-1` IDs.

- [ ] **Step 4: Add the explicit dependency**

Add `faiss-cpu==1.9.0.post1` to both `requirements.txt` and `environment.yml`.

- [ ] **Step 5: Run Task 1 tests and confirm GREEN**

Run the Task 1 test command and `/data/conda/envs/macorag/bin/python -m compileall -q src/data_processing/e5_faiss.py`.

- [ ] **Step 6: Commit only safely separable Task 1 files**

Run `git add src/data_processing/e5_faiss.py requirements.txt environment.yml tests/test_retrieval_env.py` and `git commit -m "feat: add native e5 faiss retrieval"`. If a listed file has pre-existing edits, do not stage it wholesale; defer the commit and report why.

---

### Task 2: Retrieval CLI, configuration, and backend factory

**Files:**
- Modify: `src/data_processing/retrieval_cli.py`
- Modify: `src/rl_training/retrieval.py`
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Modify: `src/evaluation/config.py`
- Modify: `src/evaluation/evaluate_rag_model.py`
- Modify: `config/build_retrieval.yml`
- Create: `config/build_retrieval_eval_e5.yml`
- Create: `config/build_retrieval_train_e5.yml`
- Modify: `config/train_grpo.yml`
- Modify: `config/eval_macorag.yml`
- Test: `tests/test_retrieval_env.py`
- Test: `tests/test_evaluation.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Consumes: Task 1 E5 classes.
- Produces: `create_retrieval_env()` and backend-aware `validate_retrieval_assets()`.

- [ ] **Step 1: Write failing CLI, parser, factory, cache, and validation tests**

Parse E5 fields from eval and GRPO YAML, reject unknown backends, and inject fake engines into the factory. Assert batch cache misses are deduplicated and results retain original order. Verify E5 validation requires only `e5_Flat.index`, `corpus.jsonl`, and metadata, while LinearRAG keeps its existing files.

```python
env = create_retrieval_env(
    backend="e5_faiss",
    retrieval_root=tmp_path,
    embedding_model="intfloat/e5-base-v2",
    device="cpu",
    top_k=5,
    max_length=512,
    batch_size=32,
    query_cache_size=8,
    e5_engine_factory=fake_factory,
)
first = env.query_batch("hotpotqa", ["q", "q"])
second = env.query_batch("hotpotqa", ["q"])
assert first[0] == first[1] == second[0]
```

- [ ] **Step 2: Run Task 2 tests and confirm RED**

Run: `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_retrieval_env.py tests/test_evaluation.py tests/test_rl_training.py -k 'retrieval_backend or e5_faiss_config or e5_faiss_factory'`

- [ ] **Step 3: Extend the retrieval CLI**

Add `--backend {e5_faiss,linear_rag}`, `--data-root`, `--device`, and `--max-length`. Build mode dispatches E5 once per dataset; query mode instantiates `E5FaissQueryEngine`. Keep current LinearRAG behavior under `linear_rag`.

- [ ] **Step 4: Implement the runtime factory and cache contract**

Implement `create_retrieval_env()` with keyword-only parameters `backend`, `retrieval_root`, `embedding_model`, `device`, `top_k`, `max_length`, `batch_size`, `query_cache_size=0`, `spacy_model=None`, `max_workers=4`, `use_vectorized_retrieval=True`, and injectable `e5_engine_factory=E5FaissQueryEngine`. Implement `validate_retrieval_assets()` with keyword-only `backend`, `retrieval_root`, `datasets`, and `embedding_model`.

The E5 environment exposes `query`, `query_batch`, `prewarm`, and `stats` and uses existing deep-copy LRU semantics.

- [ ] **Step 5: Wire evaluation and GRPO arguments**

Add defaults/CLI fields `retrieval_backend=e5_faiss`, `retrieval_embedding_model=intfloat/e5-base-v2`, `retrieval_device=cpu`, and `retrieval_max_length=512`. Update both `_build_retrieval_env()` functions to call the factory. Persist backend/model/device/max length/root/top-k in run metadata. Preserve existing resume/retry edits in overlapping files.

- [ ] **Step 6: Add build configs and switch runtime roots**

Use `data/eval_1000 -> data/eval_1000_e5_faiss` and `data/rl_train_2000 -> data/rl_train_2000_e5_faiss`, with model E5-base-v2, CPU default, max length 512, and top-k 5.

- [ ] **Step 7: Run Task 2 tests and confirm GREEN**

Run the Task 2 test command and `/data/conda/envs/macorag/bin/python -m data_processing.retrieval_cli --config config/build_retrieval_eval_e5.yml --help`.

---

### Task 3: Final-round forced-answer contract

**Files:**
- Modify: `src/rag/prompts.py`
- Modify: `src/rag/parser.py`
- Modify: `src/rag/executor.py`
- Modify: `src/rl_training/policy.py`
- Modify: `src/rl_training/batched_rollout.py`
- Modify: `src/evaluation/evaluate_rag_model.py`
- Test: `tests/test_rag.py`
- Test: `tests/test_evaluation.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: keyword-only `build_answer_generator_prompt(force_final_answer: bool = False)` in addition to its existing arguments, plus `PolicyGenerationRequest.force_final_answer`.

- [ ] **Step 1: Write failing prompt, parser, synchronous, and batched tests**

```python
prompt = build_answer_generator_prompt(
    question="q",
    state=RAGState(question="q"),
    force_final_answer=True,
)
assert "can_answer=true" in prompt
assert "fallback_guess:" in prompt
assert "can_answer=false" not in prompt
```

Fake policies capture the final flag and return false in an early round, then a marked answer in the final round. Violation tests return final false/null and expect `final_answer_required`.

- [ ] **Step 2: Run Task 3 tests and confirm RED**

Run: `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rag.py tests/test_evaluation.py tests/test_rl_training.py -k 'force_final_answer or fallback_guess or final_answer_required'`

- [ ] **Step 3: Implement prompt and validation helpers**

```python
def validate_final_answer(answer: dict[str, Any]) -> None:
    if answer.get("can_answer") is not True:
        raise ValueError("final_answer_required: answer.can_answer must be true in the final round")
    if not str(answer.get("answer") or "").strip():
        raise ValueError("final_answer_required: answer.answer must be non-empty in the final round")

def is_fallback_guess(answer: dict[str, Any]) -> bool:
    return str(answer.get("rationale") or "").strip().casefold().startswith("fallback_guess:")
```

The forced prompt keeps the same outer `<answer>{JSON object}</answer>` format.

- [ ] **Step 4: Propagate the flag through every policy path**

Add the flag to `PolicyGenerationRequest`, HF prompt construction, batch generation, and evaluation's OpenAI policy. Executors compute the final round before Answer generation, validate final responses, and record:

```python
turn["force_final_answer"] = is_final_round
turn["fallback_guess"] = is_fallback_guess(answer)
```

Do not make a second policy call.

- [ ] **Step 5: Run Task 3 tests and confirm GREEN**

Run the Task 3 command and `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rag.py`.

---

### Task 4: Per-agent cross-round decision returns and advantages

**Files:**
- Modify: `src/rl_training/policy.py`
- Modify: `src/rl_training/trainer.py`
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Modify: `config/train_grpo.yml`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `GeneratedAction.decision_return`, global-weight config, and per-role statistics.

- [ ] **Step 1: Replace role-round tests with failing role-only tests**

Use variable-length rollouts with Query actions at multiple rounds. Assert all Query actions share one bucket and roles remain separate.

```python
stats = assign_action_advantages(
    rollouts,
    global_weights={
        "query_retriever": 1 / 3,
        "evidence_updater": 3 / 7,
        "answer_generator": 7 / 3,
    },
    epsilon=1e-8,
)
assert first_q1.decision_return == pytest.approx(5.0 + (1 / 3) * 4.0)
assert stats["query_retriever"]["count"] == 3
assert sum(action.advantage for action in query_actions) == pytest.approx(0.0)
```

Include singleton, zero-variance, non-finite weight/epsilon, and parse-error action cases.

- [ ] **Step 2: Run Task 4 tests and confirm RED**

Run: `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py -k 'assign_action_advantages or global_reward_weight or decision_return'`

- [ ] **Step 3: Implement combined returns and role-only normalization**

Change `assign_action_advantages()` to accept `rollouts`, keyword-only `global_weights`, and `epsilon=1e-8`, and return `dict[str, dict[str, float]]` role statistics.

Look up local rewards by `(role, round_index)`, set terminal reward, compute `decision_return = local + lambda_j * terminal`, and bucket only by role. Use population mean/std once per role; singleton/zero-std buckets receive zero. Return count/mean/std/min/max statistics.

- [ ] **Step 4: Replace configuration semantics and log diagnostics**

```yaml
query_global_reward_weight: 0.3333333333333333
evidence_global_reward_weight: 0.42857142857142855
answer_global_reward_weight: 2.3333333333333335
advantage_epsilon: 1.0e-8
```

Remove runtime use of old local weights. Store role stats in step metrics and weights/epsilon in `train_meta.json`; serialize `decision_return`. Do not alter clipped policy or KL math.

- [ ] **Step 5: Run Task 4 tests and confirm GREEN**

Run the Task 4 command and `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py -k 'grpo_loss or action_advantage'`.

---

### Task 5: Full index construction and integrated verification

**Files:**
- Generate: `data/eval_1000_e5_faiss/<dataset>/`
- Generate: `data/rl_train_2000_e5_faiss/<dataset>/`
- Verify: all Task 1-4 files

**Interfaces:**
- Consumes: completed code and local E5 weights.
- Produces: validated evaluation and training indexes.

- [ ] **Step 1: Install and verify FAISS**

Run `/data/conda/envs/macorag/bin/python -m pip install faiss-cpu==1.9.0.post1`, then `/data/conda/envs/macorag/bin/python -c "import faiss; print(faiss.__version__)"`. Request approval if environment installation is blocked; never copy packages from another environment.

- [ ] **Step 2: Run focused tests before expensive builds**

Run: `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_retrieval_env.py tests/test_rag.py tests/test_evaluation.py tests/test_rl_training.py`

- [ ] **Step 3: Build evaluation indexes**

Run: `/data/conda/envs/macorag/bin/python -m data_processing.retrieval_cli --config config/build_retrieval_eval_e5.yml build`

Expected historical corpus counts are 6,292 / 10,586 / 11,535; if sources changed, validate and report current counts/hashes rather than forcing them.

- [ ] **Step 4: Build training indexes**

Run: `/data/conda/envs/macorag/bin/python -m data_processing.retrieval_cli --config config/build_retrieval_train_e5.yml build`

Expected: three validated 768-dimensional FlatIP indexes with matching metadata.

- [ ] **Step 5: Run parity and real-query smoke checks**

For fixed queries, compare evaluation top-5 ordered passage texts/IDs against existing R-Search artifacts. Differences fail unless a verified corpus hash differs. Query both roots through `create_retrieval_env()` and assert five passages, finite descending scores, and stable cached results.

- [ ] **Step 6: Run integrated verification**

Run `/data/conda/envs/macorag/bin/python -m compileall -q src`, `bash -n scripts/build_retrieval.sh scripts/eval_macorag.sh scripts/run_train_grpo.sh`, `/data/conda/envs/macorag/bin/python -m pytest -q`, and `git diff --check`.

Parse both runtime YAML files and instantiate/prewarm E5 retrieval without loading the 7B policy.

- [ ] **Step 7: Review and report**

Confirm no unrelated dirty changes were overwritten. Report tests, index paths/counts/hashes/dimensions, retrieval device, fallback behavior, lambda values, and any unrun GPU-scale validation. Do not commit generated FAISS indexes unless repository policy explicitly tracks large data artifacts.
