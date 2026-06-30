# vLLM RAG Evaluation Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vLLM/OpenAI-compatible backend and sample-level threaded prediction to the RAG evaluation module.

**Architecture:** Keep the existing RAG loop and retrieval environment. Add a `VLLMOpenAIPolicy` that implements the same policy interface as `HFSharedPolicy`, and use `ThreadPoolExecutor` only when `inference_backend` is `vllm_openai`.

**Tech Stack:** Python standard library `urllib.request`, `concurrent.futures`, existing MACORAG `RAGLoopExecutor`, existing YAML config parser, pytest.

## Global Constraints

- Python logic stays in `src/`.
- Shell launchers stay in `scripts/`.
- User-facing parameters are passed through `config/evaluate_rag_model.yml`.
- Existing `hf_local` evaluation behavior remains the default.
- `eval_request_workers` controls vLLM prediction concurrency; `judge_workers` continues to control Bailian judging concurrency.
- No test may require a live vLLM service.

---

### Task 1: Backend Configuration

**Files:**
- Modify: `src/evaluation/config.py`
- Modify: `config/evaluate_rag_model.yml`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: parsed args with `inference_backend`, `vllm_base_urls`, `vllm_model`, `vllm_api_key_env`, `vllm_timeout`, `vllm_retries`, `vllm_retry_sleep_seconds`, and `eval_request_workers`.

- [ ] **Step 1: Write the failing test**

Add a test that writes YAML containing:

```yaml
inference_backend: "vllm_openai"
vllm_base_urls:
  - "http://127.0.0.1:8000/v1"
  - "http://127.0.0.1:8001/v1"
vllm_model: "macorag-lora"
vllm_api_key_env: ""
vllm_timeout: 30
vllm_retries: 2
vllm_retry_sleep_seconds: 0.1
eval_request_workers: 8
```

Assert `parse_args(["--config", str(config), "--eval-request-workers", "4"])` returns `inference_backend == "vllm_openai"`, both base URLs, `vllm_model == "macorag-lora"`, and `eval_request_workers == 4`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_parse_eval_config_loads_vllm_backend_fields`

Expected: FAIL because the config keys and CLI option are unknown.

- [ ] **Step 3: Implement config fields**

Add defaults to `DEFAULT_ARG_VALUES`, add parser arguments, and append default commented or active values to `config/evaluate_rag_model.yml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_parse_eval_config_loads_vllm_backend_fields`

Expected: PASS.

### Task 2: vLLM OpenAI Policy

**Files:**
- Modify: `src/evaluation/evaluate_rag_model.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `VLLMOpenAIPolicy.generate(role, question, state, observation=None, endpoint_index=0) -> str`.

- [ ] **Step 1: Write failing tests**

Add tests that monkeypatch `urllib.request.urlopen` and assert:

- The policy posts to `http://127.0.0.1:8000/v1/chat/completions`.
- Payload includes `model`, `messages`, `temperature`, `top_p`, and `max_tokens`.
- Returned content is parsed from `choices[0].message.content`.
- The second sample index uses the second endpoint when two endpoints are configured.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_vllm_policy_posts_chat_completion_request tests/test_evaluation.py::test_vllm_policy_round_robins_base_urls`

Expected: FAIL because `VLLMOpenAIPolicy` does not exist.

- [ ] **Step 3: Implement policy**

Implement the class using existing prompt builders imported from `rag`. Normalize base URLs by trimming trailing slashes, build request headers with a local placeholder key unless `vllm_api_key_env` points to an env var, and retry request failures according to config.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_vllm_policy_posts_chat_completion_request tests/test_evaluation.py::test_vllm_policy_round_robins_base_urls`

Expected: PASS.

### Task 3: Backend Selection

**Files:**
- Modify: `src/evaluation/evaluate_rag_model.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `_load_policy(args)` returning either `HFSharedPolicy` or `VLLMOpenAIPolicy`.

- [ ] **Step 1: Write failing tests**

Add tests that:

- Set `inference_backend="vllm_openai"` and assert `_load_dependencies` is not called.
- Set `inference_backend="bad"` and assert `_load_policy` raises `SystemExit` with supported backend names.
- Set `vllm_base_urls=[]` and assert `_load_policy` raises `SystemExit`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_load_policy_uses_vllm_without_loading_local_model tests/test_evaluation.py::test_load_policy_rejects_invalid_backend tests/test_load_policy_rejects_vllm_without_base_urls`

Expected: FAIL because backend selection is not implemented.

- [ ] **Step 3: Implement backend selection**

Branch at the top of `_load_policy(args)`. Keep current HF code inside the `hf_local` branch and instantiate `VLLMOpenAIPolicy` in the `vllm_openai` branch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_load_policy_uses_vllm_without_loading_local_model tests/test_evaluation.py::test_load_policy_rejects_invalid_backend tests/test_evaluation.py::test_load_policy_rejects_vllm_without_base_urls`

Expected: PASS.

### Task 4: Concurrent Prediction Runner

**Files:**
- Modify: `src/evaluation/evaluate_rag_model.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `run_predictions()` that uses threads only for vLLM and preserves final `predictions.json` order.

- [ ] **Step 1: Write failing tests**

Add tests that:

- Use two samples, `inference_backend="vllm_openai"`, `eval_request_workers=2`, and a fake executor that blocks both tasks until two threads have entered. Assert both samples complete and `predictions.json` order is `[q1, q2]`.
- Use two samples, `inference_backend="hf_local"`, `eval_request_workers=2`, and assert fake executor calls happen sequentially on one thread.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_run_predictions_uses_threads_for_vllm_backend tests/test_evaluation.py::test_run_predictions_keeps_hf_backend_sequential`

Expected: FAIL because `run_predictions()` is sequential and does not read backend fields.

- [ ] **Step 3: Implement concurrency**

Add a `_run_one_prediction()` helper. In the vLLM branch, submit indexed samples to a `ThreadPoolExecutor`, append JSONL under a lock as futures finish, and write final JSON sorted by original index. In the HF branch, keep the current sequential loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_run_predictions_uses_threads_for_vllm_backend tests/test_evaluation.py::test_run_predictions_keeps_hf_backend_sequential`

Expected: PASS.

### Task 5: vLLM Server Helper Script

**Files:**
- Create: `scripts/evaluate_rag_model_vllm_servers.sh`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces: shell-only helper that reads config through `evaluation.config.parse_args` and prints or starts one `vllm serve` process per configured base URL port.

- [ ] **Step 1: Write failing test**

Add a test that reads the script and asserts it contains `vllm serve`, `--enable-lora`, `--lora-modules`, `model_path`, `adapter_path`, `vllm_model`, and `MACORAG_VLLM_DRY_RUN`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_vllm_server_helper_script_exists`

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Create script**

Create an executable script that supports dry-run output, derives ports from `vllm_base_urls`, and starts one server per URL with matching `CUDA_VISIBLE_DEVICES` entries when provided.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_vllm_server_helper_script_exists`

Expected: PASS.

### Task 6: Full Verification

**Files:**
- Modify only files touched by Tasks 1-5.

**Interfaces:**
- Produces: verified evaluation test suite and shell syntax.

- [ ] **Step 1: Run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py`

Expected: PASS.

- [ ] **Step 2: Run related regression tests**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py tests/test_rag.py tests/test_retrieval_env.py`

Expected: PASS, or report environment/dependency failures exactly.

- [ ] **Step 3: Run shell syntax checks**

Run: `bash -n scripts/evaluate_rag_model.sh scripts/evaluate_rag_model_vllm_servers.sh`

Expected: PASS.

- [ ] **Step 4: Commit**

Commit only the evaluation concurrency changes and docs, leaving unrelated dirty files untouched.
