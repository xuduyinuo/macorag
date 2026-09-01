# Local vLLM Evaluation Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local vLLM evaluation independent of ambient HTTP proxies and preserve every successful concurrent prediction when another request suffers an infrastructure failure.

**Architecture:** Use a proxy-free urllib opener only for loopback endpoints, leaving non-loopback behavior and global proxy settings unchanged. In threaded collection, retain the first exception, drain successful futures into the append-only progress file, and then re-raise so incomplete runs remain failed and resumable.

**Tech Stack:** Python 3.9, `urllib.request`, `concurrent.futures`, `pytest`, Bash.

## Global Constraints

- Do not change prompts, sampling, retrieval, metrics, dataset order, adapter selection, or vLLM server configuration.
- Do not modify the user's dirty training files: `config/eval_vllm_server.yml`, `config/train_grpo.yml`, `src/rl_training/train_grpo_macorag.py`, or `tests/test_rl_training.py`.
- Keep retry count and delay, resume validation, qid validation, and non-zero exit on persistent infrastructure failures.
- Do not launch the full 3000-sample evaluation during verification.

---

## File Structure

- Modify `src/evaluation/evaluate_rag_model.py`: select direct loopback HTTP transport and drain successful futures.
- Modify `tests/test_evaluation.py`: add proxy-bypass and concurrent-progress regression tests.
- Read `outputs/eval/2026-09-01_14-54-23` only for the resume audit and command handoff.

### Task 1: Bypass ambient proxies for loopback vLLM endpoints

**Files:**
- Modify: `src/evaluation/evaluate_rag_model.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `VLLMOpenAIPolicy._endpoint() -> str` and `_post_chat_completion(payload) -> dict`.
- Produces: `_open(request)` using a proxy-free opener for `127.0.0.1`, `localhost`, and `::1`; other hosts still use `urllib.request.urlopen`.

- [ ] **Step 1: Write the failing proxy regression test**

Add `from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer` and this test to `tests/test_evaluation.py`:

```python
def test_vllm_policy_bypasses_http_proxy_for_loopback_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    class ProxyHandler(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self) -> None:
            type(self).calls += 1
            self.send_error(502, "proxy must not receive loopback traffic")

        def log_message(self, format: str, *args: object) -> None:
            return None

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    threads = [
        threading.Thread(target=target.serve_forever, daemon=True),
        threading.Thread(target=proxy.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        proxy_url = f"http://127.0.0.1:{proxy.server_port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("http_proxy", proxy_url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        policy = VLLMOpenAIPolicy(
            base_urls=[f"http://127.0.0.1:{target.server_port}/v1"], model="macorag",
            api_key_env="", system_prompt=None, max_prompt_length=128,
            max_completion_length=16, temperature=0.0, top_p=1.0,
            timeout=2, retries=1, retry_sleep_seconds=0.0,
        )
        response = policy._post_chat_completion({"model": "macorag", "messages": []})
        assert response["choices"][0]["message"]["content"] == "ok"
        assert ProxyHandler.calls == 0
    finally:
        target.shutdown()
        proxy.shutdown()
        target.server_close()
        proxy.server_close()
        for thread in threads:
            thread.join(timeout=2)
```

- [ ] **Step 2: Verify RED**

Run: `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_evaluation.py::test_vllm_policy_bypasses_http_proxy_for_loopback_endpoint`

Expected: FAIL with a `RuntimeError` caused by proxy HTTP 502.

- [ ] **Step 3: Implement loopback opener selection**

Import `urllib.parse`, initialize `self._loopback_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))`, add:

```python
def _open(self, request: urllib.request.Request):
    hostname = (urllib.parse.urlparse(request.full_url).hostname or "").lower()
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        return self._loopback_opener.open(request, timeout=self.timeout)
    return urllib.request.urlopen(request, timeout=self.timeout)
```

Then replace `urllib.request.urlopen(request, timeout=self.timeout)` in `_post_chat_completion` with `self._open(request)`.

- [ ] **Step 4: Verify GREEN and existing policy construction**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_evaluation.py::test_vllm_policy_bypasses_http_proxy_for_loopback_endpoint \
  tests/test_evaluation.py::test_load_policy_uses_vllm_without_loading_local_model \
  tests/test_evaluation.py::test_load_policy_rejects_vllm_without_base_urls
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/evaluate_rag_model.py tests/test_evaluation.py
git commit -m "fix: bypass proxies for local vllm evaluation"
```

### Task 2: Persist successful futures before propagating failure

**Files:**
- Modify: `src/evaluation/evaluate_rag_model.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `_run_one_prediction(...) -> tuple[int, dict]` futures and `_append_jsonl`.
- Produces: threaded `run_predictions` that appends every success, then raises the first exception.

- [ ] **Step 1: Write the failing concurrent-progress test**

```python
def test_run_predictions_drains_successful_futures_before_raising_infrastructure_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = [
        EvalSample("fails", "musique", "Question 1?", "Answer 1", [], [], {}),
        EvalSample("succeeds", "musique", "Question 2?", "Answer 2", [], [], {}),
    ]
    failure_released = threading.Event()

    def fake_run_one_prediction(*, index: int, sample: EvalSample, **kwargs):
        if sample.qid == "fails":
            failure_released.set()
            raise RuntimeError("vLLM chat completion failed after 3 attempt(s): connection refused")
        assert failure_released.wait(timeout=2)
        return index, {
            "qid": sample.qid, "dataset": sample.dataset, "question": sample.question,
            "pred_answer": sample.answer, "gold_answer": sample.answer, "answer_aliases": [],
            "trajectory": [], "parse_errors": [], "retrieval_count": 0,
        }

    monkeypatch.setattr("evaluation.evaluate_rag_model._run_one_prediction", fake_run_one_prediction)
    args = SimpleNamespace(eval_request_workers=2, disable_tqdm=True, resume=False)
    with pytest.raises(RuntimeError, match="vLLM chat completion failed"):
        run_predictions(args, samples, object(), object(), tmp_path)

    rows = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert [row["qid"] for row in rows] == ["succeeds"]
```

- [ ] **Step 2: Verify RED**

Run: `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_evaluation.py::test_run_predictions_drains_successful_futures_before_raising_infrastructure_error`

Expected: FAIL because `predictions.jsonl` is absent after the first future raises.

- [ ] **Step 3: Drain successes and re-raise the first failure**

Replace the threaded result loop with:

```python
first_error: Exception | None = None
for future in iterator:
    try:
        index, prediction = future.result()
    except Exception as exc:
        if first_error is None:
            first_error = exc
        continue
    predictions_by_index[index] = prediction
    with progress_lock:
        _append_jsonl(progress_path, prediction)
if first_error is not None:
    raise first_error
```

Do not perform the final ordered rewrite after an exception; the append-only file remains the incomplete-run recovery source.

- [ ] **Step 4: Verify GREEN and resume behavior**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_evaluation.py::test_run_predictions_drains_successful_futures_before_raising_infrastructure_error \
  tests/test_evaluation.py::test_run_predictions_uses_threads_when_multiple_eval_workers_are_configured \
  tests/test_evaluation.py::test_run_predictions_re_raises_vllm_service_errors \
  tests/test_evaluation.py::test_run_predictions_resumes_completed_qids \
  tests/test_evaluation.py::test_run_predictions_flushes_jsonl_progress
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/evaluate_rag_model.py tests/test_evaluation.py
git commit -m "fix: preserve completed concurrent evaluations"
```

### Task 3: Verify and audit the resume handoff

**Files:**
- Verify: `src/evaluation/evaluate_rag_model.py`, `tests/test_evaluation.py`, `scripts/eval_macorag.sh`
- Read: `outputs/eval/2026-09-01_14-54-23/*/predictions.jsonl`

**Interfaces:**
- Consumes: Tasks 1-2 and existing `--output-dir` plus `--resume` options.
- Produces: fresh tests, retained-qid counts, exact model identity evidence, and the recovery command.

- [ ] **Step 1: Run the full evaluation test module**

Run: `/data/conda/envs/macorag/bin/python -m pytest -q tests/test_evaluation.py`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run static and launcher checks**

```bash
/data/conda/envs/macorag/bin/python -m compileall -q src/evaluation
bash -n scripts/eval_macorag.sh scripts/eval_vllm_server.sh
git diff --check
```

Expected: all exit 0; `git diff --check` prints nothing.

- [ ] **Step 3: Audit persisted counts and unique qids read-only**

```bash
/data/conda/envs/macorag/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/eval/2026-09-01_14-54-23")
for dataset, expected in {"2wiki": 1000, "hotpotqa": 1000, "musique": 145}.items():
    rows = [json.loads(line) for line in (root / dataset / "predictions.jsonl").read_text().splitlines() if line]
    qids = [str(row["qid"]) for row in rows]
    assert len(rows) == expected and len(qids) == len(set(qids))
    print(dataset, len(rows), "unique_qids", len(set(qids)))
print("RESUME_ARTIFACT_AUDIT_OK")
PY
```

Expected: `1000`, `1000`, `145`, then `RESUME_ARTIFACT_AUDIT_OK`.

- [ ] **Step 4: Check the live endpoint and exact model identity**

Run: `curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models`

Expected: a model entry with `id` exactly `macorag`. If absent, report the need to restart `bash scripts/eval_vllm_server.sh`; do not start it automatically.

- [ ] **Step 5: Dry-run the exact resume command**

```bash
MACORAG_EVAL_DRY_RUN=1 bash scripts/eval_macorag.sh \
  --output-dir outputs/eval/2026-09-01_14-54-23 --resume
```

Expected: exit 0 with the canonical config path and `CUDA_VISIBLE_DEVICES=1`.

- [ ] **Step 6: Hand off without launching the full evaluation**

```bash
cd /data/xudu/macorag
bash scripts/eval_macorag.sh \
  --output-dir outputs/eval/2026-09-01_14-54-23 \
  --resume
```

Expected formal behavior: skip 2145 retained qids, evaluate 855 remaining MuSiQue qids, then publish MuSiQue and aggregate metrics.
