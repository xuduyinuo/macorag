# Local vLLM Evaluation Recovery Design

## Problem

`src/evaluation/evaluate_rag_model.py` sends requests to the local vLLM OpenAI-compatible endpoint with `urllib.request.urlopen`. The process environment defines `HTTP_PROXY` and `http_proxy` as `http://127.0.0.1:10900` but does not define `NO_PROXY`. Consequently, requests intended for `http://127.0.0.1:8000/v1` are routed through the local proxy. A transient proxy refusal is surfaced as a vLLM transport failure even while the vLLM process, model identity, and port remain healthy.

The threaded prediction path also stops consuming completed futures as soon as one future raises. The `ThreadPoolExecutor` context waits for already-submitted work, but successful results completed after the first raised future are never appended to `predictions.jsonl`. This wastes completed inference and weakens resume behavior.

The failed run at `outputs/eval/2026-09-01_14-54-23` retained 1000 2Wiki rows, 1000 HotpotQA rows, and 145 MuSiQue rows. Recovery must reuse those rows and generate only the missing MuSiQue qids.

## Scope

The change is limited to the evaluation HTTP client, threaded result collection, focused evaluation tests, and operator recovery instructions. It does not change prompts, sampling, retrieval, metrics, dataset order, adapter selection, vLLM server configuration, or unrelated training code.

## Design

### Direct local endpoint access

`VLLMOpenAIPolicy` will use an explicit urllib opener that bypasses environment proxies for loopback vLLM endpoints. The local evaluation client therefore talks directly to the configured `127.0.0.1` or `localhost` service even when proxy variables are present.

Proxy bypass belongs in the Python client rather than only in `scripts/eval_macorag.sh`, because the module is also a supported direct entrypoint. No global proxy environment variables will be mutated, so unrelated external traffic in the process remains unaffected.

### Preserve completed concurrent results

The threaded collector will remember the first infrastructure exception but continue consuming every submitted future. Each successful prediction will still be appended immediately. After all submitted futures have been consumed, the first infrastructure exception will be re-raised, so the run remains visibly failed and incomplete metrics are not published.

This preserves the existing fail-fast semantic at the dataset boundary while ensuring that work already accepted by the thread pool is not discarded. Resume continues to validate and skip completed qids from `predictions.jsonl`.

### Recovery

The existing output directory and resume identity will be reused with:

```bash
cd /data/xudu/macorag
bash scripts/eval_macorag.sh \
  --output-dir outputs/eval/2026-09-01_14-54-23 \
  --resume
```

The launcher must use the same evaluation config and live `macorag` model identity. Completed 2Wiki and HotpotQA rows and the 145 completed MuSiQue rows will be validated and skipped.

## Error Handling

- HTTP status and response-shape errors keep their existing behavior.
- Transport failures retain the configured retry count and delay.
- A persistent infrastructure failure still terminates the run without publishing aggregate completion artifacts.
- Successful concurrent predictions are persisted before the remembered infrastructure failure is raised.
- Resume identity mismatches and malformed or duplicate resumed qids remain hard failures.

## Tests

Focused regression tests will prove:

1. A loopback vLLM request succeeds directly when `HTTP_PROXY` points to an unavailable proxy.
2. In threaded mode, a successful future is persisted even when another future raises an infrastructure error first.
3. The original infrastructure exception is still raised after completed futures are drained.
4. Existing qid resume tests, evaluation tests, launcher syntax checks, and diff checks remain green.

The proxy test will use local ephemeral HTTP servers only and will not require a GPU or a real vLLM process.

## Acceptance Criteria

- Loopback vLLM calls do not depend on `HTTP_PROXY`, `http_proxy`, or `NO_PROXY`.
- Concurrent successful results are never lost merely because another future fails.
- A persistent vLLM infrastructure failure still produces a non-zero evaluation exit.
- The interrupted run can resume from its existing output directory without regenerating completed qids.
- No unrelated dirty worktree files are modified.
