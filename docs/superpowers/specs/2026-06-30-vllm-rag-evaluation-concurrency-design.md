# vLLM RAG Evaluation Concurrency Design

## Goal

Add a vLLM/OpenAI-compatible inference backend to `src/evaluation/evaluate_rag_model.py` so evaluation can process test samples concurrently while keeping the existing MACORAG RAG loop, retrieval assets, output schema, and Bailian `qwen-plus` judging flow unchanged.

## Approved Approach

Use external vLLM model services for GPU-heavy generation. The evaluation process stays as a lightweight client, dispatches sample-level work with a thread pool, and sends chat-completion requests to one or more vLLM OpenAI-compatible endpoints.

This is preferred over multi-threading the in-process HuggingFace model because PyTorch/Transformers generation and the shared model object are not a good concurrency boundary. The existing local HuggingFace backend remains supported and defaults to sequential prediction.

## Configuration

Add these YAML/CLI keys:

- `inference_backend`: `"hf_local"` or `"vllm_openai"`, default `"hf_local"`.
- `vllm_base_urls`: list of OpenAI-compatible base URLs such as `http://127.0.0.1:8000/v1`.
- `vllm_model`: model name or LoRA alias served by vLLM, for example `"macorag-lora"`.
- `vllm_api_key_env`: optional environment variable for an API key; empty means use a placeholder key for local vLLM.
- `vllm_timeout`: request timeout in seconds.
- `vllm_retries`: number of generation request attempts.
- `vllm_retry_sleep_seconds`: sleep between retries.
- `eval_request_workers`: sample-level prediction threads for the vLLM backend; values below 1 are rejected.

The existing `model_path`, `adapter_path`, `max_completion_length`, `temperature`, `top_p`, and prompt settings remain the source of truth for local HF generation. vLLM services are started outside the Python evaluator and can use `--enable-lora --lora-modules`.

## Architecture

Create a `VLLMOpenAIPolicy` with the same `generate(role, question, state, observation)` interface as `HFSharedPolicy`. It builds the same role prompts used by `HFSharedPolicy`, wraps them as OpenAI chat messages, sends `/chat/completions` requests, retries transient failures, and returns the assistant content.

`_load_policy(args)` chooses the backend:

- `hf_local`: load tokenizer, base model, and PEFT adapter as today.
- `vllm_openai`: validate vLLM config and create `VLLMOpenAIPolicy` without loading a local model.

`run_predictions()` adds a concurrency branch:

- For `vllm_openai` and `eval_request_workers > 1`, use `ThreadPoolExecutor`.
- Each sample run creates its own `RAGLoopExecutor` and calls the shared vLLM policy.
- Results are written to `predictions.jsonl` as futures complete for durability.
- `predictions.json` is written in original sample order for stable downstream evaluation.
- Infrastructure errors still fail fast instead of being swallowed into per-sample errors.

For `hf_local`, prediction remains sequential even if `eval_request_workers` is greater than 1. This avoids unsafe concurrent access to a single local model.

## Endpoint Strategy

`vllm_base_urls` can contain one endpoint or several endpoints. The policy chooses an endpoint by sample index for round-robin balancing. Both deployment styles are supported:

- One vLLM endpoint with internal batching/data parallelism; all threads call the same URL.
- Multiple vLLM server instances on different ports; worker threads distribute requests across URLs.

Example server commands:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve model/Qwen2.5-7B-Instruct \
  --port 8000 \
  --enable-lora \
  --lora-modules macorag-lora=outputs/lora_qwen2.5-7b_trajectory_20260627_203027/adapter

CUDA_VISIBLE_DEVICES=1 vllm serve model/Qwen2.5-7B-Instruct \
  --port 8001 \
  --enable-lora \
  --lora-modules macorag-lora=outputs/lora_qwen2.5-7b_trajectory_20260627_203027/adapter
```

## Script Support

Keep `scripts/evaluate_rag_model.sh` as the main evaluator launcher. Add a helper script under `scripts/` for starting vLLM servers from YAML-derived values when the user wants local multi-port serving. The helper should remain shell-only and should not embed Python evaluation logic.

## Testing

Add focused tests for:

- YAML and CLI parsing of the new vLLM/backend fields.
- vLLM policy request payload, response parsing, retry behavior, and endpoint round-robin.
- `_load_policy()` selecting vLLM without importing/loading local Transformers models.
- `run_predictions()` using sample-level threads for `vllm_openai`, preserving final prediction order.
- `run_predictions()` keeping `hf_local` sequential despite a high `eval_request_workers` value.
- Shell script dry-run behavior remains compatible with config-driven GPU selection.

No test should require a live vLLM server.

## Non-Goals

- Do not implement distributed HuggingFace inference inside this evaluator.
- Do not change LinearRAG retrieval semantics or Bailian judging semantics.
- Do not merge prediction and judge concurrency knobs; `eval_request_workers` is for model inference, `judge_workers` is for Bailian metric evaluation.
