# RAG Model Evaluation Design

## Goal

Build an independent evaluation module for a MACORAG model after SFT and RL. The module will run the configured RAG loop on `data/eval_1000` with retrieval from `data/eval_1000_retrieval`, generate predictions with the trained local adapter, and evaluate answers with the same metric shape as `LinearRAG/src/evaluate.py`, using Alibaba Bailian `qwen-plus` as the LLM judge.

## Confirmed Inputs

- Test data root: `data/eval_1000`
- Retrieval root: `data/eval_1000_retrieval`
- Dataset directories: `2wiki`, `hotpotqa`, `musique`
- Test files:
  - `data/eval_1000/2wiki/2wiki_dev.jsonl`
  - `data/eval_1000/hotpotqa/hotpotqa_dev.jsonl`
  - `data/eval_1000/musique/musique_dev.jsonl`
- Retrieval index files exist per dataset under `data/eval_1000_retrieval/<dataset>/`, including `LinearRAG.graphml`, `chunks.json`, `passage_embedding.parquet`, `entity_embedding.parquet`, and `sentence_embedding.parquet`.
- The trained model is loaded as a base HuggingFace causal LM plus a LoRA adapter path. The adapter may be an SFT adapter or the final adapter produced by RL.

## Architecture

Add a new package under `src/evaluation/` so inference and evaluation stay separate from SFT and RL training code. The module will reuse the existing RAG primitives instead of introducing a parallel RAG implementation:

- `rl_training.policy.HFSharedPolicy` for model-backed role generation.
- `rag.RAGLoopExecutor` for query, evidence, and answer loop execution.
- `rl_training.retrieval.CachedLinearRAGRetrievalEnv` for LinearRAG-backed retrieval.

The command-line entrypoint will parse a YAML config, load samples, run prediction, write prediction artifacts, and then run answer evaluation. Shell launch remains thin and config-driven, matching the existing `scripts/run_train_sft_lora_gpu1.sh` and `scripts/run_train_grpo.sh` style.

## Files

- Create `src/evaluation/__init__.py`
  - Export only stable module-level helpers needed by tests.
- Create `src/evaluation/config.py`
  - Parse `config/evaluate_rag_model.yml`.
  - Allow CLI overrides for the same fields.
  - Reject unknown YAML keys.
- Create `src/evaluation/data.py`
  - Load JSONL samples from the configured dataset files.
  - Skip `corpus.jsonl`.
  - Normalize records into a small evaluation sample dataclass with `qid`, `dataset`, `question`, `answer`, `answer_aliases`, `supporting_facts`, and `metadata`.
- Create `src/evaluation/bailian_evaluator.py`
  - Implement an OpenAI-compatible Bailian chat client using `urllib.request`, aligned with the existing teacher-generation style.
  - Provide `calculate_llm_accuracy`, `calculate_contain`, and `evaluate_predictions`.
  - Write `evaluation_results.json` and update `predictions.json` with per-sample `llm_accuracy` and `contain_accuracy`.
- Create `src/evaluation/evaluate_rag_model.py`
  - Configure visible GPUs from YAML unless `CUDA_VISIBLE_DEVICES` is already set.
  - Load tokenizer, base model, and adapter.
  - Run `RAGLoopExecutor` over evaluation samples.
  - Append/flush predictions incrementally to a JSONL progress file and write final `predictions.json`.
  - Invoke Bailian evaluation unless `skip_judge` is true.
- Create `config/evaluate_rag_model.yml`
  - Default paths:
    - `data_root: "data/eval_1000"`
    - `retrieval_root: "data/eval_1000_retrieval"`
    - `output_dir: "outputs/eval_rag_model"`
  - Default judge:
    - `judge_model: "qwen-plus"`
    - `judge_endpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"`
    - `judge_api_key_env: "DASHSCOPE_API_KEY"`
- Create `scripts/evaluate_rag_model.sh`
  - Set `PYTHONPATH=${REPO_ROOT}/src`.
  - Set `MACORAG_SILENT_RETRIEVAL=1` by default.
  - Read `gpu_indices` or `gpu_index` from YAML and export `CUDA_VISIBLE_DEVICES`.
  - Run `python -m evaluation.evaluate_rag_model --config "${CONFIG_PATH}" "$@"`.
- Create `tests/test_evaluation.py`
  - Cover config parsing, sample loading, prediction schema formatting, script GPU parsing, and Bailian judge response parsing.

## Configuration Fields

Core model and data fields:

- `model_path`: base HuggingFace model directory.
- `adapter_path`: trained SFT or RL adapter directory.
- `data_root`: evaluation dataset root.
- `data_files`: optional explicit dataset-relative files. When empty, load non-corpus JSONL files under `data_root`.
- `retrieval_root`: LinearRAG retrieval root.
- `output_dir`: base output directory.
- `max_samples`: optional cap for smoke tests.
- `seed`: random seed.
- `gpu_index`: fallback single GPU.
- `gpu_indices`: visible GPU list for the launcher.

Generation and RAG fields:

- `system_prompt`: system prompt passed to `HFSharedPolicy`.
- `max_rounds`: RAG loop round cap.
- `max_prompt_length`: prompt token cap.
- `max_completion_length`: generation token cap.
- `temperature`, `top_p`, `top_k`: generation settings.
- `load_4bit`, `bf16`, `fp16`: model loading settings.
- `retrieval_embedding_model`, `retrieval_spacy_model`, `retrieval_top_k`, `retrieval_max_workers`, `retrieval_batch_size`, `use_vectorized_retrieval`: retrieval settings.

Evaluation fields:

- `skip_judge`: if true, generate predictions but do not call Bailian.
- `judge_model`: default `qwen-plus`.
- `judge_endpoint`: Bailian OpenAI-compatible endpoint.
- `judge_api_key_env`: environment variable holding the API key.
- `judge_temperature`: default `0.0`.
- `judge_max_tokens`: default `8`.
- `judge_timeout`: request timeout in seconds.
- `judge_retries`: retry count.
- `judge_workers`: concurrent judge requests.

## Output Format

Each run creates a timestamped output directory under `output_dir`, unless the config requests a fixed output directory.

Required artifacts:

- `run_config.json`: resolved config snapshot.
- `predictions.jsonl`: append-only progress records, one sample per successful or failed prediction.
- `predictions.json`: final JSON list compatible with `LinearRAG/src/evaluate.py`.
- `evaluation_results.json`: aggregate `llm_accuracy`, `contain_accuracy`, `num_samples`, and judge metadata.

Each prediction item contains:

- `qid`
- `dataset`
- `question`
- `pred_answer`
- `gold_answer`
- `answer_aliases`
- `trajectory`
- `parse_errors`
- `retrieval_count`
- `error`, present only when a sample failed.

## Error Handling

- Missing config file passed explicitly should fail fast.
- Missing data files should fail fast.
- Missing retrieval index files should fail before model generation for the affected dataset.
- A single sample failure should be recorded in `predictions.jsonl` and should not discard previous successful samples.
- Missing `DASHSCOPE_API_KEY` should fail only when `skip_judge` is false.
- Bailian judge failures retry per config; after retries, the sample receives `llm_accuracy: 0.0` and an evaluation error field, while the rest of the evaluation continues.

## Testing

Use test-first implementation:

- Config parsing test verifies YAML defaults, CLI override, and unknown-key rejection.
- Data loading test verifies recursive JSONL discovery skips `corpus.jsonl` and normalizes `gold_answer`/`answer`.
- Prediction formatting test uses fake policy and retrieval objects to verify `predictions.json` fields without loading a model.
- Script test verifies YAML-driven `CUDA_VISIBLE_DEVICES` parsing and default `MACORAG_SILENT_RETRIEVAL=1`.
- Bailian evaluator test stubs the client response and verifies `correct` maps to `1.0`, `incorrect` maps to `0.0`, and contain accuracy follows `LinearRAG/src/evaluate.py` normalization semantics.

## Non-Goals

- Do not rebuild retrieval indexes.
- Do not extract or preprocess datasets.
- Do not train or update the adapter.
- Do not change the existing SFT or RL training entrypoints.
- Do not replace `RAGLoopExecutor` behavior.
