# E5-FAISS Retrieval, Final-Round Fallback, and Agent Credit Design

## Goal

Replace MACORAG's default LinearRAG retrieval path with a repository-native implementation of the R-Search `intfloat/e5-base-v2` plus FAISS FlatIP retriever, require a best-effort answer in the final interaction round, and align action-level credit assignment with the paper definition

\[
G_{i,t}^{j}=r_{i,t}^{j}+\lambda_jR(\tau_i)
\]

followed by group-relative normalization over all valid decisions from the same agent type for the same question.

The change must apply consistently to GRPO training and model evaluation without changing the shared Qwen2.5 model architecture. Existing LinearRAG assets remain available as an explicit fallback backend, but E5-FAISS becomes the configured default.

## Scope and constraints

- Implement E5-FAISS inside MACORAG; runtime code must not import from or call `/data/xudu/baseline/R-Search`.
- Build independent indexes for `data/eval_1000` and `data/rl_train_2000`.
- Match the R-Search dense retrieval contract: `intfloat/e5-base-v2`, `query:` and `passage:` prefixes, attention-mask mean pooling, L2 normalization, 512-token truncation, FAISS `IndexFlatIP`, and top-k 5 by default.
- Use CPU E5 query encoding and CPU FAISS by default during GRPO training to avoid competing with vLLM on GPU 0 and the trainer on GPU 1. Index building may select another device explicitly.
- Preserve the existing LinearRAG implementation behind `retrieval_backend: linear_rag`; do not delete old assets.
- Keep one Answer Agent decision per round. The final round uses a forced-answer prompt instead of generating a second retry action.
- Use agent-global reward coefficients
  - `lambda_Q = 1/3`,
  - `lambda_E = 3/7`,
  - `lambda_A = 7/3`.
- Normalize agent decision returns across trajectories and rounds for one question, separately for Query, Evidence, and Answer.
- Preserve all unrelated dirty-worktree changes.

## 1. Native E5-FAISS retrieval

### 1.1 Asset format

For each dataset, the index builder reads `<data_root>/<dataset>/corpus.jsonl` in stable file order. Each passage is formatted as `title + "\n" + text` when a title exists, otherwise as `text`. It emits a dataset directory containing:

- `e5_Flat.index`: a CPU-serializable FAISS `IndexFlatIP` index;
- `corpus.jsonl`: the ordered passage records used to build the index;
- `index_metadata.json`: schema version, dataset, E5 model identifier, vector dimension, pooling method, maximum token length, FAISS type, corpus count, and corpus SHA256.

The loader validates the metadata, corpus count, corpus hash, FAISS vector count, and vector dimension before serving queries. A mismatch is a hard error; it must never silently load another backend or stale index.

Evaluation indexes and training indexes live under separate configured roots so that their corpora cannot be mixed accidentally.

### 1.2 Encoding and search

The encoder uses the E5 tokenizer and encoder directly:

1. Prefix queries with `query: ` and passages with `passage: `.
2. Tokenize with padding, truncation, and `max_length=512`.
3. Apply attention-mask mean pooling to the final hidden state.
4. L2-normalize every embedding.
5. Convert embeddings to contiguous `float32` arrays.

The query engine loads one encoder and lazily loads one FAISS index/corpus pair per dataset. It supports both `query()` and `query_batch()`. Batch rollout retrieval must encode all cache misses together and perform one FAISS batch search.

The query engine may retain the complete corpus row and global FAISS ID internally, but the runtime observation exposed to agents contains exactly `passage_id`, `title`, `text`, and `score`. `passage_id` is the zero-based rank within the current retrieval result (`0..top_k-1`), not the global corpus ID. Every retrieval starts numbering from zero. When the Evidence Agent selects a local ID, the executor immediately copies the corresponding title, text, and score into accumulated evidence, so later retrievals cannot reinterpret an earlier local ID. Corpus metadata such as `chunk_id`, `doc_id`, `dataset`, and the text-duplicating `sentences` field must not enter the agent prompt.

The existing LRU query cache and retrieval timing counters remain available at the backend-neutral environment boundary.

### 1.3 Configuration and dependency behavior

Training, evaluation, and retrieval build configuration gain explicit fields:

- `retrieval_backend: e5_faiss`;
- `retrieval_embedding_model: intfloat/e5-base-v2`;
- `retrieval_device: cpu` for query-time encoding;
- `retrieval_max_length: 512`;
- backend-specific index/data roots where needed.

`faiss-cpu` becomes an explicit environment and requirements dependency. Imports remain lazy so configuration help and unrelated tests can run without initializing FAISS or transformer weights. Selecting `e5_faiss` without FAISS, local E5 weights, or complete indexes produces a fail-fast message with the missing requirement and expected build command.

## 2. Final-round forced fallback answer

### 2.1 Request and prompt contract

Answer generation requests gain a `force_final_answer` boolean. Both the synchronous `RAGLoopExecutor` used by evaluation and the batched rollout executor used by GRPO set it only when `round_index == max_rounds - 1`.

Normal rounds retain the current contract: when accumulated evidence is insufficient and retrieval budget remains, the Answer Agent may return `can_answer=false` with a null answer.

The final-round prompt instead requires:

- `can_answer=true`;
- a non-empty string in `answer`;
- a normal evidence-grounded rationale when evidence is sufficient;
- a best-effort answer and a rationale beginning with `fallback_guess:` when evidence is insufficient.

There is no second Answer Agent call. This preserves one decision and one completion-token sequence for each `(trajectory, round, agent)` entry in the credit-assignment equations.

### 2.2 Trajectory and reward semantics

Each final-round answer turn records `force_final_answer`. It also records `fallback_guess` based on the structured rationale marker so evaluation and training logs can report fallback frequency separately from ordinary answers.

A fallback answer receives no special correctness bonus. It is treated as `can_answer=true`, scored against the real gold answers, and contributes its normal answer F1 and terminal reward. This changes coverage without rewarding unsupported guesses merely for being non-empty.

If the model violates the final-round prompt by returning `can_answer=false` or an empty answer, parsing records a distinct `final_answer_required` protocol error. The system does not fabricate an answer in post-processing.

## 3. Agent credit assignment

### 3.1 Decision returns

For every generated action with a valid completion-token trace, the trainer attaches the action's local reward and its trajectory's terminal reward, then computes

\[
G_{i,t}^{j}=r_{i,t}^{j}+\lambda_jR(\tau_i),
\]

using:

\[
\lambda_Q=\frac{1}{3},\qquad
\lambda_E=\frac{3}{7},\qquad
\lambda_A=\frac{7}{3}.
\]

Malformed structured output is still a generated and trainable decision when completion tokens and old-policy log probabilities exist. It receives the appropriate parse-error local penalty. Agents that were never called after an upstream termination do not enter any decision set.

`GeneratedAction` stores `local_reward`, `terminal_reward`, `decision_return`, and `advantage` for diagnostics and loss construction.

### 3.2 Per-agent cross-round normalization

For the current question, all decisions from all sampled trajectories are bucketed only by agent role:

\[
\mathcal S_j(q)=\{(i,t)\mid 1\leq i\leq N,\ t\in T_i^j\}.
\]

The current `(role, round_index)` bucket is removed. Each role's decision returns are normalized once across all trajectories and all valid rounds:

\[
A_{i,t}^{j}=
\frac{G_{i,t}^{j}-\mu_j}{\sigma_j+10^{-8}}.
\]

The implementation uses population variance. A singleton role bucket or a zero-variance bucket receives all-zero advantages. Longer trajectories contribute more decisions exactly as specified by `S_j(q)`; there is no additional trajectory-level reweighting.

The legacy `*_local_credit_weight` configuration is replaced by:

- `query_global_reward_weight: 0.3333333333333333`;
- `evidence_global_reward_weight: 0.42857142857142855`;
- `answer_global_reward_weight: 2.3333333333333335`;
- `advantage_epsilon: 1.0e-8`.

Run metadata and training logs record these values and per-role count, mean, standard deviation, and normalized-advantage statistics.

### 3.3 GRPO optimization

The existing token-level GRPO path remains structurally unchanged:

- each action's scalar advantage is broadcast to its valid generated tokens;
- padding and invalid tokens are excluded by the action mask;
- the current/old-policy token probability ratio feeds the clipped surrogate objective;
- sampled-token KL regularization against the fixed reference policy is applied over valid tokens;
- Query, Evidence, and Answer actions jointly update the same shared policy parameters;
- no learned value model is introduced.

The material change is that the loss now consumes the paper-aligned per-agent, cross-round normalized `decision_return` advantages rather than separately normalized local and terminal rewards in role-round buckets.

## 4. Testing and acceptance

### 4.1 Retrieval tests

Unit and integration tests must cover:

- E5 query/passage prefixes;
- masked mean pooling and L2 normalization;
- deterministic corpus ordering and SHA256 metadata;
- tiny-corpus build, save, load, and batch search;
- top-k passage ordering and stable IDs;
- per-dataset lazy loading and LRU query caching;
- fail-fast behavior for missing FAISS, missing weights, incomplete assets, hash mismatch, count mismatch, and dimension mismatch;
- fixed-query top-k comparison against the existing R-Search E5-FAISS assets.

### 4.2 Fallback tests

Tests must cover synchronous and batched rollout paths:

- non-final rounds can still return `can_answer=false`;
- final-round Answer requests set `force_final_answer=true`;
- the final prompt requires `can_answer=true` and a non-empty answer;
- unsupported final answers carry the `fallback_guess:` rationale marker;
- trajectory flags distinguish normal and fallback answers;
- a final false/null response produces `final_answer_required` instead of a silent blank.

### 4.3 Credit and loss tests

Tests must construct variable-length rollout groups and verify:

- `decision_return = local_reward + lambda_j * terminal_reward`;
- role-only buckets combine different rounds and trajectories;
- Query, Evidence, and Answer normalize independently;
- the configured population mean, standard deviation, and epsilon behavior;
- singleton and zero-variance buckets yield zero advantages;
- parse-error actions remain trainable while ungenerated downstream actions are absent;
- decision advantages are broadcast only to valid action tokens;
- clipped policy loss and reference KL remain finite and use the new advantages.

### 4.4 Operational acceptance

After code-level tests pass:

1. Build complete E5-FAISS assets for `data/eval_1000` and `data/rl_train_2000`.
2. Validate each dataset's corpus hash, count, index size, and dimension.
3. Run real fixed-query retrieval smoke checks.
4. Run evaluation and GRPO configuration parsing/check-only paths with `retrieval_backend=e5_faiss`.
5. Run targeted tests followed by the complete pytest suite.
6. Report any validation not run because of dependency installation, GPU availability, or prohibitive runtime; do not represent partial validation as complete.

## Non-goals

- No change to the Qwen2.5 base architecture or shared-policy ownership.
- No learned value model.
- No second final-round Answer retry action.
- No deletion or conversion in place of existing LinearRAG assets.
- No reuse of R-Search output predictions as MACORAG training labels.
- No unrelated cleanup of the existing resume/vLLM worktree changes.
