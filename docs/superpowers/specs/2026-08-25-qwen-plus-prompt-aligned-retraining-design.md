# Qwen-Plus Prompt-Aligned SFT and RL Retraining Design

## Goal

Regenerate a clean teacher-trajectory dataset with Qwen-Plus, retrain the
Qwen2.5-7B SFT adapter from the base model, and then retrain the MACORAG GRPO
adapter without carrying forward the current answer-format regression.

The teacher generator, SFT renderer, GRPO rollout policy, validation policy,
and final evaluator must share one versioned prompt contract. For the same
question, state, and round metadata, SFT, RL, and evaluation must construct
byte-identical system and user messages. The only intentional exception is
the Qwen-Plus teacher API wrapper: it may request a top-level JSON object for
API-level structured output, but its answer-decision rules must be rendered
from the same contract used by the student-facing prompt.

## Scope and fixed decisions

- Use Scheme A: regenerate all teacher trajectories instead of converting the
  existing `data/sft/teacher_qwen_plus_trajectory_train` records in place.
- Generate exactly 1,000 valid trajectories for each of `2wiki`, `hotpotqa`,
  and `musique` from the 3,000 candidates per dataset under
  `data/trajectory_train`.
- Use `qwen-plus` as the teacher, with temperature `0.2`, four retrieval
  rounds, and resumable requests.
- Use MACORAG's repository-native `intfloat/e5-base-v2` plus CPU FAISS
  `IndexFlatIP` retrieval for teacher generation, SFT trajectory semantics,
  GRPO, validation, and evaluation.
- Freeze `retrieval_top_k: 5` for this retraining cycle. A top-k change is a
  separate retrieval experiment because it changes observation length and
  the student prompt distribution.
- Use query and passage prefixes, attention-mask mean pooling, L2
  normalization, and 512-token E5 truncation as specified by the existing
  E5-FAISS design.
- Use `max_rounds: 4` in teacher generation, SFT metadata, GRPO, validation,
  and evaluation.
- Use `model/Qwen2.5-7B-Instruct` as the SFT base. Do not initialize the new
  SFT run from an older SFT or RL adapter.
- Initialize both the trainable GRPO policy and frozen GRPO reference from the
  newly accepted SFT adapter. Do not warm-start from
  `outputs/grpo_qwen2.5-7b/2026-08-24_06-12-01/adapter`.
- Preserve all current datasets, indexes, adapters, checkpoints, and outputs.
  New artifacts use new versioned paths.
- Preserve LinearRAG as an explicit comparison backend, but do not mix it into
  this formal teacher/SFT/RL/evaluation chain.
- Do not make paid Qwen-Plus requests until local unit tests, prompt snapshots,
  E5 index validation, a no-network dry-run, and a small real teacher smoke run
  pass.

## 1. Versioned prompt contract

### 1.1 Single source of truth

`config/prompts.yml` becomes the configuration root for:

- `prompt_contract_version: macorag-rag-v2`;
- one student system prompt for each of `query_retriever`,
  `evidence_updater`, and `answer_generator`;
- stable role names and output tags.

`src/prompt_config.py` loads and validates this contract. Runtime role prompt
construction remains in `src/rag/prompts.py`. SFT must import all three
runtime builders rather than maintaining private `_build_*_prompt` copies.
The runtime policy selects the system prompt by role; it must not send an
answer-specific system message to Query or Evidence calls.

Evaluation must stop hard-coding its own copy of the system prompt. SFT, RL,
and evaluation may accept an explicit prompt configuration path, but they all
resolve through `src/prompt_config.py`. An unknown contract version or a
contract/version mismatch with an adapter is a fail-fast error.

### 1.2 Shared answer context

Answer prompt construction consumes an explicit immutable context:

```python
@dataclass(frozen=True)
class AnswerPromptContext:
    round_index: int
    max_rounds: int

    @property
    def is_final_round(self) -> bool:
        return self.round_index == self.max_rounds - 1

    @property
    def remaining_rounds(self) -> int:
        return self.max_rounds - self.round_index - 1
```

All stages call one interface:

```python
build_answer_generator_prompt(
    *,
    question: str,
    state: RAGState,
    context: AnswerPromptContext,
) -> str
```

No caller passes an independently computed `force_final_answer` boolean. This
prevents SFT, batched RL, synchronous evaluation, and teacher generation from
disagreeing about which round is final.

### 1.3 Query and evidence prompt consistency

Query and Evidence prompts are part of the same versioned contract even
though the observed regression is concentrated in Answer. Their SFT, RL, and
evaluation renderings must also be byte-identical.

The Query system prompt requires exactly one complete
`<query-retriever>...</query-retriever>` block containing a valid JSON object
with non-empty string `sub_goal` and string `query`. The user prompt supplies
the question, canonical state, explicit round metadata, and rules that forbid
unsupported intermediate entities, unresolved placeholders, and repeated
queries. An empty query is allowed only when no further retrieval is needed.

The Evidence system prompt requires exactly one complete
`<update-evidence>...</update-evidence>` block containing integer local
passage IDs and a short rationale. The user prompt supplies the question,
canonical state, latest observation, explicit round metadata, and requires
IDs to come only from that observation.

Neither role uses JSON ellipses, Markdown fences, or stage-specific examples.
The Qwen-Plus planning and evidence-selection requests may use a top-level
JSON transport wrapper, but their semantic rules are generated from these
same Query and Evidence contracts.

### 1.4 Shared Answer student system prompt

The student-facing system prompt is:

```text
You are the answer_generator in a multi-round retrieval QA system.
Follow the decision rule in the user message.
Return exactly one complete <answer>...</answer> block and nothing else.
The content inside <answer> must be one valid JSON object.
Required fields:
- "can_answer": JSON boolean
- "answer": non-empty string when can_answer=true; null when can_answer=false
- "rationale": string
Do not use Markdown fences.
Do not output text before or after the <answer> block.
Always close the </answer> tag.
```

The format contract also remains as a suffix at the end of the user prompt so
left truncation cannot remove the only copy of the output requirement.

### 1.5 Normal-round answer prompt

For rounds zero through `max_rounds - 2`, the user prompt states:

```text
Task: decide whether the accumulated evidence is sufficient to answer.
Decision mode: CONTINUE
Round: {round_index + 1} of {max_rounds}
Remaining retrieval rounds after this decision: {remaining_rounds}

Rules:
1. Use only selected evidence in <state>.
2. If the evidence is sufficient, return can_answer=true and a concise answer.
3. If the evidence is insufficient, return can_answer=false and answer=null.
4. Do not make a fallback guess in CONTINUE mode.

Question:
{question}

<state>
{state_json}
</state>

Valid output when answering:
<answer>{"can_answer":true,"answer":"short answer","rationale":"supported by selected evidence"}</answer>

Valid output when waiting:
<answer>{"can_answer":false,"answer":null,"rationale":"more evidence is required"}</answer>

Return exactly one complete <answer>...</answer> block and nothing else.
```

The examples contain valid JSON. Ellipsis tokens are not used as JSON values.

### 1.6 Final-round answer prompt

For `round_index == max_rounds - 1`, the same renderer switches to:

```text
Task: produce the final answer from the accumulated evidence.
Decision mode: FINAL
Round: {max_rounds} of {max_rounds}
Remaining retrieval rounds after this decision: 0

Rules:
1. You must return can_answer=true.
2. The answer must be a non-empty concise string.
3. Use selected evidence whenever it supports an answer.
4. If evidence is insufficient, make the best supported guess.
5. For a guess, rationale must start exactly with "fallback_guess:".
6. Never return can_answer=false or answer=null.

Question:
{question}

<state>
{state_json}
</state>

Valid supported output:
<answer>{"can_answer":true,"answer":"short answer","rationale":"supported by selected evidence"}</answer>

Valid fallback output:
<answer>{"can_answer":true,"answer":"best supported guess","rationale":"fallback_guess: evidence is incomplete"}</answer>

Return exactly one complete <answer>...</answer> block and nothing else.
```

The final round is a distinct semantic mode, not a separately maintained
template. The shared renderer owns both branches.

### 1.7 Shared state transition

SFT currently builds its answer state differently from the runtime executor.
The new path centralizes construction of the post-retrieval, post-evidence
answer state. It must update the same fields in every stage:

- `question`;
- current `sub_goal`;
- cumulative selected `evidence`;
- cumulative `retrieval_history`, including the current query, sub-goal, and
  top score;
- incremented `retrieval_count`.

Given identical inputs, teacher record rendering, SFT record rendering, RL,
and evaluation must serialize identical state JSON with stable key ordering.

### 1.8 Token-budget behavior

No stage may enforce `max_prompt_length` by blindly retaining the last tokens
of a fully serialized chat message. A shared prompt-budget component preserves
the complete role system prompt, task rules, question, round metadata, and
output-contract suffix. It reduces only bounded state payloads, using the same
deterministic policy in SFT, RL, and evaluation:

1. remove oldest retrieval-history entries first while preserving the current
   query context;
2. retain selected evidence in recency order and trim passage text to a
   configured per-passage token budget;
3. fail before generation if the protected prompt components still cannot fit.

SFT records that cannot fit under the configured student prompt limit are
reported and excluded, never truncated with a different policy from runtime.
Prompt-budget statistics include original tokens, retained tokens, evidence
items removed, and passage tokens clipped.

### 1.9 Prompt provenance

Every generated artifact records:

- prompt contract version;
- SHA256 of the canonical prompt configuration;
- E5 model and index metadata hashes;
- `max_rounds` and `retrieval_top_k`;
- base model and adapter paths where applicable.

The fields are written to teacher `summary.json`, SFT `train_meta.json`, RL
`train_meta.json`, checkpoint manifests, evaluation `run_config.json`, and
evaluation summaries. Resume and evaluation reject mismatches rather than
silently mixing prompt contracts.

## 2. Full E5 teacher-trajectory regeneration

### 2.1 Dedicated E5 indexes

The existing `data/rl_train_2000_e5_faiss` indexes cover only the 2,000-row RL
subset and cannot serve the 3,000 teacher candidates. Build new indexes from:

```text
data/trajectory_train/{dataset}/corpus.jsonl
```

into:

```text
data/trajectory_train_e5_faiss/{dataset}/
```

Each index must validate corpus SHA256, corpus count, FAISS vector count,
dimension 768, model `intfloat/e5-base-v2`, maximum length 512, mean pooling,
L2 normalization, and `IndexFlatIP` before teacher requests begin.

### 2.2 New output namespace

Teacher generation writes only to:

```text
data/sft/teacher_qwen_plus_trajectory_train_v2/
```

The existing teacher directory remains unchanged. The v2 directory contains:

- one trajectory JSONL per dataset;
- append-only raw teacher request/response traces with secrets excluded;
- append-only failure records;
- `summary.json`;
- `run_config.json` with prompt/index fingerprints;
- per-dataset progress files sufficient for exact qid-level resume.

Resume is allowed only when the stored run identity exactly matches the active
contract, source files, teacher model, retrieval settings, and generation
settings. A trailing partial JSON line may be repaired; earlier corruption is
a hard error.

### 2.3 Teacher semantics

The Qwen-Plus API may continue using `response_format={"type":"json_object"}`.
Its answer request returns:

```json
{
  "answer": {
    "can_answer": true,
    "answer": "short answer",
    "rationale": "supported by selected evidence"
  }
}
```

or, on a non-final round only:

```json
{
  "answer": {
    "can_answer": false,
    "answer": null,
    "rationale": "more evidence is required"
  }
}
```

Teacher requests use the same normal/final decision rules and round metadata
as the student prompt. The JSON-only transport difference is explicit and is
not copied into student SFT targets.

The current no-op handling of `force_final_answer` in the teacher request and
finalizer must be removed. A final teacher answer with `can_answer=false`, a
null/empty answer, an invalid fallback marker, or invalid JSON is retried and
never written as a valid trajectory.

### 2.4 Teacher data correctness

Qwen-Plus never receives the gold answer or gold supporting-fact labels in its
prompt. Gold data may be used only after generation for acceptance checks.

A valid trajectory must satisfy all of the following:

- unique qid within its dataset;
- one to four rounds;
- schema-valid query, evidence, and answer actions in every recorded turn;
- no unsupported intermediate entity in a query;
- selected passage IDs belong to the current observation;
- state transitions replay exactly;
- normal-round false answers use `answer=null`;
- the terminal turn has `can_answer=true` and a non-empty concise answer;
- a terminal fallback rationale starts exactly with `fallback_guess:`;
- the final answer matches a normalized gold answer or normalized alias;
- no forbidden training-label wording or gold leakage appears in assistant
  output;
- no raw API error, partial response, or parser repair is used as supervision.

If an otherwise valid trajectory ends with a wrong answer, it is rejected and
another source example is attempted. The pipeline never overwrites the
teacher answer with the gold answer.

### 2.5 Scale-up gates

Generation proceeds in observable stages:

1. no-network dry-run for two samples per dataset;
2. real Qwen-Plus smoke for two samples per dataset in a disposable output
   namespace;
3. audited pilot of 32 valid samples per dataset;
4. full resumable generation to 1,000 valid samples per dataset.

The pilot must achieve zero schema failures after accepted-record filtering,
100% valid final-round answers, no gold leakage, and successful replay through
the exact SFT renderer. Otherwise full generation does not start.

## 3. SFT-v2 training

### 3.1 Dataset construction

`src/sft_training/data.py` no longer owns private role prompt strings. It
converts each accepted v2 trajectory into action records using the shared
runtime prompt builders and shared state transition.

For answer actions:

- a normal intermediate turn is rendered in `CONTINUE` mode;
- the fourth round is rendered in `FINAL` mode;
- an earlier supported terminal answer remains a valid `CONTINUE`-mode true
  answer;
- no synthetic final-mode clone is added because Scheme A generates real
  final semantics directly.

Assistant targets are serialized deterministically as exactly one role tag
containing compact valid JSON. No Markdown, prefix, suffix, or invalid example
is present.

### 3.2 Training identity

The formal SFT run uses:

- base model `model/Qwen2.5-7B-Instruct`;
- the v2 teacher data root;
- LoRA rank 16, alpha 32, dropout 0.05, and the existing target modules;
- qid-level deterministic train/eval split with no action records from one qid
  crossing the split;
- validation enabled and best adapter selected by eval loss;
- a new output namespace under `outputs/sft_qwen2.5-7b-instruct/`.

Exact optimizer and epoch settings remain configured in `config/train_sft.yml`
but are recorded in metadata. Changing them does not change the prompt
contract identity.

### 3.3 SFT acceptance

Before GRPO starts, evaluate the SFT adapter on a fixed held-out set excluded
from SFT and RL optimization. Required gates are:

- overall role parse-error rate at most 1%;
- `Missing required tag: answer` rate at most 0.2%;
- final-round `can_answer=true` and non-empty-answer compliance at least 99%;
- raw generated response saved for every parse failure;
- no prompt-contract or index fingerprint mismatch;
- per-dataset answer coverage, EM, F1, supporting-fact recall, selected-evidence
  recall, and average rounds reported.

These are protocol gates, not claims that the SFT model has reached final task
quality. Failure of a protocol gate blocks GRPO.

## 4. GRPO-v2 training

### 4.1 Initialization and consistency

The new SFT adapter initializes:

- the trainable policy adapter;
- the frozen reference adapter;
- the initial vLLM LoRA state.

Only the policy is updated. The reference remains frozen at the accepted SFT
adapter. The current broken RL adapter is not used by any of these roles.

GRPO uses the same prompt contract, `max_rounds: 4`, E5 model, E5 query
semantics, and `retrieval_top_k: 5`. Training corpus indexes remain under the
RL-specific root and must match the configured corpus hashes.

### 4.2 Failure observability

Every generated action trace stores the raw response before parsing. A parse
failure remains a trainable sampled action with its format penalty when token
and old-logprob data exist. It is never converted into a fabricated valid
answer.

Training metrics report by role and rolling window:

- missing-tag rate;
- invalid-JSON rate;
- final-answer protocol violation rate;
- blank-answer rate;
- fallback rate;
- answer coverage;
- answer F1;
- KL and reward;
- supporting-fact and evidence-selection coverage.

### 4.3 Checkpoint selection

The formal output retains full checkpoints with policy, optimizer, trainer,
RNG, prompt/index fingerprints, and generation-counter state. Fixed held-out
evaluation runs at configured checkpoint intervals.

The final adapter is selected by validation quality subject to hard protocol
gates; it is not automatically the last step. Any checkpoint with overall
parse errors above 1%, missing answer tags above 0.2%, or final-round
compliance below 99% is ineligible even if its reward is higher.

If rolling training parse errors exceed 2% for two consecutive 100-step
windows, training stops for diagnosis rather than continuing to an obviously
degraded final adapter.

## 5. Evaluation-v2

Evaluation loads the shared prompt contract through `prompt_config`, not a
hard-coded string. It verifies that the adapter manifest, prompt contract,
E5 index, `max_rounds`, and top-k match the requested run.

Every prediction records the original raw role responses in addition to the
parsed trajectory. Evaluation reports separately:

- overall EM, strict one-way Contain-Acc, repository contain accuracy, and F1;
- LLM judge accuracy as a secondary metric;
- answer coverage and answered-only EM/F1;
- parse failures by role and error message;
- final forced-answer compliance and fallback quality;
- supporting-fact Recall@5 by round and cumulative chain coverage, grouped by
  2-hop, 3-hop, and 4-hop questions.

LLM judge accuracy never substitutes for deterministic metrics or protocol
compliance.

## 6. Testing and acceptance

### 6.1 Prompt contract tests

Tests must prove:

- SFT, RL, and evaluation produce byte-identical system and user messages for
  identical inputs;
- teacher normal/final decision rules are semantically identical to the
  student contract;
- all embedded examples are valid JSON;
- final mode cannot render a false/null example;
- fallback uses exactly `fallback_guess:`;
- evaluation no longer has a private system-prompt constant;
- prompt version and SHA mismatch are rejected;
- long-state truncation retains question, decision mode, round metadata, and
  the complete output suffix.

### 6.2 Teacher pipeline tests

Tests cover:

- full use of `force_final_answer` semantics;
- invalid final answers are retried/rejected;
- no gold appears in teacher messages;
- gold/alias validation occurs only after generation;
- deterministic qid sampling and dataset quotas;
- exact resume identity and partial-line recovery;
- raw response logging without API keys;
- separate old/v2 output namespaces;
- E5 corpus/index mismatch fails before any teacher request.

### 6.3 Training and evaluation tests

Tests cover:

- qid-level SFT split isolation;
- shared state transition equivalence;
- new SFT and frozen-reference adapter ownership;
- raw failed-response persistence;
- rolling protocol metrics and stop conditions;
- full checkpoint prompt/index provenance;
- checkpoint eligibility gates;
- evaluation metric separation and hop-aware retrieval reporting.

### 6.4 End-to-end acceptance sequence

1. Run prompt and teacher unit tests.
2. Run all affected SFT, RAG, RL, retrieval, checkpoint, and evaluation tests.
3. Build and validate full `trajectory_train` E5 indexes.
4. Run no-network teacher dry-run.
5. Run six-sample real teacher smoke.
6. Run 96-sample teacher pilot and audit it.
7. Generate and validate 3,000 accepted trajectories.
8. Run SFT check-only and a tiny training smoke.
9. Train formal SFT-v2 and pass held-out protocol gates.
10. Run GRPO check-only and a tiny real rollout/update smoke.
11. Train GRPO-v2 with checkpoint validation and early protocol-stop rules.
12. Select the best eligible checkpoint and run the full 3x1,000 evaluation.

Each long stage is resumable and writes a new run ID. A failed gate stops the
next stage. No stage silently falls back to old data, LinearRAG, an old prompt,
or an old adapter.

## Non-goals

- Do not alter the Qwen2.5 architecture.
- Do not use the current broken RL adapter as a warm start.
- Do not overwrite or delete old teacher data, indexes, SFT adapters, RL
  checkpoints, or evaluations.
- Do not silently repair malformed teacher/student outputs into supervision or
  scored predictions.
- Do not add constrained decoding in this cycle; first verify whether the
  aligned prompt and clean retraining restore protocol compliance.
- Do not change retrieval top-k during the formal SFT-to-RL-to-evaluation
  chain.
- Do not include unrelated cleanup from the existing dirty worktree.
