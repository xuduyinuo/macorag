# MACORAG MAPPO

This directory is a self-contained MAPPO implementation. It does not import
code from `src/rag`, `src/rl_training`, `src/data_processing`, or other local
packages.

The three roles (`query_retriever`, `evidence_updater`, `answer_generator`) are
cooperative agents. They share a role-conditioned LoRA language-model actor for
decentralized execution. A separate centralized critic observes the full joint
RAG state. Each role receives local credit plus the shared terminal reward, and
GAE is computed along that role's own turn sequence. Optimization uses clipped
MAPPO actor and value objectives, entropy regularization, multiple PPO epochs,
gradient clipping, and a target-KL early stop.

The default rollout backend is a vLLM service on GPU 1. Start it first from the
repository root (terminal 1):

```bash
/data/conda/envs/macorag/bin/python -m src.rl_mappo.launch_vllm --dry-run
/data/conda/envs/macorag/bin/python -m src.rl_mappo.launch_vllm
```

Then validate and start the MAPPO optimizer on GPU 0 (terminal 2):

```bash
/data/conda/envs/macorag/bin/python -m src.rl_mappo.train_mappo --check-only
/data/conda/envs/macorag/bin/python -m src.rl_mappo.train_mappo
```

For normal training, the recommended one-command launcher starts both sides,
waits for exact model readiness, and cleans up only its own vLLM process:

```bash
src/rl_mappo/run_mappo.sh
```

Preview without starting a service or loading a model:

```bash
MAPPO_LAUNCH_DRY_RUN=1 src/rl_mappo/run_mappo.sh
```

The console shows a sample-level `tqdm` bar with rollout/PPO/sync stages and
the latest reward, F1 and invalid-action rate. vLLM output is captured in a
temporary `/tmp/mappo_vllm_*.log` file only for startup diagnostics and is
deleted on exit. No new logs are written to `outputs/mappo_launcher_logs`.
Set `MAPPO_VLLM_LOG=/some/path.log` only when a persistent service log is
explicitly needed.

The configured sample pool is split deterministically by dataset and by each
dataset's question type/hop stratum, producing disjoint training and validation
sets. The formal configuration loads 500 questions per dataset, reserves exactly
50 per dataset for the fixed holdout, and gives rare MuSiQue `4hop2`/`4hop3`
questions 1.5 times their natural selection weight without duplicating QIDs.
Validation uses temperature 0 every 50 steps, writes `validation_metrics.jsonl` and
`validation_episodes.jsonl`, exports protocol-eligible macro-F1 improvements to
`best_answer_actor/` (and the compatible `best_actor/` alias), tracks protocol
quality separately in `best_protocol_actor/`, and can
stop after `early_stopping_patience` validation checks without quality improvement.
Early-stopping patience is an independent scheduled-validation state: the step-0
baseline initializes it, only validations at `validation_steps` update it, and
the final validation cannot increment, reset, or move its best score. KL emergency
uses controller recovery without running the full holdout by default, preventing
clusters of expensive validations a few steps apart.
Before the first PPO update it also evaluates the untouched SFT-initialized
policy at step 0 and writes `baseline_validation.json`, so later checkpoints
measure actual RL gain against the same fixed holdout. Step 0 participates in
best-policy selection, preventing a worse RL checkpoint from replacing SFT.
`best_validation.json` records macro answer F1 and protocol eligibility.
An ineligible validation result cannot update `best_actor`, even when its scalar
score is higher. Eligibility uses the complete validation set with 2% parse,
1% missing-answer, and 98% final-compliance gates. Per-dataset rates remain
diagnostics rather than independently vetoing a checkpoint; full recovery
checkpoints are still saved independently.
Evidence coverage and per-stratum quality remain diagnostics and are not added to
the selection score. Protocol compliance and parse thresholds are hard checkpoint
gates over the complete validation set; per-dataset values are diagnostics. Early
stopping requires three bad scheduled quality validations and cannot trigger before
step 150. Otherwise the trainer consumes the complete configured training split for
each `num_train_epochs`; there is no independent optimizer-step cap.

Training also records rolling protocol failure metrics and the raw response for
each invalid action. Valid tagged JSON receives `format_reward_weight`; invalid
actions receive `invalid_action_penalty`. A format failure gets one conservative
recovery opportunity first: known JSON serialization artifacts are repaired and
re-tokenized/re-scored, otherwise the action is generated once more. Semantic
violations remain invalid without retry. Per-role repair/retry rates and episode
recovery events are logged explicitly. In particular, the final-round answer
prompt now requires and demonstrates `can_answer=true`, matching the strict
parser instead of showing a contradictory non-final example. Non-final Answer
prompts contain one syntactically valid refusal example and one syntactically
valid supported-answer example; final-round prompts show only forced answering.
The vLLM path also applies guided-regex decoding on the final Answer action, fixing
`can_answer=true` and requiring a non-empty answer. Evidence actions use a dynamic
finite grammar built from the current observation. Each compact passage is labeled
`P0`...`P4` immediately beside its title/text, and only unique subsets of those
observed pointers can be generated, so duplicate and out-of-range selections are blocked
before parsing. A non-empty observation requires at least one selected passage,
preventing the empty-selection collapse seen with the original grammar. Evidence
rationale remains in the shared SFT/MAPPO protocol and is capped at 64 characters,
but its tokens are excluded from Evidence PPO and reference-KL objectives. A startup
tokenizer check verifies the worst valid pointer action fits the dedicated 192-token
action budget. Across rounds, passages already
present in accumulated evidence are removed before the Evidence prompt and guided
grammar are constructed. State merging performs the same stable-ID deduplication
again, so repeated retrievals cannot consume the Answer evidence budget. A malformed or refusing final
Answer receives `-3`, receives no positive terminal reward, and gates off the
evidence component of the global reward. A correct non-final wait is only mildly
positive (`0.2`), while a legal final answer receives an independent `+1` bonus.
Prompt construction is shared by SFT, MAPPO, and evaluation. It compacts state,
retrieval history, accumulated evidence, and the latest observation before
tokenization while retaining passage pointers,
titles, item counts, and both early and recent multi-hop context. All role
contracts are placed at the prompt tail; Answer few-shots are also after the
state. If a prompt still exceeds 1024 tokens, encoding preserves the complete
system-message prefix and the contract/few-shot tail instead of deleting the
system prompt with a plain tail slice.

The trainer sends exact prompt token IDs to `/v1/completions` and consumes the
returned token IDs and server-side behavior log probabilities. Before each
MAPPO update, the sampled token IDs are re-scored without gradients by the
local trainable actor. PPO old/new log probabilities therefore use the same
4-bit numerical path; the BF16-vs-4-bit difference remains available in
`behavior_logprob_abs_diff` and `behavior_approx_kl`. After each MAPPO
update it saves a versioned LoRA snapshot and reloads `mappo_policy` through
vLLM's runtime LoRA API. Local loopback requests explicitly bypass proxy
environment variables.

The learner keeps a second frozen copy of the initialization SFT LoRA and adds a
token-level reference-KL penalty. Query, Evidence, and Answer now have independent
adaptive controllers and checkpointed state. For Evidence, both policy and frozen
reference logits are normalized over the same legal pointer-token trie used by
guided decoding. Only genuine pointer-choice tokens contribute to PPO and the
controller; `selection_kl`, `selection_raw_kl`, `rationale_kl`, and `format_kl`
separate the decision signal from diagnostic drift. A role-specific emergency now
requires both the current KL and its EMA to stay above threshold. It boosts the
affected role's KL beta and temporarily reduces that role's gradient contribution;
checkpoint ineligibility alone never terminates training. Optional emergency
validation remains disabled in the formal configuration.
Only the trainable policy
adapter is exported to vLLM/checkpoints. Entropy starts at `0.002`, anneals after
step 100, and reaches zero at step 150.

The optimizer-side Qwen2.5-7B base is loaded in 4-bit QLoRA mode by default.
The LoRA forward/backward compute remains BF16; vLLM on GPU 1 keeps its own
BF16 base for fast rollout generation. A full-BF16 optimizer-side 7B base does
not leave enough activation memory for long PPO re-scoring on a 24 GiB GPU.
The verified launch defaults therefore use 1024 prompt tokens, 128 Query tokens,
and dedicated 192-token Evidence and Answer ceilings. Every update prints a
`mappo_update_preflight` record with actual maximum token lengths and CUDA
allocated/reserved memory before the first policy forward.

For conservative small-batch updates, the default is one PPO epoch per rollout
batch. Advantages are normalized independently for Query, Evidence, and Answer
roles; singleton or tied role groups keep their original signal. Actor gradients
are accumulated from one homogeneous microbatch per available role and then applied
in one shared optimizer step. Configured role weights are renormalized when a
trajectory omits a role; Evidence defaults to half the Query/Answer weight because
its pointer loss contains fewer, higher-variance tokens. `train_metrics.jsonl` records per-role validity, local/team rewards, raw and
normalized advantages, action lengths, evidence-selection behavior, actor loss,
entropy, PPO KL, reference KL, and clip fraction under `role_metrics`. The terminal
answer-F1 weight is 1.5 and the already-mastered format reward is 0.1, shifting
optimization toward answer quality without removing protocol supervision.

Role-gradient diagnostics are observational and do not modify the optimizer. Every
`gradient_diagnostics_steps`, the first configured logical role-balanced group uses
the same weighted actor/entropy/reference-KL objectives and writes their individual
gradients into the unchanged summed `.grad`. `gradient_diagnostics.jsonl` records
pairwise cosine/dot products, conflict rate, role and combined norms, cancellation
ratio, and each role's cosine with the final combined update.

External evaluation can use `evaluation_algorithm: mappo` with
`mappo_config_path: src/rl_mappo/train_mappo.yml`. This reuses `RolloutCollector`
and inherits the training validation contract: role prompts and compaction, query /
evidence / answer token budgets, retrieval settings, cross-round evidence deduplication,
guided Evidence pointers, forced final Answer decoding, format recovery, and MAPPO
protocol thresholds. The evaluation contract stores the training YAML SHA-256 so a
result cannot silently masquerade as an aligned run after configuration drift.

Resume a full checkpoint:

```bash
# Restart terminal 1 with the checkpoint actor.
/data/conda/envs/macorag/bin/python -m src.rl_mappo.launch_vllm \
  --adapter-path outputs/mappo_Qwen2.5-7B-Instruct/RUN/checkpoint-N/actor

# Resume optimization in terminal 2.
/data/conda/envs/macorag/bin/python -m src.rl_mappo.train_mappo \
  --resume-from-checkpoint outputs/mappo_Qwen2.5-7B-Instruct/RUN/checkpoint-N
```

The one-command resume form is:

```bash
src/rl_mappo/run_mappo.sh \
  --resume-from-checkpoint outputs/mappo_Qwen2.5-7B-Instruct/RUN/checkpoint-N
```

Configuration lives in `train_mappo.yml`. Outputs include `run_config.json`,
`episodes.jsonl`, `train_metrics.jsonl`, full optimizer/RNG checkpoints, actor
adapters, critic weights, and `summary.json`.
