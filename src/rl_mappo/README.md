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
stop after `early_stopping_patience` validation checks without improvement.
Before the first PPO update it also evaluates the untouched SFT-initialized
policy at step 0 and writes `baseline_validation.json`, so later checkpoints
measure actual RL gain against the same fixed holdout. Step 0 participates in
best-policy selection, preventing a worse RL checkpoint from replacing SFT.
`best_validation.json` records macro answer F1 and protocol eligibility.
An ineligible validation result cannot update `best_actor`, even when its scalar
score is higher. Eligibility must pass both the aggregate thresholds and every
dataset's thresholds; full recovery checkpoints are still saved independently.
Evidence coverage and per-stratum quality remain diagnostics and are not added to
the selection score. Protocol compliance and parse thresholds are hard gates,
including per-dataset gates. Early stopping requires three bad validations, cannot
trigger before step 150, and the run is capped at 200 updates to limit drift.

Training also records rolling protocol failure metrics and the raw response for
each invalid action. Valid tagged JSON receives `format_reward_weight`; invalid
actions receive `invalid_action_penalty`. In particular, the final-round answer
prompt now requires and demonstrates `can_answer=true`, matching the strict
parser instead of showing a contradictory non-final example. Non-final Answer
prompts contain one syntactically valid refusal example and one syntactically
valid supported-answer example; final-round prompts show only forced answering.
The vLLM path also applies guided-regex decoding on the final Answer action, fixing
`can_answer=true` and requiring a non-empty answer. A malformed or refusing final
Answer receives `-3`, receives no positive terminal reward, and gates off the
evidence component of the global reward. A correct non-final wait is only mildly
positive (`0.2`), while a legal final answer receives an independent `+1` bonus.
Prompt construction compacts state, retrieval history, accumulated evidence,
and the latest observation before tokenization while retaining passage IDs,
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
token-level reference-KL penalty (`beta=0.01`). Only the trainable policy adapter
is exported to vLLM/checkpoints. Entropy starts at `0.002`, anneals after step 100,
and reaches zero at step 150.

The optimizer-side Qwen2.5-7B base is loaded in 4-bit QLoRA mode by default.
The LoRA forward/backward compute remains BF16; vLLM on GPU 1 keeps its own
BF16 base for fast rollout generation. A full-BF16 optimizer-side 7B base does
not leave enough activation memory for long PPO re-scoring on a 24 GiB GPU.
The verified launch defaults therefore cap the shared vLLM/PPO contract at
1024 prompt tokens and 128 completion tokens. Every update prints a
`mappo_update_preflight` record with actual maximum token lengths and CUDA
allocated/reserved memory before the first policy forward.

For conservative small-batch updates, the default is one PPO epoch per rollout
batch. Advantages are normalized independently for Query, Evidence, and Answer
roles; singleton or tied role groups keep their original signal. The terminal
answer-F1 weight is 1.5 and the already-mastered format reward is 0.1, shifting
optimization toward answer quality without removing protocol supervision.

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
