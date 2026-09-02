# MACORAG Reward, Credit Assignment, and Cross-Model Ablation Design

## Goal

Design the smallest formal experiment package that answers three reviewer-facing
questions:

1. Do local action rewards and the global trajectory reward provide
   complementary supervision?
2. Does role-round action-level credit assignment outperform broadcasting one
   trajectory-level advantage to every action?
3. Does the complete method improve both Qwen2.5-7B and Llama-3-8B relative to
   their own SFT initializations?

The package uses one training seed, five GRPO training configurations, and seven
full evaluations. Every GRPO comparison uses the same 3,000 training questions
and the same 3,000-question evaluation set. The complete Qwen method continues
from 3,000 to 6,000 consumed training samples only for the convergence study;
the ablation table compares all Qwen variants at the common 3,000-sample
boundary.

## Claims and hypotheses

### H1: Reward complementarity

For an action from role `j` at round `t`, the complete decision return is

\[
G_{i,t}^{j}=r_{i,t}^{j}+\lambda_jR(\tau_i).
\]

Local reward `r` supplies process supervision for query formulation, evidence
selection, and answer timing. Global reward `R` aligns those decisions with the
final trajectory outcome. The hypothesis is not merely that the combined
reward has the highest answer F1; it should also retain or improve the
role-specific process metrics that each isolated reward source favors.

### H2: Action-level credit assignment

The complete method normalizes decision returns among comparable actions from
the same role and round. This should outperform a trajectory-level baseline
that uses the same underlying reward information but assigns one normalized
advantage to every action in a rollout.

### H3: Cross-model effect

The complete RL method should improve both Qwen2.5-7B and Llama-3-8B relative
to the corresponding SFT adapter. The claim is based on within-backbone deltas,
not on which backbone has the larger absolute score.

## Formal experiment matrix

| ID | Backbone | Local reward | Global reward | Credit assignment | Training samples |
|---|---|---:|---:|---|---:|
| `qwen_full` | Qwen2.5-7B | yes | yes | role-round action-level | 3,000, then continue to 6,000 |
| `qwen_local` | Qwen2.5-7B | yes | no | role-round action-level | 3,000 |
| `qwen_global` | Qwen2.5-7B | no | yes | role-round action-level | 3,000 |
| `qwen_trajectory` | Qwen2.5-7B | yes | yes | trajectory-level broadcast | 3,000 |
| `llama_full` | Llama-3-8B | yes | yes | role-round action-level | 3,000 |

The evaluation-only SFT controls are:

| ID | Backbone | Adapter |
|---|---|---|
| `qwen_sft` | Qwen2.5-7B | `outputs/sft_qwen2.5-7b-instruct-v2/2026-08-26_19-21-27/adapter` |
| `llama_sft` | Llama-3-8B | final exported `adapter/` under `outputs/sft_Meta-Llama-3-8B/2026-09-01_23-12-10` |

Formal Llama evaluation and GRPO training must not begin until that run exports
its final adapter and prompt contract. An intermediate checkpoint is not
silently substituted for the final SFT control.

## Exact variant semantics

### `qwen_full` and `llama_full`

Use the configured local action rewards, terminal trajectory reward, role
coefficients, and role-round normalization:

\[
A_{i,t}^{j}=\operatorname{Norm}_{q,j,t}
\left(r_{i,t}^{j}+\lambda_jR(\tau_i)\right).
\]

The current coefficients remain fixed:

- Query Retriever: `1/3`;
- Evidence Updater: `3/7`;
- Answer Generator: `7/3`.

### `qwen_local`

Use only the local action reward:

\[
A_{i,t}^{j}=\operatorname{Norm}_{q,j,t}(r_{i,t}^{j}).
\]

The terminal reward is still computed and logged for analysis, but its
coefficient in the trainable decision return is exactly zero.

### `qwen_global`

Use only the terminal trajectory reward:

\[
A_{i,t}^{j}=\operatorname{Norm}_{q,j,t}
\left(\lambda_jR(\tau_i)\right).
\]

Local rewards are still computed and logged as diagnostics, but they do not
enter the trainable decision return. Because normalization is performed within
each role-round bucket, positive role-specific scaling alone does not create a
preference; the preference comes from differences in terminal outcomes among
rollouts.

### `qwen_trajectory`

Compute the same per-action decision returns as `qwen_full`, then aggregate
them without changing their reward content:

\[
\bar G_i=\frac{1}{|\mathcal A_i|}
\sum_{(j,t)\in\mathcal A_i}
\left(r_{i,t}^{j}+\lambda_jR(\tau_i)\right).
\]

Normalize `bar G` across rollouts sampled for the same question and broadcast
the resulting scalar advantage to every valid generated action in that
rollout. The mean is required rather than the sum so longer trajectories do
not receive larger-magnitude returns solely because they contain more actions.

This baseline preserves generated trajectories, rewards, KL, clipping,
sampling, retrieval, and optimizer behavior. Only the credit-assignment
mapping changes.

## Controlled training protocol

### Shared data manifest

Build one immutable 6,000-question master manifest from every record in
`data/rl_train_2000_stratified_v2`. Its ordered first half contains exactly
1,000 questions from each of 2Wiki, HotpotQA, and MuSiQue; its ordered second
half contains the remaining 1,000 questions from each dataset. Both halves are
proportionally stratified within each dataset. Store the ordered QIDs, dataset
labels, strata, source fingerprints, and selection seed `20260826`.

All five training configurations consume the exact same 3,000-question prefix
in the exact same order. Only `qwen_full` subsequently consumes the ordered
3,000-question suffix. Model-specific tokenization is allowed; reselection or
reordering is not. The training seed and generation-seed derivation are
identical across variants.

### Shared optimization and environment

Except for backbone/adapter identity and the declared ablation variable, keep
the following fixed:

- one pass over the shared 3,000-question prefix;
- initial group size, adaptive expansion policy, sampling temperature, top-p,
  maximum rounds, prompt/completion budgets, and retrieval top-k;
- E5-FAISS retrieval assets and retrieval parameters;
- LoRA rank, alpha, dropout, target-module family, optimizer, learning-rate
  schedule, KL coefficient, clipping, and gradient accumulation;
- prompt contract and final-round answer behavior;
- checkpoint boundaries and logging schema.

Adaptive group expansion remains enabled for all five configurations so it is
not confounded with reward or credit assignment. Group informativeness must be
defined using a validated numerical tolerance shared by every variant.

Backbone-specific target-module names may differ only when the architectures
require it. Such differences must be recorded in run metadata and must not be
described as an ablation variable.

### Checkpoints and convergence

Save complete checkpoints at consumed-sample boundaries 1,000, 2,000, and
3,000 for every GRPO configuration. The main ablation comparison uses only the
3,000 boundary.

Continue `qwen_full` from its complete 3,000 checkpoint to 6,000 total consumed
samples using the ordered 3,000-question suffix of the master manifest. Preserve optimizer,
scheduler, RNG, generation counter, and data cursor. Evaluate its 1,000,
2,000, 3,000, and 6,000 boundaries for a convergence plot. Do not compare a
6,000-sample full model against 3,000-sample ablations in the main ablation
table.

### Pilot gate

Before formal runs, execute 100--300 consumed samples for each backbone using
the complete reward and credit configuration. A pilot passes only if:

- all gradients and rewards are finite;
- parse-failure rate is below 1%;
- missing-answer-tag rate is below 0.2%;
- tied decision-return buckets do not become informative from floating-point
  residue;
- effective-update and expansion-success metrics use the same nonzero
  advantage tolerance;
- complete checkpoints can be resumed without changing the selected QIDs or
  generation-seed sequence.

A failed pilot is repaired and rerun. Formal variants are not launched with a
known credit-signal or resume defect.

## Evaluation protocol

Evaluate the following seven models on all 3,000 examples in
`data/eval_1000_stratified_v2`:

1. `qwen_sft`;
2. `qwen_full` at 3,000;
3. `qwen_local` at 3,000;
4. `qwen_global` at 3,000;
5. `qwen_trajectory` at 3,000;
6. `llama_sft`;
7. `llama_full` at 3,000.

The Qwen convergence study additionally evaluates `qwen_full` at 1,000,
2,000, and 6,000. All evaluations use the same ordered QIDs, E5-FAISS indexes,
prompt contract, maximum rounds, retrieval top-k, generation settings, metric
implementation, and output schema. Contract fingerprints must match before
paired comparisons are accepted.

## Metrics and analysis

### Primary answer-quality metrics

Report per-dataset and macro-average values for:

- Exact Match (EM, higher is better);
- Contain-Accuracy (higher is better);
- token F1 (higher is better, primary selection metric).

The main table contains absolute metrics. Each ablation row also reports the
paired delta from `qwen_full` at the same 3,000-sample boundary.

### Process and training metrics

Report metrics tied to the mechanism rather than relying on answer F1 alone:

- Query Retriever: newly retrieved supporting-fact rate and query novelty;
- Evidence Updater: selected-evidence support coverage, invalid-selection rate,
  and evidence precision when labels are available;
- Answer Generator: answer rate, final fallback rate, answer F1, and premature
  answer rate;
- trajectory: mean rounds, retrieval calls, parse-failure rate, and final
  protocol compliance;
- optimization: zero-advantage rate, effective optimizer-update rate,
  role-specific nonzero-advantage fraction, mean effective group size,
  expansion trigger/success rate, KL, clipping, and runtime per consumed
  sample.

Missing support labels remain missing; they are not replaced by inferred gold
evidence for the process tables.

### Statistical analysis under one training seed

Training-seed variance cannot be estimated and must be stated as a limitation.
For evaluation uncertainty, use paired stratified bootstrap over the identical
QIDs:

- resample within each dataset;
- use 10,000 bootstrap replicates and analysis seed `20260902`;
- recompute per-dataset and macro F1 differences for each paired comparison;
- report the paired delta and 95% percentile confidence interval.

The bootstrap supports uncertainty over evaluation questions, not over RL
training seeds. Do not describe it as evidence of training stability.

## Paper tables and figures

### Table 1: Reward and credit ablation on Qwen2.5-7B

Rows: `qwen_full`, `qwen_local`, `qwen_global`, `qwen_trajectory`, and
`qwen_sft`. Columns: per-dataset F1, macro F1, EM, Contain-Accuracy, evidence
coverage, effective-update rate, and zero-advantage rate. The caption states
that all GRPO rows use the same 3,000 training questions and one seed.

### Table 2: Cross-model effect

For each backbone, report SFT, Full-RL, absolute scores, and within-backbone
delta. The claim is supported only if the complete method yields a meaningful
gain over its own SFT initialization without unacceptable protocol regression.

### Figure 1: Reward-source mechanism analysis

Show final macro F1 together with Query support-hit rate and Evidence support
coverage for `qwen_full`, `qwen_local`, and `qwen_global`. This connects reward
source to the stage it is intended to supervise.

### Figure 2: Credit and convergence analysis

Use two panels:

- `qwen_full` versus `qwen_trajectory`: role-specific nonzero-advantage and
  effective-update rates;
- `qwen_full` checkpoints 1,000/2,000/3,000/6,000: macro F1 and process
  coverage.

## Decision rules

H1 is supported when the combined reward improves macro F1 over both isolated
reward sources and its process metrics show the expected complementary pattern.
If one isolated reward wins, report that result and weaken the complementarity
claim rather than selecting a different checkpoint post hoc.

H2 is supported when action-level credit improves paired macro F1 and/or the
role-specific process outcomes over trajectory broadcasting while using the
same reward information. Effective-update rate alone is diagnostic evidence,
not sufficient proof of better task performance.

H3 is supported when Full-RL improves over SFT for both backbones. If only one
backbone improves, report model sensitivity and do not claim backbone-agnostic
generalization.

No formal configuration is rerun with altered hyperparameters solely because
its result is unfavorable. Any failed or incomplete run is resumed from its
latest compatible complete checkpoint; semantic changes require a new run ID.

## Required implementation surface

Before experiments, add explicit, checkpoint-critical configuration fields for
reward mode and credit mode, rather than maintaining separate hand-edited code
branches. The intended values are:

```yaml
reward_mode: local_global  # local_global | local_only | global_only
credit_assignment_mode: role_round  # role_round | trajectory
```

Run metadata, checkpoint fingerprints, dry-run output, and training JSONL must
record both fields. Focused tests must verify each decision-return formula,
trajectory mean aggregation, broadcast behavior, numerical zero tolerance,
resume rejection on semantic mismatch, and disabled-reward diagnostic logging.

Thin launch configurations should name outputs by experiment ID and backbone.
They must share one immutable training manifest and must never reuse an RL
checkpoint from another variant.

## Scope boundaries

- One training seed is used because of the chosen compute budget.
- Reward and credit ablations are run on Qwen2.5-7B only.
- Llama-3-8B validates the complete method, not every ablation.
- Retriever, final-answer fallback, KL weight, group size, adaptive expansion,
  role coefficients, and prompt design are controlled variables, not ablations
  in this package.
- The 6,000-sample Qwen continuation is a convergence experiment, not an extra
  row in the equal-budget ablation table.
- No formal training starts as part of implementing the experiment harness.
