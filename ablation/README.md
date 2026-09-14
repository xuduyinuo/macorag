# MACORAG framework ablations

All six variants use the local `model/Qwen2.5-7B-Instruct` base model. This directory contains the complete launch surface and generated artifacts for the six framework variants. The fixed training-data contract is 1,000 questions: 400 2WikiMultiHopQA, 400 HotpotQA, and 200 MuSiQue. `prepare_data.py` deterministically materializes that split under `ablation/data/` with seed `20260905`. SFT validation splitting is disabled so all 1,000 selected questions participate in training.

Every experiment script evaluates its resulting model automatically after training. All variants share one fixed, train-disjoint evaluation manifest containing 500 questions from each dataset (1,500 total), selected with seed `20260905` by proportional difficulty/type strata from `data/eval_1000_stratified_v2`.

## Variant contract

| Script | Variant | Changed component |
|---|---|---|
| `01_full.sh` | MACORAG Full | Four retrieval rounds, role/round action credit, local + terminal reward, SFT + GRPO |
| `02_wo_multi_round_retrieval.sh` | w/o Multi-round Retrieval | `max_rounds=1`; all other GRPO settings match Full |
| `03_wo_fine_grained_credit.sh` | w/o Fine-grained Credit | One normalized terminal-return advantage is broadcast to every action in a rollout |
| `04_wo_local_reward.sh` | w/o Local Reward | All Query/Evidence/Answer local rewards are zero; terminal reward remains |
| `05_wo_grpo.sh` | w/o GRPO | SFT only on the fixed 1,000-question split |
| `06_wo_sft_grpo.sh` | w/o SFT & GRPO | Untrained base-model control; no training data are consumed |

The GRPO variants deliberately share the exact adapter produced by `05_wo_grpo.sh`. This avoids retraining a different SFT initialization for every row of the ablation table. The one-round variant disables only the strict adapter `max_rounds` metadata check because its shared SFT adapter records the Full setting (`max_rounds=4`).

## Run order

From the repository root, first build the common SFT control:

```bash
bash ablation/05_wo_grpo.sh
```

Then run each GRPO variant separately:

```bash
bash ablation/01_full.sh
bash ablation/02_wo_multi_round_retrieval.sh
bash ablation/03_wo_fine_grained_credit.sh
bash ablation/04_wo_local_reward.sh
```

Register the base-model control:

```bash
bash ablation/06_wo_sft_grpo.sh
```

Each command follows `start training vLLM -> train -> stop training vLLM -> start variant-specific evaluation vLLM -> evaluate -> stop evaluation vLLM`. Both services run in isolated process groups. Normal completion, errors, and shell interruption all trigger group-wide cleanup; graceful shutdown uses a 30-second timeout before forced termination. Evaluation artifacts are stored beside the corresponding model run under `evaluation/eval_500_each/`, including per-dataset predictions and metrics plus aggregate metrics. The one-round ablation is also evaluated with `max_rounds=1`; all other variants use four rounds.

Before starting vLLM, the launcher validates whether all fixed QIDs and metrics are already complete. A completed evaluation is skipped safely. For an incomplete evaluation it resumes existing per-QID predictions and automatically chooses the GPU with the most free memory (at least 18,000 MiB by default).

The fixed evaluation strata are:

- 2WikiMultiHopQA: `question_type` (`compositional`, `comparison`, `bridge_comparison`, `inference`).
- HotpotQA: difficulty and question type (`hard/bridge`, `hard/comparison`).
- MuSiQue: hop/composition type (`2hop`, `3hop1`, `3hop2`, `4hop1`, `4hop2`, `4hop3`).

The selected QIDs, source hashes, manifest fingerprint, and exact stratum counts are recorded in `ablation/data/eval_500_each/manifest_meta.json`. The current overlap audit reports zero QID overlap with the fixed ablation training data.

To use an already trained compatible SFT adapter instead of the recorded `05_wo_grpo.sh` result:

```bash
SFT_ADAPTER_PATH=/absolute/path/to/adapter bash ablation/01_full.sh
```

Launch-only validation does not start training:

```bash
MACORAG_LAUNCH_DRY_RUN=1 bash ablation/05_wo_grpo.sh
MACORAG_LAUNCH_DRY_RUN=1 SFT_ADAPTER_PATH=/absolute/path/to/adapter bash ablation/01_full.sh
```

Useful operational overrides:

```bash
# Temporarily run only training.
SKIP_EVALUATION=1 bash ablation/01_full.sh

# Evaluate the latest model artifact only; do not repeat training. This is the
# recovery path when training succeeded but vLLM/evaluation previously failed.
EVAL_ONLY=1 EVAL_GPU_INDEX=0 bash ablation/01_full.sh

# Evaluate all six latest model artifacts sequentially. Completed evaluations
# are skipped; missing/failed variants are reported and the rest are still tried.
EVAL_GPU_INDEX=0 bash ablation/evaluate_all.bash

# Evaluation automatically chooses the GPU with the most free memory (minimum
# 18,000 MiB by default). Override the selection or threshold if needed.
EVAL_GPU_INDEX=1 EVAL_PORT=8101 bash ablation/01_full.sh
EVAL_MIN_FREE_MIB=16000 bash ablation/01_full.sh

# Re-run even when the fixed evaluation is already complete.
FORCE_EVALUATION=1 EVAL_GPU_INDEX=0 bash ablation/evaluate_variant.bash \
  wo_grpo adapter ablation/outputs/wo_grpo 4

# Resume GRPO training from a checkpoint; evaluation follows after training.
bash ablation/01_full.sh --resume-from-checkpoint /path/to/checkpoint
```

All new datasets, manifests, checkpoints, adapters, logs, and control descriptors are written below `ablation/data/`, `ablation/artifacts/`, and `ablation/outputs/`. The large read-only base model, E5 retrieval indexes, and shared prompt configuration remain referenced from their existing repository paths rather than duplicated.
