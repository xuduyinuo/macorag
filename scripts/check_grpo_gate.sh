#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
: "${TRAIN_METRICS:?Set TRAIN_METRICS to train_metrics.jsonl}"
: "${SFT_EVAL:?Set SFT_EVAL to the fixed SFT evaluation directory}"
: "${RL_EVAL:?Set RL_EVAL to the fixed RL evaluation directory}"
REPORT="${REPORT:-${RL_EVAL}/gate_report.json}"
IFS=':' read -r -a TRAIN_METRIC_PATHS <<< "${TRAIN_METRICS}"
exec "${PYTHON:-python}" -m rl_training.stability_gate \
  --train-metrics "${TRAIN_METRIC_PATHS[@]}" --sft-eval "${SFT_EVAL}" \
  --rl-eval "${RL_EVAL}" --report "${REPORT}"
