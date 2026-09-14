#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  bash "${HERE}/launch_grpo.bash" wo_multi_round_retrieval "${HERE}/configs/grpo_wo_multi_round_retrieval.yml" "$@"
fi
if [[ "${SKIP_EVALUATION:-0}" != "1" ]]; then
  bash "${HERE}/evaluate_variant.bash" wo_multi_round_retrieval adapter "${HERE}/outputs/wo_multi_round_retrieval" 1
fi
