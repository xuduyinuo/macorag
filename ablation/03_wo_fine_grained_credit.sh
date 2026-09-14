#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  bash "${HERE}/launch_grpo.bash" wo_fine_grained_credit "${HERE}/configs/grpo_wo_fine_grained_credit.yml" "$@"
fi
if [[ "${SKIP_EVALUATION:-0}" != "1" ]]; then
  bash "${HERE}/evaluate_variant.bash" wo_fine_grained_credit adapter "${HERE}/outputs/wo_fine_grained_credit" 4
fi
