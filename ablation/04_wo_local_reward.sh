#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  bash "${HERE}/launch_grpo.bash" wo_local_reward "${HERE}/configs/grpo_wo_local_reward.yml" "$@"
fi
if [[ "${SKIP_EVALUATION:-0}" != "1" ]]; then
  bash "${HERE}/evaluate_variant.bash" wo_local_reward adapter "${HERE}/outputs/wo_local_reward" 4
fi
