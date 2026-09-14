#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  bash "${HERE}/launch_grpo.bash" full "${HERE}/configs/grpo_full.yml" "$@"
fi
if [[ "${SKIP_EVALUATION:-0}" != "1" ]]; then
  bash "${HERE}/evaluate_variant.bash" full adapter "${HERE}/outputs/full" 4
fi
