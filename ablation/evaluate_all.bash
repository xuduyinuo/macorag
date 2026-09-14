#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Completed evaluations are validated and skipped by evaluate_variant.bash.
# A missing adapter is reported, while the remaining variants are still tried.
failed=0
run_eval() {
  local variant="$1"
  local model_kind="$2"
  local max_rounds="$3"
  printf '\n[ablation-eval-all] variant=%s\n' "${variant}"
  if ! bash "${HERE}/evaluate_variant.bash" \
      "${variant}" "${model_kind}" "${HERE}/outputs/${variant}" "${max_rounds}"; then
    printf '[ablation-eval-all] FAILED variant=%s\n' "${variant}" >&2
    failed=1
  fi
}

run_eval full adapter 4
run_eval wo_multi_round_retrieval adapter 1
run_eval wo_fine_grained_credit adapter 4
run_eval wo_local_reward adapter 4
run_eval wo_grpo adapter 4
run_eval wo_sft_grpo base 4

if [[ "${failed}" == "1" ]]; then
  printf '\n[ablation-eval-all] Some variants could not be evaluated. Check missing model artifacts or errors above.\n' >&2
  exit 1
fi
printf '\n[ablation-eval-all] All six evaluations are complete.\n'
