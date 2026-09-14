#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${REPO_ROOT}"
if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  "${PYTHON_BIN}" -m ablation.prepare_data
  if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
    printf '[ablation-sft] variant=wo_grpo config=%s output_root=%s\n' \
      "${HERE}/configs/sft_1000.yml" "${HERE}/outputs/wo_grpo"
  else
    CONFIG_PATH="${HERE}/configs/sft_1000.yml" bash "${REPO_ROOT}/scripts/run_train_sft.sh" "$@"
    "${PYTHON_BIN}" -m ablation.ablation_cli record-adapter \
      --root "${HERE}/outputs/wo_grpo" \
      --pointer "${HERE}/artifacts/sft_shared_adapter.path"
  fi
fi
if [[ "${SKIP_EVALUATION:-0}" != "1" ]]; then
  bash "${HERE}/evaluate_variant.bash" wo_grpo adapter "${HERE}/outputs/wo_grpo" 4
fi
