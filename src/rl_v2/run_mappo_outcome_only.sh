#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${SCRIPT_DIR}/train_mappo_outcome_only.yml"
exec bash "${SCRIPT_DIR}/run_mappo.sh" "$@"
