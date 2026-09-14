#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-/data/conda/envs/macorag/bin/python}" -m rl_training.optimization_validation "$@"
