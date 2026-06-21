#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "No python found in current environment. Activate macorag first, e.g.: conda activate macorag" >&2
  exit 1
fi

if ! python -m pip --version >/dev/null 2>&1; then
  echo "pip is not available in current environment." >&2
  exit 1
fi

export PIP_NO_BUILD_ISOLATION=1

python -m pip install --no-build-isolation --no-deps -e .
python -m pip install --no-build-isolation -r LinearRAG/requirements.txt
