#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v python >/dev/null 2>&1; then
  echo "No python found in current environment. Activate macorag first, e.g.: conda activate macorag" >&2
  exit 1
fi

if ! python -m pip --version >/dev/null 2>&1; then
  echo "pip is not available in current environment." >&2
  exit 1
fi

# Require Python version compatible with pyproject.toml (>=3.9).
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "当前环境 Python 版本为 ${PYTHON_VERSION}，项目需要 >= 3.9（pyproject.toml: requires-python). 请先切换/新建 >=3.9 环境后重试。" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
if [ ! -f "pyproject.toml" ]; then
  echo "未在仓库根目录找到 pyproject.toml: ${PROJECT_ROOT}" >&2
  exit 1
fi

export PIP_NO_BUILD_ISOLATION=1

python -m pip install --no-build-isolation --no-deps -e .
python -m pip install --no-build-isolation -r LinearRAG/requirements.txt

# Install spaCy language model for retrieval env entities.
python -m spacy download en_core_web_sm
