#!/usr/bin/env bash
set -euo pipefail

# Reproducibly build FlashAttention against the already-installed macorag Torch.
# This intentionally does not resolve or upgrade dependencies.
MACORAG_PYTHON="${MACORAG_PYTHON:-/data/conda/envs/macorag/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
MAX_JOBS="${MAX_JOBS:-4}"
NVCC_THREADS="${NVCC_THREADS:-4}"
FLASH_ATTENTION_FORCE_BUILD=TRUE
FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE
FLASH_ATTN_CUDA_ARCHS=80

FLASH_ATTN_VERSION="2.8.3.post1"
FLASH_ATTN_URL="https://files.pythonhosted.org/packages/01/7a/92a46e7cd6bbb4d7b2855a457c3b855df54a97af5656d98fc92e58e61065/flash_attn-2.8.3.post1.tar.gz"
FLASH_ATTN_SHA256="55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348"

if [[ ! -x "${MACORAG_PYTHON}" ]]; then
  echo "macorag Python not found or not executable: ${MACORAG_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${CUDA_HOME}" ]]; then
  echo "CUDA toolkit not found: ${CUDA_HOME}" >&2
  exit 1
fi

"${MACORAG_PYTHON}" - <<'PY'
from pathlib import Path
import sys
import torch

assert Path(sys.prefix).name == "macorag", f"expected macorag environment, got {sys.prefix}"
assert torch.__version__ == '2.6.0+cu124', torch.__version__
assert torch._C._GLIBCXX_USE_CXX11_ABI is False
print("preflight PASS", sys.executable, torch.__version__, "cxx11_abi", False)
PY

BUILD_DIR="$(mktemp -d -t macorag-flash-attn-XXXXXX)"
trap 'rm -rf "${BUILD_DIR}"' EXIT
SDIST="${BUILD_DIR}/flash_attn-${FLASH_ATTN_VERSION}.tar.gz"

curl --fail --location --retry 3 --output "${SDIST}" "${FLASH_ATTN_URL}"
printf '%s  %s\n' "${FLASH_ATTN_SHA256}" "${SDIST}" | sha256sum --check --strict

export CUDA_HOME MAX_JOBS NVCC_THREADS
export FLASH_ATTENTION_FORCE_BUILD FLASH_ATTENTION_FORCE_CXX11_ABI FLASH_ATTN_CUDA_ARCHS
"${MACORAG_PYTHON}" -m pip install \
  --force-reinstall \
  --no-build-isolation \
  --no-deps \
  --no-cache-dir \
  "${SDIST}"

# CUDA is mandatory here: a CPU-only success would not validate the installed kernel.
"${MACORAG_PYTHON}" - <<'PY'
import flash_attn
from flash_attn import flash_attn_func
import torch

assert flash_attn.__version__ == "2.8.3.post1", flash_attn.__version__
assert torch.__version__ == '2.6.0+cu124', torch.__version__
assert torch._C._GLIBCXX_USE_CXX11_ABI is False
assert torch.cuda.is_available(), "CUDA is required for FlashAttention verification"
torch.cuda.set_device(0)
assert torch.cuda.is_bf16_supported(), "selected CUDA device does not support BF16"

q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
out = flash_attn_func(q, k, v, causal=True)
loss = out.float().square().mean()
loss.backward()
assert out.shape == q.shape and torch.isfinite(out).all()
for tensor in (q, k, v):
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
print("flash-attn GPU smoke PASS", flash_attn.__version__, torch.cuda.get_device_name(0))
PY
