# SFT BF16 and FlashAttention 2 - Task 2 Report

## Status

**DONE**

Task 2 is complete. `flash-attn==2.8.3.post1` is pinned in both dependency manifests and is importable in `/data/conda/envs/macorag`. PyTorch, Transformers, and the CUDA toolchain were not upgraded or replaced. No training job was started, stopped, or resumed by this task.

## Authorized file changes

- Added `flash-attn==2.8.3.post1` to `requirements.txt` alongside the training/runtime dependencies.
- Added `flash-attn==2.8.3.post1` to the pip training stack in `environment.yml`.
- Preserved the pre-existing uncommitted `faiss-cpu==1.9.0.post1` additions and all other unrelated working-tree changes.
- No commit was created.

Final pin checks:

```text
environment.yml:40:      - flash-attn==2.8.3.post1
requirements.txt:14:flash-attn==2.8.3.post1
```

`git diff --check -- requirements.txt environment.yml` exited 0.

## Installation chronology and diagnosis

### 1. Planned install and initial ABI failure

The first installation used the plan-prescribed dependency isolation controls:

```bash
CUDA_HOME=/usr/local/cuda-12.9 MAX_JOBS=4 \
  /data/conda/envs/macorag/bin/python -m pip install \
  --no-build-isolation --no-deps flash-attn==2.8.3.post1
```

pip downloaded the 2.8.3.post1 sdist, then the package's custom `CachedWheelsCommand` selected a prebuilt wheel. Installation itself exited 0, but import failed with:

```text
ImportError: .../flash_attn_2_cuda.cpython-39-x86_64-linux-gnu.so:
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

Root-cause evidence:

- Existing PyTorch reported `torch._C._GLIBCXX_USE_CXX11_ABI == False`.
- PyTorch `libc10.so` exported the old-ABI `c10::Error` symbol ending in `ESs`.
- The initially installed extension referenced the new-ABI `std::__cxx11::basic_string` symbol.
- Therefore the installed extension and the fixed existing Torch ABI were incompatible; Torch was not changed to accommodate the extension.

### 2. pip cache bypass issue

Setting `FLASH_ATTENTION_FORCE_BUILD=TRUE` alone did not rebuild the extension because pip first reused the newly cached wheel:

```text
Using cached flash_attn-2.8.3.post1-cp39-cp39-linux_x86_64.whl
```

This confirmed that `--no-cache-dir` was also necessary to reach the source package's force-build path. No dependency was upgraded or re-resolved during this attempt.

### 3. Source-build attempts and intentional termination

A cache-disabled, force-source build was started with the fixed plan settings. The package defaulted to `FLASH_ATTN_CUDA_ARCHS=80;90;100;120`, so that broad architecture build was intentionally terminated by the main agent after the unnecessary architecture fan-out was identified.

The main agent then requested this exact Ada-compatible sm80-limited build:

```bash
CUDA_HOME=/usr/local/cuda-12.9 \
MAX_JOBS=4 \
NVCC_THREADS=4 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_CUDA_ARCHS=80 \
  /data/conda/envs/macorag/bin/python -m pip install \
  --force-reinstall --no-cache-dir --no-build-isolation --no-deps \
  flash-attn==2.8.3.post1
```

No `FLASH_ATTENTION_FORCE_CXX11_ABI` value was set. Live compiler evidence showed all required properties:

```text
-gencode arch=compute_80,code=sm_80
--threads 4
-D_GLIBCXX_USE_CXX11_ABI=0
```

The build advanced from 20/72 to 25/72 CUDA object files, with `cicc` consuming CPU and source-file PIDs changing, so it was not stalled. The main agent intentionally terminated this long build after locating an already cached, independently symbol-checked ABI=False wheel. The terminated pip process exited 143; this was an external SIGTERM, not a compiler failure. The process group was cleared, and the final check found no residual flash-attn pip/nvcc/ninja build processes.

### 4. Final exact-wheel installation

Two same-version cached wheels were inspected at the binary-symbol level:

- The wheel under `.../wheels/5c/...`, generated during this task, referenced the incompatible `__cxx11` symbol and was rejected.
- The wheel under `.../wheels/b4/6b/74/...` referenced the old-ABI `c10::Error(..., std::string)` symbol ending in `ESs`, matching Torch ABI=False.

The compatible wheel was installed directly without dependency resolution:

```bash
/data/conda/envs/macorag/bin/python -m pip install \
  --force-reinstall --no-deps \
  /data/conda/pip-cache/wheels/b4/6b/74/f859cb1233d634acda327e933475cbd1909c350ec659f24f07/flash_attn-2.8.3.post1-cp39-cp39-linux_x86_64.whl
```

Result: exit 0, `Successfully installed flash-attn-2.8.3.post1`.

Wheel SHA256:

```text
83abb9f78260ebdc5b059623ccbae4928394acd805f0ffb3e5d9bf9045751f00
```

The installed extension's dynamic symbol table contains:

```text
U _ZN3c105ErrorC2ENS_14SourceLocationESs
```

This matches the existing Torch ABI=False symbol family.

## Final verification

### Import and immutable version check

Command:

```bash
/data/conda/envs/macorag/bin/python -c \
  "import flash_attn, torch, transformers; print(flash_attn.__version__, torch.__version__, transformers.__version__); print('torch_cxx11_abi', torch._C._GLIBCXX_USE_CXX11_ABI)"
```

Result: exit 0.

```text
2.8.3.post1 2.6.0+cu124 4.57.6
torch_cxx11_abi False
```

This proves `flash_attn` imports successfully and that the existing PyTorch and Transformers versions remain unchanged.

### Dependency consistency

Command:

```bash
/data/conda/envs/macorag/bin/python -m pip check
```

Result: exit 0.

```text
No broken requirements found.
```

pip also emitted a non-fatal warning that `/data/conda/pip-cache` is not owned or writable by the current user. This did not affect the check result or the direct wheel installation.

## Validation boundary

This task verified package pinning, binary importability, fixed Torch/Transformers versions, ABI compatibility, and dependency consistency. It did not run the BF16 FlashAttention forward/backward GPU kernel smoke or repository regression suite; those are Task 3. It made no end-to-end SFT throughput claim and did not start training.
