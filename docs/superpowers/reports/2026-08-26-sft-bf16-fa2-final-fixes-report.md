# SFT BF16 and FlashAttention 2 Final Fixes Report

## Status

PASS. All four final-review findings were implemented with focused regression coverage. The requested test suites, shell syntax check, and repository whitespace check pass. No package was downloaded, rebuilt, reinstalled, or removed; no 7B model or training process was started; no commit was created.

## Findings resolved

### 1. Strict FlashAttention import and ABI failure reporting

`_validate_acceleration_runtime` now performs a real `import flash_attn`, retrieves and verifies the callable `flash_attn_func`, and catches `ImportError`, `OSError`, and malformed-module `AttributeError`. Startup raises a clear `SystemExit` that includes the original exception class and message, so an extension ABI error such as `undefined symbol` can no longer pass the previous `find_spec`-only check.

The existing missing-package check remains in place, and no SDPA fallback was added.

### 2. Reproducible source-build installer

Added executable `scripts/install_flash_attn.sh`. Its contract is fixed to:

- macorag Python, default `/data/conda/envs/macorag/bin/python`;
- exact Torch `2.6.0+cu124` and Torch CXX11 ABI `False` preflight;
- official PyPI `flash_attn-2.8.3.post1.tar.gz` URL;
- SHA256 `55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348`;
- `CUDA_HOME=/usr/local/cuda-12.9` by default;
- `FLASH_ATTENTION_FORCE_BUILD=TRUE`;
- `FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE`;
- `FLASH_ATTN_CUDA_ARCHS=80`;
- bounded `MAX_JOBS` and `NVCC_THREADS` defaults;
- pip `--no-build-isolation --no-deps --no-cache-dir`;
- mandatory post-install import/version/ABI checks and a real CUDA BF16 `flash_attn_func` forward/backward smoke.

The official PyPI JSON for version `2.8.3.post1` was inspected to use the exact uploaded sdist URL and matching supplied SHA256. The script does not refer to `/data/conda/pip-cache` or another local wheel/cache. `requirements.txt` and `environment.yml` retain the exact pin and point ABI-sensitive rebuilds to this script.

Per task constraint, the installer itself was not executed in this task. Only its syntax and static contract were tested.

### 3. BF16 and FP16 conflict coverage

Added an independent test proving simultaneous `bf16=True` and `fp16=True` terminates with the existing explicit conflict message before acceleration setup.

### 4. Multi-GPU BF16 device selection

When BF16 is enabled with `WORLD_SIZE > 1` and CUDA is available, runtime validation now calls `torch.cuda.set_device(LOCAL_RANK)` before `torch.cuda.is_bf16_supported()`. The test records call ordering and proves local rank 1 is selected first.

## TDD evidence

### RED

Focused command before implementation:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_sft_training.py -k 'abi_import_failure or bf16_and_fp16 or selects_local_rank' \
  tests/test_retraining_launchers.py -k 'flash_attention_installer or abi_import_failure or bf16_and_fp16 or selects_local_rank'
```

Observed:

```text
3 failed, 1 passed, 49 deselected
```

The intended failures were:

- `_validate_acceleration_runtime` did not accept or perform a real import;
- BF16 capability was queried without `set_device(LOCAL_RANK)`;
- `scripts/install_flash_attn.sh` did not exist.

The BF16/FP16 conflict test already passed against the existing behavior, proving that branch independently rather than requiring a production change.

### GREEN

Focused acceleration and installer-contract tests:

```text
7 passed, 46 deselected in 1.53s
```

Final requested regression:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_sft_training.py tests/test_retraining_launchers.py
```

Result:

```text
53 passed in 9.22s
```

Additional final checks:

```bash
bash -n scripts/install_flash_attn.sh
git diff --check
```

Both exited 0 with no output.

## Concerns and boundaries

- The new installer contract is reproducible from the official sdist, but the source build can be lengthy and was deliberately not rerun because the current environment has already passed the separate real GPU kernel smoke.
- The installer intentionally fails when CUDA is unavailable; it does not silently skip the required GPU forward/backward validation.
- The installer smoke selects visible CUDA device 0. Operators can constrain the physical GPU using `CUDA_VISIBLE_DEVICES` before invoking the script.
- The runtime validation tests cover import/ABI error propagation and local-rank ordering without loading the 7B model. They do not establish end-to-end distributed training throughput or stability.
- The worktree contains extensive pre-existing uncommitted changes. This task used narrow patches and did not reset, clean, commit, or alter unrelated files.
