# SFT BF16 / FlashAttention 2 Task 1 Report

## Status

DONE

Task 1 is implemented and its focused and full SFT test suite pass. An expected Task 2 boundary remains: `flash-attn` was not installed or GPU-smoked in this task, so the newly active `flash_attention_2` configuration will fail fast until Task 2 installs the pinned package.

## Scope and constraints followed

- Read the complete Task 1 plan and the complete design document before editing.
- Modified only the four Task 1 files:
  - `config/train_sft.yml`
  - `src/sft_training/config.py`
  - `src/sft_training/train_sft_lora_macorag.py`
  - `tests/test_sft_training.py`
- Added this explicitly requested report file.
- Preserved the repository's extensive pre-existing uncommitted changes; no reset, checkout, commit, or unrelated cleanup was performed.
- Did not install dependencies, start training, invoke the SFT entrypoint, or run a GPU smoke.

## Changes

### Active configuration

- Set `bf16: true`.
- Set `fp16: false`.
- Set `attn_implementation: "flash_attention_2"`.
- Updated the active-YAML contract test so all three acceleration keys are required active keys.

### Parser contract

- Added runtime default `attn_implementation: "sdpa"`.
- Added `--attn-implementation` with the exact allowed values `eager`, `sdpa`, and `flash_attention_2`.
- YAML values continue to feed the existing defaults-first CLI parser, so explicit CLI values override YAML as before.

### Strict runtime validation

- Added `_validate_acceleration_runtime(args, torch, find_spec=importlib.util.find_spec)`.
- It raises the plan-specified `SystemExit` messages for:
  - simultaneous BF16 and FP16;
  - requested FlashAttention 2 with no importable `flash_attn`;
  - requested FlashAttention 2 with CUDA unavailable;
  - requested BF16 with CUDA unavailable or BF16 unsupported.
- Called validation immediately after training dependencies load and before tokenizer/model loading.
- No fallback to SDPA was added.

### Model and resume contract

- `_model_kwargs` always passes the selected `attn_implementation` to Transformers model loading.
- Added `attn_implementation` to `sft_run_manifest.json`; the existing all-key resume compatibility check therefore rejects an attention-backend mismatch.

### Tests

- Added active config parsing assertions for BF16, FP16, and FlashAttention 2.
- Added a model kwargs propagation test.
- Added focused fail-fast tests for missing `flash_attn`, unavailable CUDA, and unsupported BF16.

## TDD evidence

### RED

Command:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_sft_training.py -k 'attention or acceleration_runtime or active_sft_config'
```

Observed before implementation:

```text
.FFFFF                                                                   [100%]
5 failed, 1 passed, 39 deselected in 0.25s
```

The failures were the intended missing-contract failures:

- active YAML parsed `bf16=False` instead of `True`;
- `_model_kwargs` lacked `attn_implementation`;
- `_validate_acceleration_runtime` did not exist for each of the three prerequisite cases.

### GREEN: focused

Fresh command after implementation and test cleanup:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_sft_training.py -k 'attention or acceleration_runtime or active_sft_config'
```

Observed:

```text
......                                                                   [100%]
6 passed, 39 deselected in 0.15s
```

### GREEN: full Task 1 suite

Fresh command:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_sft_training.py
```

Observed:

```text
.............................................                            [100%]
45 passed in 1.75s
```

## Self-review

- Compared the implementation line by line with Task 1's exact validation rules and error strings.
- Confirmed validation occurs after dependency loading and before `AutoTokenizer.from_pretrained` and model loading.
- Confirmed `attn_implementation` is passed for both ordinary and 4-bit model loading paths because it is inserted before the quantization branch.
- Confirmed the active YAML keys moved from the forbidden low-frequency list to the required active-key list.
- Confirmed the manifest field participates in the existing strict resume compatibility check.
- Ran `git diff --check` on the four Task 1 files successfully before writing this report.

## Concerns and validation boundaries

- `flash-attn` installation and version pinning belong to Task 2 and were explicitly not performed. Until then, a real non-check-only launch with the active YAML should terminate with the new missing-package error.
- GPU BF16/FlashAttention forward/backward proof belongs to Task 3 and was not performed.
- No complete SFT run or throughput claim is made.
- The working tree contained substantial pre-existing changes in all four Task 1 files. This task used narrow patches and did not attempt to attribute, rewrite, or revert those existing edits.
