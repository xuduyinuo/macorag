# Manual adapter paths and local evaluation metrics design

## Goal

Prevent training and evaluation from silently selecting the most recently
completed adapter, because recency does not imply best validation quality.
Remove the external LLM-as-judge metric so evaluation uses only deterministic
local answer metrics.

## Adapter path contract

Adapter paths are explicit YAML configuration inputs. Configuration files expose
`sft_adapter_path` and `adapter_path` with clear placeholder values that the
operator replaces manually. They must not contain `AUTO_FROM_SFT_V2`,
`AUTO_FROM_GRPO_V2`, or any directory-scanning fallback.

GRPO training and its LoRA vLLM service both read `sft_adapter_path` from the
selected training YAML. The two launchers validate that the configured directory
contains `adapter_config.json` and `prompt_contract.json`, print the effective
path during a dry run, and fail before starting GPU work when the field is missing
or invalid. Both launchers use the same YAML field so the trainer and generation
service cannot accidentally select different SFT adapters.

The evaluation vLLM service similarly reads `adapter_path` from its YAML. It
validates the adapter artifacts, exposes the selected path in dry-run output,
and fails before model startup when the field is missing or invalid. The
repository launcher passes the validated configured value to the Python service.

All directory-scanning logic that chooses the lexicographically latest completed
SFT or GRPO run is removed. No fallback path is used.

## Evaluation metric contract

The evaluation pipeline removes the Bailian/Qwen judge integration and the
`llm_accuracy` result. It no longer reads a judge API key, submits answers to an
external model, or accepts `skip_judge` and `judge_*` configuration fields.

After predictions are generated, evaluation computes deterministic local
metrics only:

- exact match (`exact_match`)
- containment accuracy (`contain_accuracy`)
- token F1 (`f1`)
- sample count (`num_samples`)

The existing metric implementations and prediction files remain unchanged.
`evaluation_results.json` no longer contains `llm_accuracy` or judge metadata.
Historical result files are not rewritten.

## Error handling

Missing adapter configuration fields produce a concise error naming the required
YAML key. Invalid paths report which required artifact is absent.
These checks run in normal launches before any CUDA process starts. Dry-run mode
also requires a valid configured path but does not require starting a model or writing
outputs.

Unknown obsolete judge fields in an evaluation YAML are rejected by the existing
strict configuration validation. This makes stale configurations visible rather
than silently ignoring them.

## Compatibility and scope

Base-model-only evaluation is intentionally not supported by
`scripts/eval_vllm_server.sh` after this change because the agreed workflow
requires a manually selected adapter. Direct use of the lower-level Python
vLLM server remains outside this launcher contract.

Training algorithms, checkpoint resume behavior, retrieval, prompting, and the
definitions of EM, containment accuracy, and F1 are not changed.

## Tests and verification

Launcher tests first demonstrate failure when `sft_adapter_path` or
`adapter_path` is absent from its YAML, success in dry-run mode with a valid
explicitly configured adapter, and absence of automatic discovery markers.

Evaluation tests demonstrate that configuration no longer exposes judge fields,
no judge client is constructed, local metrics are written without
`llm_accuracy`, and stale judge YAML keys fail validation. Verification includes
the focused launcher and evaluation tests, Bash syntax checks for modified
launchers, and the relevant existing test modules.
