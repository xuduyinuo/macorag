# GRPO Prompt Contract JSON Import Fix

## Problem

`src/rl_training/train_grpo_macorag.py::_validate_sft_prompt_contract()` reads
`prompt_contract.json` with `json.loads()`, but the module does not import the
Python standard-library `json` module. A valid SFT adapter therefore fails
during startup validation with `NameError` before model loading begins.

## Design

Add the missing top-level `import json` to
`src/rl_training/train_grpo_macorag.py`. Do not change prompt-contract
semantics, launcher behavior, distributed setup, model loading, or training.

Add one focused regression test to `tests/test_rl_training.py`. The test will
create a valid temporary prompt contract metadata file, call
`_validate_sft_prompt_contract()`, and assert that the parsed metadata is
returned. Before the production import is added, this test must fail with the
observed `NameError`; after the import is added, it must pass.

## Verification

Run the focused regression test first, then the relevant RL training test file.
Also compile the modified module and run the configured GRPO `--check-only`
preflight when the local adapter and runtime prerequisites allow it. Report any
later independent startup blocker separately from this fixed exception.

## Scope

The change is limited to one standard-library import and one regression test.
Existing unrelated working-tree changes are preserved.
