# GRPO JSON Import Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent GRPO startup prompt-contract validation from raising `NameError` when it reads valid JSON metadata.

**Architecture:** Keep `_validate_sft_prompt_contract()` and its validation semantics unchanged. Add the missing standard-library import and protect the successful JSON-reading path with one focused function-level regression test.

**Tech Stack:** Python 3.9, standard-library `json`, pytest

## Global Constraints

- Do not change prompt-contract semantics, launcher behavior, distributed setup, model loading, or training.
- Preserve all unrelated working-tree changes.
- Limit production code to one top-level standard-library import.

---

### Task 1: Reproduce and fix prompt-contract JSON loading

**Files:**
- Modify: `tests/test_rl_training.py`
- Modify: `src/rl_training/train_grpo_macorag.py:4-12`

**Interfaces:**
- Consumes: `_validate_sft_prompt_contract(args: Any) -> dict[str, Any]` and `load_prompt_contract(path)`.
- Produces: A validated metadata dictionary without `NameError` when `prompt_contract.json` contains valid JSON.

- [ ] **Step 1: Write the failing regression test**

Add `_validate_sft_prompt_contract` to the existing imports from
`rl_training.train_grpo_macorag`, then add:

```python
def test_validate_sft_prompt_contract_reads_valid_json(tmp_path: Path) -> None:
    from prompt_config import load_prompt_contract

    contract = load_prompt_contract("config/prompts.yml")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    metadata = {
        "prompt_contract_version": contract.version,
        "prompt_contract_fingerprint": contract.fingerprint,
        "max_rounds": 4,
        "retrieval_top_k": 5,
    }
    (adapter / "prompt_contract.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    args = Namespace(
        prompt_config_path="config/prompts.yml",
        sft_adapter_path=str(adapter),
        require_sft_prompt_contract=True,
        max_rounds=4,
        retrieval_top_k=5,
    )

    assert _validate_sft_prompt_contract(args) == metadata
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/test_rl_training.py::test_validate_sft_prompt_contract_reads_valid_json
```

Expected: FAIL at `json.loads(...)` in `train_grpo_macorag.py` with
`NameError: name 'json' is not defined`.

- [ ] **Step 3: Add the minimal production fix**

Add this import beside the existing standard-library imports in
`src/rl_training/train_grpo_macorag.py`:

```python
import json
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
pytest -q tests/test_rl_training.py::test_validate_sft_prompt_contract_reads_valid_json
```

Expected: `1 passed`.

- [ ] **Step 5: Run relevant regression and static checks**

Run:

```bash
pytest -q tests/test_rl_training.py
python -m py_compile src/rl_training/train_grpo_macorag.py
git diff --check -- src/rl_training/train_grpo_macorag.py tests/test_rl_training.py
```

Expected: the RL training test file passes, compilation exits with status 0,
and `git diff --check` emits no errors.

- [ ] **Step 6: Run GRPO startup preflight**

Run:

```bash
PYTHONPATH=src python -m rl_training.train_grpo_macorag \
  --config config/train_grpo.yml --check-only --disable-tqdm
```

Expected: the command no longer raises the JSON `NameError`. If a later
independent prerequisite fails, capture and report that exact blocker.

- [ ] **Step 7: Commit the bug fix**

```bash
git add src/rl_training/train_grpo_macorag.py tests/test_rl_training.py
git commit -m "fix: import json for GRPO prompt contract validation"
```
