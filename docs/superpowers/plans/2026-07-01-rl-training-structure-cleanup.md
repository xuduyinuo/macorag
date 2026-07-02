# RL Training Structure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `src/rl_training` readability and RL YAML organization without changing runnable behavior.

**Architecture:** Keep `src/rl_training/train_grpo_macorag.py` as the public entrypoint and orchestration layer. Move GPU/vLLM validation and training logging helpers into focused modules, then re-export the existing helper names from the entrypoint so current tests and scripts keep working.

**Tech Stack:** Python, argparse, PyYAML, pytest, torch/PEFT/vLLM integration contracts already present in this repo.

## Global Constraints

- Do not change current runnable functionality or execution logic.
- Preserve `src/rl_training/train_grpo_macorag.py`, `config/train_grpo.yml`, and `scripts/run_train_grpo.sh` contracts.
- Keep runtime parameters YAML-driven under `config/`.
- Add Chinese comments in code/config where they clarify module boundaries and RL parameters.
- Avoid editing unrelated dirty files in the working tree.

---

### Task 1: Extract Runtime and Logging Helpers

**Files:**
- Create: `src/rl_training/runtime.py`
- Create: `src/rl_training/logging_utils.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Consumes: existing helper signatures from `train_grpo_macorag.py`.
- Produces: `_parse_gpu_indices`, `_validate_vllm_gpu_placement`, `_validate_local_vllm_server_model`, `_write_json`, `_append_jsonl`, `_write_train_event` remain importable from `train_grpo_macorag.py`.

- [ ] **Step 1: Write failing structure tests**

```python
def test_rl_runtime_helpers_are_extracted_and_reexported() -> None:
    import rl_training.runtime as runtime
    import rl_training.train_grpo_macorag as entrypoint

    assert entrypoint._parse_gpu_indices is runtime.parse_gpu_indices
    assert entrypoint._validate_vllm_gpu_placement is runtime.validate_vllm_gpu_placement
    assert entrypoint._validate_local_vllm_server_model is runtime.validate_local_vllm_server_model


def test_rl_logging_helpers_are_extracted_and_reexported() -> None:
    import rl_training.logging_utils as logging_utils
    import rl_training.train_grpo_macorag as entrypoint

    assert entrypoint._write_json is logging_utils.write_json
    assert entrypoint._append_jsonl is logging_utils.append_jsonl
    assert entrypoint._write_train_event is logging_utils.write_train_event
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py::test_rl_runtime_helpers_are_extracted_and_reexported tests/test_rl_training.py::test_rl_logging_helpers_are_extracted_and_reexported`

Expected: FAIL because `rl_training.runtime` and `rl_training.logging_utils` do not exist.

- [ ] **Step 3: Move helpers with compatibility aliases**

Move GPU parsing, vLLM process/model validation, and JSON/JSONL event helpers into the new modules. Import them back into `train_grpo_macorag.py` with aliases matching the old private names.

- [ ] **Step 4: Run targeted tests**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py::test_rl_runtime_helpers_are_extracted_and_reexported tests/test_rl_training.py::test_rl_logging_helpers_are_extracted_and_reexported tests/test_rl_training.py::test_parse_gpu_indices_normalizes_comma_lists tests/test_rl_training.py::test_write_train_event_records_sample_progress`

Expected: PASS.

### Task 2: Clean Config Parser and YAML Organization

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `config/train_grpo.yml`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Consumes: existing `parse_args(argv: list[str] | None) -> argparse.Namespace`.
- Produces: same accepted YAML keys and CLI flags, grouped internally and documented in YAML.

- [ ] **Step 1: Write failing config grouping test**

```python
def test_train_grpo_yaml_has_documented_sections() -> None:
    text = Path("config/train_grpo.yml").read_text(encoding="utf-8")

    for heading in [
        "# 基础路径",
        "# rollout 与采样",
        "# GRPO 优化",
        "# vLLM 生成与权重同步",
        "# 检索环境",
        "# 日志与检查点",
        "# 运行环境",
    ]:
        assert heading in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py::test_train_grpo_yaml_has_documented_sections`

Expected: FAIL because the current YAML has no Chinese section headings.

- [ ] **Step 3: Group parser declarations and YAML**

Group `DEFAULT_ARG_VALUES` and parser additions with Chinese comments. Reorder YAML into the same sections without renaming or deleting consumed keys.

- [ ] **Step 4: Run parser and YAML tests**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py::test_parse_args_loads_train_grpo_yaml tests/test_rl_training.py::test_parse_args_loads_vllm_generation_config tests/test_rl_training.py::test_train_grpo_yaml_has_documented_sections`

Expected: PASS.

### Task 3: Final Regression

**Files:**
- Verify only.

**Interfaces:**
- Existing RL training and vLLM LoRA tests remain valid.

- [ ] **Step 1: Run full RL test file**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py`

Expected: all tests pass.

- [ ] **Step 2: Run vLLM LoRA server tests**

Run: `PYTHONPATH=src pytest -q tests/test_vllm_lora_server.py`

Expected: all tests pass.

- [ ] **Step 3: Run shell syntax checks**

Run: `bash -n scripts/run_train_grpo.sh scripts/run_grpo_vllm_server.sh scripts/run_grpo_vllm_lora_server.sh`

Expected: exit code 0.
