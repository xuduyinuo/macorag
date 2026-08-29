# Nonfatal RL Protocol Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep GRPO training and recovery checkpoints running when rollout protocol quality is temporarily poor, while preserving accurate parse and final-compliance metrics.

**Architecture:** `rag.protocol_metrics` remains responsible for classifying rollout protocol outcomes and tracking fixed windows. The GRPO entrypoint converts a persistent bad-window condition into a logged warning rather than an exception, and checkpoint scheduling remains based only on step and optimizer-boundary state. True runtime, CUDA, vLLM, and invariant failures retain their existing fatal behavior.

**Tech Stack:** Python 3.9, dataclasses, pytest, PyTorch/PEFT GRPO training, JSONL logs.

## Global Constraints

- Errors beginning with `final_answer_required:` lower `final_compliance_rate` but do not lower `parse_failure_rate`.
- Do not rewrite answers, coerce `can_answer`, or alter reward and loss computation.
- Replace `should_stop` with `should_warn`; no protocol-quality `SystemExit` remains.
- Recovery checkpoints are saved when due at optimizer-safe boundaries regardless of `checkpoint_eligible`.
- Keep `checkpoint_eligible` as an observational metric.
- Do not catch or suppress CUDA, vLLM, numerical, data, or invariant failures.
- Preserve unrelated changes in the dirty working tree.

---

### Task 1: Classify final-round semantic violations separately from parse failures

**Files:**
- Modify: `tests/test_protocol_metrics.py`
- Modify: `src/rag/protocol_metrics.py`

**Interfaces:**
- Consumes: rollout dictionaries containing `parse_errors`, `trajectory`, and final `answer` data.
- Produces: `compute_protocol_metrics(rollouts: list[dict[str, Any]]) -> dict[str, Any]` with corrected `parse_failure_rate` and unchanged result keys.

- [ ] **Step 1: Write the failing semantic-classification test**

Add this test to `tests/test_protocol_metrics.py`:

```python
def test_final_answer_required_is_not_a_parse_failure() -> None:
    metrics = compute_protocol_metrics(
        [
            _rollout(
                error="final_answer_required: answer.can_answer must be true in the final round",
                final_ok=False,
            )
        ]
    )
    assert metrics["parse_failure_rate"] == 0.0
    assert metrics["missing_answer_tag_rate"] == 0.0
    assert metrics["final_compliance_rate"] == 0.0
    assert metrics["checkpoint_eligible"] is False
```

Update `_rollout()` so semantic errors still emit a valid tagged answer:

```python
def _rollout(*, error: str | None = None, final_ok: bool = True) -> dict:
    raw = (
        "plain text"
        if error and not error.startswith("final_answer_required:")
        else f'<answer>{{"can_answer":{str(final_ok).lower()},"answer":null}}</answer>'
    )
    return {
        "parse_errors": [error] if error else [],
        "trajectory": [
            {
                "force_final_answer": True,
                "raw_responses": {"answer_generator": raw},
                "answer": {"can_answer": final_ok, "answer": "x" if final_ok else None},
            }
        ],
    }
```

- [ ] **Step 2: Run the new test and verify the old classification fails**

Run:

```bash
pytest -q tests/test_protocol_metrics.py::test_final_answer_required_is_not_a_parse_failure
```

Expected: FAIL because `parse_failure_rate` is currently `1.0`.

- [ ] **Step 3: Add the minimal structural-error classifier**

Add above `compute_protocol_metrics()` in `src/rag/protocol_metrics.py`:

```python
def _is_parse_error(error: str) -> bool:
    return not error.startswith("final_answer_required:")
```

Replace the unconditional error count with:

```python
        if any(_is_parse_error(error) for error in errors):
            parse_failures += 1
```

Leave missing-tag detection, final compliance, and `checkpoint_eligible` calculation unchanged.

- [ ] **Step 4: Run protocol metric tests**

Run:

```bash
pytest -q tests/test_protocol_metrics.py
```

Expected: the new classification test passes; the existing stop-window test still fails only after Task 2 renames the monitor API.

---

### Task 2: Convert the protocol stop signal into a logged warning

**Files:**
- Modify: `tests/test_protocol_metrics.py`
- Modify: `tests/test_rl_training.py`
- Modify: `src/rag/protocol_metrics.py`
- Modify: `src/rl_training/train_grpo_macorag.py`

**Interfaces:**
- Consumes: `ProtocolWindowMonitor.add(rollouts)` status dictionaries.
- Produces: `should_warn: bool` in monitor status and `_protocol_warning_event(step: int, protocol_status: dict[str, Any]) -> dict[str, Any] | None` for the training log.

- [ ] **Step 1: Replace the monitor stop test with warning-only assertions**

Replace `test_protocol_monitor_stops_after_two_bad_nonoverlapping_windows()` in `tests/test_protocol_metrics.py` with:

```python
def test_protocol_monitor_warns_after_two_bad_nonoverlapping_windows() -> None:
    monitor = ProtocolWindowMonitor(
        window_size=2,
        max_parse_failure_rate=0.02,
        bad_windows_to_warn=2,
    )
    first = monitor.add([_rollout(error="bad"), _rollout()])
    second = monitor.add([_rollout(error="bad"), _rollout()])
    assert first["should_warn"] is False
    assert second["should_warn"] is True
    assert "should_stop" not in first
    assert "should_stop" not in second
```

Add this test to `tests/test_rl_training.py` and include `_protocol_warning_event` in its imports:

```python
def test_protocol_warning_event_is_nonfatal_log_payload() -> None:
    status = {"should_warn": True, "parse_failure_rate": 0.03}
    assert _protocol_warning_event(step=49, protocol_status=status) == {
        "event": "protocol_warning",
        "step": 49,
        "protocol_metrics": status,
    }
    assert _protocol_warning_event(
        step=49,
        protocol_status={"should_warn": False},
    ) is None
```

- [ ] **Step 2: Run both new tests and verify they fail**

Run:

```bash
pytest -q \
  tests/test_protocol_metrics.py::test_protocol_monitor_warns_after_two_bad_nonoverlapping_windows \
  tests/test_rl_training.py::test_protocol_warning_event_is_nonfatal_log_payload
```

Expected: FAIL because the constructor still accepts `bad_windows_to_stop` and `_protocol_warning_event` does not exist.

- [ ] **Step 3: Rename the monitor decision without changing its window behavior**

In `src/rag/protocol_metrics.py`, change the dataclass field and returned key:

```python
    bad_windows_to_warn: int = 2
```

```python
            "should_warn": self.consecutive_bad_windows >= self.bad_windows_to_warn,
```

Do not return `should_stop`.

- [ ] **Step 4: Add a pure warning-event builder and remove the abort path**

Add near the other small helpers in `src/rl_training/train_grpo_macorag.py`:

```python
def _protocol_warning_event(
    *,
    step: int,
    protocol_status: dict[str, Any],
) -> dict[str, Any] | None:
    if not protocol_status.get("should_warn"):
        return None
    return {
        "event": "protocol_warning",
        "step": step,
        "protocol_metrics": protocol_status,
    }
```

Construct the monitor with `bad_windows_to_warn=2`. Replace the `protocol_abort`/`SystemExit` block after `protocol_monitor.add()` with:

```python
                protocol_warning = _protocol_warning_event(
                    step=global_step,
                    protocol_status=latest_protocol_status,
                )
                if protocol_warning is not None and _is_main_process():
                    _append_jsonl(log_path, protocol_warning)
```

This leaves rollout training, optimizer stepping, synchronization, metrics, and persistence on the normal path.

- [ ] **Step 5: Run focused monitor and training-helper tests**

Run:

```bash
pytest -q tests/test_protocol_metrics.py tests/test_rl_training.py -k 'protocol or prompt_contract'
```

Expected: PASS.

---

### Task 3: Make recovery checkpoints independent of protocol quality

**Files:**
- Modify: `tests/test_rl_training.py`
- Modify: `src/rl_training/train_grpo_macorag.py`

**Interfaces:**
- Consumes: `_checkpoint_save_decision(global_step, save_steps, did_optimizer_step, pending) -> tuple[bool, bool]`.
- Produces: scheduled and final deferred calls to `save_full_checkpoint()` that depend only on recovery scheduling and optimizer safety.

- [ ] **Step 1: Add a regression test for forbidden quality gating**

Add imports for `ast`, `inspect`, and `rl_training.train_grpo_macorag as train_grpo_module` to `tests/test_rl_training.py`, then add:

```python
def test_main_does_not_gate_recovery_checkpoints_on_protocol_quality() -> None:
    tree = ast.parse(inspect.getsource(train_grpo_module.main))
    guarded_checkpoint_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if any(
            isinstance(child, ast.Call)
            and getattr(child.func, "id", None) == "save_full_checkpoint"
            for statement in node.body
            for child in ast.walk(statement)
        ):
            guarded_checkpoint_calls.append(ast.unparse(node.test))
    assert all("checkpoint_eligible" not in test for test in guarded_checkpoint_calls)
```

This checks both the scheduled and final deferred checkpoint call sites without depending on whitespace.

- [ ] **Step 2: Run the regression test and verify both current gates are detected**

Run:

```bash
pytest -q tests/test_rl_training.py::test_main_does_not_gate_recovery_checkpoints_on_protocol_quality
```

Expected: FAIL and show conditions containing `checkpoint_eligible`.

- [ ] **Step 3: Remove quality predicates from checkpoint call sites**

In `src/rl_training/train_grpo_macorag.py`, change:

```python
                if should_save:
                    save_full_checkpoint(
```

and:

```python
        if checkpoint_pending:
            save_full_checkpoint(
```

Keep `_checkpoint_save_decision()`, full checkpoint arguments, atomic writes, pruning, and `checkpoint_pending = False` unchanged.

- [ ] **Step 4: Run checkpoint and control-flow regressions**

Run:

```bash
pytest -q \
  tests/test_rl_training.py::test_main_does_not_gate_recovery_checkpoints_on_protocol_quality \
  tests/test_rl_checkpointing.py
```

Expected: PASS.

---

### Task 4: Verify the complete focused change and perform a short GPU smoke run

**Files:**
- Verify: `src/rag/protocol_metrics.py`
- Verify: `src/rl_training/train_grpo_macorag.py`
- Verify: `tests/test_protocol_metrics.py`
- Verify: `tests/test_rl_training.py`
- Verify: `tests/test_rl_checkpointing.py`

**Interfaces:**
- Consumes: the production GRPO launcher and current canonical configuration.
- Produces: test evidence plus a fresh one-step run directory containing final adapter metadata.

- [ ] **Step 1: Run focused CPU regression suites**

Run:

```bash
pytest -q \
  tests/test_protocol_metrics.py \
  tests/test_prompt_budget.py \
  tests/test_rl_checkpointing.py \
  tests/test_rl_training.py -k 'not active_config_exposes_manual_sft_adapter_path'
```

Expected: PASS. The excluded assertion is an existing active-config/manual-adapter mismatch outside this fix.

- [ ] **Step 2: Compile changed Python modules**

Run:

```bash
python -m py_compile \
  src/rag/protocol_metrics.py \
  src/rl_training/train_grpo_macorag.py \
  tests/test_protocol_metrics.py \
  tests/test_rl_training.py
```

Expected: exit status 0 with no output.

- [ ] **Step 3: Check patch hygiene and forbidden terminal names**

Run:

```bash
git diff --check -- \
  src/rag/protocol_metrics.py \
  src/rl_training/train_grpo_macorag.py \
  tests/test_protocol_metrics.py \
  tests/test_rl_training.py
rg -n 'should_stop|bad_windows_to_stop|protocol_abort' \
  src/rag/protocol_metrics.py \
  src/rl_training/train_grpo_macorag.py \
  tests/test_protocol_metrics.py \
  tests/test_rl_training.py
```

Expected: both commands produce no findings and exit successfully for the diff check; `rg` exits 1 because no forbidden names remain.

- [ ] **Step 4: Run a one-step GPU smoke training in a new output root**

Run:

```bash
/home/being/anaconda3/bin/conda run --no-capture-output \
  -p /data/conda/envs/macorag \
  bash scripts/run_train_grpo.sh \
  --max-steps 1 \
  --output-root outputs/grpo_qwen2.5-7b-v2-smoke-nonfatal-protocol
```

Expected: exit status 0, one completed optimization step, and a timestamped directory under `outputs/grpo_qwen2.5-7b-v2-smoke-nonfatal-protocol/` containing `adapter/prompt_contract.json` and `train_meta.json`.

- [ ] **Step 5: Inspect the smoke artifacts**

Run:

```bash
smoke_dir=$(find outputs/grpo_qwen2.5-7b-v2-smoke-nonfatal-protocol \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
test -f "$smoke_dir/adapter/prompt_contract.json"
test -f "$smoke_dir/train_meta.json"
tail -n 1 "$smoke_dir/train_metrics.jsonl"
```

Expected: both file checks succeed and the final JSONL row reports `global_step` equal to `1`.

---

## Completion Boundary

The change is complete only when the focused tests, compilation, diff hygiene, forbidden-name scan, and one-step GPU smoke all pass. Report the exact smoke output directory. Do not claim that the historical 49-step process is resumable; it produced no full-state checkpoint.
