# GRPO Resume and vLLM Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the interrupted GRPO policy from checkpoint-3600 at the original shuffled-data position and make transient vLLM generation disconnects recoverable.

**Architecture:** Add explicit resume fields to the GRPO CLI, resolve and validate a deterministic resume position before model loading, load the policy and reference adapters from distinct paths, and slice the reconstructed epoch order. Add narrowly scoped retry logic around only the `/generate/` POST, rebuilding the Requests session between attempts. A new wrapper script supplies the checkpoint-3600 arguments while retaining the existing launcher and YAML.

**Tech Stack:** Python 3.9, PyTorch, PEFT, Requests, argparse/PyYAML, pytest, Bash.

## Global Constraints

- Preserve the user's existing uncommitted `config/train_grpo.yml` changes.
- Keep the frozen reference adapter on the original `sft_adapter_path`.
- Treat checkpoint-3600 as policy-only and initialize a fresh AdamW optimizer.
- Skip exactly 3,600 samples from deterministic epoch 1 and resume at original step 3,601.
- Retry only Requests connection and timeout failures from `/generate/`; never retry HTTP/model/payload/LoRA-sync failures.
- Create a new timestamped run directory and never overwrite the interrupted run.

---

### Task 1: vLLM generation retry

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Modify: `src/rl_training/vllm_client.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Consumes: the existing `VLLMGenerationClient.generate_batch()` request payload and TRL backend session.
- Produces: `vllm_generate_max_attempts: int`, `vllm_generate_retry_backoff_seconds: float`, and generation-only retry behavior.

- [ ] **Step 1: Write failing parser and retry tests**

Add tests that parse both new YAML values, then construct a client whose first session raises `requests.ConnectionError` and whose replacement session returns a valid payload. Assert two posts, one session close, one replacement, and one recorded sleep. Add separate tests proving exhaustion re-raises and HTTP 500 is not retried.

```python
client = VLLMGenerationClient(
    host="127.0.0.1",
    port=8000,
    timeout_seconds=5,
    max_generate_attempts=3,
    retry_backoff_seconds=1.0,
    backend=backend,
    sleep_fn=sleeps.append,
    session_factory=lambda: replacement,
)
outputs = client.generate_batch(["prompt"], max_tokens=8, temperature=0.7, top_p=0.9, top_k=5)
assert outputs[0].completion_ids == [10]
assert sleeps == [1.0]
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
pytest -q tests/test_rl_training.py -k 'vllm_generation_retry or loads_vllm_generation_config'
```

Expected: failures because constructor/config fields and retry behavior do not exist.

- [ ] **Step 3: Implement minimal generation-only retry**

Add defaults and parser arguments in `config.py`, pass them from `_build_policy()`, and extend the client constructor with injected sleep/session creation. Wrap only the session POST:

```python
for attempt in range(self.max_generate_attempts):
    try:
        response = session.post(f"{base_url}/generate/", json=request_payload)
        break
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        if attempt + 1 >= self.max_generate_attempts:
            raise
        close = getattr(session, "close", None)
        if callable(close):
            close()
        session = self._session_factory()
        self.backend.session = session
        self._sleep_fn(self.retry_backoff_seconds * (2**attempt))
```

Validate attempts are at least one and backoff is nonnegative. Preserve existing response status and JSON validation unchanged.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the same pytest command and expect all selected tests to pass.

---

### Task 2: deterministic checkpoint resume

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Consumes: `resume_from_checkpoint`, `resume_epoch`, `resume_samples_consumed`, `resume_global_step`, deterministic `epoch_sample_order()`, and the original SFT adapter path.
- Produces: validated `ResumeState`, distinct policy/reference loading, skipped rank-local samples, continued global step, and `resume_meta.json`.

- [ ] **Step 1: Write failing resume tests**

Add parser tests, `ResumeState` validation tests, a policy/reference load-path test using fake PEFT dependencies, and a deterministic slicing test. Assert that checkpoint-3600 is used only for tokenizer/policy while `reference` loads from the original SFT adapter.

```python
state = _resolve_resume_state(
    Namespace(
        resume_from_checkpoint=str(checkpoint),
        resume_epoch=1,
        resume_samples_consumed=3600,
        resume_global_step=3600,
    ),
    rank_epoch_size=6000,
    total_epochs=1,
)
assert state.samples_consumed == 3600
assert state.global_step == 3600
assert state.optimizer_state_restored is False
```

Verify that the reconstructed sequence still maps step 3,600 to `2hop__767045_160851` and step 3,622 to `5ab60bbc554299110f2199c7`, then assert the resumed slice starts at original step 3,601.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
pytest -q tests/test_rl_training.py -k 'resume_state or resume_policy or resume_sequence or parse_args_loads_resume'
```

Expected: failures because resume configuration and helpers do not exist.

- [ ] **Step 3: Implement resume state and adapter separation**

Add CLI defaults and arguments. Introduce:

```python
@dataclass(frozen=True)
class ResumeState:
    checkpoint_path: Path | None
    epoch: int
    samples_consumed: int
    global_step: int
    optimizer_state_restored: bool = False
```

Implement `_resolve_resume_state()` with pre-model validation. In `_load_policy_and_reference()`, choose the checkpoint for tokenizer and the trainable `default` adapter only, while always loading the frozen `reference` adapter from `sft_adapter_path`.

- [ ] **Step 4: Implement loop slicing and metadata**

Initialize `global_step` from the resume state and start the epoch loop at `resume_state.epoch`. For the first resumed epoch, slice rank samples by `resume_state.samples_consumed`; later epochs start at zero. Write `resume_meta.json` when the output directory is created:

```json
{
  "resume_from_checkpoint": ".../checkpoint-3600",
  "resume_epoch": 1,
  "resume_samples_consumed": 3600,
  "resume_global_step": 3600,
  "optimizer_state_restored": false
}
```

Also embed this object in final `train_meta.json`.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run the same resume-focused pytest command and expect all selected tests to pass.

---

### Task 3: checkpoint-3600 launcher

**Files:**
- Create: `scripts/run_train_grpo_resume_3600.sh`
- Modify: `tests/test_rl_training.py`

**Interfaces:**
- Consumes: `scripts/run_train_grpo.sh` and the resume/retry CLI flags from Tasks 1-2.
- Produces: a strict, override-friendly single-command launcher for this interrupted run.

- [ ] **Step 1: Write a failing launcher contract test**

Read the new script and assert strict mode, the exact default checkpoint path, resume epoch/sample/global step values, retry defaults, `CONFIG_PATH` support, and delegation to `run_train_grpo.sh`.

- [ ] **Step 2: Run the launcher test and confirm RED**

Run:

```bash
pytest -q tests/test_rl_training.py -k 'resume_3600_script'
```

Expected: failure because the script does not exist.

- [ ] **Step 3: Create the launcher**

Create an executable script with strict mode and repository-relative defaults:

```bash
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${REPO_ROOT}/outputs/grpo_qwen2.5-7b/2026-08-18_23-11-03/checkpoint-3600}"
RESUME_EPOCH="${RESUME_EPOCH:-1}"
RESUME_SAMPLES_CONSUMED="${RESUME_SAMPLES_CONSUMED:-3600}"
RESUME_GLOBAL_STEP="${RESUME_GLOBAL_STEP:-3600}"

exec bash "${SCRIPT_DIR}/run_train_grpo.sh" \
  --resume-from-checkpoint "${RESUME_CHECKPOINT}" \
  --resume-epoch "${RESUME_EPOCH}" \
  --resume-samples-consumed "${RESUME_SAMPLES_CONSUMED}" \
  --resume-global-step "${RESUME_GLOBAL_STEP}" \
  --vllm-generate-max-attempts "${VLLM_GENERATE_MAX_ATTEMPTS:-3}" \
  --vllm-generate-retry-backoff-seconds "${VLLM_GENERATE_RETRY_BACKOFF_SECONDS:-1.0}" \
  "$@"
```

Fail early if the checkpoint directory is absent.

- [ ] **Step 4: Verify launcher syntax and test GREEN**

Run:

```bash
bash -n scripts/run_train_grpo_resume_3600.sh
pytest -q tests/test_rl_training.py -k 'resume_3600_script'
```

Expected: syntax success and passing launcher test.

---

### Task 4: integrated verification

**Files:**
- Verify: all modified files

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: evidence that the resume/retry feature works without starting a large CUDA run.

- [ ] **Step 1: Run static checks**

```bash
bash -n scripts/run_train_grpo.sh scripts/run_train_grpo_resume_3600.sh
python -m compileall -q src/rl_training
git diff --check
```

- [ ] **Step 2: Run the complete RL test file**

```bash
pytest -q tests/test_rl_training.py
```

Expected: all repository-present RL tests pass, except any pre-existing fixture failure must be reported explicitly rather than hidden.

- [ ] **Step 3: Run a no-model resume configuration check**

Use the parser/data helpers to resolve checkpoint-3600, regenerate the order, and print the first resumed qid. Confirm it equals the original epoch-1 item at step 3,601 and that step 3,600/3,622 still match the historical log.

- [ ] **Step 4: Review the final diff**

Confirm no edits to the user's existing `config/train_grpo.yml`, no optimizer-restoration claim, no reference-adapter reset, and no retry around LoRA synchronization.

