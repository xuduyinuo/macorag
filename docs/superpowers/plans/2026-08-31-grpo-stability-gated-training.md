# GRPO Stability-Gated Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Canonicalize GRPO old/current probabilities in HF, add clipped and scheduled resumable optimization, and gate a deterministic 300-step stability stage before continuation to 1000 steps.

**Architecture:** vLLM generates tokens and supplies diagnostics only; HF deterministically rescores behavior tokens before backward. Training/checkpoint code owns optimization state, while evaluation modules own the fixed manifest, resumable predictions, and pass/fail reports.

**Tech Stack:** Python 3.9, PyTorch, Transformers/PEFT, vLLM 0.8.5, pytest, YAML, Bash.

## Global Constraints

- HF and vLLM explicitly load BF16: `bf16: true`, `fp16: false`, `load_4bit: false`, `vllm_dtype: bfloat16`.
- Fix one 1000-example training selection; consume positions 1-300 and then 301-1000.
- Keep `max_steps: 1000` critical and exclude only operational `run_until_step` from the fingerprint.
- Use cosine schedule, 3% warmup, 10% LR floor, and `max_grad_norm: 1.0`.
- Scheduler advances only after a successful optimizer update.
- vLLM log-probabilities never enter the GRPO ratio.
- Fixed validation contains 100 examples per dataset.
- Do not launch full 300/1000-step jobs during implementation.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Configuration and staged-run contract

**Files:** Modify `src/rl_training/config.py`, `src/rl_training/checkpointing.py`, `config/train_grpo.yml`; test `tests/test_rl_training.py`, `tests/test_rl_checkpointing.py`.

**Produces:** parsed `run_until_step`, `max_grad_norm`, `lr_scheduler_type`, `min_lr_ratio`; fingerprint tracks scheduler/clipping/precision but not `run_until_step`.

- [ ] Write failing tests:

```python
def test_stage_scheduler_config(tmp_path):
    p = tmp_path / "x.yml"
    p.write_text("max_steps: 1000\nrun_until_step: 300\nmax_grad_norm: 1.0\n"
                 "lr_scheduler_type: cosine\nwarmup_ratio: 0.03\nmin_lr_ratio: 0.1\n")
    a = parse_args(["--config", str(p)])
    assert (a.max_steps, a.run_until_step, a.max_grad_norm) == (1000, 300, 1.0)

def test_run_until_is_not_critical():
    assert fingerprint_config(SimpleNamespace(max_steps=1000, run_until_step=300)) == \
           fingerprint_config(SimpleNamespace(max_steps=1000, run_until_step=1000))
```

- [ ] Verify RED:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py tests/test_rl_checkpointing.py -k 'stage_scheduler or run_until_is_not_critical'
```

- [ ] Add defaults/CLI validation: `0 <= run_until_step <= max_steps`, positive grad norm, `0 <= warmup_ratio < 1`, `0 <= min_lr_ratio <= 1`; add all semantic fields to `_CRITICAL_CONFIG_FIELDS`, excluding `run_until_step`.
- [ ] Set canonical YAML to 1000 selected samples, `max_steps: 1000`, `run_until_step: 300`, BF16, cosine/0.03/0.1, grad norm 1.0.
- [ ] Re-run focused tests; expected PASS.
- [ ] Commit only these files: `git commit -m "feat: define staged GRPO optimization contract"`.

### Task 2: Canonical HF behavior rescoring

**Files:** Modify `src/rl_training/policy.py`, `src/rl_training/train_grpo_macorag.py`, `src/rl_training/trainer.py`, `tests/test_rl_training.py`.

**Produces:** `GeneratedAction.server_logprobs`; `_rescore_behavior_logprobs(...)`; masked ratio diagnostics.

- [ ] Write failing tests that vLLM stores `[-9]` only in `server_logprobs`, HF batched rescore overwrites `old_logprobs` with `[-0.25]`, model mode observed during rescore/current is `False`, and unchanged HF parameters yield `clip_fraction == 0`.
- [ ] Verify RED:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py -k 'server_logprobs or behavior_rescore or deterministic_ratio'
```

- [ ] In `VLLMSharedPolicy`, retain server values separately and initialize old values empty; HF-only generation remains compatible.
- [ ] Implement batched no-grad rescore after early skip checks and before reference/current forwards; validate token masks, lengths, and finite values; store detached CPU values.
- [ ] Keep policy adapter in `eval()` for gradient-bearing current forward; autograd remains enabled while LoRA dropout is disabled.
- [ ] Add masked `preupdate_logratio_mean/max_abs`, `ratio_mean/p95`, server/HF MAE/max; fix clip fraction to exclude padding.
- [ ] Run policy/loss/train tests; expected PASS.
- [ ] Commit: `git commit -m "fix: canonicalize GRPO behavior logprobs with HF"`.

### Task 3: Gradient clipping and cosine scheduler

**Files:** Create `src/rl_training/scheduling.py`, `tests/test_rl_scheduling.py`; modify trainer and `tests/test_rl_training.py`.

**Produces:** `build_cosine_scheduler(optimizer, total_updates, warmup_ratio, min_lr_ratio)` and scheduler-aware `_train_on_rollouts`/final flush.

- [ ] Write failing tests:

```python
def test_cosine_schedule_reaches_warmup_peak_and_floor():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1e-5)
    sch = build_cosine_scheduler(opt, total_updates=1000, warmup_ratio=.03, min_lr_ratio=.1)
    values=[]
    for _ in range(1000): opt.step(); sch.step(); values.append(opt.param_groups[0]["lr"])
    assert values[29] == pytest.approx(1e-5)
    assert values[-1] == pytest.approx(1e-6)
```

Add a trainer test with norm >1 asserting before >1, after <=1, optimizer/scheduler each step once; add skip tests asserting scheduler does not step.
- [ ] Verify RED with `pytest -q tests/test_rl_scheduling.py tests/test_rl_training.py -k 'cosine_schedule or clips_gradients or scheduler_skip'`.
- [ ] Implement warmup plus floored cosine `LambdaLR`; total slots are `ceil(max_steps / gradient_accumulation_steps)`.
- [ ] Clip finite trainable grads immediately before optimizer step, then optimizer, scheduler, zero-grad; use identical order in final flush.
- [ ] Log pre/post norm, clipped boolean, live LR, successful update count.
- [ ] Re-run focused tests and commit `feat: clip GRPO gradients and schedule successful updates`.

### Task 4: Schema-v2 resumable optimization state

**Files:** Modify `src/rl_training/checkpointing.py`, trainer, `tests/test_rl_checkpointing.py`, `tests/test_rl_checkpoint_seeds.py`.

**Produces:** `scheduler.pt`; manifest update count/horizon/warmup/optimization contract; exact restore.

- [ ] Write a failing round-trip test that steps optimizer/scheduler once, saves, restores into new objects, and asserts identical scheduler state plus `successful_optimizer_updates == 1`.
- [ ] Verify RED: `pytest -q tests/test_rl_checkpointing.py -k scheduler`.
- [ ] Bump schema to 2; write scheduler before COMPLETE; include it in expected files; validate `hf_rescore_v1`, precision, clipping, scheduler horizon and warmup.
- [ ] Construct scheduler before restore, restore count, increment only on successful step, and include state in periodic/final saves.
- [ ] Run both checkpoint suites; expected PASS.
- [ ] Commit `feat: resume GRPO scheduler state exactly`.

### Task 5: Safe step-300 stop and sample-301 continuation

**Files:** Modify trainer, `scripts/run_train_grpo_resume.sh`, `tests/test_rl_training.py`, `tests/test_retraining_launchers.py`.

**Produces:** `_run_step_limit(args)`; resume launcher `RUN_UNTIL_STEP` override.

- [ ] Write failing tests that limit resolves 300/1000 correctly and deterministic first 300 plus resumed 700 are disjoint and cover all fixed 1000 qids.
- [ ] Replace all progress/inner/outer max-step checks with one operational ceiling; preserve dataset fingerprint and sample cursor.
- [ ] Add resume dry-run output and `--run-until-step "$RUN_UNTIL_STEP"` without mutating YAML.
- [ ] Run focused pytest plus `bash -n scripts/run_train_grpo_resume.sh`; commit `feat: stop and resume GRPO at stage boundaries`.

### Task 6: Fixed manifest and resumable evaluation

**Files:** Create `src/evaluation/fixed_manifest.py`, `scripts/build_grpo_validation_manifest.sh`, `config/eval_grpo_fixed.yml`; modify `src/evaluation/data.py`, `config.py`, `evaluate_rag_model.py`, `local_evaluator.py`, `tests/test_evaluation.py`.

**Produces:** `data/eval_300_grpo_fixed/manifest.jsonl`, `manifest_meta.json`; explicit resumable evaluation output and contract.

- [ ] Write failing tests: same seed produces same fingerprint; counts equal 100/100/100; every row has stratum; existing completed qid is skipped on resume; unknown/duplicate/mismatched qid fails.
- [ ] Verify RED: `pytest -q tests/test_evaluation.py -k 'fixed_manifest or resume_predictions or evaluation_contract'`.
- [ ] Promote current stratum derivation to a public helper without changing RL selection; allocate each dataset quota proportionally by stratum with deterministic SHA seeds/tie breaks.
- [ ] Add eval args `output_dir`, `resume`, `manifest_meta_path`, `adapter_label`; explicit output bypasses timestamps.
- [ ] Resume valid JSONL rows, generate only missing qids, atomically rewrite ordered output; write aggregate macro metrics and contract fingerprints.
- [ ] Build the real manifest:

```bash
PYTHON=/data/conda/envs/macorag/bin/python bash scripts/build_grpo_validation_manifest.sh
```

- [ ] Assert metadata counts and fingerprint; run full evaluation tests; commit `feat: add resumable fixed GRPO validation` including generated manifest.

### Task 7: Stability/F1 gate and thin launchers

**Files:** Create `src/rl_training/stability_gate.py`, `scripts/check_grpo_gate.sh`, `scripts/evaluate_grpo_fixed.sh`, `tests/test_grpo_stability_gate.py`; modify launcher tests and metrics payload.

**Produces:** `gate_report.json`; exit 0 pass, 2 threshold failure, 3 incomplete/contract mismatch.

- [ ] Write failing tests for every threshold boundary, missing rows, contract mismatch, macro-F1 gain with one dataset regression, and valid pass.
- [ ] Verify RED: `pytest -q tests/test_grpo_stability_gate.py`.
- [ ] Aggregate exactly 300 numbered train rows: clip mean/P95, previous/final KL windows, protocol counts, and finite successful updates.
- [ ] Require overall macro F1 strictly above SFT, every dataset non-regressing, parse `<.01`, missing tag `<.002`, and identical manifest/prompt/retrieval/generation contracts.
- [ ] Always write report before returning status; add dry-run-safe thin wrappers that reuse existing training/eval entrypoints.
- [ ] Add validated `advantage_granularity` (`role_round` default, `role_only` ablation) and record/fingerprint it; do not launch ablations.
- [ ] Run gate/launcher/trainer tests and shell syntax; commit `feat: gate GRPO continuation on stability and F1`.

### Task 8: Verification and bounded GPU smoke

**Files:** Modify only task-owned files if verification reveals a defect.

- [ ] Run focused suites:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py tests/test_rl_scheduling.py tests/test_rl_checkpointing.py tests/test_rl_checkpoint_seeds.py tests/test_evaluation.py tests/test_grpo_stability_gate.py tests/test_retraining_launchers.py
```

- [ ] Run static checks:

```bash
/data/conda/envs/macorag/bin/python -m compileall -q src
bash -n scripts/*.sh
git diff --check
```

- [ ] Run trainer check-only and all new dry-runs; expected exit 0 with BF16/stage/manifest paths printed.
- [ ] When both GPUs are genuinely free, run one sample with a temporary config (`max_total_samples/max_steps/run_until_step: 1`) and temporary output root. Do not stop unrelated processes.
- [ ] Verify BF16 loading, server/HF diagnostics, near-zero clip, finite clipped grad, one scheduler step, LoRA sync, and complete schema-v2 checkpoint.
- [ ] Hand off exact commands for manifest, SFT baseline, step 300, gate, resume to 1000, and checkpoint evaluations; do not claim full experiments ran.
