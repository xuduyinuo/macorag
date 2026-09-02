# SFT Stratified Sampling and Step Early Stopping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the canonical SFT run select exactly 400 2Wiki, 400 HotpotQA, and 300 MuSiQue trajectories before a per-dataset 95/5 split, evaluate token-level `eval_loss` every 200 optimizer steps, stop after three non-meaningful improvements, and export the numerically best checkpoint within a three-epoch cap.

**Architecture:** Extend the existing YAML/CLI contract, keep quota selection and stratified splitting in `src/sft_training/data.py`, and let Hugging Face `Trainer` own checkpoint comparison, callback persistence, stopping, and best-model restoration. Record every data and stopping input in `sft_run_manifest.json`, then record the observed terminal Trainer state in `train_meta.json`.

**Tech Stack:** Python 3.10+, PyYAML, pytest, Hugging Face Transformers `Trainer`/`EarlyStoppingCallback`, PEFT LoRA, JSON/JSONL run artifacts.

## Global Constraints

- Apply quotas to usable original trajectories, not action records, before validation splitting.
- Keep every `(dataset, qid)` trajectory wholly in train or validation.
- Use stable digest-derived dataset seeds; never use Python's randomized `hash()`.
- Treat `early_stopping_threshold=0.001` only as the patience-reset threshold. Any strictly lower `eval_loss` remains eligible as the best checkpoint.
- Preserve legacy scalar `max_samples` for smoke runs, but reject using it together with non-empty per-dataset quotas.
- Do not change RL code, prompt construction, target masking, LoRA structure, or evaluation metric semantics.
- Preserve unrelated dirty-worktree changes and stage only files owned by each task.

---

### Task 1: Add and validate the explicit configuration contract

**Files:**
- Modify: `src/sft_training/config.py`
- Modify: `config/train_sft.yml`
- Test: `tests/test_sft_training.py`

- [ ] **Step 1: Write failing parser and active-config tests**

Add tests that assert the checked-in canonical configuration parses to:

```python
assert args.max_samples is None
assert args.max_samples_by_dataset == {"2wiki": 400, "hotpotqa": 400, "musique": 300}
assert args.data_sampling_seed == 42
assert args.num_train_epochs == 3.0
assert args.eval_strategy == "steps"
assert args.eval_steps == 200
assert args.save_steps == 200
assert args.early_stopping_enabled is True
assert args.early_stopping_patience == 3
assert args.early_stopping_threshold == pytest.approx(0.001)
assert args.metric_for_best_model == "eval_loss"
assert args.greater_is_better is False
assert args.restore_callback_states_from_checkpoint is True
```

Add negative cases for: simultaneous scalar and mapping limits, non-positive/non-integer quotas, enabled early stopping without validation, non-step evaluation, non-positive intervals, `save_steps % eval_steps != 0`, non-positive patience, a metric other than `eval_loss`, and `greater_is_better=True`.

- [ ] **Step 2: Run the new tests and confirm they fail for missing fields/validation**

Run:

```bash
pytest -q tests/test_sft_training.py -k 'active_sft_config or sample_limit_config or early_stopping_config'
```

Expected: failures identify absent arguments or configurations that are currently accepted.

- [ ] **Step 3: Implement normalization and cross-field validation**

Add these defaults and BooleanOptionalAction flags in `config.py`:

```python
DATA_DEFAULTS.update({
    "max_samples_by_dataset": {},
    "data_sampling_seed": 42,
})
EVAL_DEFAULTS.update({
    "early_stopping_enabled": False,
    "restore_callback_states_from_checkpoint": True,
})
```

Normalize YAML mappings and JSON CLI input into `dict[str, int]`, canonicalize dataset keys to `2wiki`, `hotpotqa`, and `musique`, and reject unknown names. After parsing, validate all cross-field invariants in one `_validate_args(args)` function. Error messages must name the offending field and include its actual value.

Keep the library defaults backward-compatible (`early_stopping_enabled=False`, empty quotas); put the formal-run values only in `config/train_sft.yml`.

- [ ] **Step 4: Update the canonical YAML without disturbing unrelated tuning choices**

Set precisely:

```yaml
max_samples: null
max_samples_by_dataset:
  2wiki: 400
  hotpotqa: 400
  musique: 300
data_sampling_seed: 42
num_train_epochs: 3.0
eval_strategy: "steps"
eval_steps: 200
save_steps: 200
validation_split: true
eval_split_ratio: 0.05
train_test_seed: 777
early_stopping_enabled: true
early_stopping_patience: 3
early_stopping_threshold: 0.001
metric_for_best_model: "eval_loss"
greater_is_better: false
restore_callback_states_from_checkpoint: true
save_total_limit: 3
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest -q tests/test_sft_training.py -k 'config'
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the configuration contract**

```bash
git add src/sft_training/config.py config/train_sft.yml tests/test_sft_training.py
git commit -m "feat: define SFT sampling and early-stop contract"
```

---

### Task 2: Select exact deterministic per-dataset quotas

**Files:**
- Modify: `src/sft_training/data.py`
- Modify: `src/sft_training/train_sft_lora_macorag.py`
- Test: `tests/test_sft_training.py`

- [ ] **Step 1: Write failing selection tests**

Build in-memory `TrainingSample` lists for each dataset and test a public helper such as `select_training_samples_by_dataset(...)` for:

- exact selected counts `400/400/300` from larger inputs;
- identical selected qids for repeated calls with seed 42;
- at least one changed qid with seed 43;
- stable source order after sampling;
- duplicate `(dataset, qid)` rejection;
- requested quota larger than usable count rejection with dataset, requested, and available values in the message;
- scalar `max_samples` retaining its current global-cap order.

- [ ] **Step 2: Run selection tests and confirm the helper is missing**

```bash
pytest -q tests/test_sft_training.py -k 'quota or deterministic_selection or duplicate_qid'
```

Expected: collection or assertion failures because quota selection is not implemented.

- [ ] **Step 3: Implement stable seed derivation and selection**

Use a digest-based helper:

```python
def _dataset_seed(seed: int, dataset: str, namespace: str) -> int:
    payload = f"{namespace}\0{seed}\0{dataset}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
```

Implement selection by grouping usable `TrainingSample` objects, validating qid uniqueness, shuffling source indices with `random.Random(_dataset_seed(...))`, taking the requested count, and sorting the chosen indices before returning samples. Require a quota for every dataset present in formal mapping mode, and reject quota keys whose data file produced no usable trajectories.

Extend the loader signature without changing the scalar behavior:

```python
def build_training_data(
    data_root: Path,
    max_samples: int | None = None,
    max_samples_by_dataset: dict[str, int] | None = None,
    data_sampling_seed: int = 42,
) -> TrainingData:
```

Keep `source_sample_counts_by_dataset` as pre-selection usable counts and add `selected_sample_counts_by_dataset` to `TrainingData` so metadata distinguishes availability from selection.

- [ ] **Step 4: Pass the new arguments through the canonical entry point**

Change the call in `main()` to:

```python
training_data = build_training_data(
    data_root,
    max_samples=args.max_samples,
    max_samples_by_dataset=args.max_samples_by_dataset,
    data_sampling_seed=args.data_sampling_seed,
)
```

- [ ] **Step 5: Run selection and existing data tests**

```bash
pytest -q tests/test_sft_training.py -k 'training_data or quota or selection or duplicate_qid or flatten_training_samples'
```

Expected: all selected tests pass, including legacy scalar-cap coverage.

- [ ] **Step 6: Commit deterministic quota selection**

```bash
git add src/sft_training/data.py src/sft_training/train_sft_lora_macorag.py tests/test_sft_training.py
git commit -m "feat: sample SFT quotas per dataset"
```

---

### Task 3: Split the reduced set independently by dataset

**Files:**
- Modify: `src/sft_training/data.py`
- Modify: `src/sft_training/train_sft_lora_macorag.py`
- Test: `tests/test_sft_training.py`

- [ ] **Step 1: Write failing stratified-split tests**

Create 400 `2wiki`, 400 `hotpotqa`, and 300 `musique` samples, with two action records on selected qids. Assert:

```python
assert counts(train_samples) == {"2wiki": 380, "hotpotqa": 380, "musique": 285}
assert counts(val_samples) == {"2wiki": 20, "hotpotqa": 20, "musique": 15}
assert {(s.dataset, s.qid) for s in train_samples}.isdisjoint(
    {(s.dataset, s.qid) for s in val_samples}
)
```

Also assert repeated splits are identical, a different seed changes membership, returned items preserve source order, and a dataset yielding zero train or validation items fails with its dataset name. Retain the existing single-dataset `split_training_samples` regression test or route it through the new implementation without changing its promised output.

- [ ] **Step 2: Run the split tests and confirm global splitting gives wrong per-dataset counts**

```bash
pytest -q tests/test_sft_training.py -k 'split_training_samples or stratified_split'
```

Expected: the new exact-count test fails under the current global shuffle.

- [ ] **Step 3: Implement stratified splitting**

Add:

```python
def split_training_samples_by_dataset(
    samples: list[TrainingSample], ratio: float, seed: int
) -> tuple[list[TrainingSample], list[TrainingSample]]:
```

Group source indices by dataset, derive each shuffle seed with `_dataset_seed(seed, dataset, "split")`, and select exactly `math.floor(len(dataset_indices) * ratio)` validation indices per group. Fail when any non-empty dataset has zero train or validation samples. Build both returned lists by scanning original indices once so source order remains stable.

- [ ] **Step 4: Switch only the canonical SFT split call**

Import the new helper and update `_split_train_eval_samples()` to use it. Do not split flattened action records and do not change the target-only loss code.

- [ ] **Step 5: Run focused split and masking regressions**

```bash
pytest -q tests/test_sft_training.py -k 'split or target_only or masking or flatten'
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit stratified splitting**

```bash
git add src/sft_training/data.py src/sft_training/train_sft_lora_macorag.py tests/test_sft_training.py
git commit -m "feat: stratify SFT validation split"
```

---

### Task 4: Wire explicit early stopping, best restoration, resume state, and metadata

**Files:**
- Modify: `src/sft_training/train_sft_lora_macorag.py`
- Test: `tests/test_sft_training.py`

- [ ] **Step 1: Write failing TrainingArguments and callback tests**

Using a recording `DummyTrainingArguments`, assert enabled validation produces:

```python
assert kwargs["eval_strategy"] == "steps"
assert kwargs["eval_steps"] == 200
assert kwargs["save_strategy"] == "steps"
assert kwargs["save_steps"] == 200
assert kwargs["load_best_model_at_end"] is True
assert kwargs["metric_for_best_model"] == "eval_loss"
assert kwargs["greater_is_better"] is False
assert kwargs["restore_callback_states_from_checkpoint"] is True
```

Assert disabling `early_stopping_enabled` yields no `EarlyStoppingCallback` and `load_best_model_at_end=False`, even if patience is non-zero. Extract callback construction into `_build_early_stopping_callback(args, has_eval, callback_cls)` so it is unit-testable without loading a model.

- [ ] **Step 2: Add a synthetic loss-sequence regression**

Drive the installed Transformers `EarlyStoppingCallback` with fake `TrainingArguments`, `TrainerState`, and `TrainerControl` objects across losses:

```python
losses = [0.0800, 0.0750, 0.0746, 0.0748, 0.0751]
```

Before each callback invocation, update `state.best_metric` exactly as Trainer does for strict lower-is-better checkpoint selection. Assert the patience counter evolves `0, 0, 1, 2, 3`, stopping is requested on the fifth evaluation, and the independently tracked numeric best is `0.0746` at step 600. This protects the distinction between thresholded patience and strict best-checkpoint selection.

- [ ] **Step 3: Run the new early-stopping tests and confirm current implicit behavior fails**

```bash
pytest -q tests/test_sft_training.py -k 'training_arguments or early_stopping or best_checkpoint'
```

Expected: failures show the missing explicit enable switch and callback-state restoration argument.

- [ ] **Step 4: Implement Trainer wiring**

In `_training_arguments()`, derive:

```python
load_best_model = bool(has_eval and args.early_stopping_enabled)
```

Pass `restore_callback_states_from_checkpoint=args.restore_callback_states_from_checkpoint`. Register exactly one callback only when `has_eval and args.early_stopping_enabled`; keep its reference for terminal metadata. Continue exporting `model` only after `_run_trainer()` returns, because Trainer reloads the best model at the end of training when `load_best_model_at_end=True`.

- [ ] **Step 5: Expand the resume manifest with deterministic fingerprints**

Add helpers that count samples by dataset and compute a SHA-256 fingerprint over ordered `dataset\0qid` pairs. Bump `schema_version` to 2 and add:

```python
"max_samples_by_dataset": args.max_samples_by_dataset,
"data_sampling_seed": args.data_sampling_seed,
"source_sample_counts_by_dataset": training_data.source_sample_counts_by_dataset,
"selected_sample_counts_by_dataset": training_data.selected_sample_counts_by_dataset,
"selected_qid_fingerprint": sample_fingerprint(training_data.samples),
"train_sample_counts_by_dataset": sample_counts(train_samples),
"eval_sample_counts_by_dataset": sample_counts(val_samples),
"train_qid_fingerprint": sample_fingerprint(train_samples),
"eval_qid_fingerprint": sample_fingerprint(val_samples),
"eval_strategy": args.eval_strategy,
"eval_steps": args.eval_steps,
"save_steps": args.save_steps,
"early_stopping_enabled": args.early_stopping_enabled,
"early_stopping_patience": args.early_stopping_patience,
"early_stopping_threshold": args.early_stopping_threshold,
"metric_for_best_model": args.metric_for_best_model,
"greater_is_better": args.greater_is_better,
"restore_callback_states_from_checkpoint": args.restore_callback_states_from_checkpoint,
```

The existing `_validate_resume_compatibility()` exact-value comparison then rejects changed selection, split, or early-stop contracts. Add a test changing one field at a time, including sampling seed and threshold.

- [ ] **Step 6: Record observed terminal state**

After `_run_trainer()`, read `trainer.state` and the retained early-stop callback. Add to `train_meta.json`:

```python
"stopped_early": bool(trainer.control.should_training_stop and trainer.state.global_step < total_optimizer_steps),
"best_metric": trainer.state.best_metric,
"best_model_checkpoint": trainer.state.best_model_checkpoint,
"stopped_epoch": trainer.state.epoch,
"global_step": trainer.state.global_step,
"early_stopping_enabled": args.early_stopping_enabled,
"early_stopping_patience": args.early_stopping_patience,
"early_stopping_threshold": args.early_stopping_threshold,
"restore_callback_states_from_checkpoint": args.restore_callback_states_from_checkpoint,
"output_dir": str(output_dir / "adapter"),
```

Prefer a small `_trainer_completion_metadata(...)` helper and unit-test both an early-stopped state and a normal three-epoch completion. Do not infer early stopping solely from the callback counter because a run can end naturally on the same evaluation.

- [ ] **Step 7: Test callback-state resume behavior**

Construct a temporary checkpoint `trainer_state.json` containing serialized callback state (or use a tiny CPU Trainer fixture if the installed Transformers serializer requires it). Verify `restore_callback_states_from_checkpoint=True` restores a non-zero patience counter and that disabling restoration leaves a fresh counter. This test must exercise the installed Transformers behavior rather than a repository-local imitation.

- [ ] **Step 8: Run early-stop and resume tests**

```bash
pytest -q tests/test_sft_training.py -k 'early_stopping or training_arguments or trainer_completion or resume_manifest or callback_state'
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Trainer behavior and observability**

```bash
git add src/sft_training/train_sft_lora_macorag.py tests/test_sft_training.py
git commit -m "feat: restore best SFT checkpoint on early stop"
```

---

### Task 5: Make check-only output prove the formal data contract

**Files:**
- Modify: `src/sft_training/train_sft_lora_macorag.py`
- Test: `tests/test_sft_training.py`

- [ ] **Step 1: Write a failing check-only output test**

Construct a `TrainingData` object with the formal selected counts and capture `_print_check_only()`. Require machine-readable JSON lines for source, selected, train, and validation sample counts plus the effective early-stop contract. Assert the output includes exactly:

```text
selected: 2wiki=400, hotpotqa=400, musique=300
train:    2wiki=380, hotpotqa=380, musique=285
eval:     2wiki=20,  hotpotqa=20,  musique=15
```

- [ ] **Step 2: Run the test and confirm current output lacks split counts**

```bash
pytest -q tests/test_sft_training.py -k 'check_only'
```

Expected: failure because the current check-only path prints only original and action-record counts.

- [ ] **Step 3: Compute and print the same split used by training**

Have `_print_check_only(args, training_data)` call `_split_train_eval_samples(args, training_data.samples)` and print deterministic JSON objects with sorted keys. Include `eval_strategy`, `eval_steps`, `save_steps`, `early_stopping_enabled`, patience, threshold, metric, direction, and callback-state restoration. Keep the existing sample prompt/target leakage check.

- [ ] **Step 4: Run focused tests**

```bash
pytest -q tests/test_sft_training.py -k 'check_only or stratified_split or active_sft_config'
```

Expected: all selected tests pass and captured output proves `400/400/300`, `380/380/285`, and `20/20/15`.

- [ ] **Step 5: Commit check-only observability**

```bash
git add src/sft_training/train_sft_lora_macorag.py tests/test_sft_training.py
git commit -m "feat: report SFT split and early-stop contract"
```

---

### Task 6: Run final regression and launch-readiness verification

**Files:**
- Verify: `config/train_sft.yml`
- Verify: `src/sft_training/config.py`
- Verify: `src/sft_training/data.py`
- Verify: `src/sft_training/train_sft_lora_macorag.py`
- Verify: `tests/test_sft_training.py`

- [ ] **Step 1: Run the complete SFT test module**

```bash
pytest -q tests/test_sft_training.py
```

Expected: zero failures.

- [ ] **Step 2: Run the adjacent SFT/RL regression suite**

```bash
pytest -q tests/test_sft_training.py tests/test_rl_training.py
```

Expected: zero failures and no RL regression.

- [ ] **Step 3: Run the real-data check-only path**

Use the repository launcher/config path so environment setup matches a real run:

```bash
bash scripts/run_train_sft.sh --check-only --check-only-max-samples 3
```

Expected output includes selected sample counts `{"2wiki": 400, "hotpotqa": 400, "musique": 300}`, train counts `{"2wiki": 380, "hotpotqa": 380, "musique": 285}`, validation counts `{"2wiki": 20, "hotpotqa": 20, "musique": 15}`, and step evaluation/saving at 200 with patience 3 and threshold 0.001. If the actual launcher has a different filename, locate the canonical SFT launcher with `rg --files scripts | rg 'sft'` and use its supported argument-forwarding syntax; do not start full training.

- [ ] **Step 4: Inspect the effective diff and scope**

```bash
git diff --check
git diff --stat HEAD~5..HEAD
git diff HEAD~5..HEAD -- src/rl_training tests/test_rl_training.py
```

Expected: no whitespace errors and no RL diff introduced by these five implementation commits. Account separately for any pre-existing user changes already present before implementation.

- [ ] **Step 5: Verify there are no placeholders or stale epoch-mode assertions**

```bash
rg -n 'TODO|TBD|PLACEHOLDER|evaluates_once_per_epoch|eval_strategy.*epoch' \
  src/sft_training config/train_sft.yml tests/test_sft_training.py
```

Expected: no new placeholder markers; any remaining `epoch` references are deliberate backward-compatibility tests, not assertions about the active formal config.

- [ ] **Step 6: Record the handoff evidence**

Report the exact passing test counts, the check-only count lines, and the files changed. Do not claim a full GPU training run or successful early termination from real loss curves unless that run was actually executed.
