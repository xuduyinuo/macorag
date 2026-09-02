# SFT Per-Dataset Sampling and Step-Based Early Stopping Design

Date: 2026-09-02

## Objective

Reduce the SFT source set to fixed per-dataset quotas, split validation only after that reduction, and stop training when token-level validation loss no longer improves meaningfully. The final exported adapter must always come from the checkpoint with the lowest validation loss, not merely the last optimizer step.

## Scope

This change covers the canonical SFT path:

- `config/train_sft.yml`
- `src/sft_training/config.py`
- `src/sft_training/data.py`
- `src/sft_training/train_sft_lora_macorag.py`
- focused SFT tests

It does not change trajectory conversion, prompt construction, target masking, LoRA structure, the RL trainer, retrieval, or evaluation metrics.

## Confirmed Training Contract

The source quotas apply to usable original trajectory samples before the train/validation split:

| Dataset | Reduced total | Train at 95% | Validation at 5% |
|---|---:|---:|---:|
| 2Wiki | 400 | 380 | 20 |
| HotpotQA | 400 | 380 | 20 |
| MuSiQue | 300 | 285 | 15 |
| Total | 1100 | 1045 | 55 |

All action records belonging to one original `(dataset, qid)` trajectory remain in the same split. Splitting action records independently is forbidden because it would leak trajectory state across train and validation.

Training is capped at three epochs. Validation runs every 200 optimizer steps. Training stops when three consecutive validations fail to improve the historical best `eval_loss` by more than `0.001`. At completion, the Trainer restores the checkpoint with the lowest validation loss before the repository exports `adapter/`.

## Configuration Interface

The canonical YAML will use the following values:

```yaml
# Source-sample selection before validation splitting.
max_samples: null
max_samples_by_dataset:
  2wiki: 400
  hotpotqa: 400
  musique: 300
data_sampling_seed: 42

# Training horizon.
num_train_epochs: 3.0

# Validation, checkpointing, and early stopping.
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

`max_samples` remains available for existing smoke commands. A run must not set both a non-null `max_samples` and a non-empty `max_samples_by_dataset`; argument validation will reject that ambiguous configuration.

`early_stopping_enabled` is the authoritative switch. `early_stopping_patience` is no longer overloaded as an implicit enable/disable flag. When the switch is disabled, no early-stopping callback is registered and best-checkpoint restoration is not implicitly enabled.

## Deterministic Per-Dataset Selection

For each configured dataset, the data loader will:

1. Load and validate the dataset JSONL.
2. Convert rows to `TrainingSample` objects and retain only usable trajectories with at least one action record.
3. Reject duplicate qids within that dataset.
4. Fail before training if fewer usable samples exist than the requested quota.
5. Derive a stable dataset-specific selection seed from `data_sampling_seed` and the canonical dataset name.
6. Sample without replacement to the exact requested quota.
7. Restore selected samples to source-file order before downstream processing, so selection is deterministic without coupling training order to the random draw order.

Dataset-specific seed derivation must use a stable digest rather than Python's process-randomized `hash()`. This keeps selected qids identical across processes and Python invocations.

The legacy scalar `max_samples` retains its existing global smoke-test behavior. Formal training uses `max_samples_by_dataset`.

## Stratified Train/Validation Split

Validation splitting happens after quota selection and independently within each dataset:

1. Group reduced `TrainingSample` objects by dataset.
2. Derive a stable dataset-specific split seed from `train_test_seed` and the dataset name.
3. Shuffle indices deterministically inside each dataset.
4. Assign `floor(dataset_count * eval_split_ratio)` samples to validation.
5. Return train and validation samples in stable source order.

For the confirmed quotas and ratio, the split must be exactly `380/20`, `380/20`, and `285/15`. Both splits must be non-empty for every configured dataset. The code will fail fast if rounding or a future quota violates that condition.

## Step-Based Validation and Early Stopping

Early stopping uses the existing target-only evaluation loss produced by the custom SFT Trainer. Its metric semantics remain the macro mean of per-action target-token mean loss.

When `early_stopping_enabled` is true, training arguments must satisfy:

- validation is enabled;
- `eval_strategy` is `steps`;
- `eval_steps` is positive;
- `save_strategy` is `steps`;
- `save_steps` is positive and divisible by `eval_steps`;
- `load_best_model_at_end` is true;
- `metric_for_best_model` resolves to `eval_loss`;
- `greater_is_better` is false;
- callback state restoration is enabled for full checkpoint resume.

At every 200th optimizer step, the Trainer evaluates and saves at the same boundary. A new result resets the patience counter only when:

```text
historical_best_eval_loss - current_eval_loss > 0.001
```

Otherwise the counter increases by one. A counter value of three requests training termination. This comparison is against the historical best, not merely the immediately preceding result.

The threshold controls only whether patience resets. Best-checkpoint selection remains strict lower-is-better without applying the threshold: any numerically lower `eval_loss` becomes the checkpoint restored at the end, even when the decrease is too small to reset patience.

Example:

```text
step 200:  0.0800 -> new best, counter 0
step 400:  0.0750 -> new best, counter 0
step 600:  0.0746 -> improvement below threshold, counter 1
step 800:  0.0748 -> no improvement, counter 2
step 1000: 0.0751 -> no improvement, counter 3, stop
```

The best checkpoint is step 600 in this example because `0.0746` is the lowest observed loss, even though its improvement was too small to reset patience. After `trainer.train()` returns, Hugging Face Trainer reloads that checkpoint, and the existing final `model.save_pretrained(output_dir / "adapter")` exports those restored weights.

## Checkpoint Retention and Resume

`save_total_limit: 3` bounds ordinary checkpoint storage. Hugging Face Trainer's best-checkpoint retention must remain active so the best checkpoint is not pruned merely because it is older than the most recent checkpoints.

A full resume must restore:

- model and LoRA weights;
- optimizer and scheduler state;
- RNG state;
- global step and epoch progress;
- historical best metric and best checkpoint;
- `EarlyStoppingCallback.early_stopping_patience_counter`.

The installed Transformers version exposes `restore_callback_states_from_checkpoint`; it must be passed through `TrainingArguments` when the configured field is true.

The SFT run manifest becomes part of the resume compatibility contract for:

- per-dataset quotas;
- sampling seed;
- selected-qid fingerprints;
- per-dataset train/validation counts;
- split seed and ratio;
- evaluation interval;
- early-stopping enable flag, patience, threshold, metric, and direction.

Changing any of these fields invalidates a claim of lossless continuation. The launcher should fail with a contract-mismatch message rather than silently resetting selection or early-stopping state.

## Metadata and Observability

`sft_run_manifest.json` will record the requested quotas, actual usable counts, selected-qid fingerprints, exact per-dataset split counts, and the complete early-stopping contract.

`train_meta.json` will additionally report final Trainer state:

- `stopped_early`;
- `best_metric`;
- `best_model_checkpoint`;
- `stopped_epoch` and `global_step`;
- early-stopping patience and threshold;
- whether callback state restoration was enabled;
- the final exported adapter path.

Existing `eval_metrics.jsonl` remains the authoritative chronological validation-loss stream. No existing metrics file is removed or renamed.

## Error Handling

Training fails before model loading when:

- a requested dataset quota is missing, non-integer, or non-positive;
- requested quota exceeds the number of usable samples;
- scalar and per-dataset limits are both active;
- duplicate qids make deterministic selection ambiguous;
- a configured dataset would have an empty train or validation split;
- early stopping is enabled without validation;
- step evaluation/save intervals are invalid or incompatible;
- a resume manifest disagrees with the current sampling, split, or early-stopping contract.

Failures must identify the dataset or configuration field and show expected versus actual values.

## Testing Strategy

Focused tests will cover:

1. Exact `400/400/300` selection from larger source files.
2. Deterministic selection for repeated runs with the same seed.
3. A changed sampling seed changes at least one selected qid.
4. Exact stratified split counts `380/20`, `380/20`, and `285/15`.
5. No `(dataset, qid)` overlap between train and validation.
6. All actions from a trajectory remain in one split.
7. Rejection of insufficient quotas, duplicate qids, and conflicting limit modes.
8. Enabled and disabled early-stopping callback registration.
9. Step evaluation and save alignment at 200 optimizer steps.
10. Synthetic losses `0.0800, 0.0750, 0.0746, 0.0748, 0.0751` stop after the third consecutive validation without meaningful improvement and retain step 600 as the numerically best checkpoint.
11. Final `adapter/` export uses the Trainer-restored best weights.
12. Resume restores callback patience state and rejects an incompatible manifest.

The existing SFT tokenization, target-only loss, launcher syntax, and resume tests remain required regression coverage.

## Acceptance Criteria

The design is implemented successfully when a check-only or dry-run inspection proves:

- reduced source counts are exactly `400/400/300`;
- train counts are exactly `380/380/285`;
- validation counts are exactly `20/20/15`;
- selected qids and splits are deterministic;
- effective Trainer arguments show step evaluation and saving every 200 optimizer steps;
- early stopping uses `eval_loss`, patience 3, threshold `0.001`, and lower-is-better semantics;
- a synthetic regression proves termination and best-checkpoint restoration;
- full resume preserves early-stopping state;
- no RL files or behavior are changed.
