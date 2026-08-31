# SFT Throughput and Resume Design

## Goal

Reduce validation overhead in the MACORAG SFT pipeline without changing the training objective, make future runs safely resumable, and expose timing evidence that separates training, evaluation, and saving.

## Approved behavior

- Run validation once at the end of each epoch when a validation split exists.
- Shuffle the training dataset each epoch using the Trainer-compatible random/distributed sampler.
- Batch validation examples by nearby token length to reduce padding while evaluating every example exactly once.
- Accept an explicit SFT checkpoint path from YAML or CLI and pass it to `Trainer.train`.
- Record phase timing and globally reduced non-padding train-token throughput in a dedicated JSONL file. Label evaluation throughput as logical-dataset throughput.
- Do not stop or mutate the already-running SFT process. Changes apply only to later launches.

## Components

- `config/train_sft.yml` selects epoch validation and exposes an optional resume path.
- `src/sft_training/config.py` parses the validation strategy and resume path.
- `src/sft_training/trainer.py` owns train shuffling and length-grouped evaluation sampling.
- `src/sft_training/callbacks.py` records phase timing without mixing it into per-sample loss logs.
- `src/sft_training/train_sft_lora_macorag.py` validates arguments, wires callbacks, and resumes Trainer state.

## Error handling

- A resume path must contain adapter, optimizer, scheduler, Trainer, all-rank RNG, and applicable FP16 scaler state. New runs also persist an atomic compatibility manifest covering model, ordered tokenized train/eval fingerprints, split inputs, prompt, optimization, LoRA, precision, world size, and sampler contracts.
- Legacy checkpoints without a manifest preserve the historical sequential sampler; new checkpoints preserve and validate random sampling across resume.
- Resume repairs a truncated final JSONL record, removes metrics newer than the selected checkpoint, and appends an explicit resume-segment marker to every metric stream.
- Epoch validation requires a non-empty validation dataset but does not require `eval_steps`.
- Early stopping remains valid with epoch evaluation; best-model loading still follows the configured metric.
- Length-grouped evaluation reports a macro mean of each action record's target-token mean loss, so re-bucketing cannot change `eval_loss` by itself.
- Evaluation covers every logical example once; Accelerate owns any physical tail handling and rank sharding in distributed execution.

## Verification

- Unit tests prove argument parsing, epoch strategy, sampler behavior, timing output, and resume propagation.
- The focused SFT suite runs under `/data/conda/envs/macorag/bin/python`.
- Compile, shell syntax, config check-only, and diff whitespace checks form the final gate.
