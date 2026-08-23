# GRPO Full-State Checkpoint and Resume Design

## Goal and recovery guarantee

MACORAG GRPO checkpoints will resume from the latest completed optimizer boundary with the same policy LoRA, AdamW state, data cursor, random-number-generator states, and deterministic vLLM generation seeds. Recovery must not skip or replay any sample committed in that checkpoint. Work performed after the latest completed checkpoint may be repeated after an abrupt interruption.

The guarantee is semantic reproducibility on the same software and hardware configuration. Bit-for-bit equality across different CUDA, PyTorch, transformers, PEFT, or vLLM versions is outside scope.

## Full checkpoint format

Every new `checkpoint-<step>` contains:

- policy LoRA and tokenizer files;
- `optimizer.pt`, containing the AdamW state dictionary;
- `trainer_state.pt`, containing schema version, epoch, rank-local samples consumed in that epoch, global step, vLLM generation counter, gradient-accumulation boundary, world size, dataset fingerprint, and critical configuration fingerprint;
- one `rng_state_rank<N>.pt` per trainer rank, containing Python, NumPy, PyTorch CPU, and CUDA RNG states;
- `checkpoint_manifest.json`, containing the expected files and scalar recovery metadata;
- `COMPLETE`, written last.

The trainer writes into `.checkpoint-<step>.tmp`, synchronizes all ranks, validates the expected files, and atomically renames the temporary directory. A directory without `COMPLETE` is never resumable.

## Save boundary and retention

The periodic save request is honored only after `optimizer.step()` and `optimizer.zero_grad()`. This avoids serializing partial accumulated gradients. With the current `gradient_accumulation_steps: 1`, each completed sample is an optimizer boundary. If a configured `save_steps` lands inside a future multi-step accumulation window, saving is deferred until the next optimizer boundary.

After a complete checkpoint becomes visible, pruning retains:

- the newest `save_total_limit` checkpoints, default 3;
- every checkpoint whose step is divisible by `save_milestone_steps`, default 1000.

Pruning applies only to the current timestamped run directory and never deletes the resume source from a previous run.

## Resume behavior

For a full checkpoint, `--resume-from-checkpoint` is the only required recovery argument. The trainer reads epoch, consumed samples, global step, generation counter, and fingerprints from the checkpoint. It loads the policy LoRA before optimizer construction, then restores the optimizer and per-rank RNG state after all model/runtime initialization and immediately before the next rollout. The frozen reference adapter continues to load from the original SFT adapter.

Resume fails before training when the checkpoint is incomplete, a state file is missing, the world size or gradient-accumulation setting differs, the selected dataset/order fingerprint differs, or critical rollout/optimization configuration differs. Existing policy-only checkpoints remain available through the legacy explicit epoch/sample/global-step warm-start path and continue to record `optimizer_state_restored: false`.

## Deterministic vLLM generation

`VLLMSharedPolicy` owns a monotonically increasing generation counter. Each prompt receives a stable seed derived from the configured training seed and its counter value. The client sends one seed per prompt, and the LoRA server creates one `SamplingParams` instance per prompt. The counter is stored in `trainer_state.pt`; transport retries reuse the same seed payload. This makes the next rollout seed sequence recoverable without depending on vLLM process-global RNG state.

## Validation

Tests must prove:

- atomic full checkpoint contents and rejection of incomplete checkpoints;
- optimizer and Python/NumPy/PyTorch RNG round-trip;
- automatic cursor/global-step restoration and legacy warm-start compatibility;
- dataset/config/world-size mismatch rejection;
- retention of the newest three plus 1000-step milestones;
- checkpoint deferral outside optimizer boundaries;
- per-prompt vLLM seeds, retry seed stability, and generation-counter restoration;
- continuous training and interrupted-then-resumed toy training reach identical policy and optimizer tensors under a deterministic fake rollout backend.
