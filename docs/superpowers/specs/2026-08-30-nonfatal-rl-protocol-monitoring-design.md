# Nonfatal RL Protocol Monitoring and Recovery Checkpoints

## Problem

The run at `outputs/grpo_qwen2.5-7b-v2/2026-08-29_22-38-47`
completed 49 optimization steps and then intentionally exited after two
100-rollout protocol windows exceeded a hard-coded 2% parse-failure threshold.

The six logged failures were all valid `<answer>` JSON objects from three
MuSiQue questions. They violated the final-round semantic requirement by
returning `can_answer=false`; none was a missing tag or malformed JSON.
`compute_protocol_metrics()` currently treats every item in `parse_errors` as
a parse failure, even though `final_answer_required:*` is already represented
by `final_compliance_rate`.

The same training loop gates recovery checkpoints on `checkpoint_eligible`, a
strict model-quality signal. When quality is below the gate, an otherwise due
checkpoint is skipped and its pending state is cleared. The protocol
`SystemExit` then bypasses final adapter and metadata saving, leaving no
recoverable artifact for the completed 49 steps.

## Design

### Protocol metric classification

Classify errors beginning with `final_answer_required:` as final-round
semantic noncompliance, not parsing failures. They continue to lower
`final_compliance_rate`; malformed JSON, missing tags, invalid fields, and
other structural protocol errors continue to lower `parse_failure_rate`.

Do not rewrite final answers or coerce `can_answer`. Noncompliant trajectories
remain visible to reward and logging as model behavior.

### Nonfatal monitoring

Retain fixed-size, non-overlapping windows and consecutive-bad-window tracking,
but rename the terminal decision from `should_stop` to `should_warn`. When the
warning threshold is reached, training appends a `protocol_warning` event and
continues through optimization, vLLM synchronization, metrics logging, and
rollout persistence.

Remove the automatic protocol-quality `SystemExit`. Unexpected runtime and
invariant failures, exhausted vLLM retries, and CUDA errors remain fatal rather
than being silently swallowed. Preserve the existing explicit handling for
non-finite gradients: clear gradients, record
`skipped_update_reason=nonfinite_gradients`, and continue training.

### Recovery checkpoint separation

Treat full-state checkpoints solely as recovery artifacts. Whenever
`_checkpoint_save_decision()` says a checkpoint is due at an optimizer-safe
boundary, call `save_full_checkpoint()` regardless of protocol quality.
Likewise, save a deferred checkpoint at the final optimizer boundary without
consulting `checkpoint_eligible`.

Keep `checkpoint_eligible` in protocol metrics as an observational quality
field. It may be used later for model selection, but never for resumability.
Keep the existing atomic checkpoint layout, pruning, fingerprints, optimizer
state, RNG state, and generation-counter contract unchanged.

## Testing

Add tests proving that:

1. `final_answer_required:*` produces zero parse failures while lowering final
   compliance.
2. Two consecutive bad parsing windows return `should_warn=true` and never
   expose a stop decision.
3. The training control path records `protocol_warning` without raising.
4. A scheduled recovery checkpoint is not filtered by
   `checkpoint_eligible=false`.
5. Existing checkpoint deferral, full-state resume, prompt-budget, RAG, and RL
   regression tests remain valid.

Run focused deterministic unit tests, compilation, and diff checks. This
control-flow-only correction does not require another GPU smoke run; the
50-sample historical failure pattern is covered by unit tests rather than
repeating a 30-minute run.

## Scope and Limitations

This change prevents stochastic protocol-quality noise from intentionally
terminating RL training and prevents quality gates from suppressing recovery
checkpoints. It does not claim that hardware failures, CUDA OOM, corrupt data,
or an unavailable vLLM server can never interrupt a process.

The failed 49-step run has no adapter or full-state checkpoint and cannot be
resumed losslessly. A new run is required after the fix.
