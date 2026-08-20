# GRPO checkpoint-3600 resume and vLLM retry design

## Goal

Resume the interrupted 6,000-sample GRPO run from the policy weights saved at
`outputs/grpo_qwen2.5-7b/2026-08-18_23-11-03/checkpoint-3600`, without replaying
the first 3,600 shuffled samples, and prevent a transient vLLM generation
connection failure from terminating another long run.

## Resume semantics

The resume is an intentional warm start rather than a bit-for-bit continuation.
The existing checkpoint contains the policy LoRA and tokenizer but no optimizer
or random-number-generator state. The resumed process therefore initializes a
new AdamW optimizer and records that optimizer state was not restored.

The frozen reference adapter must continue to load from the original SFT
adapter configured by `sft_adapter_path`. Only the trainable policy adapter
loads from `resume_from_checkpoint`. Loading both adapters from checkpoint-3600
would reset the KL reference and change the training objective.

The data sequence is reconstructed with the existing deterministic selection
and epoch shuffle. For epoch 1 and seed 42, the resumed process skips exactly
the first 3,600 rank-local samples and starts with the original step 3,601.
It sets `global_step` to 3,600 so logging, weight synchronization, and checkpoint
names continue at their original step numbers. Samples 3,601 through 3,622 are
retrained because their updates are absent from checkpoint-3600.

Resume validation fails before model loading when any of these conditions is
invalid: the checkpoint directory is missing, the resume epoch is outside the
configured epoch range, the consumed-sample count is negative or exceeds the
rank-local epoch length, or the checkpoint adapter cannot be loaded. The run
metadata records the checkpoint path, resume epoch, consumed count, starting
global step, and `optimizer_state_restored: false`.

## vLLM generation retry

Retry applies only to `/generate/` transport failures represented by
`requests.exceptions.ConnectionError` or `requests.exceptions.Timeout`.
HTTP responses, malformed response payloads, model/server errors, and LoRA
weight-update endpoints are not retried.

Generation gets three total attempts by default. Before a retry, the client
closes the current Requests session and creates a fresh session so a stale
keep-alive socket is not reused. Delays use bounded exponential backoff starting
at one second: one second before attempt two and two seconds before attempt
three. After the final failure, the original exception propagates with no
silent sample skipping.

The retry count and initial backoff are normal GRPO configuration/CLI fields,
with defaults of three attempts and one second. Tests inject the sleep and
session creation dependencies so retry behavior is deterministic and fast.

## Launcher

Create `scripts/run_train_grpo_resume_3600.sh`. It reuses
`scripts/run_train_grpo.sh` and the normal YAML configuration while passing the
checkpoint-3600 resume arguments explicitly. The script uses strict Bash mode,
resolves paths relative to the repository, and supports environment overrides
for the config path and resume checkpoint. A new timestamped output directory
is created by the existing trainer; the interrupted directory is never
overwritten.

## Tests and verification

Unit tests cover deterministic skipping, distinct policy/reference adapter
paths, resume validation and metadata, retry success after a transient failure,
retry exhaustion, and non-retryable HTTP failures. A launcher test verifies the
checkpoint and position arguments without starting CUDA work. Verification runs
the focused RL tests, Bash syntax checking, Python compilation, and the existing
RL test suite.

