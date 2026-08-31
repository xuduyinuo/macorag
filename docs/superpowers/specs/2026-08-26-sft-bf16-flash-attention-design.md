# SFT BF16 and FlashAttention 2 Design

## Goal

Enable BF16 mixed-precision training and FlashAttention 2 for the Qwen2.5-7B SFT path in the `macorag` conda environment, with explicit configuration, fail-fast dependency checks, and GPU smoke verification.

## Scope

- Set the active SFT configuration to BF16 and disable FP16.
- Add an explicit `attn_implementation` SFT setting whose active value is `flash_attention_2`.
- Pass the selected attention implementation to `AutoModelForCausalLM.from_pretrained`.
- Install a Torch 2.6 / CUDA 12.4 compatible `flash-attn` build in the `macorag` environment.
- Validate parsing, model-loading kwargs, dependency availability, BF16 device support, and a short GPU forward/backward smoke.
- Do not start, stop, or resume the full SFT job.

## Configuration Contract

The active YAML will contain:

```yaml
bf16: true
fp16: false
attn_implementation: "flash_attention_2"
```

The CLI/config parser accepts `attn_implementation`. The setting is included in the run manifest so that an incompatible resume is rejected rather than silently changing attention kernels.

## Runtime Behavior

Model loading passes `attn_implementation` through the standard Transformers `from_pretrained` interface. When `flash_attention_2` is selected, startup performs two checks before loading the full model:

1. `flash_attn` must be importable in the selected Python environment.
2. CUDA must be available and the selected device must report BF16 support when BF16 is enabled.

Failure produces a clear `SystemExit` message. There is no silent fallback to SDPA because that would make the requested performance optimization unverifiable.

## Installation Strategy

Use `/data/conda/envs/macorag/bin/python` for all package and verification commands. Prefer a compatible binary wheel. If no compatible wheel is available, build `flash-attn` against the environment's existing Torch 2.6 and CUDA 12.4 toolchain with build isolation disabled. Do not upgrade or replace PyTorch, CUDA, Transformers, or unrelated dependencies.

## Test Strategy

Follow red-green testing:

- Parser test for the new attention setting and active BF16/FP16 values.
- Model-kwargs test proving `flash_attention_2` reaches model loading.
- Runtime validation tests for missing `flash_attn`, missing CUDA, and unsupported BF16.
- Existing SFT regression suite, compile check, shell syntax check, check-only, and diff check.
- Host GPU smoke proving `flash_attn` imports and a small BF16 FlashAttention forward/backward succeeds.

The GPU smoke is a bounded validation, not an end-to-end throughput claim. Actual speedup must be measured in a later comparable SFT run.

## Rollback

Set `attn_implementation: "sdpa"` to return to the existing attention backend. Set `bf16: false` to disable Trainer BF16 autocast. Package removal is not required for rollback because an installed but unselected `flash_attn` does not change model behavior.
