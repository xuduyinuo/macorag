# vLLM LoRA Hot Sync for MACORAG GRPO

## Context

The current GRPO + vLLM training path works, but the cost profile is dominated by weight synchronization. In
`outputs/grpo_qwen2.5-7b_20260630_235405/train_metrics.jsonl`, each sample spends about 55 seconds in
`time_weight_sync_seconds`, roughly 59% of total runtime. The current implementation trains only LoRA, but syncs by:

1. calling `merge_adapter()` on the trainer model,
2. dequantizing merged 4-bit base weights,
3. sending dense base-model tensors to TRL's `update_named_param`,
4. calling `unmerge_adapter()`.

This is correct but expensive because it updates large 7B base weights instead of the small trainable LoRA adapter.

## Goal

Replace dense base-weight vLLM synchronization during GRPO with LoRA-only hot synchronization:

- Trainer continues to train only LoRA adapter weights.
- vLLM continues to generate rollouts on GPU 0.
- After optimizer updates, trainer sends only LoRA A/B tensors to the vLLM server.
- The vLLM server updates the active LoRA adapter in GPU memory.
- The next rollout uses the updated LoRA adapter without reloading the base model or merging full dense weights.

The success criterion is that `time_weight_sync_seconds` drops substantially from the current ~55 seconds per sample
while generated rollouts still use the latest policy LoRA.

## Non-Goals

- Do not replace GRPO with PPO in this change.
- Do not change reward shaping, retrieval logic, SFT adapters, or rollout XML/JSON parsing behavior.
- Do not require re-downloading the base model.
- Do not remove the existing dense sync path until LoRA hot sync has a working fallback.

## Architecture

Add a project-owned server instead of invoking TRL's stock CLI directly:

- `src/rl_training/vllm_lora_server.py`
- `scripts/run_grpo_vllm_lora_server.sh`

The server will start from TRL 0.18.2's `trl.scripts.vllm_serve` structure, but adds LoRA-specific behavior:

- initialize vLLM with LoRA support enabled,
- load or create a fixed adapter identity such as `macorag_train`,
- use `LoRARequest("macorag_train", 1, adapter_path)` for all `/generate/` requests,
- expose a LoRA update endpoint for trainer-side hot synchronization.

The existing `scripts/run_grpo_vllm_server.sh` can remain as the dense-sync fallback.

## Trainer Data Flow

The trainer will gain a LoRA sync mode:

```yaml
vllm_sync_mode: "lora"
vllm_lora_name: "macorag_train"
vllm_lora_int_id: 1
vllm_lora_adapter_path: "outputs/lora_qwen2.5-7b_trajectory_20260627_203027/adapter"
```

When `vllm_sync_mode: "lora"`:

1. collect only trainable PEFT LoRA tensors from `raw_policy_model`,
2. map PEFT parameter names to vLLM LoRA module names,
3. send tensor metadata to the server,
4. transfer tensor data to the server,
5. server updates the active LoRA adapter tensors in memory,
6. trainer records sync timing and count of LoRA tensors.

The current dense sync path remains available behind:

```yaml
vllm_sync_mode: "dense"
```

## Server Update Strategy

The first implementation should use the least invasive reliable path:

1. Keep TRL's existing NCCL communicator for tensor transfer if possible.
2. Add a new endpoint such as `/update_lora_param/`.
3. Server receives `name`, `dtype`, and `shape`.
4. Server allocates the incoming tensor on its vLLM device.
5. Server broadcasts from trainer to server through NCCL.
6. Server writes the received tensor into the active LoRA adapter parameter.

If vLLM's LoRA parameter objects are not directly addressable in a stable way, the implementation should stop and
fall back to a reload-based LoRA adapter path rather than silently reverting to dense base-weight sync.

## Name Mapping

The trainer PEFT names are expected to look like:

```text
base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
```

The server should normalize names into the representation used by vLLM's LoRA manager. The implementation must include
tests for the Qwen2.5 LoRA target modules used by the existing adapter:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

The mapping must preserve layer index, adapter side (`lora_A` or `lora_B`), tensor shape, dtype, and scaling behavior.

## Error Handling

The trainer should fail fast with actionable messages for:

- server does not expose LoRA hot-sync endpoints,
- server model path does not match trainer model path,
- server adapter name or int id does not match trainer config,
- LoRA tensor name cannot be mapped,
- LoRA tensor shape mismatches server-side adapter tensor,
- server confirms update but generation endpoint is not using the LoRA request.

For shape mismatch errors, logs must include:

- trainer parameter name,
- mapped server name,
- trainer tensor shape,
- server expected shape if available.

## Compatibility

Keep the existing vLLM + TRL environment:

- `vllm==0.8.5.post1`
- `trl==0.18.2`
- `torch==2.6.0+cu124`

Do not install incompatible packages that downgrade `transformers` below the vLLM/TRL requirements.

The implementation may copy selected logic from TRL's installed `vllm_serve.py` into the project, but must keep it
local to `src/rl_training/` so the conda package install is not patched in place.

## Testing Plan

Unit tests:

- config parsing for `vllm_sync_mode`, `vllm_lora_name`, `vllm_lora_int_id`, and adapter path,
- LoRA trainable tensor collection excludes frozen base weights,
- PEFT-to-server LoRA name mapping for Qwen2.5 modules,
- trainer uses LoRA sync client when `vllm_sync_mode: "lora"`,
- dense sync remains available as fallback,
- server request parsing validates LoRA tensor metadata.

Integration smoke tests:

- server starts with the configured base model and LoRA adapter path,
- `/health/` passes,
- `/generate/` uses the configured LoRA request,
- one LoRA tensor update round-trip succeeds,
- a one-sample GRPO run records much lower `time_weight_sync_seconds` than dense sync.

## Rollout Plan

1. Add config and tests while defaulting to dense sync until the server path exists.
2. Add custom LoRA server launcher and health endpoints.
3. Add trainer LoRA sync client and tensor mapping.
4. Add server-side in-memory LoRA tensor replacement.
5. Run one-sample smoke test on GPU 0/1.
6. Switch `config/train_grpo.yml` to `vllm_sync_mode: "lora"` only after smoke verification.

## Risks

The main risk is vLLM 0.8.5's internal LoRA manager not exposing a stable direct parameter update path. If that happens,
the implementation should explicitly report the blocker and use a reload-based LoRA adapter fallback for measurement,
not silently keep dense sync.

The second risk is LoRA adapter naming mismatch between PEFT and vLLM. This must be tested against the actual local
Qwen2.5 adapter in `outputs/lora_qwen2.5-7b_trajectory_20260627_203027/adapter`.

The third risk is generation accidentally using the base model without the active LoRA. The server must always pass a
`LoRARequest` into `LLM.generate()` in LoRA sync mode.
