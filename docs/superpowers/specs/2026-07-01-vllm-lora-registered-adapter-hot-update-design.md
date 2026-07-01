# vLLM LoRA Registered Adapter Hot Update Design

## Goal

Implement real LoRA-only hot synchronization between the GRPO trainer and the
project-local vLLM server for the current stack:

- vLLM 0.8.5.post1
- TRL 0.18.2 client protocol
- PEFT LoRA training on Qwen2.5-7B-Instruct

After each optimizer step, the trainer sends only trainable LoRA tensors to the
vLLM server. The server updates the in-memory LoRA adapter and refreshes the
active GPU LoRA slot so the next rollout uses the updated policy without dense
base-weight synchronization.

## Current Problem

The current branch has these pieces:

- vLLM generation through a project-local LoRA server.
- Trainer-side LoRA tensor collection and `/update_lora_param/` calls.
- Startup validation that refuses `vllm_sync_mode: "lora"` because the server
  currently returns HTTP 501 for LoRA parameter updates.

Dense sync still takes about 53-55 seconds per optimizer step because it
updates merged/base model weights through vLLM's dense `load_weights` path.
That is not the intended LoRA-only hot update.

## Recommended Approach

Use vLLM's internal LoRA adapter lifecycle instead of directly writing
`lora_a_stacked` and `lora_b_stacked`.

The server worker will:

1. Receive one PEFT-style LoRA tensor name and metadata through
   `/update_lora_param/`.
2. Allocate a tensor on the worker device.
3. Receive the real tensor from the trainer through the existing NCCL
   communicator.
4. Map the PEFT name to a vLLM LoRA module and side:
   - `model.layers.N.self_attn.q_proj.lora_A.weight`
   - `model.layers.N.self_attn.q_proj.lora_B.weight`
   - and the corresponding MLP projection names.
5. Find the registered vLLM LoRA model at:
   `self.model_runner.lora_manager._adapter_manager._registered_adapters[lora_int_id]`.
6. Update that registered LoRA layer weight after strict shape validation.
   vLLM stores LoRA weights transposed compared with PEFT, so the incoming
   tensor is copied as `incoming.T`.
7. If the adapter is active on a GPU slot, deactivate and reactivate it. This
   reuses vLLM's own `activate_adapter()` and `module.set_lora()` logic to copy
   the registered adapter into the active GPU stacked buffers.

This avoids manually handling packed QKV buffers, tensor-parallel slicing, and
other vLLM-specific GPU buffer layouts.

## Server Contract

`POST /update_lora_param/` keeps the existing request body:

```json
{
  "name": "model.layers.0.self_attn.q_proj.lora_A.weight",
  "dtype": "torch.float16",
  "shape": [64, 3584]
}
```

The endpoint calls:

```python
llm.collective_rpc(
    method="update_lora_param",
    args=(name, dtype, shape, args.lora_int_id),
)
```

The worker extension returns normally only after:

- NCCL broadcast succeeds,
- registered LoRA tensor shape is validated and updated,
- active GPU adapter slot is refreshed if needed.

If any step fails, the endpoint returns a non-200 error and the trainer does
not broadcast the next tensor as a fake success.

`GET /health/` reports:

```json
{
  "sync_mode": "lora",
  "supports_lora_param_update": true,
  "lora_name": "macorag_train",
  "lora_int_id": 1,
  "lora_adapter_path": "..."
}
```

## Trainer Contract

The trainer keeps sending only trainable LoRA tensors through
`VLLMGenerationClient.sync_lora_parameters()`.

`collect_lora_named_tensors()` must continue to:

- include only `requires_grad=True` LoRA A/B parameters,
- preserve dtype,
- move tensors only to the communicator device.

`vllm_sync_mode: "lora"` should validate server identity and update capability
before rollout. Once the server supports updates, validation should pass.

## Shape And Name Rules

Supported incoming tensor names are the normalized PEFT names already produced
by `normalize_peft_lora_name()`:

- `model.layers.<N>.self_attn.q_proj.lora_A.weight`
- `model.layers.<N>.self_attn.q_proj.lora_B.weight`
- `model.layers.<N>.self_attn.k_proj.lora_A.weight`
- `model.layers.<N>.self_attn.k_proj.lora_B.weight`
- `model.layers.<N>.self_attn.v_proj.lora_A.weight`
- `model.layers.<N>.self_attn.v_proj.lora_B.weight`
- `model.layers.<N>.self_attn.o_proj.lora_A.weight`
- `model.layers.<N>.self_attn.o_proj.lora_B.weight`
- `model.layers.<N>.mlp.gate_proj.lora_A.weight`
- `model.layers.<N>.mlp.gate_proj.lora_B.weight`
- `model.layers.<N>.mlp.up_proj.lora_A.weight`
- `model.layers.<N>.mlp.up_proj.lora_B.weight`
- `model.layers.<N>.mlp.down_proj.lora_A.weight`
- `model.layers.<N>.mlp.down_proj.lora_B.weight`

The server strips `.lora_A.weight` or `.lora_B.weight` to get the vLLM module
name, then looks up `lora_model.loras[module_name]`.

Validation:

- for `lora_A`, registered shape must equal `incoming.T.shape`,
- for `lora_B`, registered shape must equal `incoming.T.shape`,
- unsupported names fail fast,
- missing registered adapters fail fast,
- missing modules fail fast.

## Testing

Unit tests must not load the real 7B model. They should cover:

- PEFT name parsing into module name and side.
- Registered adapter tensor update with transpose and dtype/device conversion.
- Shape mismatch error.
- Active adapter refresh calls deactivate then activate.
- `/health/` advertises update support.
- `/update_lora_param/` calls `collective_rpc()` with `lora_int_id`.
- Client validation passes when health says update support is true.
- `vllm_sync_mode: "lora"` reaches `sync_lora_parameters()`.

Smoke verification after unit tests:

1. Start `scripts/run_grpo_vllm_lora_server.sh` on GPU0.
2. Run one-sample GRPO with `--vllm-sync-mode lora --max-samples 1 --max-steps 1`.
3. Confirm `time_weight_sync_seconds` is far below dense sync's 53-55 seconds.

## Risks

This approach uses vLLM internal fields:

- `model_runner.lora_manager`
- `_adapter_manager`
- `_registered_adapters`
- `_active_adapters`
- `activate_adapter()`
- `_deactivate_adapter()`

These are stable enough for the pinned vLLM 0.8.5.post1 environment, but they
are not public API. Future vLLM upgrades may require adaptation.

The design intentionally avoids direct writes to active GPU stacked buffers
because that path is more fragile under packed modules and tensor parallelism.

## Rollback

If hot update fails in smoke testing, keep:

```yaml
vllm_sync_mode: "dense"
```

and keep the explicit server error. Do not ship a mode that reports success
while using stale vLLM adapter weights.
