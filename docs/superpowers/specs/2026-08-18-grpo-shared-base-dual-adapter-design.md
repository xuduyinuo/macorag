# GRPO Shared Base Dual Adapter Design

## Goal

Reduce GRPO training-side GPU memory by replacing the two separately loaded Qwen base models with one shared base carrying two LoRA adapters, without changing policy optimization, KL reference semantics, vLLM rollout synchronization, or saved adapter compatibility.

## Architecture

- Load `model_path` exactly once.
- Load the SFT adapter as trainable adapter `default`; this remains the GRPO policy and the only optimizer/vLLM synchronization target.
- Load the same SFT adapter a second time as frozen adapter `reference`; this remains fixed for KL scoring.
- Return the same PEFT model object as both policy and reference handles. Reference scoring temporarily activates `reference`, switches the model to evaluation mode, runs under `torch.no_grad()`, then restores `default` and the previous training mode.
- Re-freeze all `reference` parameters after every adapter switch because PEFT `set_adapter()` may make the activated adapter trainable.

## Training and synchronization invariants

- The base weights and `reference` adapter never require gradients.
- `default` is active before policy forward, optimizer construction, vLLM hot synchronization, and checkpoint saving.
- Reference scoring is sequential with policy scoring; concurrent forwards on the shared model are unsupported.
- LoRA hot sync continues to expose only trainable `.default.` tensors, so vLLM receives only the current policy adapter.
- Dense sync, when selected, merges only the active `default` adapter.

## Saving

Intermediate and final checkpoints call PEFT `save_pretrained(..., selected_adapters=["default"])`. The output remains a directly loadable single LoRA adapter at the existing checkpoint/adapter paths and does not include the frozen `reference` copy.

## Gradient checkpointing

The existing `gradient_checkpointing` option controls both k-bit preparation and the explicit PEFT checkpointing enable call. It defaults to enabled. Reference scoring runs in evaluation/no-gradient mode, so checkpoint recomputation is not used there.

## Failure handling

- Fail early when the loaded PEFT object lacks `load_adapter()` or `set_adapter()`.
- Restore the policy adapter and original model mode in a `finally` block if reference scoring raises.
- Before synchronization or saving, explicitly restore and validate policy adapter trainability.

## Verification

- Unit-test single base loading, dual adapter names, trainability, and effective gradient-checkpointing toggle.
- Unit-test reference context switching, mode restoration, exception restoration, and reference re-freezing.
- Unit-test reference-before-policy forward ordering with the same model object.
- Unit-test policy-only saving and policy-only vLLM tensor collection.
- Run the RL training test module and the broader test suite while excluding the known unrelated single-GPU script fixture failure.
