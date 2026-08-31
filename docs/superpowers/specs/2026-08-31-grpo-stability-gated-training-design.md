# GRPO Stability-Gated Training Design

## Goal

Repair GRPO old/current policy consistency, gradient clipping, and learning-rate scheduling; establish a deterministic 300-step stability gate and a fixed 300-example validation comparison against the initializing SFT adapter; then permit continuation to 1000 total steps only when stability, quality, and protocol requirements pass.

## Scope

This design changes only the repository-native GRPO training, checkpoint, staged-run, fixed-validation, and gate-report paths. It does not launch the full 300-step or 1000-step experiment during implementation, change retrieval semantics, redesign rewards, or run ablations before the main 1000-step experiment passes.

The main run uses the existing Qwen2.5-7B SFT adapter, the existing stratified-v2 RL source, and the existing stratified-v2 evaluation source and retrieval indexes.

## Policy and Log-Probability Contract

vLLM generates completion tokens with the latest hot-synchronized LoRA adapter. Its returned token log-probabilities are retained only as diagnostic `server_logprobs`; they are never used in the GRPO importance ratio.

After a rollout group is complete and before any backward or optimizer operation, the HF policy performs a batched, no-gradient rescore of every generated action. These detached token log-probabilities replace `action.old_logprobs` and are the only behavior-policy probabilities consumed by the GRPO loss.

Both behavior rescore and gradient-bearing current-policy forward run with the policy in evaluation mode. Evaluation mode disables the SFT adapter's LoRA dropout while preserving autograd, so two forwards with unchanged parameters represent the same deterministic policy. Reference-adapter scoring remains no-gradient evaluation.

The ratio is therefore:

```
ratio = exp(hf_current_logprob - hf_old_logprob)
```

The trainer records server-versus-HF log-probability MAE and maximum absolute difference, pre-update log-ratio mean and maximum absolute difference, and ratio mean and P95. Any completion-token length mismatch or non-finite HF log-probability stops training before backward.

Both model backends explicitly load the same BF16 source checkpoint:

```yaml
bf16: true
fp16: false
load_4bit: false
vllm_dtype: "bfloat16"
```

Different HF and vLLM kernels may still produce diagnostic differences, but those differences do not enter the optimization ratio.

## Gradient and Scheduler Contract

The trainer clips trainable-policy gradients with `max_grad_norm: 1.0` immediately before each valid optimizer update. It logs gradient norms before and after clipping and whether clipping occurred. Existing non-finite handling remains fail-safe: clear gradients, skip optimizer and scheduler updates, do not synchronize vLLM, and retain a recoverable checkpoint boundary.

The scheduler is cosine decay with a 3% warmup and a minimum learning-rate ratio of 10%. With an initial learning rate of `1e-5`, the minimum is `1e-6`. Scheduler time is measured in successful optimizer updates, not consumed samples. Zero-advantage skips, non-finite-gradient skips, and accumulation-only samples do not advance it.

Checkpoint state adds the scheduler state, successful optimizer-update count, scheduler total-update count, warmup-update count, maximum gradient norm, log-probability source version, and precision contract. Full restore validates all training-semantic fields before accepting optimizer and scheduler state.

## Deterministic Staged Training

The trainer selects exactly 1000 training examples at run initialization with the current fixed-seed proportional-stratified and cross-dataset balancing logic. With three datasets this yields approximately 334, 333, and 333 examples. The selected qids, order, and dataset fingerprint remain identical across both stages.

`max_steps: 1000` defines the experiment and scheduler horizon. A new operational parameter, `run_until_step`, defines where the current process stops:

```yaml
max_total_samples: 1000
num_train_epochs: 1
max_steps: 1000
run_until_step: 300
```

The stability process consumes positions 1 through 300 and saves a complete `checkpoint-300`. After the gate passes, the continuation process restores that checkpoint, changes only `run_until_step` to 1000, and consumes positions 301 through 1000 without repetition or reselection.

`run_until_step` is excluded from the critical configuration fingerprint because it does not change training semantics. `max_steps`, scheduler configuration, data selection, seed, model precision, optimizer settings, and reward settings remain critical and cannot change across a full-state resume.

## Fixed Validation Set

A manifest builder selects 300 validation examples from `eval_1000_stratified_v2` using a fixed seed and the dataset's existing strata:

- 100 2Wiki examples
- 100 HotpotQA examples
- 100 MuSiQue examples

The manifest stores qid, dataset, stratum, source identity, and source fingerprint. The SFT baseline and every RL checkpoint use the same manifest, retrieval indexes, prompts, generation parameters, and metric implementation. Per-qid predictions are resumable, and each model/checkpoint has an isolated output directory.

The SFT adapter is evaluated once. RL evaluation runs at total steps 300, 500, 700, and 1000. The validation system refuses to compare outputs if the manifest, source fingerprint, prompt contract, generation contract, or retrieval contract differs.

## Stability and Expansion Gates

The 300-step stability gate passes only when all conditions hold:

- mean training `clip_fraction` is at most `0.001`;
- training `clip_fraction` P95 is at most `0.005`;
- mean KL over the final 50 steps is at most `0.1`;
- final-50 mean KL minus previous-50 mean KL is at most `max(0.01, previous_mean * 0.20)`;
- rollout `parse_failure_rate` is strictly below `0.01`;
- rollout `missing_answer_tag_rate` is strictly below `0.002`;
- every successful optimizer update has finite gradients.

A checkpoint is eligible for expanded training only when the stability gate passes and fixed-validation results also satisfy all conditions:

- overall macro F1 is strictly greater than the SFT baseline;
- each of 2Wiki, HotpotQA, and MuSiQue has F1 greater than or equal to its SFT baseline;
- validation `parse_failure_rate` is strictly below `0.01`;
- validation `missing_answer_tag_rate` is strictly below `0.002`.

The gate tool writes machine-readable `gate_report.json` and a concise human-readable summary. Exit code `0` means pass, `2` means complete results that fail a threshold, and `3` means missing results or contract/fingerprint mismatch. It never deletes checkpoints or automatically launches the next stage.

## Configuration and Ablations

Separate stage launchers or overlays represent the 300-step stability stage and the continuation to 1000 total steps, avoiding manual mutation of the canonical training configuration beyond the explicitly operational resume fields.

Ablations are enabled only after the main 1000-step run passes validation and protocol gates. The planned factors are KL coefficient, role reward weights, role-only advantage, and role-round advantage. Every ablation starts independently from the same SFT adapter and uses the same training qid order and validation manifest; no ablation inherits a main-run RL checkpoint.

## Failure Handling

Training fails before backward for behavior/current token misalignment, non-finite HF rescoring, invalid precision configuration, or incompatible resume state. A non-finite gradient follows the existing controlled skip path and cannot advance the optimizer, scheduler, or vLLM weights. Validation fails closed on incomplete predictions or contract drift. All failures preserve the latest complete safe checkpoint and diagnostic logs.

## Verification Strategy

Unit tests cover HF rescoring overriding vLLM probabilities, deterministic old/current behavior with LoRA dropout disabled, near-unit pre-update ratios, gradient clipping, scheduler advancement only after successful optimizer updates, and non-finite skip behavior.

Checkpoint tests cover scheduler and update-count round trips, permission to change only `run_until_step`, rejection of critical configuration changes, and exact continuation from sample 301. Data tests prove that the first 300 and remaining 700 positions are disjoint and jointly equal the fixed 1000-example selection.

Validation and gate tests cover deterministic stratified manifest construction, resumable predictions, threshold boundaries, incomplete output, fingerprint mismatch, aggregate F1 improvement, and per-dataset non-regression. Launcher tests cover dry-run output for stability, continuation, validation, and ablations.

After CPU tests, a small real-GPU smoke run verifies BF16 HF/vLLM loading, one rollout, batched HF rescore, backward, clipping, scheduler advancement, LoRA hot synchronization, and safe checkpoint creation. Implementation verification does not automatically launch the full 300-step or 1000-step experiment.
