# SFT BF16 and FlashAttention 2 - Task 3 Report

## Status

**PASS**

The bounded BF16 FlashAttention 2 forward/backward smoke passed on an idle-of-compute-work GPU, and every repository verification command required by Task 3 exited 0. This task did not load the 7B model, start or stop training, modify production code or dependencies, or create a commit.

## Inputs and constraints reviewed

Before execution, the complete Global Constraints and Task 3 sections of `docs/superpowers/plans/2026-08-26-sft-bf16-flash-attention.md`, the complete design document, and the Task 1 and Task 2 reports were reviewed.

The following constraints were maintained:

- All Python commands used `/data/conda/envs/macorag/bin/python`.
- No PyTorch, Transformers, CUDA, FlashAttention, or unrelated dependency was installed, upgraded, removed, or replaced.
- No complete SFT job was started, stopped, resumed, or benchmarked.
- No 7B model was loaded. The SFT entrypoint was invoked only with `--check-only`.
- Existing uncommitted changes were preserved.
- No commit was created.

## GPU availability decision

The pre-smoke `nvidia-smi` snapshot at `2026-08-26 18:34:59` showed:

- GPU 0: NVIDIA GeForce RTX 4090, 0% utilization, 16 MiB / 24564 MiB, with no compute process and no training process. Its only listed allocation was the 4 MiB Xorg display baseline.
- GPU 1: NVIDIA GeForce RTX 4090, 22% utilization, 1265 MiB / 24564 MiB, with multiple graphical processes.

GPU 0 was selected and fixed with `CUDA_VISIBLE_DEVICES=0`; GPU 1 was not used. The selection criterion was absence of any compute/training process and 0% utilization, so the smoke did not contend with another training run.

At the post-task read-only check at `18:38:20`, a new external process, PID `3506240` using `/home/being/anaconda3/bin/python`, had appeared on GPU 0 and was using about 16-18 GiB. This process appeared after the 3.11-second smoke had exited and is not the task's `/data/conda/envs/macorag/bin/python` process. It was not interrupted, stopped, or otherwise touched.

## BF16 FlashAttention kernel smoke

The exact planned tensor shape `[2, 128, 8, 64]`, BF16 dtype, and causal mode were used:

```python
q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
out = flash_attn_func(q, k, v, causal=True)
out.float().square().mean().backward()
```

The first attempt inside the restricted execution sandbox exited 1 at `assert torch.cuda.is_available()` because that execution context hid CUDA even though host `nvidia-smi` was healthy. The identical bounded command was then run with host GPU access, still pinned to GPU 0. This is recorded as an execution-boundary observation, not a kernel failure.

Host smoke result: exit 0.

```text
gpu_name NVIDIA GeForce RTX 4090
compute_capability 8.9
bf16_supported True
shape (2, 128, 8, 64)
dtype torch.bfloat16
output_finite True
q_grad_present True
q_grad_shape (2, 128, 8, 64)
q_grad_finite True
k_grad_present True
k_grad_shape (2, 128, 8, 64)
k_grad_finite True
v_grad_present True
v_grad_shape (2, 128, 8, 64)
v_grad_finite True
smoke PASS
```

This proves a real `flash_attn_func(..., causal=True)` BF16 forward and backward kernel execution on the selected RTX 4090. It checks output shape, output finiteness, q/k/v gradient presence, gradient shape, and gradient finiteness.

## Final repository verification

### Related pytest suite

Command:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_sft_training.py \
  tests/test_sft_data_generation.py \
  tests/test_prompt_config.py \
  tests/test_retraining_launchers.py
```

Result: exit 0.

```text
89 passed, 1 warning in 7.09s
```

The sole warning was `torch.cuda` reporting `Can't initialize NVML` inside the restricted pytest execution context. It did not fail a test; host CUDA functionality was independently proven by the successful kernel smoke.

### Python compilation

```bash
/data/conda/envs/macorag/bin/python -m compileall -q src/sft_training
```

Result: exit 0, no output.

### Launcher syntax

```bash
bash -n scripts/run_train_sft.sh
```

Result: exit 0, no output.

### Check-only data/config validation

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python \
  -m sft_training.train_sft_lora_macorag \
  --config config/train_sft.yml \
  --check-only \
  --check-only-max-samples 2
```

Result: exit 0. The command loaded and validated the SFT data contract, printed two masked sample records, and reported:

```text
Loaded 20445 SFT action records from data/sft/teacher_qwen_plus_trajectory_train_v2
Loaded 3000 original trajectory samples from data/sft/teacher_qwen_plus_trajectory_train_v2
Original sample counts: {'hotpotqa': 1000, '2wiki': 1000, 'musique': 1000}
Record counts: {'hotpotqa': 5544, '2wiki': 7584, 'musique': 7317}
Action counts: {'query_retriever': 6815, 'evidence_update': 6815, 'answer': 6815}
```

No tokenizer or 7B model loading occurred in this check-only path.

### Diff whitespace validation

```bash
git diff --check
```

Result: exit 0, no output.

## Diff and ownership boundary

The working tree was already substantially dirty before Task 3, including many unrelated modified and untracked files. Task 1's acceleration implementation is present in:

- `config/train_sft.yml`
- `src/sft_training/config.py`
- `src/sft_training/train_sft_lora_macorag.py`
- `tests/test_sft_training.py`

Task 2's package pins are present in:

- `requirements.txt`
- `environment.yml`

Task 3 made no production-code, configuration, test, launcher, or dependency changes. Its only filesystem change is this report. Existing unrelated changes cannot be attributed to Tasks 1-3 merely from the aggregate repository diff, and none were altered or cleaned up here.

## Validation boundary

This task establishes that the installed FlashAttention 2 extension can execute the planned small BF16 causal forward/backward kernel on the RTX 4090 and that the specified repository checks pass. It does not establish end-to-end SFT startup, 7B model compatibility under full memory load, distributed behavior, training stability, throughput improvement, or speedup. Those claims require a separately authorized comparable training run or benchmark.
