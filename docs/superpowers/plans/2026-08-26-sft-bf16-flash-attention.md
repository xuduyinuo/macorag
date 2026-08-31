# SFT BF16 and FlashAttention 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable strict BF16 and FlashAttention 2 execution for the active Qwen2.5-7B SFT path in the `macorag` conda environment.

**Architecture:** Extend the existing YAML/CLI runtime contract with an explicit attention backend, validate accelerator prerequisites before model loading, and pass the backend through Transformers model kwargs. Pin and install the official `flash-attn` package without changing the existing Torch/CUDA/Transformers versions, then prove the kernel with a bounded GPU smoke.

**Tech Stack:** Python 3.9, PyTorch 2.6.0+cu124, CUDA toolkit 12.9 for extension compilation, Transformers 4.57.6, flash-attn 2.8.3.post1, pytest.

## Global Constraints

- Use `/data/conda/envs/macorag/bin/python` for all Python and package commands.
- Do not upgrade or replace PyTorch 2.6.0, Transformers 4.57.6, CUDA, or unrelated dependencies.
- Do not start, stop, or resume the complete SFT job.
- Preserve all pre-existing uncommitted changes.
- FlashAttention 2 selection is strict: missing prerequisites terminate startup instead of silently falling back to SDPA.

---

### Task 1: Configuration and strict runtime contract

**Files:**
- Modify: `config/train_sft.yml`
- Modify: `src/sft_training/config.py`
- Modify: `src/sft_training/train_sft_lora_macorag.py`
- Test: `tests/test_sft_training.py`

**Interfaces:**
- Consumes: YAML/CLI values `bf16: bool`, `fp16: bool`, and `attn_implementation: str`.
- Produces: `_validate_acceleration_runtime(args, torch, find_spec=importlib.util.find_spec) -> None` and `_model_kwargs(args, torch_dtype) -> dict[str, Any]` containing the selected attention backend.

- [x] **Step 1: Write failing parser and active-config tests**

Add assertions equivalent to:

```python
args = parse_args(["--config", "config/train_sft.yml"])
assert args.bf16 is True
assert args.fp16 is False
assert args.attn_implementation == "flash_attention_2"
```

Update the active-YAML key contract so `bf16`, `fp16`, and `attn_implementation` are required active keys rather than forbidden low-frequency defaults.

- [x] **Step 2: Write failing model-kwargs and prerequisite tests**

Add focused tests proving:

```python
args = SimpleNamespace(load_4bit=False, attn_implementation="flash_attention_2")
assert _model_kwargs(args, "bf16")["attn_implementation"] == "flash_attention_2"
```

and that `_validate_acceleration_runtime` raises clear `SystemExit` errors when `flash_attn` is absent, CUDA is unavailable, or BF16 is selected on a device without BF16 support.

- [x] **Step 3: Run focused tests and verify RED**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_sft_training.py -k 'attention or acceleration_runtime or active_sft_config'
```

Expected: failures because `attn_implementation` and `_validate_acceleration_runtime` do not exist and the active YAML omits BF16/FP16.

- [x] **Step 4: Implement the minimal parser and runtime validation**

Add `attn_implementation: "sdpa"` to `RUNTIME_DEFAULTS`, expose:

```python
parser.add_argument(
    "--attn-implementation",
    choices=("eager", "sdpa", "flash_attention_2"),
    default=defaults["attn_implementation"],
)
```

Implement validation with these exact rules:

```python
if args.bf16 and args.fp16:
    raise SystemExit("bf16 and fp16 cannot both be enabled.")
if args.attn_implementation == "flash_attention_2":
    if find_spec("flash_attn") is None:
        raise SystemExit("FlashAttention 2 requested but flash_attn is not installed in the active environment.")
    if not torch.cuda.is_available():
        raise SystemExit("FlashAttention 2 requested but CUDA is unavailable.")
if args.bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
    raise SystemExit("bf16 requested but the selected CUDA device does not support bf16.")
```

Call validation after training dependencies load and before tokenizer/model loading. Add `attn_implementation` to `_model_kwargs` and the SFT run manifest. Set the active YAML to BF16 true, FP16 false, and FlashAttention 2.

- [x] **Step 5: Run focused and full SFT tests and verify GREEN**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_sft_training.py
```

Expected: all tests pass.

### Task 2: Pin and install FlashAttention

**Files:**
- Modify: `requirements.txt`
- Modify: `environment.yml`

**Interfaces:**
- Consumes: PyTorch 2.6.0+cu124, Python 3.9, `/usr/local/cuda-12.9`, installed `packaging` and `ninja`.
- Produces: importable `flash_attn==2.8.3.post1` in `/data/conda/envs/macorag` and reproducible dependency manifests.

- [x] **Step 1: Record the dependency pin**

Add `flash-attn==2.8.3.post1` alongside the training stack in both dependency manifests.

- [x] **Step 2: Install without dependency re-resolution**

Run:

```bash
CUDA_HOME=/usr/local/cuda-12.9 MAX_JOBS=4 /data/conda/envs/macorag/bin/python -m pip install --no-build-isolation --no-deps flash-attn==2.8.3.post1
```

Expected: wheel installation or successful local extension build, without replacing Torch or Transformers.

- [x] **Step 3: Verify versions and dependency consistency**

Run:

```bash
/data/conda/envs/macorag/bin/python -c "import flash_attn, torch, transformers; print(flash_attn.__version__, torch.__version__, transformers.__version__)"
/data/conda/envs/macorag/bin/python -m pip check
```

Expected: `2.8.3.post1 2.6.0+cu124 4.57.6` and no broken requirements.

### Task 3: GPU kernel smoke and final regression

**Files:**
- Verify: all files modified in Tasks 1-2.

**Interfaces:**
- Consumes: installed `flash_attn.flash_attn_func`, CUDA-visible RTX 4090, BF16 tensors.
- Produces: fresh evidence that BF16 FlashAttention forward/backward runs and repository checks remain green.

- [x] **Step 1: Run a bounded BF16 FlashAttention GPU smoke**

On a free GPU, run a small causal forward/backward using tensors shaped `[2, 128, 8, 64]`:

```python
q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
out = flash_attn_func(q, k, v, causal=True)
out.float().square().mean().backward()
assert out.shape == q.shape
assert q.grad is not None
```

Expected: exit 0 and finite output/gradient.

- [x] **Step 2: Run final repository verification**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_sft_training.py tests/test_sft_data_generation.py tests/test_prompt_config.py tests/test_retraining_launchers.py
/data/conda/envs/macorag/bin/python -m compileall -q src/sft_training
bash -n scripts/run_train_sft.sh
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m sft_training.train_sft_lora_macorag --config config/train_sft.yml --check-only --check-only-max-samples 2
git diff --check
```

Expected: zero failures and zero non-warning exit codes.

- [x] **Step 3: Inspect the final diff and report validation boundaries**

Confirm only the authorized SFT acceleration files and documentation were added to this task's diff. Report the GPU smoke result separately from any end-to-end training-speed claim; no full SFT benchmark is authorized by this plan.
