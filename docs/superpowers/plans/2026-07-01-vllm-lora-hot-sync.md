# vLLM LoRA Hot Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GRPO dense base-weight vLLM synchronization with LoRA-only hot synchronization for the active vLLM rollout server.

**Architecture:** Add a project-owned vLLM LoRA server copied from TRL's `vllm_serve.py` shape, plus a trainer-side LoRA sync client. Keep the existing dense sync path as fallback while introducing `vllm_sync_mode: "lora"` and server endpoints that receive LoRA tensor metadata and update in-memory adapter tensors.

**Tech Stack:** Python 3.9, PyTorch 2.6.0+cu124, vLLM 0.8.5.post1, TRL 0.18.2, PEFT LoRA, pytest.

## Global Constraints

- Keep existing GRPO algorithm; do not implement PPO in this change.
- Preserve dense sync fallback behind `vllm_sync_mode: "dense"`.
- Do not patch installed conda packages in place; project code lives under `src/rl_training/`.
- Do not install packages that downgrade `transformers` below vLLM/TRL requirements.
- vLLM server uses GPU 0 by default; trainer uses YAML `gpu_indices`.
- Only train and hot-sync LoRA adapter weights.
- Fail fast if LoRA hot-sync endpoints are unavailable or tensor shapes do not match.

---

## File Structure

- Modify `src/rl_training/config.py`: add LoRA hot-sync config defaults and CLI args.
- Modify `config/train_grpo.yml`: add LoRA sync config while initially leaving dense fallback available.
- Modify `src/rl_training/vllm_client.py`: split dense sync from LoRA sync; add LoRA tensor collection and HTTP/NCCL client methods.
- Create `src/rl_training/vllm_lora_mapping.py`: PEFT parameter name normalization and LoRA metadata helpers.
- Create `src/rl_training/vllm_lora_server.py`: custom TRL/vLLM server with fixed `LoRARequest`, health endpoints, and LoRA update endpoint.
- Create `scripts/run_grpo_vllm_lora_server.sh`: launcher reading `config/train_grpo.yml`.
- Modify `src/rl_training/train_grpo_macorag.py`: choose dense or LoRA sync mode and validate the matching server.
- Modify `tests/test_rl_training.py`: unit tests for config, mapping, sync selection, and client behavior.
- Create `tests/test_vllm_lora_server.py`: server parser and LoRA update helper tests.

---

### Task 1: Add LoRA Sync Config and Mode Selection

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `config/train_grpo.yml`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: args fields `vllm_sync_mode: str`, `vllm_lora_name: str`, `vllm_lora_int_id: int`, `vllm_lora_adapter_path: str`.
- Consumes: existing `parse_args(argv)` config parser.

- [ ] **Step 1: Write failing config test**

Add to `tests/test_rl_training.py` near `test_parse_args_loads_vllm_generation_config`:

```python
def test_parse_args_loads_vllm_lora_sync_config(tmp_path: Path) -> None:
    config = tmp_path / "train_grpo.yml"
    config.write_text(
        "\n".join(
            [
                "use_vllm_generation: true",
                'vllm_sync_mode: "lora"',
                'vllm_lora_name: "macorag_train"',
                "vllm_lora_int_id: 7",
                'vllm_lora_adapter_path: "outputs/adapter"',
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.vllm_sync_mode == "lora"
    assert args.vllm_lora_name == "macorag_train"
    assert args.vllm_lora_int_id == 7
    assert args.vllm_lora_adapter_path == "outputs/adapter"
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_rl_training.py::test_parse_args_loads_vllm_lora_sync_config
```

Expected: FAIL because `Namespace` has no `vllm_sync_mode` field.

- [ ] **Step 3: Add defaults and CLI args**

In `src/rl_training/config.py`, add defaults near the existing vLLM keys:

```python
"vllm_sync_mode": "dense",
"vllm_lora_name": "macorag_train",
"vllm_lora_int_id": 1,
"vllm_lora_adapter_path": "",
```

Add parser args near existing vLLM parser entries:

```python
parser.add_argument("--vllm-sync-mode", choices=("dense", "lora"), default=defaults["vllm_sync_mode"])
parser.add_argument("--vllm-lora-name", default=defaults["vllm_lora_name"])
parser.add_argument("--vllm-lora-int-id", type=int, default=defaults["vllm_lora_int_id"])
parser.add_argument("--vllm-lora-adapter-path", default=defaults["vllm_lora_adapter_path"])
```

- [ ] **Step 4: Update YAML config**

In `config/train_grpo.yml`, add:

```yaml
vllm_sync_mode: "dense"
vllm_lora_name: "macorag_train"
vllm_lora_int_id: 1
vllm_lora_adapter_path: "outputs/lora_qwen2.5-7b_trajectory_20260627_203027/adapter"
```

Keep `vllm_sync_mode: "dense"` until the custom server smoke test passes.

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_rl_training.py::test_parse_args_loads_vllm_lora_sync_config tests/test_rl_training.py::test_parse_args_loads_vllm_generation_config
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rl_training/config.py config/train_grpo.yml tests/test_rl_training.py
git commit -m "feat: add grpo vllm lora sync config"
```

---

### Task 2: Add LoRA Tensor Collection and Name Mapping

**Files:**
- Create: `src/rl_training/vllm_lora_mapping.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `normalize_peft_lora_name(name: str) -> str | None`.
- Produces: `collect_lora_named_tensors(model: Any, device: Any | None = None) -> dict[str, Any]`.
- Consumes: PEFT `named_parameters()` output from `raw_policy_model`.
- Runtime sync integration is Task 3; Task 2 only creates and tests the mapping/collection interface.

- [ ] **Step 1: Write failing name mapping tests**

Add to `tests/test_rl_training.py`:

```python
def test_normalize_peft_lora_name_maps_qwen_modules() -> None:
    from rl_training.vllm_lora_mapping import normalize_peft_lora_name

    assert normalize_peft_lora_name(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    ) == "model.layers.0.self_attn.q_proj.lora_A.weight"
    assert normalize_peft_lora_name(
        "base_model.model.model.layers.31.mlp.down_proj.lora_B.default.weight"
    ) == "model.layers.31.mlp.down_proj.lora_B.weight"


def test_normalize_peft_lora_name_ignores_non_lora_weights() -> None:
    from rl_training.vllm_lora_mapping import normalize_peft_lora_name

    assert normalize_peft_lora_name("base_model.model.model.embed_tokens.weight") is None
    assert normalize_peft_lora_name("base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight") is None
```

- [ ] **Step 2: Verify mapping tests fail**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_rl_training.py::test_normalize_peft_lora_name_maps_qwen_modules tests/test_rl_training.py::test_normalize_peft_lora_name_ignores_non_lora_weights
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rl_training.vllm_lora_mapping'`.

- [ ] **Step 3: Implement mapping file**

Create `src/rl_training/vllm_lora_mapping.py`:

```python
from __future__ import annotations

import re
from typing import Any


_LORA_RE = re.compile(
    r"^(?:base_model\.model\.)?(?P<base>model\.layers\.\d+\.(?:self_attn|mlp)\."
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))\."
    r"(?P<side>lora_[AB])\.default\.weight$"
)


def normalize_peft_lora_name(name: str) -> str | None:
    match = _LORA_RE.match(name)
    if match is None:
        return None
    return f"{match.group('base')}.{match.group('side')}.weight"


def collect_lora_named_tensors(model: Any, *, device: Any | None = None) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        mapped = normalize_peft_lora_name(name)
        if mapped is None:
            continue
        tensor = parameter.detach().float()
        if device is not None:
            tensor = tensor.to(device=device)
        else:
            tensor = tensor.cpu()
        tensors[mapped] = tensor
    return tensors
```

- [ ] **Step 4: Write failing collection test**

Add to `tests/test_rl_training.py`:

```python
def test_collect_lora_named_tensors_maps_only_lora_params() -> None:
    from rl_training.vllm_lora_mapping import collect_lora_named_tensors

    model = _TinyPeftModel()

    tensors = collect_lora_named_tensors(model)

    assert sorted(tensors) == [
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    assert all(tensor.device.type == "cpu" for tensor in tensors.values())
```

Update `_TinyPeftModel` parameter names to include `base_model.model.model.layers...` so the fake matches the real Qwen2.5 adapter.

- [ ] **Step 5: Run tests**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_rl_training.py::test_normalize_peft_lora_name_maps_qwen_modules tests/test_rl_training.py::test_collect_lora_named_tensors_maps_only_lora_params
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rl_training/vllm_lora_mapping.py tests/test_rl_training.py
git commit -m "feat: add vllm lora tensor mapping"
```

---

### Task 3: Add LoRA Sync Client Path

**Files:**
- Modify: `src/rl_training/vllm_client.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `VLLMGenerationClient.sync_lora_parameters(model: Any) -> float`.
- Produces: `_sync_vllm_after_optimizer_step(policy, raw_policy_model, args) -> float` chooses dense or LoRA mode.
- Consumes: `collect_lora_named_tensors()`.
- Implements LoRA sync in the project wrapper by posting `/update_lora_param/` metadata and using TRL's existing
  `pynccl_comm.broadcast(...)`; do not require the installed TRL `VLLMClient` to expose `update_lora_param()`.

- [ ] **Step 1: Write failing client test**

Add fake backend fields in the test setup rather than adding an `update_lora_param()` method. The real TRL backend does
not expose that method, so the wrapper must use `backend.session`, `backend.base_url`, `backend.rank`, and
`backend.pynccl_comm`.

```python
class _FakeResponse:
    status_code = 200
    text = "ok"


class _FakeSession:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict) -> _FakeResponse:
        self.posts.append((url, json))
        return _FakeResponse()
```

Add test:

```python
def test_vllm_generation_client_syncs_lora_parameters_only() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    backend = _FakeTRLClient(sync_device=torch.device("meta"))
    backend.session = _FakeSession()
    backend.base_url = "http://127.0.0.1:8000"
    backend.rank = 1
    backend.pynccl_comm = type(
        "FakeCommunicator",
        (),
        {
            "device": torch.device("meta"),
            "broadcast": lambda self, tensor, src: backend.updated.append(("broadcast", tensor)),
            "group": type("Group", (), {"barrier": lambda self: None})(),
        },
    )()
    client = VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend)
    model = _TinyPeftModel()

    elapsed = client.sync_lora_parameters(model)

    assert elapsed >= 0.0
    assert backend.communicator_initialized is True
    assert [payload["name"] for _, payload in backend.session.posts] == [
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
    ]
    assert [name for name, _ in backend.updated] == ["broadcast", "broadcast"]
    assert all(tensor.device.type == "meta" for _, tensor in backend.updated)
```

- [ ] **Step 2: Verify test fails**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_rl_training.py::test_vllm_generation_client_syncs_lora_parameters_only
```

Expected: FAIL because `sync_lora_parameters` is missing.

- [ ] **Step 3: Implement client method**

In `src/rl_training/vllm_client.py`, import:

```python
from .vllm_lora_mapping import collect_lora_named_tensors
```

Add helper:

```python
def _ensure_communicator(self) -> None:
    if not self._communicator_initialized:
        initializer = getattr(self.backend, "init_communicator", None)
        if initializer is None:
            raise SystemExit("TRL VLLMClient is missing init_communicator(); hot sync is unavailable.")
        initializer()
        self._communicator_initialized = True
```

Refactor `sync_trainable_parameters()` to call `_ensure_communicator()`.

Add helpers and method:

```python
def _post_lora_param_metadata(backend: Any, name: str, tensor: Any) -> None:
    session = getattr(backend, "session", None)
    base_url = getattr(backend, "base_url", None)
    if session is None or base_url is None:
        raise SystemExit("Installed TRL VLLMClient internals are incompatible with LoRA hot sync.")
    response = session.post(
        f"{base_url}/update_lora_param/",
        json={"name": name, "dtype": str(tensor.dtype), "shape": tuple(tensor.shape)},
    )
    if response.status_code != 200:
        raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")


def _broadcast_lora_tensor(backend: Any, tensor: Any) -> None:
    communicator = getattr(backend, "pynccl_comm", None)
    rank = getattr(backend, "rank", None)
    if communicator is None or rank is None:
        raise SystemExit("vLLM LoRA hot sync communicator is not initialized.")
    communicator.broadcast(tensor, src=rank)
    communicator.group.barrier()


def sync_lora_parameters(self, model: Any) -> float:
    self._ensure_communicator()
    sync_device = _backend_communicator_device(self.backend)
    start = time.perf_counter()
    tensors = collect_lora_named_tensors(model, device=sync_device)
    if not tensors:
        raise SystemExit("No LoRA tensors found for vLLM LoRA hot sync.")
    for name, tensor in tensors.items():
        _post_lora_param_metadata(self.backend, name, tensor)
        _broadcast_lora_tensor(self.backend, tensor)
    return time.perf_counter() - start
```

- [ ] **Step 4: Write failing trainer selection test**

Add to `tests/test_rl_training.py`:

```python
class _FakePolicyWithLoraClient:
    def __init__(self) -> None:
        self.vllm_client = type(
            "Client",
            (),
            {
                "dense_called": False,
                "lora_called": False,
                "sync_trainable_parameters": lambda self_client, model: setattr(self_client, "dense_called", True) or 1.0,
                "sync_lora_parameters": lambda self_client, model: setattr(self_client, "lora_called", True) or 2.0,
            },
        )()


def test_sync_vllm_after_optimizer_step_uses_lora_mode() -> None:
    policy = _FakePolicyWithLoraClient()
    args = Namespace(use_vllm_generation=True, vllm_sync_after_step=True, vllm_sync_mode="lora")

    elapsed = _sync_vllm_after_optimizer_step(policy, object(), args)

    assert elapsed == 2.0
    assert policy.vllm_client.lora_called is True
    assert policy.vllm_client.dense_called is False
```

- [ ] **Step 5: Implement trainer selection**

In `_sync_vllm_after_optimizer_step()`:

```python
sync_mode = getattr(args, "vllm_sync_mode", "dense")
if sync_mode == "lora":
    return float(client.sync_lora_parameters(raw_policy_model))
if sync_mode == "dense":
    return float(client.sync_trainable_parameters(raw_policy_model))
raise SystemExit(f"Unsupported vLLM sync mode: {sync_mode}")
```

- [ ] **Step 6: Run tests**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_rl_training.py::test_vllm_generation_client_syncs_lora_parameters_only tests/test_rl_training.py::test_sync_vllm_after_optimizer_step_uses_lora_mode
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rl_training/vllm_client.py src/rl_training/train_grpo_macorag.py tests/test_rl_training.py
git commit -m "feat: add trainer lora hot sync client"
```

---

### Task 4: Add Custom vLLM LoRA Server Parser and Launcher Skeleton

**Files:**
- Create: `src/rl_training/vllm_lora_server.py`
- Create: `scripts/run_grpo_vllm_lora_server.sh`
- Test: `tests/test_vllm_lora_server.py`
- Modify: `tests/test_rl_training.py`

**Interfaces:**
- Produces: `parse_server_args(argv: list[str] | None = None) -> argparse.Namespace`.
- Produces: launcher script `scripts/run_grpo_vllm_lora_server.sh`.
- Produces a parser-only server module that exits with a clear "runtime is not implemented yet" message.
- Runtime endpoints `/health/`, `/generate/`, `/get_world_size/`, `/init_communicator/`, and `/update_lora_param/` are Task 5.

- [ ] **Step 1: Write failing parser tests**

Create `tests/test_vllm_lora_server.py`:

```python
from __future__ import annotations

from rl_training.vllm_lora_server import parse_server_args


def test_parse_server_args_requires_lora_identity() -> None:
    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )

    assert args.model == "model/Qwen2.5-7B-Instruct"
    assert args.lora_name == "macorag_train"
    assert args.lora_int_id == 1
    assert args.lora_adapter_path == "outputs/adapter"
```

- [ ] **Step 2: Verify parser test fails**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_vllm_lora_server.py::test_parse_server_args_requires_lora_identity
```

Expected: FAIL because `vllm_lora_server.py` does not exist.

- [ ] **Step 3: Implement parser skeleton**

Create `src/rl_training/vllm_lora_server.py`:

```python
from __future__ import annotations

import argparse


def parse_server_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MACORAG vLLM LoRA hot-sync server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--lora-name", required=True)
    parser.add_argument("--lora-int-id", type=int, required=True)
    parser.add_argument("--lora-adapter-path", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_server_args()
    raise SystemExit(
        "vLLM LoRA server runtime is not implemented yet; parser and launcher are available for tests."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write failing launcher test**

Add to `tests/test_rl_training.py`:

```python
def test_run_grpo_vllm_lora_server_script_uses_custom_server_and_lora_config() -> None:
    script = Path("scripts/run_grpo_vllm_lora_server.sh").read_text(encoding="utf-8")

    assert "rl_training.vllm_lora_server" in script
    assert "vllm_lora_name" in script
    assert "vllm_lora_int_id" in script
    assert "vllm_lora_adapter_path" in script
    assert "--data-parallel-size" in script
    assert 'export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"' in script
```

- [ ] **Step 5: Implement launcher**

Create executable `scripts/run_grpo_vllm_lora_server.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/train_grpo.yml}"

read -r YAML_MODEL_PATH YAML_HOST YAML_PORT YAML_VLLM_GPU_INDICES YAML_TP YAML_DP YAML_GPU_UTIL YAML_MAX_LEN YAML_DTYPE YAML_LORA_NAME YAML_LORA_INT_ID YAML_LORA_ADAPTER_PATH < <(
  "${PYTHON:-python}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(
    config.get("model_path", "model/Qwen2.5-7B-Instruct"),
    config.get("vllm_host", "127.0.0.1"),
    int(config.get("vllm_port", 8000)),
    str(config.get("vllm_gpu_indices", "0")),
    int(config.get("vllm_tensor_parallel_size", 1)),
    int(config.get("vllm_data_parallel_size", 1)),
    float(config.get("vllm_gpu_memory_utilization", 0.75)),
    int(config.get("vllm_max_model_len", 8192)),
    config.get("vllm_dtype", "auto"),
    config.get("vllm_lora_name", "macorag_train"),
    int(config.get("vllm_lora_int_id", 1)),
    config.get("vllm_lora_adapter_path") or config.get("sft_adapter_path", ""),
)
PY
)

export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"

exec "${PYTHON:-python}" -m rl_training.vllm_lora_server \
  --model "${YAML_MODEL_PATH}" \
  --host "${YAML_HOST}" \
  --port "${YAML_PORT}" \
  --tensor-parallel-size "${YAML_TP}" \
  --data-parallel-size "${YAML_DP}" \
  --gpu-memory-utilization "${YAML_GPU_UTIL}" \
  --max-model-len "${YAML_MAX_LEN}" \
  --dtype "${YAML_DTYPE}" \
  --lora-name "${YAML_LORA_NAME}" \
  --lora-int-id "${YAML_LORA_INT_ID}" \
  --lora-adapter-path "${YAML_LORA_ADAPTER_PATH}" \
  "$@"
```

Run:

```bash
chmod +x scripts/run_grpo_vllm_lora_server.sh
```

- [ ] **Step 6: Run tests**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_vllm_lora_server.py tests/test_rl_training.py::test_run_grpo_vllm_lora_server_script_uses_custom_server_and_lora_config
bash -n scripts/run_grpo_vllm_lora_server.sh
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rl_training/vllm_lora_server.py scripts/run_grpo_vllm_lora_server.sh tests/test_vllm_lora_server.py tests/test_rl_training.py
git commit -m "feat: add grpo vllm lora server launcher"
```

---

### Task 5: Implement Server LoRA Runtime and Hot Update

**Files:**
- Modify: `src/rl_training/vllm_lora_server.py`
- Modify: `src/rl_training/vllm_client.py`
- Test: `tests/test_vllm_lora_server.py`

**Interfaces:**
- Produces: `build_lora_request(args: argparse.Namespace) -> LoRARequest`.
- Produces: `WeightSyncLoRAWorkerExtension.update_lora_param(name: str, dtype: torch.dtype, shape: Sequence[int]) -> None`.
- Server must pass `lora_request=build_lora_request(args)` into `LLM.generate()`.

- [ ] **Step 1: Write failing LoRARequest test**

Add to `tests/test_vllm_lora_server.py`:

```python
def test_build_lora_request_uses_configured_identity() -> None:
    from rl_training.vllm_lora_server import build_lora_request, parse_server_args

    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )

    request = build_lora_request(args)

    assert request.lora_name == "macorag_train"
    assert request.lora_int_id == 1
    assert request.lora_path == "outputs/adapter"
```

- [ ] **Step 2: Implement LoRARequest helper**

In `src/rl_training/vllm_lora_server.py`:

```python
def build_lora_request(args):
    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name=args.lora_name,
        lora_int_id=args.lora_int_id,
        lora_path=args.lora_adapter_path,
    )
```

- [ ] **Step 3: Add runtime server copied from TRL structure**

Replace `main()` with a local server implementation modeled on TRL's installed `vllm_serve.py`:

```python
def main() -> None:
    args = parse_server_args()
    try:
        import uvicorn
        from fastapi import FastAPI
        from pydantic import BaseModel
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing vLLM LoRA server dependency: {exc}") from exc

    app = FastAPI()
    lora_request = build_lora_request(args)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
    )

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        max_tokens: int = 256

    @app.get("/health/")
    async def health():
        return {"status": "ok", "lora_name": args.lora_name, "model": args.model}

    @app.post("/generate/")
    async def generate(request: GenerateRequest):
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
        )
        outputs = llm.generate(request.prompts, sampling_params=sampling_params, lora_request=lora_request)
        completion_ids = [list(output.token_ids) for outputs_item in outputs for output in outputs_item.outputs]
        return {"completion_ids": completion_ids}

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
```

- [ ] **Step 4: Add LoRA update endpoint skeleton that fails clearly**

Before implementing direct in-memory update, add endpoint:

```python
class UpdateLoRAParamRequest(BaseModel):
    name: str
    dtype: str
    shape: list[int]


@app.post("/update_lora_param/")
async def update_lora_param(request: UpdateLoRAParamRequest):
    raise RuntimeError(
        "LoRA in-memory tensor replacement is not implemented for this vLLM version yet. "
        f"Requested {request.name} shape={request.shape}."
    )
```

This makes Task 4/partial Task 5 runnable without silently falling back to dense sync.

- [ ] **Step 5: Inspect vLLM LoRA internals in the running server**

Run a controlled probe on a small or 7B model with:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n macorag python -m rl_training.vllm_lora_server \
  --model model/Qwen2.5-7B-Instruct \
  --lora-name macorag_train \
  --lora-int-id 1 \
  --lora-adapter-path outputs/lora_qwen2.5-7b_trajectory_20260627_203027/adapter \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192
```

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health/
```

Expected: JSON includes `status: ok`, `lora_name: macorag_train`, and the 7B model path.

- [ ] **Step 6: Implement direct LoRA tensor replacement only after finding stable objects**

If the running `LLM` exposes LoRA adapter tensors through a stable object path, implement:

```python
def replace_lora_tensor(server_state, name: str, tensor) -> tuple[int, ...]:
    expected = server_state.lora_tensor_index[name]
    if tuple(expected.shape) != tuple(tensor.shape):
        raise ValueError(f"LoRA tensor shape mismatch for {name}: expected {tuple(expected.shape)}, got {tuple(tensor.shape)}")
    expected.data.copy_(tensor.to(device=expected.device, dtype=expected.dtype))
    return tuple(expected.shape)
```

If no stable object path exists, stop implementation and report the blocker. Do not fake success.

- [ ] **Step 7: Commit**

If only the runnable LoRA generate server and explicit blocker endpoint are implemented:

```bash
git add src/rl_training/vllm_lora_server.py tests/test_vllm_lora_server.py
git commit -m "feat: add vllm lora generation server"
```

If direct in-memory update is also implemented and smoke tested:

```bash
git add src/rl_training/vllm_lora_server.py tests/test_vllm_lora_server.py
git commit -m "feat: add vllm lora hot update endpoint"
```

---

### Task 6: Switch Config After Smoke Test

**Files:**
- Modify: `config/train_grpo.yml`
- Test: smoke commands and `tests/test_rl_training.py`

**Interfaces:**
- Consumes: custom server from Task 5 and trainer LoRA sync from Task 3.
- Produces: active default `vllm_sync_mode: "lora"` only if direct update works.

- [ ] **Step 1: Run dense baseline check**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_rl_training.py tests/test_vllm_lora_server.py
```

Expected: PASS.

- [ ] **Step 2: Start LoRA server**

Run in terminal 1:

```bash
cd /data/xudu/macorag
conda run -n macorag bash scripts/run_grpo_vllm_lora_server.sh
```

Expected: server starts and `/health/` returns `status: ok`.

- [ ] **Step 3: Run one-sample GRPO smoke**

Run in terminal 2:

```bash
cd /data/xudu/macorag
conda run -n macorag bash scripts/run_train_grpo.sh --max-samples 1 --max-steps 1 --vllm-sync-mode lora
```

Expected:

- rollout generation succeeds through vLLM,
- optimizer step completes,
- LoRA sync endpoint succeeds,
- output `train_metrics.jsonl` includes `time_weight_sync_seconds` well below dense sync's ~55 seconds.

- [ ] **Step 4: Switch YAML default only after smoke success**

If Step 3 passes, update `config/train_grpo.yml`:

```yaml
vllm_sync_mode: "lora"
```

If Step 3 fails because vLLM has no stable in-memory LoRA tensor path, keep:

```yaml
vllm_sync_mode: "dense"
```

and document the blocker in the final response.

- [ ] **Step 5: Commit**

If YAML is switched:

```bash
git add config/train_grpo.yml
git commit -m "config: enable grpo vllm lora sync"
```

If YAML remains dense due to blocker, do not commit a config change.

---

## Self-Review

- Spec coverage: config, custom server, LoRARequest generation, LoRA tensor mapping, trainer sync selection, fallback, and smoke test are covered.
- Placeholder scan: no incomplete placeholder markers are present; Task 5 explicitly defines the condition for stopping if vLLM internals do not expose stable LoRA tensors.
- Type consistency: config field names match across YAML, parser, client, and trainer selection.
- Risk handling: dense sync fallback remains until direct LoRA hot update passes smoke verification.
