# vLLM LoRA Registered Adapter Hot Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement real LoRA-only hot synchronization by updating vLLM's registered LoRA adapter and refreshing its active GPU slot.

**Architecture:** The vLLM worker receives LoRA tensors through the existing NCCL communicator, updates the registered `LoRAModel` tensors after strict PEFT-to-vLLM mapping and shape validation, then reactivates the adapter so vLLM copies the new values into its active GPU LoRA buffers. The trainer keeps using the project-local `/update_lora_param/` endpoint and only switches to `vllm_sync_mode: "lora"` after tests and smoke validation.

**Tech Stack:** Python 3.9, PyTorch 2.6.0+cu124, vLLM 0.8.5.post1, TRL 0.18.2, PEFT LoRA, pytest.

## Global Constraints

- Do not patch installed TRL/vLLM package files; project-local code only.
- Keep dense sync fallback available behind `vllm_sync_mode: "dense"`.
- Only train and hot-sync LoRA adapter weights.
- Do not silently claim LoRA hot sync works unless tensors are really updated.
- Fail fast on unsupported LoRA names, missing registered adapters, missing modules, dtype parse errors, or shape mismatch.
- Unit tests must not load the real 7B model.
- Use vLLM 0.8.5.post1 internal LoRA manager fields deliberately: `model_runner.lora_manager`, `_adapter_manager`, `_registered_adapters`, `_active_adapters`, `_deactivate_adapter()`, and `activate_adapter()`.
- Preserve current GPU split: vLLM server GPU from `vllm_gpu_indices`, trainer GPU from `gpu_indices`.

---

## File Structure

- Modify `src/rl_training/vllm_lora_server.py`: add PEFT LoRA name parsing, registered adapter update helpers, worker update implementation, health capability flag, and endpoint wiring.
- Modify `src/rl_training/vllm_client.py`: keep current protocol; only adjust validation if server response shape changes.
- Modify `config/train_grpo.yml`: switch to LoRA sync only after smoke validation succeeds.
- Modify `tests/test_vllm_lora_server.py`: unit tests for name parsing, registered adapter update, refresh, endpoint RPC args, and health capability.
- Modify `tests/test_rl_training.py`: validation and sync-mode regression tests as needed.

---

### Task 1: Add Server-Side Registered Adapter Update Helpers

**Files:**
- Modify: `src/rl_training/vllm_lora_server.py`
- Test: `tests/test_vllm_lora_server.py`

**Interfaces:**
- Produces: `parse_vllm_lora_tensor_name(name: str) -> tuple[str, str]`.
- Produces: `update_registered_lora_tensor(adapter_manager: Any, lora_int_id: int, name: str, tensor: torch.Tensor) -> tuple[int, ...]`.
- Produces: `refresh_active_lora(adapter_manager: Any, lora_int_id: int) -> bool`.
- Consumes: vLLM registered adapter object with `loras: dict[str, LoRALayerWeights]`.

- [ ] **Step 1: Write failing name parser tests**

Add to `tests/test_vllm_lora_server.py`:

```python
def test_parse_vllm_lora_tensor_name_maps_module_and_side() -> None:
    from rl_training.vllm_lora_server import parse_vllm_lora_tensor_name

    assert parse_vllm_lora_tensor_name(
        "model.layers.0.self_attn.q_proj.lora_A.weight"
    ) == ("model.layers.0.self_attn.q_proj", "lora_A")
    assert parse_vllm_lora_tensor_name(
        "model.layers.31.mlp.down_proj.lora_B.weight"
    ) == ("model.layers.31.mlp.down_proj", "lora_B")
```

- [ ] **Step 2: Write failing unsupported-name test**

Add to `tests/test_vllm_lora_server.py`:

```python
def test_parse_vllm_lora_tensor_name_rejects_unsupported_name() -> None:
    from rl_training.vllm_lora_server import parse_vllm_lora_tensor_name

    try:
        parse_vllm_lora_tensor_name("model.embed_tokens.weight")
    except ValueError as exc:
        assert "Unsupported LoRA tensor name" in str(exc)
    else:
        raise AssertionError("expected unsupported LoRA tensor name to fail")
```

- [ ] **Step 3: Run parser tests to verify failure**

Run:

```bash
conda run -n macorag python -m pytest -q \
  tests/test_vllm_lora_server.py::test_parse_vllm_lora_tensor_name_maps_module_and_side \
  tests/test_vllm_lora_server.py::test_parse_vllm_lora_tensor_name_rejects_unsupported_name
```

Expected: FAIL with `ImportError` or missing function.

- [ ] **Step 4: Implement the parser**

In `src/rl_training/vllm_lora_server.py`, add near `_dtype_from_wire`:

```python
def parse_vllm_lora_tensor_name(name: str) -> tuple[str, str]:
    for side in ("lora_A", "lora_B"):
        suffix = f".{side}.weight"
        if name.endswith(suffix):
            module_name = name[: -len(suffix)]
            if not module_name.startswith("model.layers."):
                break
            return module_name, side
    raise ValueError(f"Unsupported LoRA tensor name: {name}")
```

- [ ] **Step 5: Write fake registered adapter update tests**

Add to `tests/test_vllm_lora_server.py`:

```python
class _FakeLoRALayer:
    def __init__(self) -> None:
        self.lora_a = torch.zeros(3, 2, dtype=torch.float16)
        self.lora_b = torch.zeros(2, 4, dtype=torch.float16)


class _FakeLoRAModel:
    def __init__(self) -> None:
        self.loras = {"model.layers.0.self_attn.q_proj": _FakeLoRALayer()}


class _FakeAdapterManager:
    def __init__(self) -> None:
        self._registered_adapters = {1: _FakeLoRAModel()}
        self._active_adapters = {1: None}
        self.deactivated: list[int] = []
        self.activated: list[int] = []

    def _deactivate_adapter(self, lora_id: int) -> None:
        self.deactivated.append(lora_id)
        self._active_adapters.pop(lora_id, None)

    def activate_adapter(self, lora_id: int) -> bool:
        self.activated.append(lora_id)
        self._active_adapters[lora_id] = None
        return True


def test_update_registered_lora_tensor_transposes_peft_a_weight() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakeAdapterManager()
    incoming = torch.arange(6, dtype=torch.float16).reshape(2, 3)

    shape = update_registered_lora_tensor(
        manager,
        1,
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        incoming,
    )

    layer = manager._registered_adapters[1].loras["model.layers.0.self_attn.q_proj"]
    assert shape == (3, 2)
    assert torch.equal(layer.lora_a, incoming.T)


def test_update_registered_lora_tensor_transposes_peft_b_weight() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakeAdapterManager()
    incoming = torch.arange(8, dtype=torch.float16).reshape(4, 2)

    shape = update_registered_lora_tensor(
        manager,
        1,
        "model.layers.0.self_attn.q_proj.lora_B.weight",
        incoming,
    )

    layer = manager._registered_adapters[1].loras["model.layers.0.self_attn.q_proj"]
    assert shape == (2, 4)
    assert torch.equal(layer.lora_b, incoming.T)
```

- [ ] **Step 6: Write missing and shape mismatch tests**

Add to `tests/test_vllm_lora_server.py`:

```python
def test_update_registered_lora_tensor_rejects_shape_mismatch() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakeAdapterManager()
    incoming = torch.zeros(5, 5, dtype=torch.float16)

    try:
        update_registered_lora_tensor(
            manager,
            1,
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            incoming,
        )
    except ValueError as exc:
        assert "LoRA tensor shape mismatch" in str(exc)
    else:
        raise AssertionError("expected shape mismatch to fail")


def test_update_registered_lora_tensor_rejects_missing_module() -> None:
    from rl_training.vllm_lora_server import update_registered_lora_tensor

    manager = _FakeAdapterManager()
    incoming = torch.zeros(2, 3, dtype=torch.float16)

    try:
        update_registered_lora_tensor(
            manager,
            1,
            "model.layers.0.self_attn.k_proj.lora_A.weight",
            incoming,
        )
    except KeyError as exc:
        assert "Registered LoRA module not found" in str(exc)
    else:
        raise AssertionError("expected missing module to fail")
```

- [ ] **Step 7: Write active refresh test**

Add to `tests/test_vllm_lora_server.py`:

```python
def test_refresh_active_lora_reactivates_active_adapter() -> None:
    from rl_training.vllm_lora_server import refresh_active_lora

    manager = _FakeAdapterManager()

    refreshed = refresh_active_lora(manager, 1)

    assert refreshed is True
    assert manager.deactivated == [1]
    assert manager.activated == [1]
```

- [ ] **Step 8: Implement update and refresh helpers**

In `src/rl_training/vllm_lora_server.py`, add:

```python
def update_registered_lora_tensor(adapter_manager: Any, lora_int_id: int, name: str, tensor: torch.Tensor) -> tuple[int, ...]:
    module_name, side = parse_vllm_lora_tensor_name(name)
    registered_adapters = getattr(adapter_manager, "_registered_adapters", {})
    if lora_int_id not in registered_adapters:
        raise KeyError(f"Registered LoRA adapter not found: {lora_int_id}")
    lora_model = registered_adapters[lora_int_id]
    loras = getattr(lora_model, "loras", {})
    if module_name not in loras:
        raise KeyError(f"Registered LoRA module not found: {module_name}")
    layer = loras[module_name]
    target = getattr(layer, "lora_a" if side == "lora_A" else "lora_b")
    source = tensor.detach().to(device=target.device, dtype=target.dtype).T.contiguous()
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"LoRA tensor shape mismatch for {name}: expected PEFT shape "
            f"{tuple(target.T.shape)}, got {tuple(tensor.shape)}"
        )
    target.copy_(source)
    return tuple(target.shape)


def refresh_active_lora(adapter_manager: Any, lora_int_id: int) -> bool:
    active_adapters = getattr(adapter_manager, "_active_adapters", {})
    if lora_int_id not in active_adapters:
        return False
    adapter_manager._deactivate_adapter(lora_int_id)
    adapter_manager.activate_adapter(lora_int_id)
    return True
```

- [ ] **Step 9: Run helper tests**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_vllm_lora_server.py -k 'parse_vllm_lora_tensor_name or update_registered_lora_tensor or refresh_active_lora'
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/rl_training/vllm_lora_server.py tests/test_vllm_lora_server.py
git commit -m "feat: add vllm lora adapter update helpers"
```

---

### Task 2: Wire Worker Extension and HTTP Endpoint to Hot Update

**Files:**
- Modify: `src/rl_training/vllm_lora_server.py`
- Test: `tests/test_vllm_lora_server.py`

**Interfaces:**
- Consumes: `update_registered_lora_tensor(adapter_manager, lora_int_id, name, tensor)`.
- Consumes: `refresh_active_lora(adapter_manager, lora_int_id)`.
- Produces: `WeightSyncLoRAWorkerExtension.update_lora_param(name, dtype, shape, lora_int_id) -> None`.
- Produces: `/update_lora_param/` endpoint that calls collective RPC with `lora_int_id`.

- [ ] **Step 1: Update fake LLM for RPC assertions**

In `tests/test_vllm_lora_server.py`, extend `_FakeLLM.collective_rpc` if needed so it records `method` and `args` unchanged.

- [ ] **Step 2: Add endpoint RPC argument test**

Add to `tests/test_vllm_lora_server.py`:

```python
def test_update_lora_param_endpoint_dispatches_collective_rpc_with_lora_id() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

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
    llm = _FakeLLM()
    app = create_app(args, llm=llm, sampling_params_cls=_FakeSamplingParams)

    response = TestClient(app).post(
        "/update_lora_param/",
        json={
            "name": "model.layers.0.self_attn.q_proj.lora_A.weight",
            "dtype": "torch.float32",
            "shape": [2, 3],
        },
    )

    assert response.status_code == 200
    assert llm.collective_rpc_calls[-1] == {
        "method": "update_lora_param",
        "args": ("model.layers.0.self_attn.q_proj.lora_A.weight", torch.float32, (2, 3), 1),
        "kwargs": {},
    }
```

- [ ] **Step 3: Replace old explicit-501 endpoint test**

Remove or rewrite `test_update_lora_param_endpoint_fails_explicitly()` so the test suite now expects HTTP 200 with a fake LLM. Keep a separate invalid dtype test:

```python
def test_update_lora_param_endpoint_rejects_bad_dtype() -> None:
    from fastapi.testclient import TestClient

    from rl_training.vllm_lora_server import create_app, parse_server_args

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
    app = create_app(args, llm=_FakeLLM(), sampling_params_cls=_FakeSamplingParams)

    response = TestClient(app, raise_server_exceptions=False).post(
        "/update_lora_param/",
        json={"name": "model.layers.0.self_attn.q_proj.lora_A.weight", "dtype": "bad", "shape": [2, 3]},
    )

    assert response.status_code == 400
```

- [ ] **Step 4: Write worker extension unit test**

Add to `tests/test_vllm_lora_server.py`:

```python
def test_weight_sync_lora_worker_extension_updates_registered_adapter(monkeypatch) -> None:
    from rl_training import vllm_lora_server
    from rl_training.vllm_lora_server import WeightSyncLoRAWorkerExtension

    manager = _FakeAdapterManager()
    extension = WeightSyncLoRAWorkerExtension()
    extension.device = torch.device("cpu")
    extension.client_rank = 1
    extension.model_runner = type(
        "Runner",
        (),
        {"lora_manager": type("WorkerManager", (), {"_adapter_manager": manager})()},
    )()

    class FakeComm:
        def __init__(self) -> None:
            self.group = type("Group", (), {"barrier": lambda self_group: None})()

        def broadcast(self, tensor, src):
            assert src == 1
            tensor.copy_(torch.arange(6, dtype=tensor.dtype).reshape(2, 3))

    extension.pynccl_comm = FakeComm()

    extension.update_lora_param(
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        torch.float16,
        (2, 3),
        1,
    )

    layer = manager._registered_adapters[1].loras["model.layers.0.self_attn.q_proj"]
    assert torch.equal(layer.lora_a, torch.arange(6, dtype=torch.float16).reshape(2, 3).T)
    assert manager.deactivated == [1]
    assert manager.activated == [1]
```

- [ ] **Step 5: Implement worker extension update**

Replace `WeightSyncLoRAWorkerExtension.update_lora_param()` in
`src/rl_training/vllm_lora_server.py`:

```python
    def update_lora_param(self, name: str, dtype: torch.dtype, shape: Sequence[int], lora_int_id: int) -> None:
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        weight = torch.empty(tuple(shape), dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()

        worker_lora_manager = getattr(self.model_runner, "lora_manager", None)
        if worker_lora_manager is None:
            raise RuntimeError("vLLM LoRA manager is not initialized.")
        adapter_manager = getattr(worker_lora_manager, "_adapter_manager", None)
        if adapter_manager is None:
            raise RuntimeError("vLLM adapter manager is not initialized.")
        update_registered_lora_tensor(adapter_manager, lora_int_id, name, weight)
        refresh_active_lora(adapter_manager, lora_int_id)
```

- [ ] **Step 6: Update health and endpoint**

In `create_app()`:

```python
"supports_lora_param_update": True,
```

Replace the `/update_lora_param/` body with:

```python
    @app.post("/update_lora_param/")
    async def update_lora_param(request: UpdateLoRAParamRequest = Body(...)):
        try:
            dtype = _dtype_from_wire(request.dtype)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            llm.collective_rpc(
                method="update_lora_param",
                args=(request.name, dtype, tuple(request.shape), args.lora_int_id),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"message": "Request received, updating LoRA parameter"}
```

- [ ] **Step 7: Run endpoint and worker tests**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_vllm_lora_server.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/rl_training/vllm_lora_server.py tests/test_vllm_lora_server.py
git commit -m "feat: implement vllm lora hot update endpoint"
```

---

### Task 3: Enable LoRA Mode Validation and Config After Smoke

**Files:**
- Modify: `src/rl_training/vllm_client.py`
- Modify: `config/train_grpo.yml`
- Test: `tests/test_rl_training.py`

**Interfaces:**
- Consumes: server health with `supports_lora_param_update: true`.
- Produces: working `vllm_sync_mode: "lora"` config if smoke passes.

- [ ] **Step 1: Add client validation success test**

Add or update in `tests/test_rl_training.py`:

```python
def test_vllm_generation_client_validate_lora_server_accepts_update_capability() -> None:
    from rl_training.vllm_client import VLLMGenerationClient

    class Response:
        status_code = 200
        text = "ok"

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class Session:
        def __init__(self) -> None:
            self.posts = []

        def get(self, url):
            return Response(
                {
                    "sync_mode": "lora",
                    "model": "model/base",
                    "lora_name": "macorag_train",
                    "lora_int_id": 1,
                    "lora_adapter_path": "outputs/adapter",
                    "supports_lora_param_update": True,
                }
            )

        def post(self, url, json):
            self.posts.append((url, json))
            return Response({"message": "ok"})

    backend = type("Backend", (), {"session": Session(), "base_url": "http://server"})()
    args = Namespace(
        model_path="model/base",
        vllm_lora_name="macorag_train",
        vllm_lora_int_id=1,
        vllm_lora_adapter_path="outputs/adapter",
    )

    VLLMGenerationClient(host="127.0.0.1", port=8000, timeout_seconds=5, backend=backend).validate_lora_server(args)

    assert backend.session.posts[0][1]["name"] == "__macorag_lora_capability_probe__"
```

- [ ] **Step 2: Ensure validation still rejects false capability**

Keep the existing unsupported endpoint test, but make it fail on either
`supports_lora_param_update: False` or non-200 probe response. Expected error
must include `LoRA hot sync is unsupported`.

- [ ] **Step 3: Run validation tests**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_rl_training.py -k 'validate_lora_server'
```

Expected: PASS.

- [ ] **Step 4: Run full focused tests before smoke**

Run:

```bash
conda run -n macorag python -m pytest -q tests/test_rl_training.py tests/test_vllm_lora_server.py
bash -n scripts/run_train_grpo.sh scripts/run_grpo_vllm_lora_server.sh
```

Expected: PASS.

- [ ] **Step 5: Start LoRA server for smoke**

In terminal 1:

```bash
cd /data/xudu/macorag
conda activate macorag
bash scripts/run_grpo_vllm_lora_server.sh
```

Expected: `/health/` returns `supports_lora_param_update: true`.

- [ ] **Step 6: Run one-sample LoRA sync smoke**

In terminal 2:

```bash
cd /data/xudu/macorag
conda activate macorag
bash scripts/run_train_grpo.sh --max-samples 1 --max-steps 1 --vllm-sync-mode lora --vllm-sync-every-steps 1
```

Expected:

- rollout generation succeeds through vLLM,
- optimizer step completes,
- LoRA sync endpoint succeeds,
- `train_metrics.jsonl` has `time_weight_sync_seconds` much lower than 53 seconds.

- [ ] **Step 7: Switch YAML only after smoke passes**

If Step 6 passes, modify `config/train_grpo.yml`:

```yaml
vllm_sync_every_steps: 1
vllm_sync_mode: "lora"
```

If Step 6 fails, keep `vllm_sync_mode: "dense"` and report the failing error.

- [ ] **Step 8: Commit**

If smoke passes:

```bash
git add src/rl_training/vllm_client.py tests/test_rl_training.py config/train_grpo.yml
git commit -m "config: enable grpo vllm lora hot sync"
```

If smoke fails but tests pass:

```bash
git add src/rl_training/vllm_client.py tests/test_rl_training.py
git commit -m "test: cover vllm lora hot sync validation"
```

---

## Self-Review

- Spec coverage: worker update helpers, endpoint wiring, client validation, config switch, tests, and smoke validation are covered.
- Placeholder scan: no TODO/TBD placeholders are used.
- Type consistency: `update_lora_param(name, dtype, shape, lora_int_id)` matches endpoint and worker extension usage.
- Risk handling: config switches to LoRA only after one-sample smoke validates the real server path.
