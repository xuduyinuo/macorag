import argparse
import os
import re
import threading
import uuid
from collections.abc import Sequence
from typing import Any, Optional

import torch


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


_VLLM_LORA_TENSOR_RE = re.compile(
    r"^(?P<module>model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj)))\.(?P<side>lora_[AB])\.weight$"
)


def parse_server_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
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


def build_lora_request(args: argparse.Namespace):
    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name=args.lora_name,
        lora_int_id=args.lora_int_id,
        lora_path=args.lora_adapter_path,
    )


def register_lora_adapter_on_workers(llm: Any, args: argparse.Namespace) -> None:
    llm.collective_rpc(
        method="register_lora_adapter",
        args=(args.lora_name, args.lora_int_id, args.lora_adapter_path),
    )


class WeightSyncLoRAWorkerExtension:
    pynccl_comm = None
    client_rank = None

    def register_lora_adapter(self, lora_name: str, lora_int_id: int, lora_adapter_path: str) -> None:
        worker_lora_manager = getattr(self.model_runner, "lora_manager", None)
        if worker_lora_manager is None:
            raise RuntimeError("vLLM LoRA manager is not initialized.")

        from vllm.lora.request import LoRARequest

        request = LoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=lora_adapter_path,
        )
        if not worker_lora_manager.add_adapter(request):
            raise RuntimeError(f"Failed to register LoRA adapter {lora_int_id}.")
        if not worker_lora_manager.pin_adapter(lora_int_id):
            raise RuntimeError(f"Failed to pin LoRA adapter {lora_int_id}.")

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_world_group
        from vllm.distributed.utils import StatelessProcessGroup

        rank = get_world_group().rank
        process_group = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(process_group, device=self.device)
        self.client_rank = world_size - 1

    def update_lora_param(self, name: str, dtype: torch.dtype, shape: Sequence[int], lora_int_id: int) -> None:
        self.update_lora_params([(name, dtype, tuple(shape))], lora_int_id)

    def update_lora_params(
        self,
        tensors: Sequence[tuple[str, torch.dtype, Sequence[int]]],
        lora_int_id: int,
    ) -> None:
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        worker_lora_manager = getattr(self.model_runner, "lora_manager", None)
        if worker_lora_manager is None:
            raise RuntimeError("vLLM LoRA manager is not initialized.")
        adapter_manager = getattr(worker_lora_manager, "_adapter_manager", None)
        if adapter_manager is None:
            raise RuntimeError("vLLM adapter manager is not initialized.")

        for name, dtype, shape in tensors:
            validate_registered_lora_tensor(adapter_manager, lora_int_id, name, shape)

        received: list[tuple[str, torch.Tensor]] = []
        for name, dtype, shape in tensors:
            weight = torch.empty(tuple(shape), dtype=dtype, device=self.device)
            self.pynccl_comm.broadcast(weight, src=self.client_rank)
            received.append((name, weight))
        self.pynccl_comm.group.barrier()
        for name, weight in received:
            update_registered_lora_tensor(adapter_manager, lora_int_id, name, weight)
        refresh_active_lora(adapter_manager, lora_int_id)

    def validate_lora_params(
        self,
        tensors: Sequence[tuple[str, torch.dtype, Sequence[int]]],
        lora_int_id: int,
    ) -> None:
        worker_lora_manager = getattr(self.model_runner, "lora_manager", None)
        if worker_lora_manager is None:
            raise RuntimeError("vLLM LoRA manager is not initialized.")
        adapter_manager = getattr(worker_lora_manager, "_adapter_manager", None)
        if adapter_manager is None:
            raise RuntimeError("vLLM adapter manager is not initialized.")

        for name, _dtype, shape in tensors:
            validate_registered_lora_tensor(adapter_manager, lora_int_id, name, shape)

    def close_communicator(self) -> None:
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


def _dtype_from_wire(value: str) -> torch.dtype:
    dtype_name = value.split(".")[-1]
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return dtype


def parse_vllm_lora_tensor_name(name: str) -> tuple[str, str]:
    match = _VLLM_LORA_TENSOR_RE.match(name)
    if match is not None:
        return match.group("module"), match.group("side")
    raise ValueError(f"Unsupported LoRA tensor name: {name}")


def _resolve_packed_lora_module(adapter_manager: Any, loras: dict[str, Any], module_name: str) -> tuple[str, int]:
    packed_modules = getattr(adapter_manager, "packed_modules", {})
    for packed_module_name, unpacked_module_names in packed_modules.items():
        if module_name in unpacked_module_names:
            return packed_module_name, list(unpacked_module_names).index(module_name)

    fallback_mappings = {
        "self_attn.qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "mlp.gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
    }
    for packed_suffix, unpacked_suffixes in fallback_mappings.items():
        for index, unpacked_suffix in enumerate(unpacked_suffixes):
            if module_name.endswith(unpacked_suffix):
                packed_module_name = module_name[: -len(unpacked_suffix)] + packed_suffix
                if packed_module_name in loras:
                    return packed_module_name, index

    raise KeyError(f"Registered LoRA module not found: {module_name}")


def _lora_b_scaling(layer: Any, list_index: Optional[int]) -> float:
    rank = getattr(layer, "rank", None)
    if not rank:
        return 1.0
    if list_index is not None:
        lora_alphas = getattr(layer, "lora_alphas", None)
        if lora_alphas is None or list_index >= len(lora_alphas) or lora_alphas[list_index] is None:
            return 1.0
        return float(lora_alphas[list_index]) / float(rank)
    lora_alpha = getattr(layer, "lora_alpha", None)
    if lora_alpha is None:
        return 1.0
    return float(lora_alpha) / float(rank)


def _mark_lora_b_scaling_merged(layer: Any, list_index: Optional[int]) -> None:
    scaling = getattr(layer, "scaling", None)
    if scaling is None:
        return
    if list_index is not None and isinstance(scaling, list) and list_index < len(scaling):
        scaling[list_index] = 1
    elif list_index is None:
        layer.scaling = 1


def _resolve_registered_lora_target(
    adapter_manager: Any,
    lora_model: Any,
    module_name: str,
    side: str,
) -> tuple[Any, torch.Tensor, float, Optional[int]]:
    loras = getattr(lora_model, "loras", {})
    list_index: Optional[int] = None
    if module_name in loras:
        layer = loras[module_name]
    else:
        packed_module_name, list_index = _resolve_packed_lora_module(adapter_manager, loras, module_name)
        layer = loras[packed_module_name]

    target_container = getattr(layer, "lora_a" if side == "lora_A" else "lora_b", None)
    if target_container is None:
        raise KeyError(f"Registered LoRA tensor side not found: {module_name}.{side}")
    if list_index is not None:
        if list_index >= len(target_container) or target_container[list_index] is None:
            raise KeyError(f"Registered packed LoRA tensor not found: {module_name}.{side}")
        target = target_container[list_index]
    else:
        target = target_container

    scaling = _lora_b_scaling(layer, list_index) if side == "lora_B" else 1.0
    return layer, target, scaling, list_index


def update_registered_lora_tensor(
    adapter_manager: Any,
    lora_int_id: int,
    name: str,
    tensor: torch.Tensor,
) -> tuple[int, ...]:
    module_name, side = parse_vllm_lora_tensor_name(name)
    registered_adapters = getattr(adapter_manager, "_registered_adapters", {})
    if lora_int_id not in registered_adapters:
        raise KeyError(f"Registered LoRA adapter not found: {lora_int_id}")

    lora_model = registered_adapters[lora_int_id]
    layer, target, scaling, list_index = _resolve_registered_lora_target(
        adapter_manager, lora_model, module_name, side
    )

    source = tensor.detach().to(device=target.device, dtype=target.dtype).T.contiguous()
    if side == "lora_B":
        source = source * scaling
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"LoRA tensor shape mismatch for {name}: expected PEFT shape "
            f"{tuple(target.T.shape)}, got {tuple(tensor.shape)}"
        )

    target.copy_(source)
    if side == "lora_B":
        _mark_lora_b_scaling_merged(layer, list_index)
    return tuple(target.shape)


def validate_registered_lora_tensor(
    adapter_manager: Any,
    lora_int_id: int,
    name: str,
    shape: Sequence[int],
) -> tuple[int, ...]:
    module_name, side = parse_vllm_lora_tensor_name(name)
    registered_adapters = getattr(adapter_manager, "_registered_adapters", {})
    if lora_int_id not in registered_adapters:
        raise KeyError(f"Registered LoRA adapter not found: {lora_int_id}")

    lora_model = registered_adapters[lora_int_id]
    _, target, _, _ = _resolve_registered_lora_target(adapter_manager, lora_model, module_name, side)
    expected_shape = tuple(target.T.shape)
    actual_shape = tuple(shape)
    if expected_shape != actual_shape:
        raise ValueError(f"LoRA tensor shape mismatch for {name}: expected PEFT shape {expected_shape}, got {actual_shape}")
    return tuple(target.shape)


def refresh_active_lora(adapter_manager: Any, lora_int_id: int) -> bool:
    active_adapters = getattr(adapter_manager, "_active_adapters", {})
    if lora_int_id not in active_adapters:
        return False

    adapter_manager._deactivate_adapter(lora_int_id)
    active_adapters.pop(lora_int_id, None)
    if not adapter_manager.activate_adapter(lora_int_id):
        raise RuntimeError(f"Failed to reactivate LoRA adapter {lora_int_id}.")
    return True


def _chosen_token_logprobs(output: Any) -> list[float]:
    token_ids = list(output.token_ids)
    token_logprobs = getattr(output, "logprobs", None)
    if token_logprobs is None or len(token_logprobs) != len(token_ids):
        raise RuntimeError(
            "vLLM completion/logprob length mismatch: "
            f"{len(token_ids)} tokens versus {0 if token_logprobs is None else len(token_logprobs)} logprobs."
        )
    chosen: list[float] = []
    for token_id, candidates in zip(token_ids, token_logprobs):
        entry = candidates.get(token_id) if isinstance(candidates, dict) else None
        if entry is None and isinstance(candidates, dict):
            entry = candidates.get(str(token_id))
        if entry is None:
            raise RuntimeError(f"vLLM logprobs missing chosen token {token_id}.")
        value = getattr(entry, "logprob", entry)
        chosen.append(float(value))
    return chosen


def create_app(
    args: argparse.Namespace,
    *,
    llm: Any,
    sampling_params_cls: type[Any],
):
    try:
        from fastapi import Body, FastAPI, HTTPException
        from pydantic import BaseModel
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing vLLM LoRA server dependency: {exc}") from exc

    app = FastAPI()
    lora_request = build_lora_request(args)
    update_statuses: dict[str, dict[str, Optional[str]]] = {}
    update_status_lock = threading.Lock()

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 256
        guided_decoding_regex: Optional[str] = None

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    class UpdateLoRAParamRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    class UpdateLoRAParamsRequest(BaseModel):
        tensors: list[UpdateLoRAParamRequest]

    @app.get("/health/")
    async def health():
        return {
            "status": "ok",
            "sync_mode": "lora",
            "model": args.model,
            "lora_name": args.lora_name,
            "lora_int_id": args.lora_int_id,
            "lora_adapter_path": args.lora_adapter_path,
            "supports_lora_param_update": True,
        }

    @app.get("/get_world_size/")
    async def get_world_size():
        return {"world_size": args.tensor_parallel_size * args.data_parallel_size}

    @app.post("/generate/")
    async def generate(request: GenerateRequest = Body(...)):
        sampling_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "logprobs": 1,
        }
        if request.guided_decoding_regex is not None:
            try:
                from vllm.sampling_params import GuidedDecodingParams
            except ModuleNotFoundError as exc:
                raise HTTPException(status_code=500, detail=f"Missing guided decoding dependency: {exc}") from exc
            sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
                backend="outlines",
                regex=request.guided_decoding_regex,
            )
        sampling_params = sampling_params_cls(**sampling_kwargs)
        outputs = llm.generate(request.prompts, sampling_params=sampling_params, lora_request=lora_request)
        flattened_outputs = [output for outputs_item in outputs for output in outputs_item.outputs]
        completion_ids = [list(output.token_ids) for output in flattened_outputs]
        logprobs = [_chosen_token_logprobs(output) for output in flattened_outputs]
        return {"completion_ids": completion_ids, "logprobs": logprobs}

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest = Body(...)):
        world_size = args.tensor_parallel_size * args.data_parallel_size + 1
        threading.Thread(
            target=llm.collective_rpc,
            kwargs={"method": "init_communicator", "args": (request.host, request.port, world_size)},
            daemon=True,
        ).start()
        return {"message": "Request received, initializing communicator"}

    @app.post("/update_lora_param/")
    async def update_lora_param(request: UpdateLoRAParamRequest = Body(...)):
        try:
            dtype = _dtype_from_wire(request.dtype)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            parse_vllm_lora_tensor_name(request.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            llm.collective_rpc(
                method="validate_lora_params",
                args=([(request.name, dtype, tuple(request.shape))], args.lora_int_id),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        update_id = uuid.uuid4().hex
        with update_status_lock:
            update_statuses[update_id] = {"state": "pending", "error": None}

        def _dispatch_update() -> None:
            try:
                llm.collective_rpc(
                    method="update_lora_param",
                    args=(request.name, dtype, tuple(request.shape), args.lora_int_id),
                )
            except Exception as exc:
                with update_status_lock:
                    update_statuses[update_id] = {"state": "error", "error": str(exc)}
            else:
                with update_status_lock:
                    update_statuses[update_id] = {"state": "ok", "error": None}

        threading.Thread(target=_dispatch_update, daemon=True).start()
        return {"message": "Request received, updating LoRA parameter", "update_id": update_id}

    @app.post("/update_lora_params/")
    async def update_lora_params(request: UpdateLoRAParamsRequest = Body(...)):
        if not request.tensors:
            raise HTTPException(status_code=400, detail="No LoRA tensors provided.")

        tensor_specs: list[tuple[str, torch.dtype, tuple[int, ...]]] = []
        for tensor in request.tensors:
            try:
                dtype = _dtype_from_wire(tensor.dtype)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            try:
                parse_vllm_lora_tensor_name(tensor.name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            tensor_specs.append((tensor.name, dtype, tuple(tensor.shape)))

        try:
            llm.collective_rpc(method="validate_lora_params", args=(tensor_specs, args.lora_int_id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        update_id = uuid.uuid4().hex
        with update_status_lock:
            update_statuses[update_id] = {"state": "pending", "error": None}

        def _dispatch_update() -> None:
            try:
                llm.collective_rpc(
                    method="update_lora_params",
                    args=(tensor_specs, args.lora_int_id),
                )
            except Exception as exc:
                with update_status_lock:
                    update_statuses[update_id] = {"state": "error", "error": str(exc)}
            else:
                with update_status_lock:
                    update_statuses[update_id] = {"state": "ok", "error": None}

        threading.Thread(target=_dispatch_update, daemon=True).start()
        return {"message": "Request received, updating LoRA parameters", "update_id": update_id}

    @app.get("/lora_update_status/{update_id}")
    async def lora_update_status(update_id: str):
        with update_status_lock:
            status = update_statuses.get(update_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Unknown LoRA update id: {update_id}")
        return status

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        resetter = getattr(llm, "reset_prefix_cache", None)
        if not callable(resetter):
            raise HTTPException(status_code=501, detail="vLLM reset_prefix_cache() is not available in this server.")
        success = resetter()
        return {"message": f"Request received, resetting prefix cache status: {bool(success)}"}

    @app.post("/close_communicator/")
    async def close_communicator():
        threading.Thread(
            target=llm.collective_rpc,
            kwargs={"method": "close_communicator"},
            daemon=True,
        ).start()
        return {"message": "Request received, closing communicator"}

    return app


def main() -> None:
    args = parse_server_args()
    if args.data_parallel_size != 1:
        raise SystemExit(
            "This project-local vLLM LoRA server currently supports --data-parallel-size 1 only. "
            "Use tensor parallelism for this endpoint until LoRA hot-sync data parallel workers are implemented."
        )
    try:
        import uvicorn
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing vLLM LoRA server dependency: {exc}") from exc

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        worker_extension_cls="rl_training.vllm_lora_server.WeightSyncLoRAWorkerExtension",
    )
    register_lora_adapter_on_workers(llm, args)
    app = create_app(args, llm=llm, sampling_params_cls=SamplingParams)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
