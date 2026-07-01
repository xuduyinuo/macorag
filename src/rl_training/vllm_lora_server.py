import argparse
import os
import re
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


class WeightSyncLoRAWorkerExtension:
    pynccl_comm = None
    client_rank = None

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

    def update_lora_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
        raise RuntimeError(
            "LoRA in-memory tensor replacement is not implemented for vLLM 0.8.5.post1 yet. "
            f"Requested {name} dtype={dtype} shape={tuple(shape)}."
        )

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
    loras = getattr(lora_model, "loras", {})
    if module_name not in loras:
        raise KeyError(f"Registered LoRA module not found: {module_name}")

    layer = loras[module_name]
    target = getattr(layer, "lora_a" if side == "lora_A" else "lora_b", None)
    if target is None:
        raise KeyError(f"Registered LoRA tensor side not found: {module_name}.{side}")

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
    return bool(adapter_manager.activate_adapter(lora_int_id))


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

    @app.get("/health/")
    async def health():
        return {
            "status": "ok",
            "sync_mode": "lora",
            "model": args.model,
            "lora_name": args.lora_name,
            "lora_int_id": args.lora_int_id,
            "lora_adapter_path": args.lora_adapter_path,
            "supports_lora_param_update": False,
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
        completion_ids = [list(output.token_ids) for outputs_item in outputs for output in outputs_item.outputs]
        return {"completion_ids": completion_ids}

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest = Body(...)):
        world_size = args.tensor_parallel_size * args.data_parallel_size + 1
        llm.collective_rpc(method="init_communicator", args=(request.host, request.port, world_size))
        return {"message": "Request received, initializing communicator"}

    @app.post("/update_lora_param/")
    async def update_lora_param(request: UpdateLoRAParamRequest = Body(...)):
        try:
            _dtype_from_wire(request.dtype)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(
            status_code=501,
            detail=(
                "LoRA in-memory tensor replacement is not implemented for this vLLM version yet. "
                f"Requested {request.name} shape={request.shape}."
            ),
        )

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        resetter = getattr(llm, "reset_prefix_cache", None)
        if not callable(resetter):
            raise HTTPException(status_code=501, detail="vLLM reset_prefix_cache() is not available in this server.")
        success = resetter()
        return {"message": f"Request received, resetting prefix cache status: {bool(success)}"}

    @app.post("/close_communicator/")
    async def close_communicator():
        llm.collective_rpc(method="close_communicator")
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
    app = create_app(args, llm=llm, sampling_params_cls=SamplingParams)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
