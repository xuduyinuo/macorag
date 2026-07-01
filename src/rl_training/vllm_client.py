from __future__ import annotations

import time
from typing import Any

from .vllm_lora_mapping import collect_lora_named_tensors


def collect_trainable_named_parameters(model: Any, *, device: Any | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if getattr(parameter, "requires_grad", False):
            tensor = parameter.detach().float()
            if device is None:
                tensor = tensor.cpu()
            else:
                tensor = tensor.to(device=device)
            params[name] = tensor
    return params


def _backend_communicator_device(backend: Any) -> Any | None:
    communicator = getattr(backend, "pynccl_comm", None)
    return getattr(communicator, "device", None)


def _is_peft_model(model: Any) -> bool:
    return callable(getattr(model, "merge_adapter", None)) and callable(getattr(model, "unmerge_adapter", None))


def _peft_parameter_name_for_vllm(model: Any, name: str) -> str | None:
    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
    if getattr(model, "prefix", "") and getattr(model, "prefix") in name:
        return None
    if "original_module" in name:
        return None
    return name.replace("modules_to_save.default.", "")


def _dequantize_bnb_weight(parameter: Any, state: Any | None = None) -> Any:
    from peft.tuners.lora.bnb import dequantize_bnb_weight

    return dequantize_bnb_weight(parameter, state=state)


def _move_tensor_for_sync(parameter: Any, *, device: Any | None) -> Any:
    if parameter.__class__.__name__ == "Params4bit":
        tensor = _dequantize_bnb_weight(parameter, state=getattr(parameter, "quant_state", None)).detach()
    else:
        tensor = parameter.detach()
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


class VLLMGenerationClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_seconds: float,
        backend: Any | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self._backend = backend
        self._communicator_initialized = False

    @property
    def backend(self) -> Any:
        if self._backend is None:
            try:
                from trl.extras.vllm_client import VLLMClient
            except ModuleNotFoundError as exc:
                raise SystemExit(
                    "TRL is required for vLLM GRPO generation. Install a TRL version that provides "
                    "trl.extras.vllm_client.VLLMClient in the macorag environment."
                ) from exc
            except ImportError as exc:
                raise SystemExit(
                    "Installed TRL does not expose trl.extras.vllm_client.VLLMClient. "
                    "Install a TRL/vLLM combination with vLLM training support."
                ) from exc
            self._backend = VLLMClient(
                host=self.host,
                server_port=self.port,
                connection_timeout=self.timeout_seconds,
            )
        return self._backend

    def check_server(self) -> None:
        checker = getattr(self.backend, "check_server", None)
        if checker is None:
            raise SystemExit("TRL VLLMClient is missing check_server(); installed TRL is incompatible.")
        checker()

    def _ensure_communicator(self) -> None:
        if self._communicator_initialized:
            return
        initializer = getattr(self.backend, "init_communicator", None)
        if initializer is None:
            raise SystemExit("TRL VLLMClient is missing init_communicator(); hot sync is unavailable.")
        initializer()
        self._communicator_initialized = True

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> tuple[list[int], str]:
        generator = getattr(self.backend, "generate", None)
        if generator is None:
            raise SystemExit("TRL VLLMClient is missing generate(); installed TRL is incompatible.")
        outputs = generator(
            prompts=[prompt],
            n=1,
            repetition_penalty=1.0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        first = outputs[0]
        if first and isinstance(first[0], list):
            first = first[0]
        return list(first), ""

    def sync_trainable_parameters(self, model: Any) -> float:
        updater = getattr(self.backend, "update_named_param", None)
        if updater is None:
            raise SystemExit("TRL VLLMClient is missing update_named_param(); hot LoRA sync is unavailable.")
        self._ensure_communicator()
        sync_device = _backend_communicator_device(self.backend)
        start = time.perf_counter()
        if _is_peft_model(model):
            model.merge_adapter()
            try:
                for name, parameter in model.named_parameters():
                    vllm_name = _peft_parameter_name_for_vllm(model, name)
                    if vllm_name is None:
                        continue
                    updater(vllm_name, _move_tensor_for_sync(parameter, device=sync_device))
            finally:
                model.unmerge_adapter()
        else:
            for name, tensor in collect_trainable_named_parameters(model, device=sync_device).items():
                updater(name, tensor)
        return time.perf_counter() - start

    def sync_lora_parameters(self, model: Any) -> float:
        self._ensure_communicator()
        sync_device = _backend_communicator_device(self.backend)
        tensors = collect_lora_named_tensors(model, device=sync_device)
        if not tensors:
            raise SystemExit("No LoRA tensors found for vLLM LoRA hot sync.")
        start = time.perf_counter()
        for name, tensor in tensors.items():
            session = getattr(self.backend, "session", None)
            base_url = getattr(self.backend, "base_url", None)
            if session is None or base_url is None:
                raise SystemExit("Installed TRL VLLMClient internals are incompatible with LoRA hot sync.")
            response = session.post(
                f"{base_url}/update_lora_param/",
                json={"name": name, "dtype": str(tensor.dtype), "shape": tuple(tensor.shape)},
            )
            if response.status_code != 200:
                raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")
            communicator = getattr(self.backend, "pynccl_comm", None)
            rank = getattr(self.backend, "rank", None)
            if communicator is None or rank is None:
                raise SystemExit("vLLM LoRA hot sync communicator is not initialized.")
            communicator.broadcast(tensor, src=rank)
            communicator.group.barrier()
        return time.perf_counter() - start
