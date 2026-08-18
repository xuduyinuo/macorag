from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .vllm_lora_mapping import collect_lora_named_tensors


@dataclass(frozen=True)
class VLLMGenerationOutput:
    completion_ids: list[int]
    logprobs: list[float] | None = None
    text: str = ""


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

    def validate_lora_server(self, args: Any) -> None:
        session = getattr(self.backend, "session", None)
        base_url = getattr(self.backend, "base_url", None)
        if session is None or base_url is None:
            raise SystemExit("Installed TRL VLLMClient internals are incompatible with LoRA server validation.")

        try:
            response = session.get(f"{base_url}/health/")
            if response.status_code != 200:
                raise SystemExit(f"vLLM LoRA server health check failed: HTTP {response.status_code}, {response.text}")
            health = response.json()
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(f"Unable to validate vLLM LoRA server health endpoint: {exc}") from exc

        mismatches: list[str] = []
        if health.get("sync_mode") != "lora":
            mismatches.append(f"sync_mode expected lora got {health.get('sync_mode')!r}")
        expected_lora_name = getattr(args, "vllm_lora_name", None)
        if health.get("lora_name") != expected_lora_name:
            mismatches.append(f"lora_name expected {expected_lora_name!r} got {health.get('lora_name')!r}")
        expected_lora_int_id = getattr(args, "vllm_lora_int_id", None)
        if health.get("lora_int_id") != expected_lora_int_id:
            mismatches.append(f"lora_int_id expected {expected_lora_int_id!r} got {health.get('lora_int_id')!r}")
        if "model" in health and health.get("model") != getattr(args, "model_path", None):
            mismatches.append(f"model expected {getattr(args, 'model_path', None)!r} got {health.get('model')!r}")
        if "lora_adapter_path" in health and health.get("lora_adapter_path") != getattr(
            args, "vllm_lora_adapter_path", None
        ):
            mismatches.append(
                "lora_adapter_path expected "
                f"{getattr(args, 'vllm_lora_adapter_path', None)!r} got {health.get('lora_adapter_path')!r}"
            )
        if mismatches:
            raise SystemExit("LoRA server identity mismatch: " + "; ".join(mismatches))

        if health.get("supports_lora_param_update") is not True:
            raise SystemExit(
                "LoRA hot sync is unsupported by the connected vLLM server: "
                "health endpoint did not report supports_lora_param_update=true"
            )

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
        outputs = self.generate_batch(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return outputs[0].completion_ids, outputs[0].text

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[VLLMGenerationOutput]:
        if not prompts:
            return []
        request_payload = {
            "prompts": prompts,
            "n": 1,
            "repetition_penalty": 1.0,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
        }
        session = getattr(self.backend, "session", None)
        base_url = getattr(self.backend, "base_url", None)
        if session is not None and base_url is not None:
            response = session.post(f"{base_url}/generate/", json=request_payload)
            if response.status_code != 200:
                raise RuntimeError(f"vLLM generation failed: HTTP {response.status_code}, {response.text}")
            try:
                payload = response.json()
            except Exception as exc:
                raise RuntimeError(f"Invalid /generate/ response: {exc}") from exc
            return self._outputs_from_payload(prompts, payload)

        generator = getattr(self.backend, "generate", None)
        if generator is None:
            raise SystemExit("TRL VLLMClient is missing generate(); installed TRL is incompatible.")
        outputs = generator(
            **request_payload,
        )
        completion_ids = []
        for output in outputs:
            if output and isinstance(output[0], list):
                if len(output) != 1:
                    raise RuntimeError("vLLM returned multiple completions when n=1.")
                output = output[0]
            completion_ids.append(list(output))
        return self._outputs_from_payload(prompts, {"completion_ids": completion_ids})

    @staticmethod
    def _outputs_from_payload(
        prompts: list[str],
        payload: dict[str, Any],
    ) -> list[VLLMGenerationOutput]:
        completion_ids = payload.get("completion_ids")
        if not isinstance(completion_ids, list) or len(completion_ids) != len(prompts):
            actual = len(completion_ids) if isinstance(completion_ids, list) else "invalid"
            raise RuntimeError(
                "vLLM returned a mismatched generation batch size: "
                f"expected {len(prompts)}, got {actual}."
            )
        payload_logprobs = payload.get("logprobs")
        if payload_logprobs is not None and (
            not isinstance(payload_logprobs, list) or len(payload_logprobs) != len(prompts)
        ):
            raise RuntimeError("vLLM returned a mismatched logprob batch size.")

        results: list[VLLMGenerationOutput] = []
        for index, token_ids in enumerate(completion_ids):
            tokens = [int(item) for item in token_ids]
            logprobs = None
            if payload_logprobs is not None:
                logprobs = [float(item) for item in payload_logprobs[index]]
                if len(logprobs) != len(tokens):
                    raise RuntimeError(
                        "vLLM completion/logprob length mismatch: "
                        f"completion {index} has {len(tokens)} tokens and {len(logprobs)} logprobs."
                    )
            results.append(VLLMGenerationOutput(completion_ids=tokens, logprobs=logprobs))
        return results

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
        session = getattr(self.backend, "session", None)
        base_url = getattr(self.backend, "base_url", None)
        if session is None or base_url is None:
            raise SystemExit("Installed TRL VLLMClient internals are incompatible with LoRA hot sync.")
        response = session.post(
            f"{base_url}/update_lora_params/",
            json={
                "tensors": [
                    {"name": name, "dtype": str(tensor.dtype), "shape": tuple(tensor.shape)}
                    for name, tensor in tensors.items()
                ]
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Invalid /update_lora_params/ response: {exc}") from exc
        update_id = payload.get("update_id")
        if not isinstance(update_id, str) or not update_id:
            raise RuntimeError("/update_lora_params/ response missing update_id.")
        communicator = getattr(self.backend, "pynccl_comm", None)
        rank = getattr(self.backend, "rank", None)
        if communicator is None or rank is None:
            raise SystemExit("vLLM LoRA hot sync communicator is not initialized.")
        for tensor in tensors.values():
            communicator.broadcast(tensor, src=rank)
        communicator.group.barrier()
        self._poll_lora_update_status(session, base_url, update_id)
        return time.perf_counter() - start

    def _poll_lora_update_status(self, session: Any, base_url: str, update_id: str) -> None:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            response = session.get(f"{base_url}/lora_update_status/{update_id}")
            if response.status_code != 200:
                raise RuntimeError(
                    f"GET /lora_update_status/{update_id} returned HTTP {response.status_code}, {response.text}"
                )
            try:
                payload = response.json()
            except Exception as exc:
                raise RuntimeError(f"Invalid LoRA update status response for {update_id}: {exc}") from exc

            state = payload.get("state")
            if state == "ok":
                return
            if state == "error":
                raise RuntimeError(f"vLLM LoRA update failed for {update_id}: {payload.get('error')}")
            if state != "pending":
                raise RuntimeError(f"Unknown vLLM LoRA update status for {update_id}: {state!r}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for vLLM LoRA update {update_id}.")
            time.sleep(0.05)
