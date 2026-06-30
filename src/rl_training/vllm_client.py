from __future__ import annotations

import time
from typing import Any


def collect_trainable_named_parameters(model: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if getattr(parameter, "requires_grad", False):
            params[name] = parameter.detach().float().cpu()
    return params


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
            self._backend = VLLMClient(host=self.host, port=self.port, connection_timeout=self.timeout_seconds)
        return self._backend

    def check_server(self) -> None:
        checker = getattr(self.backend, "check_server", None)
        if checker is None:
            raise SystemExit("TRL VLLMClient is missing check_server(); installed TRL is incompatible.")
        checker()

    def generate(
        self,
        prompt_token_ids: list[int],
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
            prompts=[prompt_token_ids],
            n=1,
            repetition_penalty=1.0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        first = outputs[0]
        token_ids = list(getattr(first, "token_ids", None) or first.get("token_ids", []))
        text = str(getattr(first, "text", None) or first.get("text", ""))
        return token_ids, text

    def sync_trainable_parameters(self, model: Any) -> float:
        updater = getattr(self.backend, "update_named_param", None)
        if updater is None:
            raise SystemExit("TRL VLLMClient is missing update_named_param(); hot LoRA sync is unavailable.")
        start = time.perf_counter()
        for name, tensor in collect_trainable_named_parameters(model).items():
            updater(name, tensor)
        return time.perf_counter() - start
