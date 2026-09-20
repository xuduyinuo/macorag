from __future__ import annotations

import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_TOKEN_ID = re.compile(r"^token_id:(\d+)$")


@dataclass(frozen=True)
class VLLMOutput:
    token_ids: list[int]
    token_logprobs: list[float]
    text: str


class VLLMClient:
    """Proxy-free client for vLLM's OpenAI and runtime-LoRA endpoints."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        model_name: str,
        lora_name: str,
        timeout: float,
        attempts: int,
        backoff: float,
        initial_adapter_path: str | Path | None = None,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        try:
            import requests
        except ModuleNotFoundError as exc:
            raise SystemExit("requests is required for vLLM MAPPO generation") from exc
        self.requests = requests
        self.base_url = f"http://{host}:{port}"
        self.model_name = model_name
        self.lora_name = lora_name
        self.timeout = timeout
        self.attempts = attempts
        self.backoff = backoff
        self.sleep = sleep
        self.session = session or requests.Session()
        self._provided_session = session is not None
        self._session_owner = threading.get_ident()
        self._thread_local = threading.local()
        # Loopback requests must never follow the user's ambient HTTP proxy.
        self.session.trust_env = False
        self.current_adapter_path = (
            Path(initial_adapter_path).resolve()
            if initial_adapter_path is not None else None
        )

    def _request_session(self) -> Any:
        """Use one requests.Session per rollout worker.

        A Session is stateful and is not guaranteed to be thread-safe. Tests and
        explicit callers may still inject one deterministic session; production
        workers get independent connection pools while retaining keep-alive.
        """
        if self._provided_session or threading.get_ident() == self._session_owner:
            return self.session
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = self.requests.Session()
            session.trust_env = False
            self._thread_local.session = session
        return session

    def _request(self, method: str, path: str, *, payload: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                response = self._request_session().request(
                    method, f"{self.base_url}{path}", json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"vLLM {path} failed: HTTP {response.status_code}: {response.text}"
                    )
                return response
            except (self.requests.ConnectionError, self.requests.Timeout) as exc:
                last_error = exc
                if attempt + 1 < self.attempts:
                    self.sleep(min(self.timeout, self.backoff * (2 ** attempt)))
        raise RuntimeError(f"Unable to reach vLLM at {self.base_url}: {last_error}")

    def check_server(self) -> dict[str, Any]:
        response = self._request("GET", "/v1/models")
        payload = response.json()
        visible = {
            str(item.get("id"))
            for item in payload.get("data", [])
            if isinstance(item, dict)
        }
        if self.lora_name not in visible:
            raise RuntimeError(
                f"vLLM server does not expose LoRA {self.lora_name!r}; visible={sorted(visible)}"
            )
        if self.model_name not in visible:
            raise RuntimeError(
                f"vLLM server base model mismatch: expected {self.model_name!r}; "
                f"visible={sorted(visible)}"
            )
        return {"visible_models": sorted(visible), "base_url": self.base_url}

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        guided_regex: str | None = None,
    ) -> VLLMOutput:
        request_payload = {
            "model": self.lora_name,
            "prompt": prompt_ids,
            "add_special_tokens": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
            "stop_token_ids": [],
        }
        if guided_regex is not None:
            request_payload["guided_regex"] = guided_regex
        response = self._request(
            "POST", "/v1/completions", payload=request_payload,
        )
        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("vLLM must return exactly one completion")
        choice = choices[0]
        logprobs = choice.get("logprobs") or {}
        wire_tokens = logprobs.get("tokens")
        token_logprobs = logprobs.get("token_logprobs")
        if not isinstance(wire_tokens, list) or not isinstance(token_logprobs, list):
            raise RuntimeError("vLLM response omitted completion token logprobs")
        token_ids: list[int] = []
        for token in wire_tokens:
            match = _TOKEN_ID.fullmatch(str(token))
            if match is None:
                raise RuntimeError(
                    "vLLM did not return token IDs; ensure return_tokens_as_token_ids is supported"
                )
            token_ids.append(int(match.group(1)))
        values = [float(value) for value in token_logprobs]
        if len(token_ids) != len(values):
            raise RuntimeError("vLLM token/logprob length mismatch")
        return VLLMOutput(
            token_ids=token_ids,
            token_logprobs=values,
            text=str(choice.get("text") or ""),
        )

    def sync_lora(
        self,
        model: Any,
        tokenizer: Any,
        *,
        sync_root: Path,
        step: int,
        keep: int,
    ) -> Path:
        """Atomically publish and reload a unique LoRA snapshot."""
        sync_root.mkdir(parents=True, exist_ok=True)
        target = sync_root / f"step-{step:08d}"
        temporary = sync_root / f".step-{step:08d}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        save_kwargs: dict[str, Any] = {"safe_serialization": True}
        policy_adapter = getattr(model, "_mappo_policy_adapter_name", None)
        if policy_adapter is not None:
            save_kwargs["selected_adapters"] = [policy_adapter]
        model.save_pretrained(temporary, **save_kwargs)
        tokenizer.save_pretrained(temporary)
        (temporary / "READY").write_text("ok\n", encoding="utf-8")
        temporary.replace(target)

        previous = self.current_adapter_path
        self._request("POST", "/v1/unload_lora_adapter", payload={
            "lora_name": self.lora_name,
        })
        try:
            self._request("POST", "/v1/load_lora_adapter", payload={
                "lora_name": self.lora_name,
                "lora_path": str(target.resolve()),
            })
        except Exception:
            if previous is not None and previous.is_dir():
                self._request("POST", "/v1/load_lora_adapter", payload={
                    "lora_name": self.lora_name,
                    "lora_path": str(previous.resolve()),
                })
            raise
        self.current_adapter_path = target
        snapshots = sorted(path for path in sync_root.glob("step-*") if (path / "READY").is_file())
        for stale in snapshots[:-keep]:
            if stale != self.current_adapter_path:
                shutil.rmtree(stale)
        self.check_server()
        return target
