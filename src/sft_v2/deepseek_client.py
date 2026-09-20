from __future__ import annotations

import json
import os
import random
import time
from typing import Any
import urllib.error
import urllib.request

from .config import TeacherConfig


class DeepSeekClient:
    """Minimal OpenAI-compatible DeepSeek chat client with bounded retries."""

    def __init__(self, config: TeacherConfig) -> None:
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {config.api_key_env}")
        self.config = config
        self._api_key = api_key
        self.endpoint = config.api_base.rstrip("/") + "/chat/completions"

    def complete_json(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.config.teacher_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.config.thinking},
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: BaseException | None = None
        for attempt in range(1, self.config.request_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choice = body["choices"][0]
                content = choice["message"]["content"]
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("DeepSeek JSON response must be an object")
                usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
                trace = {
                    "request_id": body.get("id"),
                    "model": body.get("model") or self.config.teacher_model,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": usage,
                    "raw_content": content,
                }
                return parsed, trace
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt >= self.config.request_retries:
                    break
                delay = self.config.retry_base_seconds * (2 ** (attempt - 1))
                time.sleep(delay + random.random() * min(1.0, delay / 4))
        raise RuntimeError(f"DeepSeek request failed after {self.config.request_retries} attempts: {last_error}")

