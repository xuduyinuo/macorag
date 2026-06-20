from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import json
import os
import urllib.request


class TeacherClient(Protocol):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class FakeTeacherClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError("FakeTeacherClient has no remaining responses")
        return self.responses.pop(0)


@dataclass
class OpenAICompatibleClient:
    api_key_env: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048

    def generate(self, prompt: str) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing api key env var: {self.api_key_env}")

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
