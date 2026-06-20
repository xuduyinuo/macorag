import json
import urllib.request

import pytest

from macorag.teacher_api import FakeTeacherClient, OpenAICompatibleClient


def test_fake_teacher_client_returns_configured_responses_and_records_calls():
    client = FakeTeacherClient(["first response", "second response"])

    assert client.generate("first prompt") == "first response"
    assert client.generate("second prompt") == "second response"
    assert client.calls == ["first prompt", "second prompt"]


def test_fake_teacher_client_raises_when_responses_are_exhausted():
    client = FakeTeacherClient([])

    with pytest.raises(RuntimeError, match="no remaining responses"):
        client.generate("prompt")

    assert client.calls == ["prompt"]


def test_openai_compatible_client_raises_when_api_key_env_missing(monkeypatch):
    monkeypatch.delenv("MISSING_TEACHER_API_KEY", raising=False)
    client = OpenAICompatibleClient(
        api_key_env="MISSING_TEACHER_API_KEY",
        base_url="https://example.com/v1",
        model="teacher-model",
    )

    with pytest.raises(RuntimeError, match="MISSING_TEACHER_API_KEY"):
        client.generate("prompt")


def test_openai_compatible_client_posts_chat_completion_request(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self):
            return b'{"choices":[{"message":{"content":"teacher reply"}}]}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("TEACHER_API_KEY", "secret-key")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        api_key_env="TEACHER_API_KEY",
        base_url="https://example.com/v1/",
        model="teacher-model",
        temperature=0.4,
        max_tokens=512,
    )

    result = client.generate("teacher prompt")

    request = captured["request"]
    body = json.loads(request.data.decode("utf-8"))
    assert result == "teacher reply"
    assert request.full_url == "https://example.com/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer secret-key"
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["content-type"] == "application/json"
    assert body == {
        "model": "teacher-model",
        "messages": [{"role": "user", "content": "teacher prompt"}],
        "temperature": 0.4,
        "max_tokens": 512,
    }
    assert captured["timeout"] == 120
