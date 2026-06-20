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
