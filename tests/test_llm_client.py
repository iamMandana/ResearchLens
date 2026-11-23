import json
import urllib.error

import pytest

from src import llm_client


class FakeResponse:
    """Mimics the object returned by urllib.request.urlopen."""

    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_generate_returns_model_reply(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"response": "  Hello!  "})

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)

    reply = llm_client.generate("Say hello", model="test-model")

    assert reply == "Hello!"
    assert sent["payload"]["model"] == "test-model"
    assert sent["payload"]["prompt"] == "Say hello"
    assert sent["payload"]["stream"] is False


def test_empty_prompt_raises_error():
    with pytest.raises(ValueError):
        llm_client.generate("   ")


def test_unreachable_server_gives_clear_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Ollama"):
        llm_client.generate("Hello")


def test_missing_model_gives_clear_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="ollama pull"):
        llm_client.generate("Hello", model="missing-model")