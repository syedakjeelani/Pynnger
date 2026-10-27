import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from pynnger.backends import BackendError, make_backend
from pynnger.backends.anthropic_backend import AnthropicBackend
from pynnger.backends.base import Message
from pynnger.backends.openai_compat import OpenAICompatBackend
from pynnger.config import Settings


class FakeAnthropic:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self._response, self._error = response, error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self._error:
            raise self._error
        return self._response


def response(blocks: list[Any], stop: str = "end_turn") -> Any:
    usage = SimpleNamespace(
        input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=2
    )
    return SimpleNamespace(content=blocks, usage=usage, model="claude-haiku-5-5", stop_reason=stop)


def test_anthropic_request_shape_and_usage_mapping() -> None:
    fake = FakeAnthropic(response([SimpleNamespace(type="text", text="hello")]))
    out = AnthropicBackend(client=fake).complete(
        system="SYS",
        messages=[Message("user", "body", cache_prefix="PREFIX")],
        model="claude-haiku-5-5",
        max_tokens=100,
    )
    kw = fake.kwargs
    assert kw["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    blocks = kw["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "PREFIX", "cache_control": {"type": "ephemeral"}}
    assert blocks[1] == {"type": "text", "text": "body"}
    assert "tools" not in kw and "temperature" not in kw
    assert out.text == "hello" and out.stop_reason == "end"
    assert (out.usage.input_tokens, out.usage.output_tokens) == (10, 5)
    assert (out.usage.cache_read_tokens, out.usage.cache_write_tokens) == (3, 2)


def test_anthropic_plain_message_has_no_cache_markers_when_disabled() -> None:
    fake = FakeAnthropic(response([SimpleNamespace(type="text", text="x")]))
    AnthropicBackend(client=fake).complete(
        system="S", messages=[Message("user", "plain")], model="m", max_tokens=1, cache_system=False
    )
    assert fake.kwargs["system"] == [{"type": "text", "text": "S"}]
    assert fake.kwargs["messages"][0]["content"] == "plain"


def test_anthropic_schema_forces_tool_use_and_returns_json_text() -> None:
    tool_block = SimpleNamespace(type="tool_use", input={"a": 1})
    fake = FakeAnthropic(response([tool_block], stop="max_tokens"))
    out = AnthropicBackend(client=fake).complete(
        system="S",
        messages=[Message("user", "u")],
        model="m",
        max_tokens=1,
        json_schema={"type": "object"},
    )
    assert fake.kwargs["tool_choice"] == {"type": "tool", "name": "emit"}
    assert fake.kwargs["tools"][0]["input_schema"] == {"type": "object"}
    assert json.loads(out.text) == {"a": 1} and out.stop_reason == "max_tokens"


def test_anthropic_errors_become_backend_errors() -> None:
    fake = FakeAnthropic(error=RuntimeError("overloaded"))
    with pytest.raises(BackendError, match="overloaded"):
        AnthropicBackend(client=fake).complete(
            system="S", messages=[Message("user", "u")], model="m", max_tokens=1
        )


def test_openai_compat_round_trip() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["json"], seen["auth"] = (
            str(request.url),
            json.loads(request.content),
            request.headers.get("authorization"),
        )
        body = {
            "model": "pynnger-7b",
            "choices": [
                {"message": {"content": "```python\nx = 1\n```"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 7},
        }
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = OpenAICompatBackend("http://localhost:8000/v1/", api_key="k", client=client)
    out = backend.complete(
        system="SYS",
        messages=[Message("user", "body", cache_prefix="PRE-")],
        model="pynnger-7b",
        max_tokens=50,
        json_schema={"type": "object"},
    )
    assert seen["url"] == "http://localhost:8000/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert seen["json"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "PRE-body"},
    ]
    assert seen["json"]["response_format"] == {"type": "json_object"}
    assert (out.usage.input_tokens, out.usage.output_tokens, out.stop_reason) == (
        40,
        7,
        "max_tokens",
    )


def test_openai_compat_http_errors_become_backend_errors() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(BackendError):
        OpenAICompatBackend("http://x/v1", client=client).complete(
            system="S", messages=[Message("user", "u")], model="m", max_tokens=1
        )


def test_factory() -> None:
    assert make_backend(Settings(backend="mock")).__class__.__name__ == "ScriptedBackend"
    with pytest.raises(BackendError):
        make_backend(Settings(backend="openai"))
    with pytest.raises(BackendError):
        make_backend(Settings(backend="nope"))
