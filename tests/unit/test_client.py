"""Real httpx client semantics with in-process, credential-free transports."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from strands_neuraltrust._client import TrustGuardClient
from strands_neuraltrust.config import TrustGuardConfig
from strands_neuraltrust.exceptions import (
    TrustGuardAuthenticationError,
    TrustGuardConfigurationError,
    TrustGuardProtocolError,
    TrustGuardStateError,
    TrustGuardUnavailable,
)


@pytest.fixture
def config() -> TrustGuardConfig:
    return TrustGuardConfig(api_key="PRIVATE-KEY", base_url="https://guard.example")


@pytest.mark.parametrize("status", ["allow", "report", "transform", "ask", "block"])
def test_sync_wire_contract_and_advisory_status(config: TrustGuardConfig, status: str) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        data: dict[str, Any] = {"status": status}
        if status == "transform":
            data["transformed_payload"] = {"input": "safe"}
        return httpx.Response(200, json=data)

    with httpx.Client(transport=httpx.MockTransport(handler), auth=("wrong", "wrong")) as http:
        with TrustGuardClient(config, http_client=http) as guard:
            verdict = guard.evaluate(
                {"input": "hello"},
                "output",
                session_id="session",
                consumer_id="actor",
                attributes={"tool": {"name": "test"}},
            )
        assert not http.is_closed
    assert verdict.status == status
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://guard.example/v1/evaluate"
    assert request.method == "POST"
    assert request.headers["authorization"] == "Bearer PRIVATE-KEY"
    assert request.headers["accept-encoding"] == "identity"
    assert json.loads(request.content) == {
        "payload": {"input": "hello"},
        "direction": "output",
        "protocol": "llm",
        "session_id": "session",
        "consumer_id": "actor",
        "attributes": {"tool": {"name": "test"}},
    }
    assert set(request.extensions["timeout"].values()) == {5.0}


@pytest.mark.parametrize(
    "code,error",
    [
        (400, TrustGuardProtocolError),
        (401, TrustGuardAuthenticationError),
        (403, TrustGuardAuthenticationError),
        (404, TrustGuardProtocolError),
        (201, TrustGuardProtocolError),
        (204, TrustGuardProtocolError),
        (429, TrustGuardUnavailable),
        (500, TrustGuardUnavailable),
        (502, TrustGuardUnavailable),
        (503, TrustGuardUnavailable),
        (504, TrustGuardUnavailable),
    ],
)
def test_status_failures_are_sanitized_and_never_retried(
    config: TrustGuardConfig, code: int, error: type
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(code, text="PRIVATE BODY AND KEY")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        guard = TrustGuardClient(config, http_client=http)
        with pytest.raises(error) as caught:
            guard.evaluate({"input": "PRIVATE-PROMPT"})
    assert len(requests) == 1
    assert "PRIVATE" not in repr(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize("target", ["https://evil.example/steal", "/other"])
@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirects_never_forward_credentials_or_body(
    config: TrustGuardConfig, target: str, code: int
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(code, headers={"location": target})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as http:
        with pytest.raises(TrustGuardProtocolError):
            TrustGuardClient(config, http_client=http).evaluate({"input": "PRIVATE"})
    assert len(requests) == 1
    assert requests[0].url.host == "guard.example"


@pytest.mark.parametrize(
    "exception",
    [
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
        httpx.DecodingError,
        OSError,
        ValueError,
        RuntimeError,
    ],
)
def test_sync_transport_failure_has_no_raw_exception_context(
    config: TrustGuardConfig, exception: type
) -> None:
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        raise exception("PRIVATE KEY AND BODY")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(TrustGuardUnavailable) as caught:
            TrustGuardClient(config, http_client=http).evaluate({"input": "PRIVATE"})
    assert count == 1
    assert "PRIVATE" not in repr(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"content-type": "text/plain"},
        {"content-type": "application/json", "content-encoding": "gzip"},
        {"content-type": "application/json", "content-length": "-1"},
        {"content-type": "application/json", "content-length": "1, 2"},
        {"content-type": "application/json", "content-length": "9" * 21},
        {"content-type": "application/json", "content-length": "1048577"},
    ],
)
def test_invalid_response_headers(config: TrustGuardConfig, headers: dict[str, str]) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers=headers, stream=Chunks([b'{"status":"allow"}']))
        )
    ) as http:
        with pytest.raises(TrustGuardProtocolError):
            TrustGuardClient(config, http_client=http).evaluate({})


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes], *, delay: float = 0) -> None:
        self.chunks = chunks
        self.read_count = 0
        self.closed = False
        self.delay = delay

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            if self.delay:
                time.sleep(self.delay)
            self.read_count += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


def test_chunked_body_limit_stops_consumption_and_closes_response() -> None:
    stream = Chunks([b"x" * 8, b"x" * 8, b"SHOULD NOT READ"])
    config = TrustGuardConfig(api_key="key", base_url="https://guard.example", max_response_bytes=10)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "application/json"}, stream=stream)
        )
    ) as http:
        with pytest.raises(TrustGuardProtocolError):
            TrustGuardClient(config, http_client=http).evaluate({})
    assert stream.read_count == 2
    assert stream.closed


def test_sync_elapsed_deadline_checked_between_chunks() -> None:
    stream = Chunks([b'{"status":', b'"allow"}'], delay=0.02)
    config = TrustGuardConfig(api_key="key", base_url="https://guard.example", timeout=0.01)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "application/json"}, stream=stream)
        )
    ) as http:
        with pytest.raises(TrustGuardUnavailable):
            TrustGuardClient(config, http_client=http).evaluate({})
    assert stream.read_count == 1
    assert stream.closed


def test_sync_deadline_checked_after_last_chunk(config: TrustGuardConfig, monkeypatch: Any) -> None:
    ticks = iter([0, 0, 10])
    monkeypatch.setattr("strands_neuraltrust._client.time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "allow"}))
    ) as http:
        with pytest.raises(TrustGuardUnavailable):
            TrustGuardClient(config, http_client=http).evaluate({})


def test_invalid_request_never_reaches_transport() -> None:
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, json={"status": "allow"})

    config = TrustGuardConfig(api_key="key", base_url="https://guard.example", max_request_bytes=5)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(TrustGuardConfigurationError):
            TrustGuardClient(config, http_client=http).evaluate({"input": "too large"})
    assert count == 0


def test_owned_sync_pool_ignores_env_and_is_closed_once(config: TrustGuardConfig, monkeypatch: Any) -> None:
    original = httpx.Client
    clients = []
    options = []

    def factory(**kwargs: Any) -> httpx.Client:
        options.append(kwargs)
        client = original(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "allow"})),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setenv("HTTPS_PROXY", "http://evil.example")
    monkeypatch.setattr("strands_neuraltrust._client.httpx.Client", factory)
    guard = TrustGuardClient(config)
    assert "PRIVATE" not in repr(guard)
    assert clients == []
    with guard:
        assert guard.evaluate({}).status == "allow"
        assert guard.evaluate({}).status == "allow"
    guard.close()
    assert guard.is_closed
    assert len(clients) == 1
    assert clients[0].is_closed
    assert options == [{"trust_env": False, "follow_redirects": False}]
    with pytest.raises(TrustGuardStateError):
        guard.evaluate({})
    with pytest.raises(TrustGuardStateError):
        guard.__enter__()


@pytest.mark.parametrize("status", ["allow", "report", "transform", "ask", "block"])
async def test_async_verdicts_and_injected_ownership(config: TrustGuardConfig, status: str) -> None:
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "status": status,
                **({"transformed_payload": {"input": "safe"}} if status == "transform" else {}),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with TrustGuardClient(config, async_http_client=http) as guard:
            assert (await guard.aevaluate({"input": "hello"}, "output")).status == status
        assert not http.is_closed
        with pytest.raises(TrustGuardStateError):
            await guard.aevaluate({})
        with pytest.raises(TrustGuardStateError):
            await guard.__aenter__()
    assert len(requests) == 1
    assert json.loads(requests[0].content)["direction"] == "output"


@pytest.mark.parametrize(
    "exception",
    [
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        OSError,
        RuntimeError,
        ValueError,
    ],
)
async def test_async_errors_have_no_raw_exception_context(config: TrustGuardConfig, exception: type) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise exception("PRIVATE KEY BODY")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(TrustGuardUnavailable) as caught:
            await TrustGuardClient(config, async_http_client=http).aevaluate({})
    assert "PRIVATE" not in repr(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "code,error",
    [
        (401, TrustGuardAuthenticationError),
        (403, TrustGuardAuthenticationError),
        (429, TrustGuardUnavailable),
        (500, TrustGuardUnavailable),
        (307, TrustGuardProtocolError),
        (400, TrustGuardProtocolError),
    ],
)
async def test_async_status_failures_and_redirects(config: TrustGuardConfig, code: int, error: type) -> None:
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(code, text="PRIVATE", headers={"location": "https://evil.example"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as http:
        with pytest.raises(error) as caught:
            await TrustGuardClient(config, async_http_client=http).aevaluate({})
    assert len(requests) == 1
    assert "PRIVATE" not in repr(caught.value)


async def test_async_deadline_bounds_slow_header_response() -> None:
    cancelled = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        finally:
            cancelled = True
        return httpx.Response(200, json={"status": "allow"})

    config = TrustGuardConfig(api_key="key", base_url="https://guard.example", timeout=0.02)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        started = time.monotonic()
        with pytest.raises(TrustGuardUnavailable) as caught:
            await TrustGuardClient(config, async_http_client=http).aevaluate({})
        assert time.monotonic() - started < 1
    assert cancelled
    assert caught.value.__context__ is None


class AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, wait: asyncio.Event | None = None) -> None:
        self.chunks = chunks
        self.read_count = 0
        self.closed = False
        self.wait = wait
        self.started = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.started.set()
            if self.wait is not None:
                await self.wait.wait()
            self.read_count += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


async def test_async_chunk_limit_closes_before_remaining_body() -> None:
    stream = AsyncChunks([b"x" * 8, b"x" * 8, b"SHOULD NOT READ"])
    config = TrustGuardConfig(api_key="key", base_url="https://guard.example", max_response_bytes=10)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "application/json"}, stream=stream)
        )
    ) as http:
        with pytest.raises(TrustGuardProtocolError):
            await TrustGuardClient(config, async_http_client=http).aevaluate({})
    assert stream.read_count == 2
    assert stream.closed


async def test_cancellation_propagates_and_closes_body(config: TrustGuardConfig) -> None:
    stream = AsyncChunks([b'{"status":"allow"}'], wait=asyncio.Event())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"content-type": "application/json"}, stream=stream)
        )
    ) as http:
        guard = TrustGuardClient(config, async_http_client=http)
        task = asyncio.create_task(guard.aevaluate({}))
        await stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not http.is_closed
    assert stream.closed


async def test_owned_transport_closed_on_cancellation(config: TrustGuardConfig, monkeypatch: Any) -> None:
    original = httpx.AsyncClient
    stream = AsyncChunks([b'{"status":"allow"}'], wait=asyncio.Event())
    clients = []

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        client = original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"content-type": "application/json"}, stream=stream
                )
            ),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr("strands_neuraltrust._client.httpx.AsyncClient", factory)
    task = asyncio.create_task(TrustGuardClient(config).aevaluate({}))
    await stream.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(clients) == 1
    assert clients[0].is_closed
    assert stream.closed


def test_owned_async_clients_are_closed_on_each_original_loop(
    config: TrustGuardConfig, monkeypatch: Any
) -> None:
    original = httpx.AsyncClient
    clients = []
    options = []

    def factory(**kwargs: Any) -> httpx.AsyncClient:
        options.append(kwargs)
        client = original(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "allow"})),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setenv("HTTPS_PROXY", "http://evil.example")
    monkeypatch.setattr("strands_neuraltrust._client.httpx.AsyncClient", factory)
    guard = TrustGuardClient(config)
    assert asyncio.run(guard.aevaluate({})).status == "allow"
    assert asyncio.run(guard.aevaluate({})).status == "allow"
    asyncio.run(guard.aclose())
    assert len(clients) == 2
    assert all(client.is_closed for client in clients)
    assert options == [{"trust_env": False, "follow_redirects": False}] * 2


def test_injected_async_transport_cannot_cross_loops(config: TrustGuardConfig) -> None:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "allow"}))
    )
    guard = TrustGuardClient(config, async_http_client=http)
    try:
        assert asyncio.run(guard.aevaluate({})).status == "allow"
        with pytest.raises(TrustGuardStateError):
            asyncio.run(guard.aevaluate({}))
        guard.close()
        assert not http.is_closed
    finally:
        asyncio.run(http.aclose())


async def test_concurrent_async_evaluations_keep_context_and_results_separate(
    config: TrustGuardConfig,
) -> None:
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        requests.append(data)
        await asyncio.sleep(0)
        return httpx.Response(200, json={"status": "transform", "transformed_payload": data["payload"]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        guard = TrustGuardClient(config, async_http_client=http)
        results = await asyncio.gather(
            *[
                guard.aevaluate({"input": str(index)}, session_id=f"s{index}", consumer_id=f"c{index}")
                for index in range(20)
            ]
        )
    assert [verdict.transformed_payload for verdict in results] == [
        {"input": str(index)} for index in range(20)
    ]
    assert all(row["consumer_id"] == "c" + row["payload"]["input"] for row in requests)
    assert all(row["session_id"] == "s" + row["payload"]["input"] for row in requests)
