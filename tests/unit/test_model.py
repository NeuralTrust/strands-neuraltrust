"""Provider-boundary cleanup and confidentiality, without patching the SDK."""

import asyncio
from typing import Any

import pytest

from strands_neuraltrust import TrustGuardUnavailable, TrustGuardUnsupportedContentError
from strands_neuraltrust._model import SanitizedModel
from tests.fakes import FakeModel, text_response


async def test_provider_exception_sanitized_before_sdk_sees_it() -> None:
    model = SanitizedModel(FakeModel([RuntimeError("SENSITIVE-PROVIDER-BODY")]))
    with pytest.raises(TrustGuardUnavailable) as caught:
        async for _ in model.stream([]):
            pass
    assert "SENSITIVE" not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None


async def test_partial_stream_closed_on_consumer_exit() -> None:
    closed = asyncio.Event()

    class Provider(FakeModel):
        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            try:
                yield {"messageStart": {"role": "assistant"}}
                await asyncio.Event().wait()
            finally:
                closed.set()

    stream = SanitizedModel(Provider([])).stream([])
    await anext(stream)
    await stream.aclose()
    assert closed.is_set()


async def test_provider_cancellation_propagates_and_closes() -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()

    class Provider(FakeModel):
        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            try:
                entered.set()
                await asyncio.Event().wait()
                yield {}
            finally:
                closed.set()

    async def consume() -> None:
        async for _ in SanitizedModel(Provider([])).stream([]):
            pass

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


async def test_provider_close_error_sanitized() -> None:
    class Provider(FakeModel):
        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            try:
                yield {"messageStart": {"role": "assistant"}}
            finally:
                raise RuntimeError("PRIVATE-CLEANUP-BODY")

    with pytest.raises(TrustGuardUnavailable) as caught:
        async for _ in SanitizedModel(Provider([])).stream([]):
            pass
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__context__ is None


async def test_plain_async_iterable_provider_supported() -> None:
    class Iterator:
        def __init__(self) -> None:
            self.done = False

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return {"messageStart": {"role": "assistant"}}

    class Provider(FakeModel):
        def stream(self, *args: Any, **kwargs: Any) -> Any:
            return Iterator()

    assert len([chunk async for chunk in SanitizedModel(Provider([])).stream([])]) == 1


async def test_config_and_ordinary_events_delegated_structured_refused() -> None:
    provider = FakeModel([text_response("answer")])
    model = SanitizedModel(provider)
    model.update_config(context_window_limit=2000)
    assert model.context_window_limit == 2000
    chunks = [chunk async for chunk in model.stream([])]
    assert any(chunk.get("messageStop") == {"stopReason": "end_turn"} for chunk in chunks)
    with pytest.raises(TrustGuardUnsupportedContentError):
        async for _ in model.structured_output(None, []):
            pass
