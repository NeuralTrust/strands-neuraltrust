"""Small, fail-closed HTTP adapter for the collector-key Evaluate API."""

from __future__ import annotations

import asyncio
import threading
import time
from types import TracebackType
from typing import Any

import httpx

from ._contracts import Direction, Verdict, encode_request, parse_verdict
from .config import TrustGuardConfig
from .exceptions import (
    TrustGuardAuthenticationError,
    TrustGuardError,
    TrustGuardProtocolError,
    TrustGuardStateError,
    TrustGuardUnavailable,
)


class TrustGuardClient:
    """Evaluate with bounded requests, no redirects, and no retries.

    Caller-supplied HTTP clients remain caller-owned. An injected async client
    may be used only on the first event loop that evaluates with it. Owned sync
    requests reuse a serialized pool; owned async requests use a fresh client per
    evaluation and close it on the same loop, including on cancellation. This
    deliberate first-release tradeoff supports repeated synchronous Agent calls
    that create different loops without retaining cross-loop transports.

    The return value is advisory. The intervention/facade enforces all outcomes.
    Do not interpret successful HTTP or a returned Verdict as authorization.
    """

    def __init__(
        self,
        config: TrustGuardConfig,
        *,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._sync_client = http_client
        self._owns_sync_client = http_client is None
        self._async_client = async_http_client
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        self._closed = False

    def __repr__(self) -> str:
        return f"TrustGuardClient(closed={self._closed})"

    @property
    def is_closed(self) -> bool:
        """Whether this adapter has been closed (independent of injected clients)."""
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise TrustGuardStateError("The TrustGuard client is closed.")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": "Bearer " + self.config.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "strands-neuraltrust",
        }

    def evaluate(
        self,
        payload: dict[str, Any],
        direction: Direction = "input",
        *,
        session_id: str | None = None,
        consumer_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Verdict:
        """Evaluate synchronously with phase timeouts and elapsed-time checks."""
        self._ensure_open()
        body = encode_request(
            payload,
            direction,
            max_bytes=self.config.max_request_bytes,
            session_id=session_id,
            consumer_id=consumer_id,
            attributes=attributes,
        )
        failure: TrustGuardError | None = None
        with self._lock:
            self._ensure_open()
            if self._sync_client is None:
                self._sync_client = httpx.Client(trust_env=False, follow_redirects=False)
            started = time.monotonic()
            try:
                with self._sync_client.stream(
                    "POST",
                    self.config.evaluate_url,
                    content=body,
                    headers=self._headers(),
                    auth=None,
                    follow_redirects=False,
                    timeout=self.config.timeout,
                ) as response:
                    _check_response(response, self.config.max_response_bytes)
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        if time.monotonic() - started >= self.config.timeout:
                            raise TrustGuardUnavailable("TrustGuard evaluation timed out.")
                        size += len(chunk)
                        if size > self.config.max_response_bytes:
                            raise TrustGuardProtocolError("TrustGuard response exceeds its byte limit.")
                        chunks.append(chunk)
                    if time.monotonic() - started >= self.config.timeout:
                        raise TrustGuardUnavailable("TrustGuard evaluation timed out.")
                    return parse_verdict(b"".join(chunks))
            except httpx.TimeoutException:
                failure = TrustGuardUnavailable("TrustGuard evaluation timed out.")
            except httpx.HTTPError:
                failure = TrustGuardUnavailable("TrustGuard evaluation could not be completed.")
            except (OSError, ValueError, RuntimeError):
                failure = TrustGuardUnavailable("TrustGuard transport could not be used.")
        # Raise outside the handler so raw transport exceptions are not retained
        # even through __context__, which some instrumentation inspects.
        raise failure

    async def aevaluate(
        self,
        payload: dict[str, Any],
        direction: Direction = "input",
        *,
        session_id: str | None = None,
        consumer_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Verdict:
        """Evaluate natively asynchronously with an overall HTTP deadline."""
        self._ensure_open()
        body = encode_request(
            payload,
            direction,
            max_bytes=self.config.max_request_bytes,
            session_id=session_id,
            consumer_id=consumer_id,
            attributes=attributes,
        )
        loop = asyncio.get_running_loop()
        with self._lock:
            self._ensure_open()
            if self._async_client is not None:
                if self._async_loop is None:
                    self._async_loop = loop
                elif self._async_loop is not loop:
                    raise TrustGuardStateError(
                        "An injected async client cannot be reused across event loops."
                    )
        failure: TrustGuardError | None = None
        try:
            return await asyncio.wait_for(self._aevaluate_http(body), timeout=self.config.timeout)
        except (asyncio.TimeoutError, httpx.TimeoutException):
            failure = TrustGuardUnavailable("TrustGuard evaluation timed out.")
        except httpx.HTTPError:
            failure = TrustGuardUnavailable("TrustGuard evaluation could not be completed.")
        except (OSError, ValueError, RuntimeError):
            failure = TrustGuardUnavailable("TrustGuard transport could not be used.")
        raise failure

    async def _aevaluate_http(self, body: bytes) -> Verdict:
        if self._async_client is not None:
            return await self._arequest(self._async_client, body)
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            return await self._arequest(client, body)

    async def _arequest(self, client: httpx.AsyncClient, body: bytes) -> Verdict:
        async with client.stream(
            "POST",
            self.config.evaluate_url,
            content=body,
            headers=self._headers(),
            auth=None,
            follow_redirects=False,
            timeout=self.config.timeout,
        ) as response:
            _check_response(response, self.config.max_response_bytes)
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > self.config.max_response_bytes:
                    raise TrustGuardProtocolError("TrustGuard response exceeds its byte limit.")
                chunks.append(chunk)
            return parse_verdict(b"".join(chunks))

    def close(self) -> None:
        """Close owned resources; never close either injected HTTP client."""
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._owns_sync_client and self._sync_client is not None:
                    self._sync_client.close()

    async def aclose(self) -> None:
        """Close the adapter; owned async requests close within their own calls."""
        self.close()

    def __enter__(self) -> TrustGuardClient:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> TrustGuardClient:
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _check_response(response: httpx.Response, max_bytes: int) -> None:
    code = response.status_code
    if code in (401, 403):
        raise TrustGuardAuthenticationError("TrustGuard authentication failed.")
    if code == 429 or code >= 500:
        raise TrustGuardUnavailable("TrustGuard service is unavailable.")
    if code != 200:
        raise TrustGuardProtocolError("TrustGuard returned an unexpected HTTP status.")
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise TrustGuardProtocolError("TrustGuard returned an unsupported response encoding.")
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise TrustGuardProtocolError("TrustGuard returned an unsupported response content type.")
    length = response.headers.get("content-length")
    if length is not None:
        if not length.isascii() or not length.isdecimal() or len(length) > 20:
            raise TrustGuardProtocolError("TrustGuard returned an invalid response length.")
        if int(length) > max_bytes:
            raise TrustGuardProtocolError("TrustGuard response exceeds its byte limit.")
