"""Sanitize provider failures before they cross into SDK-owned telemetry."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from strands.models import Model

from .exceptions import TrustGuardUnavailable, TrustGuardUnsupportedContentError


class SanitizedModel(Model):
    """Delegate a trusted stateless provider through its public Model API.

    The wrapper owns no provider resources. Provider-internal logs, spans and
    HTTP behavior remain the application's responsibility. Exceptions crossing
    the provider boundary are sanitized before Strands sees them.
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    def update_config(self, **model_config: Any) -> None:
        self.model.update_config(**model_config)

    def get_config(self) -> Any:
        return self.model.get_config()

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        raise TrustGuardUnsupportedContentError("Structured output is outside the guarded interface.")
        yield {}  # pragma: no cover

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        failed = False
        try:
            stream = self.model.stream(*args, **kwargs)
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
        except Exception:
            failed = True
        if failed:
            # Outside the handler: neither __cause__ nor __context__ retains the
            # raw provider exception for SDK exception.stacktrace serialization.
            raise TrustGuardUnavailable("The model provider failed.")
