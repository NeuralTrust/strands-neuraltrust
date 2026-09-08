"""Controlled complete-result invocation; no public raw stream or resume API."""

from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass
from typing import Any

from strands import Agent
from strands.hooks import AfterModelCallEvent, BeforeModelCallEvent
from strands.interventions import Proceed
from strands.models import Model
from strands.tools.executors import SequentialToolExecutor
from strands.types.content import Messages

from ._content import apply_invocation_transform, model_payload
from ._model import SanitizedModel
from ._stream import StreamMonitor
from ._telemetry import require_redacted_tracer
from .exceptions import (
    TrustGuardConfigurationError,
    TrustGuardError,
    TrustGuardStateError,
    TrustGuardUnavailable,
    TrustGuardUnsupportedContentError,
)
from .intervention import DecisionRecord, EvaluationClient, TrustGuardIntervention, protection_error


@dataclass(frozen=True, slots=True)
class GuardedResult:
    """Only assessed final text and bounded content-free decision metadata."""

    text: str
    decisions: tuple[DecisionRecord, ...]
    stop_reason: str = "end_turn"

    def __str__(self) -> str:
        return self.text


class _CheckedIntervention(TrustGuardIntervention):
    """Join the owned callback monitor with the public lifecycle boundaries."""

    def __init__(self, client: EvaluationClient, monitor: StreamMonitor, **kwargs: Any) -> None:
        super().__init__(client, **kwargs)
        self.monitor = monitor

    async def before_model_call(self, event: BeforeModelCallEvent, **kwargs: Any) -> Proceed:
        self.monitor.begin_model()
        return await super().before_model_call(event, **kwargs)

    async def after_model_call(self, event: AfterModelCallEvent, **kwargs: Any) -> Proceed:
        if event.exception is None:
            self.monitor.finish_model()
        return await super().after_model_call(event, **kwargs)


class GuardedAgent:
    """Own a Strands agent with sequential tools and complete checked results.

    Configure SDK trace redaction before any agent is created. Models/tools are
    trusted application code, not sandboxed processes. Plugins, custom hooks,
    sessions, raw streams, structured-output shortcuts and implicit resume are
    intentionally absent. After any failed invocation, create a new instance.
    The caller owns the injected evaluator and model lifetimes.
    The assessment budget includes both default passes and preflight requests.
    """

    def __init__(
        self,
        *,
        model: Model,
        client: EvaluationClient,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
        consumer_id: str | None = None,
        max_turns: int = 8,
        timeout: float = 60.0,
        max_stream_bytes: int = 1_048_576,
        max_evaluations: int = 128,
        text_assessment: bool = True,
    ) -> None:
        if not isinstance(model, Model):
            raise TrustGuardConfigurationError("An explicit Strands Model instance is required.")
        if model.stateful:
            raise TrustGuardConfigurationError("Provider-managed conversation state is not supported.")
        for value in (max_turns, max_stream_bytes):
            if type(value) is not int or value < 1:
                raise TrustGuardConfigurationError("Invocation limits must be positive integers.")
        valid_timeout = False
        try:
            valid_timeout = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
        except OverflowError:
            pass
        if not valid_timeout:
            raise TrustGuardConfigurationError("The invocation timeout must be finite and positive.")
        if system_prompt is not None and type(system_prompt) is not str:
            raise TrustGuardUnsupportedContentError("The guarded system prompt must be text.")
        require_redacted_tracer()
        self._monitor = StreamMonitor(max_stream_bytes)
        self._guard = _CheckedIntervention(
            client,
            self._monitor,
            session_id=session_id,
            consumer_id=consumer_id,
            max_evaluations=max_evaluations,
            text_assessment=text_assessment,
        )
        self._lock = threading.Lock()
        self._failed = False
        self._max_bytes = max_stream_bytes
        self._max_turns = max_turns
        self._timeout = float(timeout)
        self._agent = Agent(
            model=SanitizedModel(model),
            tools=list(tools or []),
            system_prompt=system_prompt,
            interventions=[self._guard],
            callback_handler=self._check_stream_budget,
            tool_executor=SequentialToolExecutor(),
            retry_strategy=None,
            load_tools_from_directory=False,
        )
        # Refuse unsupported tool metadata before the first real invocation.
        model_payload(
            [],
            self._agent.system_prompt_content,
            list(self._agent.tool_registry.get_all_tools_config().values()),
        )

    def _check_stream_budget(self, **event: Any) -> None:
        try:
            if "event" in event:
                self._monitor.accept(event["event"])
            if "tool_stream_event" in event:
                self._monitor.count(event["tool_stream_event"])
        except TrustGuardError:
            self._failed = True
            raise

    def invoke(self, prompt: str) -> GuardedResult:
        """Invoke synchronously; inside an async loop use invoke_async instead."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise TrustGuardConfigurationError("Use invoke_async from a running event loop.")
        return asyncio.run(self.invoke_async(prompt))

    def __call__(self, prompt: str) -> GuardedResult:
        return self.invoke(prompt)

    async def invoke_async(self, prompt: str) -> GuardedResult:
        """Evaluate and return complete accepted text, or a sanitized terminal error."""
        if type(prompt) is not str:
            raise TrustGuardUnsupportedContentError("The guarded invocation accepts text, not resume data.")
        if not self._lock.acquire(blocking=False):
            raise TrustGuardStateError("A guarded invocation is already running.")
        error: TrustGuardError | None = None
        try:
            if self._failed:
                raise TrustGuardStateError("The guarded conversation failed; create a new agent.")
            require_redacted_tracer()
            self._monitor.reset()
            try:
                return await asyncio.wait_for(self._invoke(prompt), timeout=self._timeout)
            except asyncio.CancelledError:
                self._failed = True
                raise
            except Exception as exc:
                self._failed = True
                known = protection_error(exc)
                error = type(known)() if known is not None else TrustGuardUnavailable()
        finally:
            self._lock.release()
        # Raise outside the original exception handler: caller-visible chains must
        # not retain SDK request_state or untrusted provider exception bodies.
        raise error if error is not None else TrustGuardUnavailable()

    async def _invoke(self, prompt: str) -> GuardedResult:
        messages: Messages = [{"role": "user", "content": [{"text": prompt}]}]
        payload = model_payload(
            messages,
            self._agent.system_prompt_content,
            list(self._agent.tool_registry.get_all_tools_config().values()),
        )
        verdict = await self._guard._preflight(self._agent, payload)
        if verdict.status == "transform":
            messages = apply_invocation_transform(messages, payload, verdict.transformed_payload)
            prompt = messages[0]["content"][0]["text"]
        result = await self._agent.invoke_async(prompt, limits={"turns": self._max_turns})
        if self._failed:
            raise TrustGuardStateError()
        if result.stop_reason not in ("end_turn", "stop_sequence"):
            raise TrustGuardStateError("The model did not finish a complete safe turn.")
        payload = model_payload([result.message])
        blocks = payload["messages"][0]["content"]
        if any(block["type"] != "text" for block in blocks):
            raise TrustGuardUnsupportedContentError("The final result must contain only text.")
        text = "".join(block["text"] for block in blocks)
        if len(text.encode("utf-8")) > self._max_bytes:
            raise TrustGuardUnavailable("The result exceeded its byte budget.")
        return GuardedResult(
            text=text,
            stop_reason=result.stop_reason,
            decisions=self._guard.decisions(self._agent),
        )
