"""Public Strands intervention with terminal, stage-aware enforcement."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from weakref import WeakKeyDictionary

from strands import Agent
from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)
from strands.interventions import InterventionHandler, Proceed

from ._content import (
    apply_invocation_transform,
    apply_model_transform,
    apply_tool_input_transform,
    apply_tool_result_transform,
    invocation_payload,
    model_payload,
    tool_input_payload,
    tool_result_payload,
)
from ._contracts import Direction, Status, Verdict
from ._identities import bind_tool_identities
from ._projection import apply_projection_transform, project_payload
from .exceptions import (
    TrustGuardApprovalRequired,
    TrustGuardBlocked,
    TrustGuardConfigurationError,
    TrustGuardError,
    TrustGuardProtocolError,
    TrustGuardStateError,
    TrustGuardUnavailable,
)


class EvaluationClient(Protocol):
    """Client boundary for synthetic testing and explicitly configured deployments."""

    async def aevaluate(
        self,
        payload: dict[str, Any],
        direction: Direction = "input",
        *,
        session_id: str | None = None,
        consumer_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Verdict: ...


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Content-free assessment metadata. Raw findings are intentionally omitted."""

    stage: str
    status: str


@dataclass
class _AgentState:
    failure: type[TrustGuardError] | None = None
    evaluations: int = 0
    decisions: list[DecisionRecord] = field(default_factory=list)
    cleanup_registered: bool = False
    preflight_pending: bool = False


def protection_error(error: BaseException) -> TrustGuardError | None:
    """Find only known policy failures inside the SDK's exception wrappers."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TrustGuardError):
            return current
        current = current.__cause__
    return None


class TrustGuardIntervention(InterventionHandler):
    """Assess text/tool content at five Strands lifecycle boundaries.

    All denials and evaluation failures are terminal exceptions, including tool
    denials. Raw SDK streaming, arbitrary middleware and external instrumentation
    are outside this component's protection boundary. Use GuardedAgent for a
    controlled complete-result interface. A failed agent must be reconstructed.
    By default each stage requires both structured and text-projected checks.
    The request budget counts both; text_assessment=False is an explicit opt-out.
    """

    name = "neuraltrust:trustguard"

    def __init__(
        self,
        client: EvaluationClient,
        *,
        session_id: str | None = None,
        consumer_id: str | None = None,
        max_evaluations: int = 128,
        text_assessment: bool = True,
    ) -> None:
        if type(max_evaluations) is not int or max_evaluations < 1:
            raise TrustGuardConfigurationError("max_evaluations must be a positive integer.")
        if type(text_assessment) is not bool:
            raise TrustGuardConfigurationError("text_assessment must be a boolean.")
        self._client = client
        self._session_id = session_id
        self._consumer_id = consumer_id
        self._max_evaluations = max_evaluations
        self._text_assessment = text_assessment
        self._states: WeakKeyDictionary[Agent, _AgentState] = WeakKeyDictionary()
        self._lock = threading.RLock()

    def _state(self, agent: Agent) -> _AgentState:
        with self._lock:
            state = self._states.setdefault(agent, _AgentState())
            if state.failure is not None:
                raise TrustGuardStateError("The guarded conversation failed; create a new agent.")
            return state

    def _fail(self, agent: Agent, error: TrustGuardError) -> None:
        with self._lock:
            state = self._states.setdefault(agent, _AgentState())
            if state.failure is None:
                state.failure = type(error)

    def _after_invocation(self, event: AfterInvocationEvent) -> None:
        # Public cleanup event also runs for cancellation and provider/tool errors
        # outside an assessment. Never permit replay of their pending tool history.
        if event.result is None or event.result.stop_reason not in ("end_turn", "stop_sequence"):
            self._fail(event.agent, TrustGuardStateError())

    def decisions(self, agent: Agent) -> tuple[DecisionRecord, ...]:
        """Return only the bounded, content-free decisions for the last invocation."""
        with self._lock:
            return tuple(self._states.get(agent, _AgentState()).decisions)

    async def _call(self, payload: dict[str, Any], direction: Direction) -> Verdict:
        """One real assessment, with sanitized failures and terminal enforcement."""
        error: TrustGuardError | None = None
        try:
            verdict = await self._client.aevaluate(
                payload, direction, session_id=self._session_id, consumer_id=self._consumer_id
            )
        except Exception as exc:
            known = protection_error(exc)
            error = type(known)() if known is not None else TrustGuardUnavailable()
        if error is not None:
            raise error
        if not isinstance(verdict, Verdict) or verdict.status not in (
            "allow",
            "report",
            "transform",
            "ask",
            "block",
        ):
            raise TrustGuardProtocolError()
        if verdict.status == "block":
            raise TrustGuardBlocked("TrustGuard blocked protected content.")
        if verdict.status == "ask":
            raise TrustGuardApprovalRequired(
                "TrustGuard requires approval; this integration cannot grant it."
            )
        if verdict.status == "transform" and not isinstance(verdict.transformed_payload, dict):
            raise TrustGuardProtocolError()
        return verdict

    async def _assess(
        self,
        payload: dict[str, Any],
        direction: Direction,
        charge: Callable[[], None],
        check: Callable[[], None],
    ) -> Verdict:
        # The evaluator sees opaque wire aliases in structural correlation-ID
        # fields. Content and tool declarations retain their original values.
        binding = bind_tool_identities(payload)
        charge()
        canonical = await self._call(binding.payload, direction)
        check()
        current = payload
        if canonical.status == "transform":
            # Validate the unchanged aliases before restoring their original
            # SDK identities. No state is committed before all checks succeed.
            current = binding.restore(canonical.transformed_payload)
            canonical = Verdict("transform", current, canonical.request_id, canonical.trace_id)
        if not self._text_assessment:
            return canonical
        projection = project_payload(current, direction)
        charge()
        projected = await self._call(projection, direction)
        check()
        if projected.status == "transform":
            current = apply_projection_transform(current, projection, projected.transformed_payload)
        check()
        if canonical.status == "transform" or projected.status == "transform":
            return Verdict("transform", current, projected.request_id, projected.trace_id)
        status: Status = "report" if canonical.status == "report" or projected.status == "report" else "allow"
        return Verdict(status, request_id=projected.request_id, trace_id=projected.trace_id)

    async def assess(self, payload: dict[str, Any], direction: Direction, stage: str) -> Verdict:
        """Assess without SDK state using a separate, bounded per-call budget.

        Each HTTP assessment consumes a unit, including the default text pass.
        Standalone use does not create a resumable conversation or decision log.
        """
        evaluations = 0

        def charge() -> None:
            nonlocal evaluations
            if evaluations >= self._max_evaluations:
                raise TrustGuardUnavailable("The assessment exceeded its request budget.")
            evaluations += 1

        return await self._assess(payload, direction, charge, lambda: None)

    async def _preflight(self, agent: Agent, payload: dict[str, Any]) -> Verdict:
        """Start the facade's shared budget before the SDK starts tracing."""
        with self._lock:
            state = self._state(agent)
            state.evaluations = 0
            state.decisions.clear()
            state.preflight_pending = True
        return await self._evaluate(agent, payload, "input", "preflight")

    async def _evaluate(
        self, agent: Agent, payload: dict[str, Any], direction: Direction, stage: str
    ) -> Verdict:
        def check() -> None:
            self._state(agent)

        def charge() -> None:
            with self._lock:
                state = self._state(agent)
                if state.evaluations >= self._max_evaluations:
                    raise TrustGuardUnavailable("The invocation exceeded its assessment budget.")
                state.evaluations += 1

        try:
            verdict = await self._assess(payload, direction, charge, check)
        except asyncio.CancelledError:
            self._fail(agent, TrustGuardStateError())
            raise
        except TrustGuardError as exc:
            self._fail(agent, exc)
            raise
        with self._lock:
            state = self._state(agent)
            state.decisions.append(DecisionRecord(stage, verdict.status))
        return verdict

    async def before_invocation(self, event: BeforeInvocationEvent, **kwargs: Any) -> Proceed:
        state = self._state(event.agent)
        if not state.cleanup_registered:
            event.agent.hooks.add_callback(AfterInvocationEvent, self._after_invocation)
            state.cleanup_registered = True
        with self._lock:
            if state.preflight_pending:
                state.preflight_pending = False
            else:
                state.evaluations = 0
                state.decisions.clear()
        if not event.messages:
            return Proceed()
        try:
            payload = invocation_payload(event.messages)
            verdict = await self._evaluate(event.agent, payload, "input", "invocation_input")
            if verdict.status == "transform":
                event.messages = apply_invocation_transform(
                    event.messages, payload, verdict.transformed_payload
                )
        except TrustGuardError as exc:
            self._fail(event.agent, exc)
            raise
        return Proceed()

    async def before_model_call(self, event: BeforeModelCallEvent, **kwargs: Any) -> Proceed:
        self._state(event.agent)
        try:
            specs = list(event.agent.tool_registry.get_all_tools_config().values())
            payload = model_payload(event.agent.messages, event.agent.system_prompt_content, specs)
            verdict = await self._evaluate(event.agent, payload, "input", "model_input")
            if verdict.status == "transform":
                changed = apply_model_transform(
                    event.agent.messages, payload, verdict.transformed_payload, tool_specs=specs
                )
                event.agent.messages[:] = changed
        except TrustGuardError as exc:
            self._fail(event.agent, exc)
            raise
        return Proceed()

    async def after_model_call(self, event: AfterModelCallEvent, **kwargs: Any) -> Proceed:
        if event.exception is not None:
            known = protection_error(event.exception)
            if known is not None:
                self._fail(event.agent, known)
                raise type(known)()
        self._state(event.agent)
        try:
            if event.exception is not None:
                known = protection_error(event.exception)
                raise type(known)() if known else TrustGuardUnavailable("Model execution failed.")
            if event.stop_response is None:
                raise TrustGuardProtocolError("The model returned no complete response.")
            message = event.stop_response.message
            payload = model_payload([message])
            verdict = await self._evaluate(event.agent, payload, "output", "model_output")
            if verdict.status == "transform":
                changed = apply_model_transform(
                    [message],
                    payload,
                    verdict.transformed_payload,
                    tool_specs=list(event.agent.tool_registry.get_all_tools_config().values()),
                )
                message["content"] = changed[0]["content"]
        except TrustGuardError as exc:
            self._fail(event.agent, exc)
            raise
        return Proceed()

    async def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> Proceed:
        self._state(event.agent)
        try:
            if event.selected_tool is None:
                raise TrustGuardProtocolError("The requested tool is not registered.")
            spec = event.selected_tool.tool_spec
            payload = tool_input_payload(event.tool_use, spec)
            verdict = await self._evaluate(event.agent, payload, "input", "tool_input")
            if verdict.status == "transform":
                changed = apply_tool_input_transform(
                    event.tool_use, payload, verdict.transformed_payload, tool_spec=spec
                )
                # SDK history and direct-call recording retain this object.
                event.tool_use["input"] = changed["input"]
        except TrustGuardError as exc:
            self._fail(event.agent, exc)
            raise
        return Proceed()

    async def after_tool_call(self, event: AfterToolCallEvent, **kwargs: Any) -> Proceed:
        # Check the original exception before the poisoned state so SDK redispatch
        # cannot turn a policy failure into a successful tool-error continuation.
        if event.exception is not None:
            known = protection_error(event.exception)
            if known is not None:
                self._fail(event.agent, known)
                raise type(known)()
        self._state(event.agent)
        try:
            if event.exception is not None:
                raise TrustGuardUnavailable("Tool execution failed.")
            payload = tool_result_payload(event.result)
            verdict = await self._evaluate(event.agent, payload, "input", "tool_output")
            if verdict.status == "transform":
                event.result = apply_tool_result_transform(event.result, payload, verdict.transformed_payload)
        except TrustGuardError as exc:
            self._fail(event.agent, exc)
            raise
        return Proceed()
