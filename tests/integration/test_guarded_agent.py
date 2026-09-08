"""Exercise the controlled complete-result API against the real SDK.

The indexed single-pass verdict fixtures explicitly qualify the structured-only
opt-out. Default two-pass budgets and enforcement have their own integration tests.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest
from strands import tool

from strands_neuraltrust import (
    GuardedAgent,
    TrustGuardApprovalRequired,
    TrustGuardBlocked,
    TrustGuardConfigurationError,
    TrustGuardProtocolError,
    TrustGuardStateError,
    TrustGuardUnavailable,
    TrustGuardUnsupportedContentError,
    Verdict,
)
from tests.fakes import FakeClient, FakeModel, text_response, tool_response


def test_complete_text_and_no_default_print(capsys: Any) -> None:
    client = FakeClient()
    model = FakeModel([text_response("Checked answer"), text_response("Second answer")])
    agent = GuardedAgent(model=model, client=client)
    result = agent("Hello")
    assert str(result) == "Checked answer"
    assert [d.stage for d in result.decisions] == [
        "preflight",
        "invocation_input",
        "model_input",
        "model_output",
    ]
    assert agent.invoke("Again").text == "Second answer"
    assert len(model.requests[1]["messages"]) == 3
    assert capsys.readouterr().out == ""
    assert not hasattr(agent, "stream_async")
    assert not hasattr(agent, "agent")
    assert not hasattr(agent, "structured_output")


@pytest.mark.parametrize("status,error", [("block", TrustGuardBlocked), ("ask", TrustGuardApprovalRequired)])
@pytest.mark.parametrize("at", [0, 1, 2, 3])
def test_every_boundary_terminal_and_silent(status: Any, error: Any, at: int, capsys: Any) -> None:
    secret = "unassessed-marker-93211"
    client = FakeClient(verdicts=[Verdict("allow")] * at + [Verdict(status)])
    model = FakeModel([text_response(secret)])
    agent = GuardedAgent(model=model, client=client, text_assessment=False)
    with pytest.raises(error) as caught:
        agent.invoke("input")
    assert len(model.requests) == (1 if at == 3 else 0)
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert secret not in capsys.readouterr().out
    with pytest.raises(TrustGuardStateError):
        agent.invoke("retry")


@pytest.mark.parametrize("at", [0, 1, 2, 3])
def test_text_transform_used_at_every_boundary(at: int) -> None:
    def policy(payload: Any, direction: str, index: int) -> Verdict:
        if index == at:
            changed = copy.deepcopy(payload)
            changed["messages"][0]["content"][0]["text"] = "transformed"
            return Verdict("transform", changed)
        return Verdict("allow")

    model = FakeModel([text_response("answer")])
    result = GuardedAgent(model=model, client=FakeClient(policy), text_assessment=False).invoke("question")
    assert result.text == ("transformed" if at == 3 else "answer")
    if at < 3:
        assert model.requests[0]["messages"][0]["content"][0]["text"] == "transformed"


@pytest.mark.parametrize("verdict", [Verdict("transform"), Verdict("mystery"), None])
def test_malformed_injected_verdict_refused(verdict: Any) -> None:
    model = FakeModel([text_response("answer")])
    with pytest.raises(TrustGuardProtocolError):
        GuardedAgent(model=model, client=FakeClient(lambda *_: verdict)).invoke("hello")
    assert not model.requests


def test_provider_error_sanitized_and_poisoned() -> None:
    agent = GuardedAgent(model=FakeModel([RuntimeError("private-provider-body")]), client=FakeClient())
    with pytest.raises(TrustGuardUnavailable) as caught:
        agent.invoke("hello")
    assert "private-provider-body" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    with pytest.raises(TrustGuardStateError):
        agent.invoke("retry")


def test_evaluation_budget_prevents_model_call() -> None:
    model = FakeModel([text_response("answer")])
    with pytest.raises(TrustGuardUnavailable):
        GuardedAgent(model=model, client=FakeClient(), max_evaluations=1).invoke("hello")
    assert not model.requests


@pytest.mark.parametrize(
    "arg,value",
    [
        ("timeout", True),
        ("timeout", float("nan")),
        ("timeout", float("inf")),
        ("timeout", 0),
        ("timeout", 10**1000),
        ("max_turns", True),
        ("max_turns", 0),
        ("max_stream_bytes", -1),
        ("max_evaluations", False),
    ],
)
def test_bad_limits(arg: str, value: Any) -> None:
    with pytest.raises(TrustGuardConfigurationError):
        GuardedAgent(model=FakeModel([]), client=FakeClient(), **{arg: value})


@pytest.mark.parametrize("prompt", [None, [], {}, b"hello", 4])
def test_resume_and_structured_inputs_absent(prompt: Any) -> None:
    model = FakeModel([])
    agent = GuardedAgent(model=model, client=FakeClient())
    with pytest.raises(TrustGuardUnsupportedContentError):
        agent.invoke(prompt)
    assert not model.requests


async def test_async_and_sync_inside_loop() -> None:
    agent = GuardedAgent(model=FakeModel([text_response("answer")]), client=FakeClient())
    with pytest.raises(TrustGuardConfigurationError):
        agent.invoke("hello")
    assert (await agent.invoke_async("hello")).text == "answer"


async def test_concurrent_invocation_rejected_without_poisoning_success() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def policy(*_: Any) -> Verdict:
        entered.set()
        await release.wait()
        return Verdict("allow")

    agent = GuardedAgent(model=FakeModel([text_response("answer")]), client=FakeClient(policy))
    first = asyncio.create_task(agent.invoke_async("one"))
    await entered.wait()
    with pytest.raises(TrustGuardStateError):
        await agent.invoke_async("two")
    release.set()
    assert (await first).text == "answer"


@pytest.mark.parametrize("cancel", [True, False])
async def test_timeout_or_cancellation_never_reused(cancel: bool) -> None:
    entered = asyncio.Event()

    async def policy(*_: Any) -> Verdict:
        entered.set()
        await asyncio.Event().wait()
        return Verdict("allow")

    agent = GuardedAgent(model=FakeModel([]), client=FakeClient(policy), timeout=0.02 if not cancel else 30)
    task = asyncio.create_task(agent.invoke_async("one"))
    await entered.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TrustGuardUnavailable):
        await task
    with pytest.raises(TrustGuardStateError):
        await agent.invoke_async("two")


def test_sequential_tools_stop_pending_side_effects_and_resume() -> None:
    effects: list[str] = []

    @tool
    def effect(value: str) -> str:
        """Record a synthetic local side effect."""
        effects.append(value)
        return "UNSAFE-TOOL-RESULT"

    response = tool_response("effect", {"value": "one"})
    response["content"] += tool_response("effect", {"value": "two"}, "call-2")["content"]

    def policy(payload: Any, *_: Any) -> Verdict:
        return Verdict("block" if "UNSAFE-TOOL-RESULT" in json.dumps(payload) else "allow")

    model = FakeModel([response, text_response("must not happen")])
    agent = GuardedAgent(model=model, client=FakeClient(policy), tools=[effect])
    with pytest.raises(TrustGuardBlocked):
        agent.invoke("hello")
    assert effects == ["one"]
    assert len(model.requests) == 1
    with pytest.raises(TrustGuardStateError):
        agent.invoke("retry")
    assert effects == ["one"]


class RawModel(FakeModel):
    def __init__(self, chunks: list[Any]) -> None:
        super().__init__([])
        self.chunks = chunks

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        self.index += 1
        for chunk in self.chunks:
            yield chunk


def raw_text(text: str = "answer") -> list[Any]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": text}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]


def test_implicit_text_start_supported() -> None:
    assert GuardedAgent(model=RawModel(raw_text()), client=FakeClient()).invoke("hello").text == "answer"


@pytest.mark.parametrize("chunks", [raw_text()[:-1], raw_text()[:2], [], raw_text() + raw_text()])
def test_incomplete_or_multiple_message_stream_refused(chunks: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        GuardedAgent(model=RawModel(chunks), client=FakeClient()).invoke("hello")


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoningContent": {"text": "private"}},
        {"reasoningContent": {"signature": "private"}},
        {"citation": {"title": "private"}},
        {"text": "one", "toolUse": {"input": "{}"}},
        {"image": {}},
    ],
)
def test_unsupported_stream_refused_before_parser(delta: Any, caplog: Any) -> None:
    chunks = raw_text()
    chunks[1] = {"contentBlockDelta": {"delta": delta}}
    with pytest.raises(TrustGuardUnsupportedContentError):
        GuardedAgent(model=RawModel(chunks), client=FakeClient()).invoke("hello")
    assert "private" not in caplog.text


@pytest.mark.parametrize("args", ['{"secret":', '{"x":1,"x":2}', '{"x":NaN}', "[]", ""])
def test_malformed_tool_json_never_reaches_sdk_log_or_effect(args: str, caplog: Any) -> None:
    effects: list[str] = []

    @tool
    def echo() -> str:
        """Record a synthetic call with optional empty arguments."""
        effects.append("called")
        return "answer"

    chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "id", "name": "echo"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": args}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    with pytest.raises(TrustGuardUnsupportedContentError):
        GuardedAgent(model=RawModel(chunks), client=FakeClient(), tools=[echo]).invoke("hello")
    assert not effects
    assert "Failed to parse" not in caplog.text
    assert "secret" not in caplog.text


def test_stream_budget_checks_raw_event_overhead() -> None:
    with pytest.raises(TrustGuardUnavailable):
        GuardedAgent(model=RawModel(raw_text("X" * 1000)), client=FakeClient(), max_stream_bytes=100).invoke(
            "h"
        )


def test_invalid_model_or_system_configuration() -> None:
    with pytest.raises(TrustGuardConfigurationError):
        GuardedAgent(model="bedrock", client=FakeClient())
    with pytest.raises(TrustGuardUnsupportedContentError):
        GuardedAgent(model=FakeModel([]), client=FakeClient(), system_prompt=[])


def test_stateful_model_refused() -> None:
    class StatefulModel(FakeModel):
        @property
        def stateful(self) -> bool:
            return True

    with pytest.raises(TrustGuardConfigurationError):
        GuardedAgent(model=StatefulModel([]), client=FakeClient())
