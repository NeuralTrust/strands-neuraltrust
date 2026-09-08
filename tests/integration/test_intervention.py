"""Exercise protection boundaries through Strands 1.54's actual event loop.

Explicit text_assessment=False cases retain the qualified structured-only
contract; default two-pass execution is covered in test_text_assessment.py.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from strands import Agent, tool
from strands.hooks import AfterToolCallEvent
from strands.tools.executors import SequentialToolExecutor

from strands_neuraltrust._contracts import Verdict
from strands_neuraltrust.exceptions import (
    TrustGuardApprovalRequired,
    TrustGuardBlocked,
    TrustGuardError,
    TrustGuardProtocolError,
    TrustGuardStateError,
    TrustGuardTransformError,
    TrustGuardUnavailable,
    TrustGuardUnsupportedContentError,
)
from strands_neuraltrust.intervention import TrustGuardIntervention, protection_error
from tests.fakes import FakeClient, FakeModel, text_response, tool_response


@pytest.fixture(autouse=True)
def redacted_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_unredacted_attributes=")


def assert_failure(error: pytest.ExceptionInfo[Exception], expected: type[TrustGuardError]) -> None:
    assert isinstance(protection_error(error.value), expected)


def make_echo(effects: list[str]) -> Any:
    @tool
    def echo(value: str) -> str:
        """Return a synthetic value and record the tool side effect."""
        effects.append(value)
        return "tool-output:" + value

    return echo


def agent_for(model: FakeModel, intervention: TrustGuardIntervention, **kwargs: Any) -> Agent:
    return Agent(model=model, interventions=[intervention], callback_handler=None, **kwargs)


def test_all_five_public_hooks_run_with_actual_provider_and_tool_effects(
    capsys: pytest.CaptureFixture[str],
) -> None:
    effects: list[str] = []
    model = FakeModel([tool_response(tool_input={"value": "argument"}), text_response("final-output")])
    client = FakeClient(verdicts=[Verdict("report")])
    guard = TrustGuardIntervention(
        client, session_id="synthetic-session", consumer_id="synthetic-user", text_assessment=False
    )
    agent = agent_for(model, guard, tools=[make_echo(effects)], system_prompt="system-instruction")
    result = agent("user-input")
    assert str(result) == "final-output\n"
    assert effects == ["argument"]
    assert model.index == 2
    assert [decision.stage for decision in guard.decisions(agent)] == [
        "invocation_input",
        "model_input",
        "model_output",
        "tool_input",
        "tool_output",
        "model_input",
        "model_output",
    ]
    assert [call["direction"] for call in client.calls] == [
        "input",
        "input",
        "output",
        "input",
        "input",
        "input",
        "output",
    ]
    assert all(
        call["session_id"] == "synthetic-session" and call["consumer_id"] == "synthetic-user"
        for call in client.calls
    )
    assert client.calls[1]["payload"]["system"] == [{"type": "text", "text": "system-instruction"}]
    assert client.calls[1]["payload"]["tools"][0]["name"] == "echo"
    assert "tool-output:argument" in json.dumps(model.requests[1]["messages"])
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize(
    "status,expected", [("block", TrustGuardBlocked), ("ask", TrustGuardApprovalRequired)]
)
@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_input_and_completed_output_refusals_are_terminal(
    status: str, expected: type[TrustGuardError], boundary: int
) -> None:
    model = FakeModel([text_response("private-model-output")])
    client = FakeClient(verdicts=[Verdict("allow")] * boundary + [Verdict(status)])
    guard = TrustGuardIntervention(client, text_assessment=False)
    agent = agent_for(model, guard)
    with pytest.raises(Exception) as error:
        agent("private-user-input")
    assert_failure(error, expected)
    assert model.index == (1 if boundary == 2 else 0)
    assert "private" not in str(protection_error(error.value))
    before = len(client.calls)
    with pytest.raises(Exception) as error:
        agent("try-again")
    assert_failure(error, TrustGuardStateError)
    assert len(client.calls) == before


@pytest.mark.parametrize("boundary", [3, 4])
def test_tool_denial_blocks_side_effect_or_next_model_and_survives_sdk_redispatch(boundary: int) -> None:
    effects: list[str] = []
    model = FakeModel([tool_response(tool_input={"value": "argument"}), text_response("must-not-run")])
    client = FakeClient(verdicts=[Verdict("allow")] * boundary + [Verdict("block")])
    redispatched_exceptions: list[Exception | None] = []

    class ObservedIntervention(TrustGuardIntervention):
        async def after_tool_call(self, event: AfterToolCallEvent, **kwargs: Any) -> Any:
            redispatched_exceptions.append(event.exception)
            return await super().after_tool_call(event, **kwargs)

    guard = ObservedIntervention(client, text_assessment=False)
    agent = agent_for(model, guard, tools=[make_echo(effects)], tool_executor=SequentialToolExecutor())
    with pytest.raises(Exception) as error:
        agent("run tool")
    assert_failure(error, TrustGuardBlocked)
    assert effects == (["argument"] if boundary == 4 else [])
    assert model.index == 1
    assert len(client.calls) == boundary + 1
    assert len(redispatched_exceptions) == (2 if boundary == 4 else 0)
    if boundary == 4:
        assert redispatched_exceptions[0] is None
        assert isinstance(protection_error(redispatched_exceptions[-1]), TrustGuardBlocked)
    with pytest.raises(Exception) as error:
        agent("retry")
    assert_failure(error, TrustGuardStateError)
    assert model.index == 1
    assert effects == (["argument"] if boundary == 4 else [])


def test_every_transform_reaches_the_next_real_boundary() -> None:
    effects: list[str] = []
    model = FakeModel(
        [
            tool_response(tool_input={"value": "model-original"}, text="planning-original"),
            text_response("final-original"),
        ]
    )

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if index == 0:
            payload["messages"][0]["content"][0]["text"] = "invocation-safe"
        elif index == 1:
            assert payload["messages"][0]["content"][0]["text"] == "invocation-safe"
            payload["messages"][0]["content"][0]["text"] = "provider-safe"
        elif index == 2:
            payload["messages"][0]["content"][0]["text"] = "planning-safe"
            payload["messages"][0]["content"][1]["input"]["value"] = "model-safe"
        elif index == 3:
            assert payload["messages"][0]["content"][0]["input"]["value"] == "model-safe"
            payload["messages"][0]["content"][0]["input"]["value"] = "tool-safe"
        elif index == 4:
            assert payload["messages"][0]["content"][0]["content"][0]["text"] == "tool-output:tool-safe"
            payload["messages"][0]["content"][0]["content"][0]["text"] = "tool-result-safe"
        elif index == 5:
            assert "tool-result-safe" in json.dumps(payload)
            return Verdict("allow")
        else:
            payload["messages"][0]["content"][0]["text"] = "final-safe"
        return Verdict("transform", payload)

    client = FakeClient(policy)
    guard = TrustGuardIntervention(client, text_assessment=False)
    agent = agent_for(model, guard, tools=[make_echo(effects)])
    result = agent("user-original")
    assert str(result) == "final-safe\n"
    assert result.message["content"] == [{"text": "final-safe"}]
    assert effects == ["tool-safe"]
    assert model.requests[0]["messages"][0]["content"] == [{"text": "provider-safe"}]
    history = json.dumps(model.requests[1]["messages"])
    assert "planning-safe" in history and "tool-result-safe" in history
    assert "original" not in history
    assert model.requests[1]["messages"][1]["content"][1]["toolUse"]["input"] == {"value": "tool-safe"}
    assert agent.messages[-1]["content"] == [{"text": "final-safe"}]


@pytest.mark.parametrize("boundary", [0, 1, 2, 3, 4])
def test_malformed_transform_never_releases_content_or_continues_tools(boundary: int) -> None:
    effects: list[str] = []
    model = FakeModel([tool_response(tool_input={"value": "argument"}), text_response("never")])
    client = FakeClient(verdicts=[Verdict("allow")] * boundary + [Verdict("transform", {"messages": []})])
    guard = TrustGuardIntervention(client, text_assessment=False)
    agent = agent_for(model, guard, tools=[make_echo(effects)])
    with pytest.raises(Exception) as error:
        agent("private")
    assert_failure(error, TrustGuardTransformError)
    assert model.index == (0 if boundary < 2 else 1)
    assert effects == (["argument"] if boundary == 4 else [])


@pytest.mark.parametrize(
    "fault", [Verdict("unexpected"), Verdict("transform"), RuntimeError("private-provider-error")]
)
def test_bad_injected_client_verdict_or_failure_is_sanitized_and_terminal(fault: Any) -> None:
    model = FakeModel([])
    client = FakeClient(verdicts=[fault])
    agent = agent_for(model, TrustGuardIntervention(client))
    with pytest.raises(Exception) as error:
        agent("private")
    expected = TrustGuardUnavailable if isinstance(fault, Exception) else TrustGuardProtocolError
    assert_failure(error, expected)
    assert "private" not in str(protection_error(error.value))
    assert model.index == 0


def test_model_failure_is_terminal_and_does_not_allow_reuse() -> None:
    model = FakeModel([RuntimeError("private-model-error")])
    agent = agent_for(model, TrustGuardIntervention(FakeClient()))
    with pytest.raises(Exception) as error:
        agent("hello")
    assert_failure(error, TrustGuardUnavailable)
    with pytest.raises(Exception) as error:
        agent("again")
    assert_failure(error, TrustGuardStateError)
    assert model.index == 1


def test_tool_runtime_failure_cannot_be_replayed_or_sent_to_next_model() -> None:
    effects: list[str] = []

    @tool
    def fail(value: str) -> str:
        """Raise a synthetic failure after recording one side effect."""
        effects.append(value)
        raise RuntimeError("private-tool-error")

    model = FakeModel([tool_response("fail", {"value": "once"}), text_response("never")])
    agent = agent_for(
        model, TrustGuardIntervention(FakeClient()), tools=[fail], tool_executor=SequentialToolExecutor()
    )
    with pytest.raises(Exception) as error:
        agent("run tool")
    assert_failure(error, TrustGuardUnavailable)
    with pytest.raises(Exception) as error:
        agent("retry")
    assert_failure(error, TrustGuardStateError)
    assert effects == ["once"] and model.index == 1


def test_direct_agent_tool_calls_are_guarded_without_a_model() -> None:
    effects: list[str] = []
    model = FakeModel([])
    client = FakeClient(verdicts=[Verdict("block")])
    guard = TrustGuardIntervention(client)
    agent = agent_for(model, guard, tools=[make_echo(effects)])
    with pytest.raises(Exception) as error:
        agent.tool.echo(value="private")
    assert_failure(error, TrustGuardBlocked)
    assert effects == [] and model.index == 0
    assert len(client.calls) == 1


def test_direct_tool_transform_reaches_effect_and_complete_caller_result() -> None:
    effects: list[str] = []

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        block = payload["messages"][0]["content"][0]
        if index == 0:
            block["input"]["value"] = "safe-argument"
        else:
            assert block["content"][0]["text"] == "tool-output:safe-argument"
            block["content"][0]["text"] = "safe-result"
        return Verdict("transform", payload)

    model = FakeModel([])
    guard = TrustGuardIntervention(FakeClient(policy), text_assessment=False)
    agent = agent_for(model, guard, tools=[make_echo(effects)])
    result = agent.tool.echo(value="private-argument")
    assert result["content"] == [{"text": "safe-result"}]
    assert effects == ["safe-argument"] and model.index == 0
    assert "private-argument" not in json.dumps(agent.messages)
    assert "safe-argument" in json.dumps(agent.messages)


async def test_concurrent_tool_allow_verdict_cannot_execute_after_sibling_failure() -> None:
    allow_entered = asyncio.Event()
    block_returning = asyncio.Event()
    effects: list[str] = []

    @tool
    async def echo(value: str) -> str:
        """Record a synthetic effect without yielding execution."""
        effects.append(value)
        return value

    async def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if index < 3:
            return Verdict("allow")
        value = payload["messages"][0]["content"][0]["input"]["value"]
        if value == "blocked":
            await allow_entered.wait()
            block_returning.set()
            return Verdict("block")
        allow_entered.set()
        await block_returning.wait()
        return Verdict("allow")

    response = tool_response(tool_input={"value": "blocked"}, tool_id="blocked-id")
    response["content"].extend(
        tool_response(tool_input={"value": "allowed"}, tool_id="allowed-id")["content"]
    )
    model = FakeModel([response, text_response("unused")])
    agent = agent_for(model, TrustGuardIntervention(FakeClient(policy), text_assessment=False), tools=[echo])
    with pytest.raises(Exception) as error:
        await asyncio.wait_for(agent.invoke_async("run"), timeout=3)
    assert_failure(error, TrustGuardBlocked)
    assert effects == [] and model.index == 1


def test_complete_system_blocks_and_restored_history_are_validated_before_provider() -> None:
    for options in (
        {"system_prompt": [{"text": "plain"}, {"cachePoint": {"type": "default"}}]},
        {
            "messages": [
                {"role": "user", "content": [{"image": {"format": "png", "source": {"bytes": b"private"}}}]}
            ]
        },
    ):
        model = FakeModel([])
        agent = agent_for(model, TrustGuardIntervention(FakeClient()), **options)
        with pytest.raises(Exception) as error:
            agent("new input")
        assert_failure(error, TrustGuardUnsupportedContentError)
        assert model.index == 0


def test_assessment_budget_is_per_invocation_and_fails_closed() -> None:
    model = FakeModel([text_response("one"), text_response("two")])
    guard = TrustGuardIntervention(FakeClient(), max_evaluations=3, text_assessment=False)
    agent = agent_for(model, guard)
    assert str(agent("first")) == "one\n"
    assert str(agent("second")) == "two\n"
    assert len(guard.decisions(agent)) == 3
    limited = agent_for(
        FakeModel([text_response("private")]),
        TrustGuardIntervention(FakeClient(), max_evaluations=2, text_assessment=False),
    )
    with pytest.raises(Exception) as error:
        limited("first")
    assert_failure(error, TrustGuardUnavailable)


async def test_one_shared_intervention_isolates_concurrent_agent_failures_and_decisions() -> None:
    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if direction == "output" and "blocked-output" in json.dumps(payload):
            return Verdict("block")
        return Verdict("allow")

    guard = TrustGuardIntervention(FakeClient(policy))
    failed = agent_for(FakeModel([text_response("blocked-output")]), guard)
    good = agent_for(FakeModel([text_response("good-output"), text_response("good-again")]), guard)
    results = await asyncio.gather(failed.invoke_async("a"), good.invoke_async("b"), return_exceptions=True)
    assert isinstance(results[0], Exception)
    assert isinstance(protection_error(results[0]), TrustGuardBlocked)
    assert str(results[1]) == "good-output\n"
    assert len(guard.decisions(failed)) == 2
    assert len(guard.decisions(good)) == 3
    assert str(await good.invoke_async("again")) == "good-again\n"
    with pytest.raises(Exception) as error:
        await failed.invoke_async("again")
    assert_failure(error, TrustGuardStateError)


async def test_cancelled_tool_assessment_performs_no_side_effects() -> None:
    arrived = asyncio.Event()
    waiting = asyncio.Event()
    effects: list[str] = []

    async def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if index == 3:
            arrived.set()
            await waiting.wait()
        return Verdict("allow")

    model = FakeModel([tool_response(tool_input={"value": "private"}), text_response("unused")])
    agent = agent_for(
        model, TrustGuardIntervention(FakeClient(policy), text_assessment=False), tools=[make_echo(effects)]
    )
    invocation = asyncio.create_task(agent.invoke_async("run"))
    await asyncio.wait_for(arrived.wait(), timeout=3)
    invocation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await invocation
    assert effects == [] and model.index == 1
    with pytest.raises(Exception) as error:
        await agent.invoke_async("retry after cancellation")
    assert_failure(error, TrustGuardStateError)
    assert effects == [] and model.index == 1


async def test_cancelled_provider_stream_cannot_be_resumed() -> None:
    arrived = asyncio.Event()
    waiting = asyncio.Event()

    class WaitingModel(FakeModel):
        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            self.index += 1
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": "unreviewed-partial"}}}
            arrived.set()
            await waiting.wait()

    model = WaitingModel([])
    agent = agent_for(model, TrustGuardIntervention(FakeClient()))
    task = asyncio.create_task(agent.invoke_async("run"))
    await asyncio.wait_for(arrived.wait(), timeout=3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.index == 1
    with pytest.raises(Exception) as error:
        await agent.invoke_async("retry")
    assert_failure(error, TrustGuardStateError)
    assert model.index == 1


async def test_cancelled_tool_body_is_not_replayed() -> None:
    arrived = asyncio.Event()
    waiting = asyncio.Event()
    effects: list[str] = []

    @tool
    async def waiting_tool(value: str) -> str:
        """Begin an observable synthetic operation and await cancellation."""
        effects.append(value)
        arrived.set()
        await waiting.wait()
        return "unreachable"

    model = FakeModel([tool_response("waiting_tool", {"value": "once"}), text_response("unused")])
    agent = agent_for(model, TrustGuardIntervention(FakeClient()), tools=[waiting_tool])
    task = asyncio.create_task(agent.invoke_async("run"))
    await asyncio.wait_for(arrived.wait(), timeout=3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert effects == ["once"] and model.index == 1
    with pytest.raises(Exception) as error:
        await agent.invoke_async("retry")
    assert_failure(error, TrustGuardStateError)
    assert effects == ["once"] and model.index == 1


def test_poisoned_tool_description_is_inspected_before_provider_use() -> None:
    effects: list[str] = []

    @tool(description="POISONED synthetic tool description")
    def described_tool(value: str) -> str:
        effects.append(value)
        return value

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        return Verdict("block" if "POISONED" in json.dumps(payload) else "allow")

    model = FakeModel([])
    agent = agent_for(model, TrustGuardIntervention(FakeClient(policy)), tools=[described_tool])
    with pytest.raises(Exception) as error:
        agent("run")
    assert_failure(error, TrustGuardBlocked)
    assert model.index == 0 and effects == []
