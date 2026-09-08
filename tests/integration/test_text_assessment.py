"""Qualify both actual SDK assessment passes and their shared failure boundary."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest
from strands import Agent, tool
from strands.tools.executors import SequentialToolExecutor

from strands_neuraltrust import GuardedAgent, TrustGuardIntervention, Verdict
from strands_neuraltrust._content import model_payload
from strands_neuraltrust.exceptions import (
    TrustGuardApprovalRequired,
    TrustGuardBlocked,
    TrustGuardConfigurationError,
    TrustGuardProtocolError,
    TrustGuardStateError,
    TrustGuardTransformError,
    TrustGuardUnavailable,
)
from strands_neuraltrust.intervention import protection_error
from tests.fakes import FakeClient, FakeModel, text_response, tool_response


def native(model: FakeModel, client: FakeClient, **kwargs: Any) -> tuple[Agent, TrustGuardIntervention]:
    guard = TrustGuardIntervention(client, **kwargs)
    return Agent(model=model, interventions=[guard], callback_handler=None), guard


def echo_tool(effects: list[str]) -> Any:
    @tool
    def echo(value: str) -> str:
        """Record one local synthetic effect."""
        effects.append(value)
        return "result:" + value

    return echo


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_text_assessment_requires_actual_bool(value: Any) -> None:
    with pytest.raises(TrustGuardConfigurationError):
        TrustGuardIntervention(FakeClient(), text_assessment=value)
    with pytest.raises(TrustGuardConfigurationError):
        GuardedAgent(model=FakeModel([]), client=FakeClient(), text_assessment=value)


def test_default_two_real_calls_for_every_native_stage_preserve_context() -> None:
    effects: list[str] = []
    client = FakeClient()
    model = FakeModel([tool_response(tool_input={"value": "argument"}), text_response("answer")])
    guard = TrustGuardIntervention(client, session_id="session", consumer_id="consumer")
    agent = Agent(
        model=model,
        interventions=[guard],
        tools=[echo_tool(effects)],
        callback_handler=None,
        tool_executor=SequentialToolExecutor(),
    )
    assert str(agent("question")) == "answer\n"
    assert effects == ["argument"]
    assert len(client.calls) == 14
    assert [item.stage for item in guard.decisions(agent)] == [
        "invocation_input",
        "model_input",
        "model_output",
        "tool_input",
        "tool_output",
        "model_input",
        "model_output",
    ]
    for canonical, projected in zip(client.calls[::2], client.calls[1::2], strict=True):
        assert canonical["direction"] == projected["direction"]
        for call in (canonical, projected):
            assert call["session_id"] == "session" and call["consumer_id"] == "consumer"
        assert set(projected["payload"]) == {"messages"}
        assert len(projected["payload"]["messages"]) == 1
        assert all(block["type"] == "text" for block in projected["payload"]["messages"][0]["content"])


@pytest.mark.parametrize("limit", range(1, 6))
def test_native_budget_charges_each_attempt_and_never_starts_excess_call(limit: int) -> None:
    client = FakeClient()
    model = FakeModel([text_response("answer")])
    agent, guard = native(model, client, max_evaluations=limit)
    with pytest.raises(Exception) as caught:
        agent("question")
    assert isinstance(protection_error(caught.value), TrustGuardUnavailable)
    assert len(client.calls) == limit
    assert model.index == (1 if limit >= 4 else 0)
    assert len(guard.decisions(agent)) == limit // 2
    with pytest.raises(Exception) as caught:
        agent("retry")
    assert isinstance(protection_error(caught.value), TrustGuardStateError)
    assert len(client.calls) == limit


def test_native_success_resets_actual_request_budget_each_invocation() -> None:
    client = FakeClient()
    agent, guard = native(FakeModel([text_response("one"), text_response("two")]), client, max_evaluations=6)
    assert str(agent("first")) == "one\n"
    assert str(agent("second")) == "two\n"
    assert len(client.calls) == 12 and len(guard.decisions(agent)) == 3


@pytest.mark.parametrize("limit", range(1, 8))
def test_facade_preflight_and_sdk_share_one_actual_request_budget(limit: int) -> None:
    client = FakeClient()
    model = FakeModel([text_response("answer")])
    agent = GuardedAgent(model=model, client=client, max_evaluations=limit)
    with pytest.raises(TrustGuardUnavailable):
        agent("question")
    assert len(client.calls) == limit
    assert model.index == (1 if limit >= 6 else 0)
    with pytest.raises(TrustGuardStateError):
        agent("retry")
    assert len(client.calls) == limit


def test_facade_success_resets_shared_budget_without_duplicate_preflight_record() -> None:
    client = FakeClient()
    agent = GuardedAgent(
        model=FakeModel([text_response("one"), text_response("two")]), client=client, max_evaluations=8
    )
    for prompt, answer in (("first", "one"), ("second", "two")):
        result = agent(prompt)
        assert result.text == answer
        assert [record.stage for record in result.decisions] == [
            "preflight",
            "invocation_input",
            "model_input",
            "model_output",
        ]
    assert len(client.calls) == 16


@pytest.mark.parametrize("status,error", [("block", TrustGuardBlocked), ("ask", TrustGuardApprovalRequired)])
@pytest.mark.parametrize("boundary", range(5))
def test_projected_denial_is_terminal_at_all_five_native_boundaries(
    status: str, error: type[Exception], boundary: int
) -> None:
    effects: list[str] = []
    client = FakeClient(verdicts=[Verdict("allow")] * (boundary * 2 + 1) + [Verdict(status)])
    model = FakeModel([tool_response(tool_input={"value": "argument"}), text_response("never")])
    guard = TrustGuardIntervention(client)
    agent = Agent(
        model=model,
        interventions=[guard],
        tools=[echo_tool(effects)],
        callback_handler=None,
        tool_executor=SequentialToolExecutor(),
    )
    with pytest.raises(Exception) as caught:
        agent("question")
    assert isinstance(protection_error(caught.value), error)
    assert len(client.calls) == boundary * 2 + 2
    assert model.index == (0 if boundary < 2 else 1)
    assert effects == (["argument"] if boundary == 4 else [])
    with pytest.raises(Exception) as caught:
        agent("retry")
    assert isinstance(protection_error(caught.value), TrustGuardStateError)
    assert len(client.calls) == boundary * 2 + 2


@pytest.mark.parametrize("status,error", [("block", TrustGuardBlocked), ("ask", TrustGuardApprovalRequired)])
def test_projected_preflight_denial_never_enters_sdk(status: str, error: type[Exception]) -> None:
    client = FakeClient(verdicts=[Verdict("allow"), Verdict(status)])
    model = FakeModel([])
    agent = GuardedAgent(model=model, client=client)
    with pytest.raises(error) as caught:
        agent("private-canary")
    assert model.index == 0 and agent._agent.messages == []
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert "private-canary" not in str(caught.value)
    assert len(client.calls) == 2


@pytest.mark.parametrize("boundary", [0, 1, 2, 3, 4, 6])
def test_projected_transform_reaches_actual_model_tool_history_and_result(boundary: int) -> None:
    effects: list[str] = []

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if index != boundary * 2 + 1:
            return Verdict("allow")
        for block in payload["messages"][0]["content"][1:]:
            block["text"] = block["text"].replace("-original", "-safe")
        return Verdict("transform", payload)

    client = FakeClient(policy)
    model = FakeModel(
        [
            tool_response(tool_input={"value": "argument-original"}, text="planning-original"),
            text_response("final-original"),
        ]
    )
    guard = TrustGuardIntervention(client)
    agent = Agent(
        model=model,
        interventions=[guard],
        tools=[echo_tool(effects)],
        callback_handler=None,
        tool_executor=SequentialToolExecutor(),
    )
    result = agent("question-original")
    assert str(result) == ("final-safe\n" if boundary == 6 else "final-original\n")
    assert effects == (["argument-safe"] if boundary in (2, 3) else ["argument-original"])
    assert model.requests[0]["messages"][0]["content"][0]["text"] == (
        "question-safe" if boundary in (0, 1) else "question-original"
    )
    history = model.requests[1]["messages"]
    if boundary == 2:
        assert history[1]["content"][0]["text"] == "planning-safe"
    if boundary in (2, 3):
        assert history[1]["content"][1]["toolUse"]["input"] == {"value": "argument-safe"}
    if boundary == 4:
        assert history[2]["content"][0]["toolResult"]["content"] == [{"text": "result:argument-safe"}]
    assert len(client.calls) == 14
    assert guard.decisions(agent)[boundary].status == "transform"


def test_projected_tool_argument_transform_must_still_satisfy_registered_schema() -> None:
    from pydantic import Field

    effects: list[str] = []

    @tool
    def numeric(value: str = Field(pattern="^[0-9-]+$")) -> str:
        """Accept only a numeric synthetic identifier."""
        effects.append(value)
        return value

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if index == 0:
            return Verdict("allow")
        payload["messages"][0]["content"][-1]["text"] = "MASKED"
        return Verdict("transform", payload)

    client = FakeClient(policy)
    model = FakeModel([])
    agent = Agent(
        model=model, interventions=[TrustGuardIntervention(client)], tools=[numeric], callback_handler=None
    )
    with pytest.raises(Exception) as caught:
        agent.tool.numeric(value="123-45-6789")
    assert isinstance(protection_error(caught.value), TrustGuardTransformError)
    assert len(client.calls) == 2 and effects == [] and model.index == 0


@pytest.mark.parametrize("first", ["allow", "report", "transform"])
@pytest.mark.parametrize("second", ["allow", "report", "transform"])
async def test_aggregate_status_and_ids_preserve_both_real_decisions(first: str, second: str) -> None:
    original = model_payload([{"role": "user", "content": [{"text": "original"}]}])
    before = copy.deepcopy(original)

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        status = first if index == 0 else second
        if status == "transform":
            payload["messages"][0]["content"][0 if index == 0 else 1]["text"] += ":changed"
        return Verdict(
            status,
            payload if status == "transform" else None,
            request_id="canonical" if index == 0 else "projected",
            trace_id=str(index),
        )

    client = FakeClient(policy)
    result = await TrustGuardIntervention(client).assess(original, "input", "standalone")
    assert len(client.calls) == 2 and original == before
    expected = (
        "transform"
        if "transform" in (first, second)
        else "report"
        if "report" in (first, second)
        else "allow"
    )
    assert result.status == expected
    assert result.request_id == "projected" and result.trace_id == "1"
    if expected == "transform":
        assert result.transformed_payload is not None
        assert result.transformed_payload["messages"][0]["content"][0]["text"] == (
            "original" + ":changed" * ((first == "transform") + (second == "transform"))
        )
    else:
        assert result.transformed_payload is None


async def test_standalone_budget_is_local_and_structured_only_opt_out_is_explicit() -> None:
    payload = model_payload([{"role": "user", "content": [{"text": "question"}]}])
    client = FakeClient()
    guard = TrustGuardIntervention(client, max_evaluations=1)
    for _ in range(2):
        with pytest.raises(TrustGuardUnavailable):
            await guard.assess(payload, "input", "standalone")
    assert len(client.calls) == 2
    legacy = TrustGuardIntervention(client, max_evaluations=1, text_assessment=False)
    assert (await legacy.assess(payload, "input", "standalone")).status == "allow"
    assert len(client.calls) == 3


def test_invalid_canonical_transform_stops_before_second_assessment() -> None:
    def policy(payload: dict[str, Any], *_: Any) -> Verdict:
        payload["system"][0]["text"] = "changed"
        return Verdict("transform", payload)

    client = FakeClient(policy)
    model = FakeModel([])
    agent = GuardedAgent(model=model, client=client, system_prompt="immutable")
    with pytest.raises(TrustGuardTransformError):
        agent("question")
    assert len(client.calls) == 1 and model.index == 0 and agent._agent.messages == []


def test_canonical_tool_transform_is_not_committed_when_text_pass_denies() -> None:
    effects: list[str] = []

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if index == 0:
            payload["messages"][0]["content"][0]["input"]["value"] = "staged-only"
            return Verdict("transform", payload)
        assert "staged-only" in json.dumps(payload)
        return Verdict("block")

    client = FakeClient(policy)
    model = FakeModel([])
    agent = Agent(
        model=model,
        interventions=[TrustGuardIntervention(client)],
        tools=[echo_tool(effects)],
        callback_handler=None,
    )
    original = {"value": "original"}
    with pytest.raises(Exception) as caught:
        agent.tool.echo(**original)
    assert isinstance(protection_error(caught.value), TrustGuardBlocked)
    assert original == {"value": "original"}
    assert "staged-only" not in json.dumps(agent.messages)
    assert len(client.calls) == 2 and effects == [] and model.index == 0


@pytest.mark.parametrize("surface", ["history", "tool_input", "tool_result"])
def test_latest_user_text_collector_receives_previously_unscored_surfaces(surface: str) -> None:
    """Model the observed collector's limited text scope, without synthetic bypasses."""
    effects: list[str] = []

    def latest_text_policy(payload: dict[str, Any], *_: Any) -> Verdict:
        text = ""
        for message in reversed(payload["messages"]):
            if message["role"] == "user":
                text = " ".join(block["text"] for block in message["content"] if block["type"] == "text")
                if text:
                    break
        return Verdict("block" if "forbidden-canary" in text else "allow")

    client = FakeClient(latest_text_policy)
    model = FakeModel([])
    guard = TrustGuardIntervention(client)
    if surface == "history":
        agent = Agent(
            model=model,
            interventions=[guard],
            callback_handler=None,
            messages=[
                {"role": "user", "content": [{"text": "forbidden-canary"}]},
                text_response("prior answer"),
            ],
        )
        with pytest.raises(Exception) as caught:
            agent("benign new input")
        assert len(client.calls) == 4
    else:

        @tool
        def echo(value: str) -> str:
            """Record a synthetic effect before returning tool content."""
            effects.append(value)
            return "forbidden-canary" if surface == "tool_result" else value

        agent = Agent(model=model, interventions=[guard], tools=[echo], callback_handler=None)
        with pytest.raises(Exception) as caught:
            agent.tool.echo(value="forbidden-canary" if surface == "tool_input" else "benign")
        assert len(client.calls) == (2 if surface == "tool_input" else 4)
    assert isinstance(protection_error(caught.value), TrustGuardBlocked)
    assert effects == (["benign"] if surface == "tool_result" else [])
    assert model.index == 0


@pytest.mark.parametrize("metadata", ["system", "tool"])
def test_metadata_is_in_text_projection_before_any_sdk_invocation(metadata: str) -> None:
    effects: list[str] = []

    @tool(description="metadata-canary")
    def described(value: str) -> str:
        effects.append(value)
        return value

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if index == 1:
            assert "metadata-canary" in json.dumps(payload)
            return Verdict("block")
        return Verdict("allow")

    client = FakeClient(policy)
    model = FakeModel([])
    agent = GuardedAgent(
        model=model,
        client=client,
        system_prompt="metadata-canary" if metadata == "system" else None,
        tools=[described] if metadata == "tool" else [],
    )
    with pytest.raises(TrustGuardBlocked):
        agent("question")
    assert len(client.calls) == 2 and model.index == 0 and effects == [] and agent._agent.messages == []


@pytest.mark.parametrize("fault", [Verdict("unknown"), Verdict("transform"), RuntimeError("private")])
def test_second_assessment_protocol_and_transport_errors_fail_closed(fault: Any) -> None:
    client = FakeClient(verdicts=[Verdict("allow"), fault])
    model = FakeModel([])
    agent = GuardedAgent(model=model, client=client)
    expected = TrustGuardUnavailable if isinstance(fault, Exception) else TrustGuardProtocolError
    with pytest.raises(expected) as caught:
        agent("question")
    assert "private" not in str(caught.value) and caught.value.__context__ is None
    assert len(client.calls) == 2 and model.index == 0
    with pytest.raises(TrustGuardStateError):
        agent("retry")


async def test_cancellation_during_projection_commits_no_canonical_transform() -> None:
    arrived = asyncio.Event()

    async def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if index == 0:
            payload["messages"][0]["content"][0]["text"] = "staged-only"
            return Verdict("transform", payload)
        arrived.set()
        await asyncio.Event().wait()
        return Verdict("allow")

    client = FakeClient(policy)
    model = FakeModel([])
    agent, _ = native(model, client)
    task = asyncio.create_task(agent.invoke_async("original"))
    await asyncio.wait_for(arrived.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "staged-only" not in json.dumps(agent.messages) and model.index == 0
    assert len(client.calls) == 2
    with pytest.raises(Exception) as caught:
        await agent.invoke_async("retry")
    assert isinstance(protection_error(caught.value), TrustGuardStateError)
    assert len(client.calls) == 2


@pytest.mark.parametrize("fail_at", ["canonical", "projection"])
async def test_concurrent_failure_is_checked_between_and_after_both_requests(fail_at: str) -> None:
    arrived = asyncio.Event()
    release = asyncio.Event()

    async def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        if index == (0 if fail_at == "canonical" else 1):
            arrived.set()
            await release.wait()
        return Verdict("allow")

    client = FakeClient(policy)
    model = FakeModel([])
    agent, guard = native(model, client)
    task = asyncio.create_task(agent.invoke_async("question"))
    await asyncio.wait_for(arrived.wait(), timeout=2)
    guard._fail(agent, TrustGuardBlocked())
    release.set()
    with pytest.raises(Exception) as caught:
        await task
    assert isinstance(protection_error(caught.value), TrustGuardStateError)
    assert len(client.calls) == (1 if fail_at == "canonical" else 2)
    assert model.index == 0 and guard.decisions(agent) == ()
