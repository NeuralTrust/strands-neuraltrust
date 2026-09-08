"""Real SDK correlation IDs survive assessed transformations via wire aliases."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from strands import Agent, tool
from strands.tools.executors import SequentialToolExecutor

from strands_neuraltrust import GuardedAgent, TrustGuardIntervention, Verdict
from strands_neuraltrust.exceptions import TrustGuardStateError, TrustGuardTransformError
from strands_neuraltrust.intervention import protection_error
from tests.fakes import FakeClient, FakeModel, text_response, tool_response


def make_echo(effects: list[str]) -> Any:
    @tool
    def echo(value: str) -> str:
        """Record one local synthetic effect."""
        effects.append(value)
        return "result:" + value

    return echo


def aliases(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    uses, results = [], []
    for message in payload["messages"]:
        for block in message["content"]:
            if block["type"] == "tool_use":
                uses.append(block["id"])
            elif block["type"] == "tool_result":
                results.append(block["tool_use_id"])
    return uses, results


@pytest.mark.parametrize("text_assessment", [False, True])
def test_direct_tool_transform_restores_actual_sdk_ids_in_every_surface(text_assessment: bool) -> None:
    effects: list[str] = []

    def policy(payload: dict[str, Any], *_: Any) -> Verdict:
        block = payload["messages"][0]["content"][0]
        if block["type"] == "tool_use":
            assert re.fullmatch("strands_tool_call_[a-z]+", block["id"])
            block["input"]["value"] = "safe-argument"
        elif block["type"] == "tool_result":
            assert re.fullmatch("strands_tool_call_[a-z]+", block["tool_use_id"])
            block["content"][0]["text"] = "safe-result"
        else:
            return Verdict("allow")
        return Verdict("transform", payload)

    client = FakeClient(policy)
    model = FakeModel([])
    guard = TrustGuardIntervention(client, text_assessment=text_assessment)
    agent = Agent(model=model, interventions=[guard], tools=[make_echo(effects)], callback_handler=None)
    result = agent.tool.echo(value="original")
    assert result["content"] == [{"text": "safe-result"}]
    assert effects == ["safe-argument"] and model.index == 0
    use = next(
        block["toolUse"] for message in agent.messages for block in message["content"] if "toolUse" in block
    )
    stored = next(
        block["toolResult"]
        for message in agent.messages
        for block in message["content"]
        if "toolResult" in block
    )
    assert use["toolUseId"] == stored["toolUseId"] == result["toolUseId"]
    assert not use["toolUseId"].startswith("strands_tool_call_")
    assert "strands_tool_call_" not in json.dumps(agent.messages)
    assert len(client.calls) == (4 if text_assessment else 2)
    assert [record.status for record in guard.decisions(agent)] == ["transform", "transform"]


def test_multiple_model_tool_calls_preserve_linkage_and_alias_like_original_ids() -> None:
    original_ids = ["strands_tool_call_a", "provider-1234567890"]
    effects: list[str] = []
    response = tool_response(tool_id=original_ids[0], tool_input={"value": "first"})
    response["content"] += tool_response(tool_id=original_ids[1], tool_input={"value": original_ids[1]})[
        "content"
    ]

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        use_ids, result_ids = aliases(payload)
        assert all(re.fullmatch("strands_tool_call_[a-z]+", value) for value in use_ids + result_ids)
        if direction == "output" and len(use_ids) == 2:
            assert use_ids == ["strands_tool_call_a", "strands_tool_call_b"]
            # The actual ID occurring in content is still assessed as content.
            assert payload["messages"][0]["content"][1]["input"]["value"] == original_ids[1]
            for block in payload["messages"][0]["content"]:
                block["input"]["value"] = "safe:" + block["input"]["value"]
            return Verdict("transform", payload)
        if result_ids and use_ids:
            assert use_ids == result_ids == ["strands_tool_call_a", "strands_tool_call_b"]
        return Verdict("allow")

    client = FakeClient(policy)
    model = FakeModel([response, text_response("final")])
    result = GuardedAgent(model=model, client=client, tools=[make_echo(effects)]).invoke("question")
    assert result.text == "final"
    assert effects == ["safe:first", "safe:" + original_ids[1]]
    history = model.requests[1]["messages"]
    uses = [
        block["toolUse"]["toolUseId"]
        for message in history
        for block in message["content"]
        if "toolUse" in block
    ]
    results = [
        block["toolResult"]["toolUseId"]
        for message in history
        for block in message["content"]
        if "toolResult" in block
    ]
    assert uses == results == original_ids
    assert model.responses[0] == response


@pytest.mark.parametrize("surface", ["tool_input", "tool_result"])
@pytest.mark.parametrize("text_assessment", [False, True])
def test_changed_wire_alias_is_terminal_without_invented_sdk_identity(
    surface: str, text_assessment: bool
) -> None:
    effects: list[str] = []

    def policy(payload: dict[str, Any], *_: Any) -> Verdict:
        block = payload["messages"][0]["content"][0]
        if surface == "tool_input" and block["type"] == "tool_use":
            block["id"] = "changed-server-alias"
            return Verdict("transform", payload)
        if surface == "tool_result" and block["type"] == "tool_result":
            block["tool_use_id"] = "changed-server-alias"
            return Verdict("transform", payload)
        return Verdict("allow")

    client = FakeClient(policy)
    model = FakeModel([])
    guard = TrustGuardIntervention(client, text_assessment=text_assessment)
    agent = Agent(model=model, interventions=[guard], tools=[make_echo(effects)], callback_handler=None)
    with pytest.raises(Exception) as caught:
        agent.tool.echo(value="original")
    assert isinstance(protection_error(caught.value), TrustGuardTransformError)
    assert effects == ([] if surface == "tool_input" else ["original"])
    assert model.index == 0 and "changed-server-alias" not in json.dumps(agent.messages)
    before = len(client.calls)
    with pytest.raises(Exception) as caught:
        agent("retry")
    assert isinstance(protection_error(caught.value), TrustGuardStateError)
    assert len(client.calls) == before


def test_swapped_aliases_between_model_tool_calls_are_rejected_before_any_tool() -> None:
    effects: list[str] = []
    response = tool_response(tool_id="original-first", tool_input={"value": "first"})
    response["content"] += tool_response(tool_id="original-second", tool_input={"value": "second"})["content"]

    def policy(payload: dict[str, Any], direction: str, index: int) -> Verdict:
        use_ids, _ = aliases(payload)
        if direction == "output" and len(use_ids) == 2:
            blocks = payload["messages"][0]["content"]
            blocks[0]["id"], blocks[1]["id"] = blocks[1]["id"], blocks[0]["id"]
            return Verdict("transform", payload)
        return Verdict("allow")

    model = FakeModel([response, text_response("never")])
    client = FakeClient(policy)
    guard = TrustGuardIntervention(client)
    agent = Agent(
        model=model,
        interventions=[guard],
        tools=[make_echo(effects)],
        callback_handler=None,
        tool_executor=SequentialToolExecutor(),
    )
    with pytest.raises(Exception) as caught:
        agent("question")
    assert isinstance(protection_error(caught.value), TrustGuardTransformError)
    assert model.index == 1 and effects == []
    assert model.responses[0] == response
