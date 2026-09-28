"""Qualify real MCP adapters and exported telemetry across supported SDK releases."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp.types import Tool
from strands.tools.mcp import MCPAgentTool

from strands_neuraltrust import GuardedAgent, TrustGuardBlocked, TrustGuardUnavailable, Verdict
from tests.fakes import FakeClient, FakeModel, text_response, tool_response


class _MCPConnection:
    """Keep transport offline while exercising the SDK's real MCP tool adapter."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def call_tool_async(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "toolUseId": kwargs["tool_use_id"],
            "status": "success",
            "content": [{"text": self.result}, {"json": {"source": "synthetic"}}],
        }


def _mcp_tool(connection: _MCPConnection) -> MCPAgentTool:
    # MCP 1.x and 2.x both accept these wire aliases, although their model field
    # names differ. The adapter must preserve the server's original tool name.
    descriptor = Tool.model_validate(
        {
            "name": "lookup",
            "description": "Look up a synthetic value",
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True},
        }
    )
    return MCPAgentTool(descriptor, connection, name_override="remote_lookup")


def test_mcp_adapter_preserves_validated_arguments_and_results() -> None:
    connection = _MCPConnection("original-result")
    model = FakeModel(
        [tool_response("remote_lookup", {"value": "original-argument"}), text_response("answer")]
    )

    def policy(payload: dict[str, Any], direction: str, _: int) -> Verdict:
        block = payload["messages"][0]["content"][0]
        if direction == "input" and block["type"] == "tool_use":
            block["input"]["value"] = "checked-argument"
        elif block["type"] == "tool_result":
            block["content"][0]["text"] = "checked-result"
        else:
            return Verdict("allow")
        return Verdict("transform", payload)

    result = GuardedAgent(model=model, client=FakeClient(policy), tools=[_mcp_tool(connection)]).invoke(
        "question"
    )
    assert result.text == "answer"
    assert len(connection.calls) == 1
    assert connection.calls[0]["name"] == "lookup"
    assert connection.calls[0]["arguments"] == {"value": "checked-argument"}
    tool_result = model.requests[1]["messages"][2]["content"][0]["toolResult"]
    assert tool_result["content"] == [{"text": "checked-result"}, {"json": {"source": "synthetic"}}]
    assert tool_result["toolUseId"] == connection.calls[0]["tool_use_id"]
    assert {decision.stage for decision in result.decisions if decision.status == "transform"} == {
        "tool_input",
        "tool_output",
    }


@pytest.mark.parametrize("blocked_boundary", ["tool_use", "tool_result"])
def test_mcp_denial_stops_effects_or_model_continuation(blocked_boundary: str) -> None:
    connection = _MCPConnection("tool-result")
    model = FakeModel([tool_response("remote_lookup", {"value": "argument"}), text_response("must-not-run")])

    def policy(payload: dict[str, Any], direction: str, _: int) -> Verdict:
        block = payload["messages"][0]["content"][0]
        # Selecting input direction lets the model's tool request pass its output
        # assessment, so this checks the actual before/after-tool boundary.
        return Verdict("block" if direction == "input" and block["type"] == blocked_boundary else "allow")

    agent = GuardedAgent(model=model, client=FakeClient(policy), tools=[_mcp_tool(connection)])
    with pytest.raises(TrustGuardBlocked):
        agent.invoke("question")
    assert len(connection.calls) == (0 if blocked_boundary == "tool_use" else 1)
    assert len(model.requests) == 1


def _check_exported_spans() -> None:
    """Run only in a fresh process; never replace the test runner's provider."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    markers = {
        "prompt": "private-prompt-f02c",
        "argument": "private-argument-9adc",
        "result": "private-result-2ced",
        "answer": "private-answer-ab21",
        "error": "private-provider-error-217a",
    }
    connection = _MCPConnection(markers["result"])
    model = FakeModel(
        [tool_response("remote_lookup", {"value": markers["argument"]}), text_response(markers["answer"])]
    )
    result = GuardedAgent(model=model, client=FakeClient(), tools=[_mcp_tool(connection)]).invoke(
        markers["prompt"]
    )
    assert result.text == markers["answer"]
    spans = exporter.get_finished_spans()
    assert len(spans) >= 6
    exported = "\n".join(span.to_json() for span in spans)
    assert all(marker not in exported for marker in markers.values())
    exporter.clear()

    failing = GuardedAgent(model=FakeModel([RuntimeError(markers["error"])]), client=FakeClient())
    with pytest.raises(TrustGuardUnavailable):
        failing.invoke("question")
    spans = exporter.get_finished_spans()
    assert len(spans) >= 2
    assert markers["error"] not in "\n".join(span.to_json() for span in spans)
    provider.shutdown()


@pytest.mark.parametrize("conventions", ["", "gen_ai_latest_experimental,"])
def test_sdk_exported_spans_redact_guarded_content(conventions: str) -> None:
    env = os.environ.copy()
    env["OTEL_SEMCONV_STABILITY_OPT_IN"] = conventions + "gen_ai_unredacted_attributes="
    code = """
from pytest_socket import disable_socket
disable_socket(allow_unix_socket=True)
from tests.integration.test_sdk_compatibility import _check_exported_spans
_check_exported_spans()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
