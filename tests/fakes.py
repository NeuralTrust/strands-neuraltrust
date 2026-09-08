"""Synthetic collaborators exercising the real Strands public execution APIs."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any

from strands.models import Model

from strands_neuraltrust._contracts import Verdict


def text_response(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"text": text}]}


def tool_response(
    name: str = "echo", tool_input: Any = None, tool_id: str = "call-1", text: str | None = None
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [] if text is None else [{"text": text}]
    content.append(
        {"toolUse": {"name": name, "toolUseId": tool_id, "input": {} if tool_input is None else tool_input}}
    )
    return {"role": "assistant", "content": content}


class FakeModel(Model):
    """Emit predefined public stream events and retain exact provider inputs."""

    def __init__(self, responses: Sequence[dict[str, Any] | Exception]) -> None:
        self.responses = copy.deepcopy(list(responses))
        self.requests: list[dict[str, Any]] = []
        self.index = 0
        self.config: dict[str, Any] = {"model_id": "synthetic-local", "context_window_limit": 100_000}

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return self.config.copy()

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        raise AssertionError("This model intentionally supports only complete ordinary invocations.")
        yield {}  # pragma: no cover

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: Any = None,
        *,
        system_prompt_content: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self.requests.append(
            copy.deepcopy(
                {
                    "messages": messages,
                    "tool_specs": tool_specs,
                    "system_prompt": system_prompt,
                    "system_prompt_content": system_prompt_content,
                }
            )
        )
        if self.index >= len(self.responses):
            raise AssertionError("The agent made an unexpected additional model request.")
        response = self.responses[self.index]
        self.index += 1
        await asyncio.sleep(0)
        if isinstance(response, Exception):
            raise response
        yield {"messageStart": {"role": response["role"]}}
        stop_reason = "end_turn"
        for block in response["content"]:
            if "toolUse" in block:
                use = block["toolUse"]
                stop_reason = "tool_use"
                start = {"toolUse": {"name": use["name"], "toolUseId": use["toolUseId"]}}
                yield {"contentBlockStart": {"start": start}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(use["input"])}}}}
            else:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": copy.deepcopy(block)}}
            yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": stop_reason}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                "metrics": {"latencyMs": 0},
            }
        }


class FakeClient:
    """Capture evaluation requests and apply deterministic synthetic decisions."""

    def __init__(
        self,
        policy: Callable[[dict[str, Any], str, int], Any] | None = None,
        *,
        verdicts: Sequence[Verdict | Exception] | None = None,
    ) -> None:
        self.policy = policy
        self.verdicts = list(verdicts or [])
        self.calls: list[dict[str, Any]] = []

    async def aevaluate(self, payload: dict[str, Any], direction: str = "input", **context: Any) -> Verdict:
        index = len(self.calls)
        self.calls.append(copy.deepcopy({"payload": payload, "direction": direction, **context}))
        await asyncio.sleep(0)
        if self.policy is not None:
            verdict = self.policy(copy.deepcopy(payload), direction, index)
            if inspect.isawaitable(verdict):
                verdict = await verdict
        else:
            verdict = self.verdicts[index] if index < len(self.verdicts) else Verdict("allow")
        if isinstance(verdict, Exception):
            raise verdict
        return verdict
