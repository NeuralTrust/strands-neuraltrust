"""Offline model plus in-process HTTP evaluator; no credentials or live services.

Run with OTEL_SEMCONV_STABILITY_OPT_IN='gen_ai_unredacted_attributes='.
The marker evaluator demonstrates plumbing; it is not a security classifier.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from strands.models import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent

from strands_neuraltrust import GuardedAgent, TrustGuardBlocked, TrustGuardClient, TrustGuardConfig


class OfflineModel(Model):
    """A deterministic Strands model whose output is entirely local."""

    def __init__(self) -> None:
        self.calls = 0

    def get_config(self) -> dict[str, Any]:
        return {}

    def update_config(self, **model_config: Any) -> None:
        if model_config:
            raise ValueError("The offline model has no configurable options.")

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This example supports text only.")

    async def stream(
        self,
        messages: Messages,
        tool_specs: Any = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}}
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "A locally evaluated response."},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 0},
            }
        }


def evaluate_locally(request: httpx.Request) -> httpx.Response:
    envelope = json.loads(request.content)
    assert request.url.host == "offline.example.invalid"
    assert envelope["protocol"] == "llm"
    assert envelope["direction"] in ("input", "output")
    blocked = "BLOCK_DEMO" in json.dumps(envelope["payload"])
    return httpx.Response(200, json={"status": "block" if blocked else "allow"})


async def main() -> None:
    config = TrustGuardConfig(api_key="synthetic-offline-key", base_url="https://offline.example.invalid")
    async with httpx.AsyncClient(transport=httpx.MockTransport(evaluate_locally)) as http:
        async with TrustGuardClient(config, async_http_client=http) as evaluator:
            agent = GuardedAgent(model=OfflineModel(), client=evaluator)
            print((await agent.invoke_async("A harmless synthetic request.")).text)

            blocked_model = OfflineModel()
            blocked_agent = GuardedAgent(model=blocked_model, client=evaluator)
            try:
                await blocked_agent.invoke_async("BLOCK_DEMO")
            except TrustGuardBlocked:
                print(f"Blocked before model execution: {blocked_model.calls == 0}")
            else:
                raise AssertionError("The synthetic block was not enforced.")


if __name__ == "__main__":
    asyncio.run(main())
