# strands-neuraltrust

Integrate NeuralTrust TrustGuard with the Strands Agents Python SDK to evaluate
prompts, conversation history, model output, tool arguments, and tool results.

Unpublished development version **0.1.0.dev2**. Supports Python **3.10–3.14** and
Strands Agents **1.54.0**.

`GuardedAgent` provides complete, checked text responses through a controlled
Strands agent. `TrustGuardIntervention` supplies the lower-level lifecycle
integration for applications that manage their own agent configuration.

## Quick start

Install from this checkout and configure trace redaction before any Strands
agent or tracer is created:

```bash
uv sync --locked
export OTEL_SEMCONV_STABILITY_OPT_IN='gen_ai_unredacted_attributes='
```

Provide a trusted Strands `Model` and the credentials for your chosen TrustGuard
deployment:

```python
import os

from strands.models import Model
from strands_neuraltrust import GuardedAgent, TrustGuardClient, TrustGuardConfig


async def answer(model: Model, prompt: str) -> str:
    config = TrustGuardConfig(
        api_key=os.environ["TRUSTGUARD_API_KEY"],
        base_url=os.environ["TRUSTGUARD_BASE_URL"],
    )
    async with TrustGuardClient(config) as evaluator:
        agent = GuardedAgent(model=model, client=evaluator)
        result = await agent.invoke_async(prompt)
        return result.text
```

`base_url` is the deployment root; the client appends `/v1/evaluate`. The package
selects no endpoint, collector, model provider, or AWS region automatically.
TrustGuard receives the content it evaluates. Verify your collector's policies,
detector directions, enforcement mode, and retention settings before deployment.

For a credential-free demonstration using a synthetic model and an in-process
evaluator:

```bash
uv run --locked python examples/offline.py
```

Expected output:

```text
A locally evaluated response.
Blocked before model execution: True
```

## Behavior

| TrustGuard decision | Result |
| --- | --- |
| `allow` / `report` | Continue with the assessed content. |
| `transform` | Apply a validated transformation that preserves supported structure. |
| `block` | Raise `TrustGuardBlocked` and terminate the invocation. |
| `ask` | Raise `TrustGuardApprovalRequired`; approval and resume are unsupported. |
| Invalid response or evaluation failure | Stop with a typed, sanitized `TrustGuardError`. |

Use `agent.invoke(prompt)` synchronously or `await agent.invoke_async(prompt)`
asynchronously. Both return `GuardedResult(text, decisions, stop_reason)`.
Successful calls can continue the conversation; failed or cancelled calls require
a new agent instance.

Each protected boundary uses a structured assessment followed by a text
assessment that exposes supported history and tool content to text-oriented
moderation. Both decisions must pass. These requests count toward the evaluation
budget and add service latency, quota usage, and evaluation records.

Tools execute sequentially. Arguments are checked before execution and completed
results before model continuation. Previously completed tool effects cannot be
undone. Models and tools remain trusted application code.

Supported content is text, model tool calls, and text/JSON tool results.
Multimodal content, sessions, custom middleware, raw caller streaming, and
structured-output shortcuts are outside the guarded interface. The lower-level
intervention cannot prevent raw SDK callbacks or instrumentation from observing
model output before its completed-output check. Read the
[security boundaries](docs/security-boundaries.md) for the full contract.

## Development

```bash
uv run --locked python scripts/check.py
SOURCE_DATE_EPOCH=1788796800 uv build
uv run --locked twine check dist/*
```

Default tests disable network sockets. Builds create local artifacts and do not
publish them.

See [configuration and API](docs/configuration.md),
[contributing](CONTRIBUTING.md), and [security reporting](SECURITY.md).
