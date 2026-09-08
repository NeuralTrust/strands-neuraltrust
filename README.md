# strands-neuraltrust

Integrate [NeuralTrust TrustGuard](https://neuraltrust.ai) with
[Strands Agents](https://strandsagents.com/) to evaluate prompts, conversation
history, model responses, tool arguments, and tool results against your policies.

## Install

```bash
pip install strands-neuraltrust
```

or:

```bash
uv add strands-neuraltrust
```

## Configure

Create an **Application** collector in NeuralTrust, assign its policy, and obtain
its API key and your TrustGuard deployment URL. The API key identifies the
collector; no separate collector ID or key is needed.

Set these variables before starting your application. The telemetry setting must
be in place before creating any Strands agent or tracer:

```bash
export TRUSTGUARD_API_KEY='<collector-api-key>'
export TRUSTGUARD_BASE_URL='<your-trustguard-deployment-url>'
export OTEL_SEMCONV_STABILITY_OPT_IN='gen_ai_unredacted_attributes='
```

Pass your configured, stateless Strands `Model` to a guarded invocation:

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

Configure provider credentials and model selection in your application. Pass
registered Strands tools with `GuardedAgent(..., tools=[your_tool])` to evaluate
their arguments before execution and their completed results before the next
model call. Tool-result checks cannot undo effects a tool has already produced.

| Setting | Purpose |
| --- | --- |
| `TrustGuardConfig.api_key` | Required collector API key. |
| `TrustGuardConfig.base_url` | Required HTTPS deployment root; the client appends `/v1/evaluate`. |
| `TrustGuardConfig.timeout` | Evaluation request timeout in seconds. |
| `GuardedAgent.session_id` | Optional session identifier for evaluation records. |
| `GuardedAgent.consumer_id` | Optional consumer identifier for evaluation records. |

The example reads environment variables explicitly; `TrustGuardConfig` does not
discover them automatically. The client context closes owned HTTP clients. For
synchronous code, use `agent.invoke(prompt)` inside `with TrustGuardClient(config)`.
Keep the agent and evaluator alive together for a multi-turn conversation. Create
a new agent after any failed or cancelled invocation.

## Verdicts

| TrustGuard | Integration behavior |
| --- | --- |
| `allow` | Continue with the assessed content. |
| `report` | Continue and record the decision. |
| `transform` | Apply a validated replacement to supported text or JSON. |
| `block` | Stop with `TrustGuardBlocked`. |
| `ask` | Stop with `TrustGuardApprovalRequired`; interactive resume is unsupported. |
| Evaluation failure or invalid response | Stop with a typed `TrustGuardError`. |

`GuardedResult.text` contains the accepted final text. `GuardedResult.decisions`
contains content-free stage and status records. Raw findings and model events
are not exposed through the result.

## Agent integration

| API | Use it when |
| --- | --- |
| `GuardedAgent` | You want complete, checked text responses with sequential tools and controlled callbacks. |
| `TrustGuardIntervention` | You manage a native Strands `Agent` and its execution settings, callbacks, and telemetry. |

Both paths evaluate supported content at invocation, model, and tool boundaries.
By default, each boundary uses structured and text assessments so text-oriented
detectors can also inspect history, tool content, and supported JSON values.
Both assessments must pass before staged changes are applied.

## Streaming and failures

`GuardedAgent` returns completed responses through `invoke` and `invoke_async`;
it does not expose a token stream. With a native agent, intervention checks do
not prevent raw SDK callbacks or streaming consumers from seeing content before
evaluation completes.

Evaluation errors fail closed. Models and tools remain trusted application code.
Configure provider logging and other telemetry sinks separately; the Strands
trace setting does not redact every sink or the separate system-prompt attribute.

See the [official integration guide](https://docs.neuraltrust.ai/integrations/strands)
for native-agent examples, full configuration, supported content, and security
boundaries.

## Develop

```bash
uv sync --locked --all-groups
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest -q
```

Run the self-contained example without credentials or live services:

```bash
OTEL_SEMCONV_STABILITY_OPT_IN='gen_ai_unredacted_attributes=' uv run --locked python examples/offline.py
```

Contributions follow the [NeuralTrust contribution guidelines](https://github.com/NeuralTrust/.github/blob/main/CONTRIBUTING.md).
CI and release automation use [NeuralTrust's shared workflows](https://github.com/NeuralTrust/workflows).
Stable releases are published to [PyPI](https://pypi.org/project/strands-neuraltrust/);
development publishing uses the internal registry when enabled.

## License

Licensed under the [MIT License](LICENSE).
