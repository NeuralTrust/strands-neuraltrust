# Configuration and API

Version `0.1.0.dev2` supports Python 3.10–3.14 and pins Strands Agents to 1.54.0.
SDK upgrades require lifecycle and telemetry compatibility checks. Runtime
dependencies are HTTPX `>=0.28.1,<0.29` and JSON Schema `>=4.23,<5`; `uv.lock`
records the development environment.

## Collector configuration

```python
from strands_neuraltrust import TrustGuardConfig

config = TrustGuardConfig(
    api_key="synthetic-example-key",
    base_url="https://guard.example.invalid",
    timeout=5.0,
    max_request_bytes=1_048_576,
    max_response_bytes=1_048_576,
)
```

Constructing the config performs no evaluation. The immutable config requires an
explicit API key and deployment URL; it reads no environment variables. Its repr
omits the key. The example host is non-routable.

| Setting | Meaning |
| --- | --- |
| `api_key` | Collector API key used as `Authorization: Bearer …`; service tokens and collector selectors are unsupported. |
| `base_url` | Deployment root, optionally with a path prefix. The endpoint is `<base_url>/v1/evaluate`. |
| `timeout` | Positive finite seconds. Native async evaluation has an overall HTTP deadline; sync evaluation has phase timeouts and elapsed checks between chunks. |
| `max_request_bytes` | Maximum UTF-8 JSON envelope size before a request is sent. |
| `max_response_bytes` | Maximum accepted response body size, checked against headers and streamed bytes. |
| `allow_local_http` | Default `False`. Explicitly permits HTTP only for `localhost`, literal IPv4 loopback, or `::1`, for local testing. |

Real deployment URLs require HTTPS. URLs containing credentials, query strings,
fragments, encoded path segments, dot segments, or malformed hosts are rejected.
Owned transports verify TLS, ignore environment proxy configuration, disable
redirects, and request uncompressed JSON. The response must be HTTP 200,
`application/json`, and identity encoding. Other responses fail closed; no
redirect or retry is performed.

Request envelopes contain `payload`, an explicit `direction` (`input` or
`output`), and `protocol="llm"`. Optional `session_id`, `consumer_id`, and
`attributes` use the documented collector API fields. They route evaluations;
they do not create Strands sessions. Unknown service verdicts, duplicate JSON
keys, nonfinite numbers, invalid Unicode, malformed transforms, and inconsistent
finding action precedence are rejected. Raw findings are not exposed.

## Client lifecycle

```python
from strands_neuraltrust import TrustGuardClient, TrustGuardConfig, Verdict


def evaluate_text(config: TrustGuardConfig, text: str) -> Verdict:
    with TrustGuardClient(config) as client:
        return client.evaluate({"input": text}, "input")
```

`evaluate` and `aevaluate` return advisory `Verdict` values, including `block` and
`ask`. Application enforcement belongs to the intervention/facade. A successful
HTTP response or a returned value alone is not authorization.

Owned sync clients reuse a serialized connection pool. Owned async evaluations
create and close a client on the same loop for each request, supporting repeated
sync facade calls that create distinct event loops. This lifecycle trades
connection reuse for predictable loop ownership.

`TrustGuardClient(config, http_client=..., async_http_client=...)` accepts caller
HTTPX clients. The caller must supply trustworthy transport/TLS/proxy/event-hook
configuration and close those clients itself. The adapter overrides request auth
and redirect behavior, but cannot control arbitrary custom transport code. An
injected async client binds to its first evaluation loop and refuses another
loop. For repeated synchronous facade calls, use owned transports; a sync HTTPX
injection is not used by async evaluation.

Use `close()`/`aclose()` or their context managers. Closing the adapter prevents
new evaluations and never closes injected clients. Finish or cancel outstanding
calls before closing shared resources. Async cancellation propagates and owned
per-request resources are closed. Deadline cancellation depends on cooperative
async transports/tools; arbitrary blocking user code is not forcibly interrupted.
The synchronous elapsed bound can be exceeded by the duration of an in-flight
HTTP phase. GuardedAgent's SDK hooks use native async evaluation on both its sync
and async invocation paths.

## Guarded agent

Construct `GuardedAgent` with an explicit Strands `Model` and evaluator. Optional
parameters are `tools`, text `system_prompt`, `session_id`, `consumer_id`,
`max_turns=8`, `timeout=60.0`, `max_stream_bytes=1_048_576`,
`max_evaluations=128`, and `text_assessment=True`. Pass registered tools; dynamic
tool discovery and arbitrary agent construction options are unsupported.

Models declaring `stateful=True` are refused: provider-managed conversation state
cannot be inspected as local history. Use a fresh, stateless model instance whose
implementation and provider settings the application trusts.

The timeout covers preflight and the invocation. Stream accounting includes
serialized raw model events and tool stream events; it is a conservative
resource budget, not a token count or an exact text-output byte count. Final
returned UTF-8 text is also bounded. The evaluation budget counts actual
evaluation requests, including facade preflight and additional text assessments.
Preflight checks the new prompt, system instructions, and registered tool
declarations before SDK tracing starts. Model input is re-evaluated before each
model turn, including supported conversation history, system instructions, and
tool declarations. There is no policy-decision cache.

With `text_assessment=True`, each successful boundary normally uses two
requests: the normalized structured payload, followed by an additional message
containing its protected text and the values inside supported JSON. Input
assessments use a user message; output assessments use an assistant message. A
fixed immutable prefix prevents protected text from impersonating a selectively
extracted message envelope. The collector, direction, session, and consumer remain the
same. A block, approval requirement, invalid transformation, or evaluation
failure stops immediately. Staged changes are applied to SDK objects only after
both assessments and reconstruction pass.

The intervention and facade replace opaque tool-call IDs with consistent,
nonnumeric aliases in each structured assessment. They retain the original IDs
locally and restore them only after validating the returned aliases unchanged.
This prevents generic PII detection from interpreting SDK routing IDs as phone
numbers. Tool names, arguments, results, schemas, and system text remain subject
to assessment. This normalization also applies with `text_assessment=False`.
The direct advisory `TrustGuardClient` sends its caller-supplied payload as given.

The additional request is intentional even after a structured `allow`: an
allowed verdict does not prove every content field was inspected. It consumes
quota, adds latency, creates ordinary service evaluation records, and may affect
stateful detector counters. Both requests remain subject to the client's byte
and time bounds and the invocation's total bounds. Decision records summarize
the logical boundary: transform takes precedence over report, then allow.

An accepted text-only facade invocation normally needs eight requests; the
corresponding native invocation needs six. Blocks and invalid responses can stop
earlier. Direct native tool calls share the agent's counter until a normal
invocation begins. The standalone `assess` method starts its own per-call budget.

Set `text_assessment=False` only for a collector whose structured assessment has
been independently qualified for all required history, tool, and metadata
surfaces. This explicit compatibility option restores one structured assessment
per boundary and removes the additional text coverage. It is not equivalent
protection with a collector that scans only the latest user message.

One facade supports one invocation at a time. Use separate instances for
concurrent conversations. The caller owns model/evaluator lifetimes. A successful
conversation retains local history; any failed or cancelled invocation makes the
instance unusable for later invocation. Constructing a replacement does not undo
previous tool effects or delete state in an external model provider.

## Lower-level intervention

```python
from strands import Agent
from strands.models import Model
from strands_neuraltrust import EvaluationClient, TrustGuardIntervention


def native_agent(model: Model, evaluator: EvaluationClient) -> Agent:
    return Agent(
        model=model,
        interventions=[TrustGuardIntervention(evaluator)],
        callback_handler=None,
    )
```

With the lower-level intervention, the application owns callbacks, tracing,
sessions, hooks, middleware, streaming, and tool executor configuration. Review
[security boundaries](security-boundaries.md) before using it. Each shared intervention keeps separate failure/decision state per agent;
do not mutate agent configuration while it is executing.

`TrustGuardIntervention` also defaults to `text_assessment=True` and shares the
same transformation and request-budget behavior. Its standalone `assess` method
uses a bounded assessment without an SDK agent; it does not authorize a later
unassessed execution.

## Errors

Catch `TrustGuardError` for terminal protection failures. Specific subclasses are
`TrustGuardBlocked`, `TrustGuardApprovalRequired`,
`TrustGuardAuthenticationError`, `TrustGuardConfigurationError`,
`TrustGuardProtocolError`, `TrustGuardTransformError`,
`TrustGuardUnsupportedContentError`, `TrustGuardUnavailable`, and
`TrustGuardStateError`. Error messages are content-free; raw upstream bodies and
finding evidence are deliberately unavailable. `asyncio.CancelledError` remains
cancellation and is not converted into an allowed decision.
