# Protection boundaries

TrustGuard is an evaluator whose decision this package enforces at selected
Strands boundaries. The integration does not establish classifier accuracy,
authorize its own approvals, or sandbox model and tool implementations.

## Supported interface

`GuardedAgent` constructs and owns the SDK agent. It accepts text through
`invoke`/`invoke_async` and returns only complete accepted text plus bounded
decision metadata. It exposes no raw stream, raw `AgentResult`, direct-tool
shortcut, session restore, structured-output shortcut, or resume method.
Accessing underscored internals or modifying the owned agent bypasses the
supported interface.

| Boundary | Evaluation and enforcement |
| --- | --- |
| Facade preflight | New user text, system instructions, and tool declarations are evaluated before SDK invocation and its agent tracing starts. |
| Before invocation | Supported new messages are evaluated before the model run. |
| Before each model call | Local conversation, system text, and current tool declarations are evaluated as input. |
| After each model call | Completed model content is evaluated as output, including intermediate tool-call turns. |
| Before each tool call | Exact supported arguments and registered schema are evaluated as input before the tool runs. |
| After each tool call | Completed text/JSON result, including an error result, is evaluated as input before model continuation. |

Tool results intentionally use the **input** detector direction because they are
untrusted input to the next model turn. The deployed collector must enable the
appropriate detectors/policy for this routing. Facade output receives an output
check at every completed model turn. Input and output direction are always sent
explicitly; the client never relies on the server's default direction.

By default, structured assessment is followed by one additional text assessment
at the same boundary. The latter presents supported history, arguments, results,
and relevant system/tool metadata as message text so a latest-user or
assistant-text detector can inspect them. It preserves the selected direction
and routing. Both decisions must pass. This improves content coverage; it does
not override policy matching, report-only settings, classifier decisions, or
service failure configuration. Disabling `text_assessment` requires separate
qualification of those structured surfaces.

All policy blocks, approval requests, malformed decisions, failed evaluations,
unsupported transformations, and unsafe execution failures stop the invocation.
Post-call checks raise terminal exceptions rather than relying on Strands `Deny`,
whose post-call semantics are insufficient in Strands 1.54.0. A redispatched
tool exception cannot turn a recorded policy failure into a successful tool-error
continuation. Failed conversations refuse subsequent calls to prevent replay of
pending tool state.

The facade uses sequential tools and disables agent retries. Its own evaluator
also performs no retries. These constraints prevent queued work from starting
after a terminal failure on the supported path; they do not supply all-or-nothing
batch authorization. Tool A may already have written a file when tool B is
denied. A checked tool result cannot reverse the tool's earlier effects.

## Content and transformation restrictions

Supported local content is user/assistant text, assistant `toolUse`, and user
`toolResult` with text or JSON blocks. The evaluator receives normalized
Anthropic-style messages. JSON tool-result blocks are serialized into individual
text blocks for evaluation and reconstructed into their original JSON block
types after validation.

Opaque tool-use and tool-result IDs are linked through consistent aliases for
each structured assessment. The original routing IDs stay local to the adapter;
they are not content-scanned. Aliases must remain unchanged in a transformation
before the adapter restores the exact originals. Names, arguments, results,
system instructions, and declarations are assessed normally. This mapping is
per assessment and does not change SDK history or provider requests. A returned
unknown, swapped, added, or modified alias is a refusal.

The additional text payload uses an immutable prefix and an ordered mapping to
the original content locations. JSON values are exposed as individual units so
serialized punctuation does not hide them from text tokenization. Transformations
are reconstructed by that mapping, with strict JSON and category checks. Repeated
equal text does not cause unrelated fields to be replaced. Changes remain staged
until both assessments and all structural/schema checks succeed. A changed prefix,
message role, block count, or immutable metadata is a refusal.

Encoded JSON objects, arrays, and quoted strings are decoded for this text
assessment. Unchanged encoded text keeps its original formatting. Bare
numeric-looking, boolean-looking, and null-looking SDK strings remain strings;
actual typed JSON values keep their categories. Projection traversal is bounded
to 64 levels and 100,000 nodes, including encoded layers. Recognizable JSON that
violates strict duplicate-key, nonfinite-number, or resource checks is refused.

Transformations preserve message count, order, roles, block count and kinds,
tool names and IDs, result success/error state, system instructions, and tool
declarations. Tool JSON transformations preserve object keys, array lengths,
and scalar categories. Arguments must validate against the registered schema
before execution. Invalid or ambiguous replacements fail closed rather than
leaving the original content in place. Full message evaluations require the
matching structured replacement; a generic `{"input": "replacement"}` is not a
valid rewrite of an evaluated multi-message payload.

Tool schemas must be reference-free JSON Schema 2020-12 (or omit `$schema` and use
that draft's semantics). `$ref`, `$dynamicRef`, and `$recursiveRef` are rejected,
including local references, to prevent unsupported resolution and external
retrieval. Output schemas and extra provider-specific tool declaration fields
are unsupported. Message metadata/tracking IDs are preserved locally and omitted
from evaluator payloads as non-provider content. Tool annotations are not policy
input; do not encode authorization data only in annotations.

Images, audio, video, documents, binary attachments, reasoning/signatures,
citations, cache points, guard-content blocks, and unrecognized message structures
are rejected. The facade additionally checks raw model event structure and
resource budgets before the SDK consumes those events. It rejects malformed tool
argument JSON before the SDK can repair it into an empty object. The lower-level
intervention alone does not supply that raw stream parser boundary.

Generated tool arguments must complete as a JSON object. Streamed tool identities
must be supplied in the start event; rich reasoning, provider-specific additional
response fields, and alternate identity-in-delta protocols are refused. Numeric
usage/latency metadata is validated before the SDK uses it. The monitor accepts
the supported sequential Bedrock-style event protocol, including implicit text
block starts, and refuses truncated streams even if the SDK would infer success.

## Streaming, telemetry, and logs

The native Strands intervention sees completed model output. Raw
`Agent.stream_async` events and application callbacks can already have observed
tokens when a later evaluation blocks them. Setting `callback_handler=None`
disables default printing but does not make raw streaming preventive.

The facade owns a callback that validates/bounds internal events and emits no
model text to the caller. It returns a deliberately small result rather than SDK
history, event objects, tool diagnostics, or metrics carrying unassessed nested
content. It does not offer token streaming; remote evaluation adds latency before
the complete response can be returned.

At process startup, set exactly:

```bash
export OTEL_SEMCONV_STABILITY_OPT_IN='gen_ai_unredacted_attributes='
```

The empty allowlist enables redaction of all controlled Strands content fields.
The facade verifies the initialized tracer state, not merely the environment
variable, on construction and invocation. Changing the variable after another
agent creates the tracer is insufficient; start a fresh process. The integration
does not mutate global telemetry settings. Strands 1.54 lacks a public redaction
state accessor, so the diagnostic reads three version-specific fields; an absent
or changed field refuses construction. It does not patch tracer behavior or
private execution methods.

The facade also wraps the supplied model through the public `Model` interface.
Provider stream exceptions become content-free errors before Strands records
exception messages or stack traces. The wrapper closes its underlying iterator;
the application still owns the provider's lifetime. Local token estimation uses
the base model heuristic, not a provider's optional remote token-counting method.
Provider-internal logging, instrumentation, retries, and request augmentation
remain outside this boundary.

The SDK's separate `system_prompt` span attribute is not covered by its content
redaction setting. Facade preflight screens system/tool context before invocation,
so blocked system instructions do not enter that span; **allowed system text can
still appear in telemetry**. Do not place secrets in system instructions on the
assumption that this environment setting removes every SDK attribute.

This check covers the supported Strands tracer fields. It does not control
third-party spans, exception exporters, Python logging handlers, tool-owned
stdout/stderr, HTTPX debug logging, external callbacks, or model-provider
instrumentation. Configure those application sinks separately and avoid logging
prompts, results, request bodies, credentials, and raw exceptions. The lower-level
SDK can log malformed model tool input before intervention checks; its raw
streaming/parser behavior must be assessed independently when using native Agent.
Do not infer a blanket “no data leakage” property from the facade's return value
or from OTel redaction.

## Application and deployment responsibilities

Model implementations, tool code, application construction, injected transports,
and their instrumentation are trusted application components. Provider-specific
augmentation, server-side history, hidden model state, malicious model code,
child-agent invocations, and direct MCP clients are not automatically protected
by guarding a parent. Custom plugins, hooks, model middleware, context injectors,
sessions, and arbitrary executor composition are excluded from facade
construction because they can change content outside an inspected boundary.
The facade rejects models declaring `stateful=True`. A stateless model flag is
not a sandbox or proof that custom model code has no hidden state.

The native intervention can assess direct `agent.tool` arguments and results,
but Strands also copies direct-call keyword values into
`ToolContext.invocation_state`. That separate state is outside the intervention's
transformation boundary. A tool reading invocation state may therefore observe
original arguments independently of its transformed tool input. Direct-call
state consumption is outside the supported transformation path; the facade
exposes no direct-tool method.

TrustGuard necessarily receives all content it evaluates, including supported
history, system text, tool metadata, arguments, and results. Choose and verify
its deployment, authentication, retention, and access controls deliberately.
An Observe policy, unmatched policy, or server configured to fail open can return
`allow`/`report`; client fail-closed transport behavior cannot convert that into a
validated deployment policy. Verify each required verdict and failure mode
against your configured deployment.
Synthetic tests establish adapter behavior, not deployed detector behavior.

Memory limits bound accepted request/response/event sizes, not every allocation
inside arbitrary provider or tool code. Python cancellation is cooperative.
Application code performing uninterruptible blocking work cannot be forcibly
stopped by an intervention deadline. Resource and policy limits fail closed on
the supported asynchronous boundaries.
