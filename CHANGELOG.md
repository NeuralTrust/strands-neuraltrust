# Changelog

## 0.1.0.dev2 — Unreleased

- Add a typed TrustGuard client with explicit collector configuration, bounded
  synchronous/asynchronous transport, and strict decision validation.
- Enforce decisions at Strands invocation, model, and tool boundaries through
  structured and text assessments.
- Validate transformations while preserving supported content structure, tool
  schemas, and routing identities.
- Provide complete checked responses through `GuardedAgent`, with sequential
  tools, stream validation, resource limits, and failed-conversation refusal.
- Require initialized SDK trace redaction and sanitize provider errors.
- Support Python 3.10–3.14 and Strands Agents 1.54.0.
