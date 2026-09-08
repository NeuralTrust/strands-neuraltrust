# Security

Report vulnerabilities privately to the NeuralTrust maintainers through your
existing project contact. Do not open a public issue with exploit details,
credentials, production prompts, raw service responses, or unredacted traces.
A verified public security reporting contact must be established before this
unpublished package is released.

Include the package, Strands, and Python versions; the affected lifecycle
boundary; a synthetic reproduction; expected and observed behavior; and the
relevant collector policy mode.

Read the [security boundaries](docs/security-boundaries.md) before deployment.
`GuardedAgent` protects its supported complete-result interface. Models and tools
are trusted application code, and collector accuracy and configuration remain
deployment responsibilities.

Discard the agent after a failed or cancelled invocation. Do not resume pending
tool history. The integration cannot undo completed tool effects or control
provider retention, third-party logging, exporters, or unprotected child agents.

Compatibility is limited to Python 3.10–3.14 and Strands Agents 1.54.0. Unsupported
versions and interfaces do not inherit the documented guarantees.
