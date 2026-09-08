# Contributing

## Development setup

Use Python 3.10–3.14 and install the locked environment:

```bash
uv sync --locked
uv run --locked python scripts/check.py
```

The check command runs Ruff, formatting checks, strict mypy, and tests with
network sockets disabled. To format code:

```bash
uv run --locked ruff format src tests examples scripts
```

Add behavior-focused regressions for changes to enforcement, transport,
transformations, cancellation, and caller-visible results. Use synthetic data
and credentials. Default tests must neither read credentials nor choose a live
service endpoint. Separately authorized live tests and their evidence stay
private.

Strands is pinned to 1.54.0. Dependency updates need lifecycle ordering,
streaming, telemetry, cancellation, and replay checks. New content types need
lossless normalization and reconstruction, schema and identity preservation,
refusal tests, and service compatibility checks. Never silently omit unsupported
content, fail open, automatically approve `ask`, resume failed invocations, log
raw prompts, or select a hidden production endpoint.

## Build and release

```bash
SOURCE_DATE_EPOCH=1788796800 uv build
uv run --locked twine check dist/*
uv run --locked python scripts/verify_artifacts.py --rebuild
```

Verify the wheel and sdist contents, clean installs, supported Python versions,
and offline example before release. Review dependency advisories and verify the
intended TrustGuard deployment separately. Changing distributed code, docs, or
metadata requires fresh artifacts and artifact verification.

This package remains unpublished and includes `Private :: Do Not Upload`.
Publishing a repository, pushing publicly visible changes, opening public issues
or pull requests, uploading to PyPI or TestPyPI, and deploying documentation all
require explicit project-owner approval of the reviewed artifacts and actions.
Local tests and builds do not grant that approval. Establish repository/package
ownership and a verified security reporting contact before publication.
