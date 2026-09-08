"""Check actual fresh SDK singleton configuration, including too-late changes."""

import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from strands_neuraltrust import TrustGuardConfigurationError, _telemetry


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"_redaction_enabled": False},
        {"_redaction_enabled": True, "_unredacted_exact": frozenset(), "_unredacted_globs": ("gen_ai.",)},
        {
            "_redaction_enabled": True,
            "_unredacted_exact": frozenset({"gen_ai.input.messages"}),
            "_unredacted_globs": (),
        },
    ],
)
def test_unknown_or_partial_redaction_state_fails_closed(monkeypatch: Any, attributes: Any) -> None:
    monkeypatch.setattr(_telemetry, "get_tracer", lambda: SimpleNamespace(**attributes))
    with pytest.raises(TrustGuardConfigurationError):
        _telemetry.require_redacted_tracer()


@pytest.mark.parametrize(
    "initial,late,accepted",
    [
        (None, None, False),
        (None, "gen_ai_unredacted_attributes=", False),
        ("gen_ai_unredacted_attributes=", None, True),
        ("gen_ai_unredacted_attributes=gen_ai.input.messages", None, False),
        ("gen_ai_unredacted_attributes=gen_ai.*", None, False),
        ("gen_ai_latest_experimental,gen_ai_unredacted_attributes=", None, True),
    ],
)
def test_fresh_process_sdk_configuration(initial: Any, late: Any, accepted: bool) -> None:
    code = """
import os
from strands.telemetry.tracer import get_tracer
get_tracer()
if os.environ.get("TEST_LATE_REDACTION"):
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = os.environ["TEST_LATE_REDACTION"]
from strands_neuraltrust._telemetry import require_redacted_tracer
from strands_neuraltrust import TrustGuardConfigurationError
try:
    require_redacted_tracer()
except TrustGuardConfigurationError:
    print("refused")
else:
    print("accepted")
"""
    env = os.environ.copy()
    env.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
    env.pop("TEST_LATE_REDACTION", None)
    if initial is not None:
        env["OTEL_SEMCONV_STABILITY_OPT_IN"] = initial
    if late is not None:
        env["TEST_LATE_REDACTION"] = late
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ("accepted" if accepted else "refused")
