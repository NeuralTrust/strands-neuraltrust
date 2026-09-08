"""Strict parsing of the documented collector Evaluate API envelope."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from .exceptions import TrustGuardConfigurationError, TrustGuardProtocolError

Direction = Literal["input", "output"]
Status = Literal["allow", "report", "transform", "ask", "block"]
_SEVERITY = {"allow": 0, "report": 1, "transform": 2, "ask": 3, "block": 4}
_OPAQUE_ID = re.compile(r"(?:[a-fA-F0-9]{16,64}|[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12})\Z")
_INVALID_RESPONSE = "TrustGuard returned an invalid evaluation response."


@dataclass(frozen=True, slots=True)
class Verdict:
    """Validated decision; raw findings and response bodies are never retained."""

    status: Status
    transformed_payload: dict[str, Any] | None = field(default=None, repr=False)
    request_id: str | None = None
    trace_id: str | None = None


def encode_request(
    payload: dict[str, Any],
    direction: Direction,
    *,
    max_bytes: int,
    session_id: str | None = None,
    consumer_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> bytes:
    """Serialize only supported request fields, with strict JSON and a byte bound."""
    message = "The TrustGuard evaluation request is invalid or exceeds its byte limit."
    try:
        if type(payload) is not dict or direction not in ("input", "output"):
            raise ValueError
        if attributes is not None and type(attributes) is not dict:
            raise ValueError
        envelope: dict[str, Any] = {"payload": payload, "direction": direction, "protocol": "llm"}
        for key, identifier in (("session_id", session_id), ("consumer_id", consumer_id)):
            if identifier is not None:
                if type(identifier) is not str or not identifier or len(identifier) > 1024:
                    raise ValueError
                envelope[key] = identifier
        if attributes is not None:
            envelope["attributes"] = attributes
        _validate_json(envelope)
        chunks: list[bytes] = []
        size = 0
        for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(
            envelope
        ):
            encoded = chunk.encode("utf-8")
            size += len(encoded)
            if size > max_bytes:
                raise ValueError
            chunks.append(encoded)
        return b"".join(chunks)
    except (TypeError, ValueError, RecursionError, UnicodeError, OverflowError):
        pass
    raise TrustGuardConfigurationError(message)


def _validate_json(value: Any, depth: int = 0, *, budget: list[int] | None = None) -> None:
    if budget is None:
        budget = [100_000]
    budget[0] -= 1
    if depth > 64 or budget[0] < 0:
        raise ValueError
    if value is None or type(value) in (bool, int):
        return
    if type(value) is str:
        value.encode("utf-8")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError
        return
    if type(value) is list:
        for item in value:
            _validate_json(item, depth + 1, budget=budget)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError
            key.encode("utf-8")
            _validate_json(item, depth + 1, budget=budget)
        return
    raise ValueError


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _integer(value: str) -> int:
    if len(value) > 128:
        raise ValueError
    return int(value)


def _float(value: str) -> float:
    if len(value) > 128:
        raise ValueError
    result = float(value)
    if not math.isfinite(result):
        raise ValueError
    return result


def _constant(value: str) -> Any:
    raise ValueError


def parse_verdict(body: bytes) -> Verdict:
    """Reject ambiguous JSON, malformed decisions and inconsistent action reduction."""
    try:
        data = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_int=_integer,
            parse_float=_float,
            parse_constant=_constant,
        )
        _validate_json(data)
        if type(data) is not dict:
            raise ValueError
        status = data.get("status")
        if type(status) is not str or status not in _SEVERITY:
            raise ValueError
        transformed = data.get("transformed_payload")
        if transformed is not None and type(transformed) is not dict:
            raise ValueError
        if status == "transform" and transformed is None:
            raise ValueError
        if status in ("allow", "report") and transformed is not None:
            raise ValueError
        findings = data.get("findings", [])
        if type(findings) is not list:
            raise ValueError
        for finding in findings:
            _validate_finding(finding, status)
        ids: dict[str, str | None] = {}
        for name in ("request_id", "trace_id"):
            identifier = data.get(name)
            if identifier is not None and type(identifier) is not str:
                raise ValueError
            ids[name] = identifier if identifier and _OPAQUE_ID.fullmatch(identifier) else None
        return Verdict(cast(Status, status), transformed, ids["request_id"], ids["trace_id"])
    except (TypeError, ValueError, RecursionError, UnicodeError, OverflowError):
        pass
    raise TrustGuardProtocolError(_INVALID_RESPONSE)


def _validate_finding(finding: Any, status: str) -> None:
    if type(finding) is not dict:
        raise ValueError
    for field_name in ("source", "signal", "outcome", "evidence"):
        if field_name in finding and type(finding[field_name]) is not dict:
            raise ValueError
    signal = finding.get("signal", {})
    if "confidence" in signal:
        confidence = signal["confidence"]
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise ValueError
    outcome = finding.get("outcome", {})
    if "action" in outcome:
        action = outcome["action"]
        if type(action) is not str or action not in _SEVERITY or action == "allow":
            raise ValueError
        if _SEVERITY[action] > _SEVERITY[status]:
            raise ValueError
