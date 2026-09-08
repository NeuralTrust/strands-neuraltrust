"""Decision and configuration adversarial tests using synthetic data only."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from strands_neuraltrust._contracts import encode_request, parse_verdict
from strands_neuraltrust.config import TrustGuardConfig
from strands_neuraltrust.exceptions import TrustGuardConfigurationError, TrustGuardProtocolError


@pytest.mark.parametrize("status", ["allow", "report", "transform", "ask", "block"])
def test_known_verdicts(status: str) -> None:
    data: dict[str, Any] = {"status": status}
    if status == "transform":
        data["transformed_payload"] = {"input": "safe"}
    verdict = parse_verdict(json.dumps(data).encode())
    assert verdict.status == status
    assert not hasattr(verdict, "findings")
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.status = "allow"  # type: ignore[misc]


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not-json-PRIVATE",
        b"null",
        b"[]",
        b"true",
        b"{}",
        b'{"status":null}',
        b'{"status":true}',
        b'{"status":3}',
        b'{"status":[]}',
        b'{"status":{}}',
        b'{"status":""}',
        b'{"status":"BLOCK"}',
        b'{"status":"unknown"}',
        b'{"status":"allow","status":"block"}',
        b'{"status":"allow","extra":{"same":1,"same":2}}',
        b'{"status":"allow","extra":NaN}',
        b'{"status":"allow","extra":Infinity}',
        b'{"status":"allow","extra":-Infinity}',
        b'{"status":"allow","extra":1e999}',
        b'{"status":"allow","extra":' + b"1" * 129 + b"}",
        b'{"status":"allow","extra":0.' + b"1" * 129 + b"}",
        b'{"status":"allow","findings":null}',
        b'{"status":"allow","findings":{}}',
        b'{"status":"allow","findings":[1]}',
        b'{"status":"allow","findings":[{"signal":null}]}',
        b'{"status":"allow","findings":[{"outcome":{"action":"block"}}]}',
        b'{"status":"block","findings":[{"outcome":{"action":"allow"}}]}',
        b'{"status":"block","findings":[{"outcome":{"action":false}}]}',
        b'{"status":"block","findings":[{"outcome":{"action":"unknown"}}]}',
        b'{"status":"allow","findings":[{"signal":{"confidence":true}}]}',
        b'{"status":"allow","findings":[{"signal":{"confidence":-0.1}}]}',
        b'{"status":"allow","findings":[{"signal":{"confidence":1.1}}]}',
        b'{"status":"allow","request_id":7}',
        b'{"status":"allow","trace_id":[]}',
        b'{"status":"transform"}',
        b'{"status":"transform","transformed_payload":null}',
        b'{"status":"transform","transformed_payload":"safe"}',
        b'{"status":"transform","transformed_payload":[]}',
        b'{"status":"allow","transformed_payload":{}}',
        b'{"status":"report","transformed_payload":{}}',
        b'{"status":"allow","x":"\xff"}',
        b'{"status":"allow","extra":' + b"[" * 65 + b"0" + b"]" * 65 + b"}",
    ],
)
def test_malformed_or_ambiguous_response_fails_closed(body: bytes) -> None:
    with pytest.raises(TrustGuardProtocolError) as caught:
        parse_verdict(body)
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_findings_are_validated_and_discarded() -> None:
    verdict = parse_verdict(
        json.dumps(
            {
                "status": "block",
                "transformed_payload": {"input": "MASKED"},
                "findings": [
                    {
                        "source": {"kind": "detector"},
                        "signal": {"confidence": 1},
                        "outcome": {"action": "transform"},
                        "evidence": {"private": "PRIVATE"},
                    }
                ],
                "request_id": "b4acec05-f574-45e1-8be2-abcd12345678",
                "trace_id": "f" * 32,
            }
        ).encode()
    )
    assert verdict.request_id == "b4acec05-f574-45e1-8be2-abcd12345678"
    assert verdict.trace_id == "f" * 32
    assert "MASKED" not in repr(verdict)
    assert "PRIVATE" not in repr(verdict)


def test_observational_finding_does_not_require_an_action() -> None:
    verdict = parse_verdict(
        b'{"status":"report","findings":[{"source":{"kind":"detector"},'
        b'"signal":{"confidence":0.5},"evidence":{"synthetic":true}}]}'
    )
    assert verdict.status == "report"


@pytest.mark.parametrize("identifier", ["secret request text", "PRIVATE", "\n", "x" * 1000, "", None])
def test_nonopaque_identifiers_are_discarded(identifier: Any) -> None:
    verdict = parse_verdict(json.dumps({"status": "allow", "request_id": identifier}).encode())
    assert verdict.request_id is None


@given(st.text().filter(lambda value: value not in {"allow", "report", "transform", "ask", "block"}))
def test_arbitrary_unknown_status_never_allows(status: str) -> None:
    with pytest.raises(TrustGuardProtocolError):
        parse_verdict(json.dumps({"status": status}).encode())


def test_request_matches_documented_collector_key_envelope() -> None:
    encoded = encode_request(
        {"input": "hello 🌍"},
        "output",
        max_bytes=1024,
        session_id="conversation",
        consumer_id="actor",
        attributes={"source": {"application": "test"}},
    )
    assert json.loads(encoded) == {
        "payload": {"input": "hello 🌍"},
        "direction": "output",
        "protocol": "llm",
        "session_id": "conversation",
        "consumer_id": "actor",
        "attributes": {"source": {"application": "test"}},
    }
    assert encoded == encode_request(
        {"input": "hello 🌍"},
        "output",
        max_bytes=len(encoded),
        session_id="conversation",
        consumer_id="actor",
        attributes={"source": {"application": "test"}},
    )
    with pytest.raises(TrustGuardConfigurationError):
        encode_request(
            {"input": "hello 🌍"},
            "output",
            max_bytes=len(encoded) - 1,
            session_id="conversation",
            consumer_id="actor",
            attributes={"source": {"application": "test"}},
        )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        {1: "x"},
        {"x": (1, 2)},
        {"x": object()},
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": "\ud800"},
    ],
)
def test_invalid_request_payload_rejected_without_context(payload: Any) -> None:
    with pytest.raises(TrustGuardConfigurationError) as caught:
        encode_request(payload, "input", max_bytes=1024)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": "unknown"},
        {"direction": None},
        {"session_id": ""},
        {"session_id": 1},
        {"consumer_id": "x" * 1025},
        {"attributes": []},
    ],
)
def test_invalid_request_context(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TrustGuardConfigurationError):
        encode_request({}, **{"direction": "input", "max_bytes": 1024, **kwargs})


def test_cyclic_request_fails_closed() -> None:
    payload: dict[str, Any] = {}
    payload["cycle"] = payload
    with pytest.raises(TrustGuardConfigurationError):
        encode_request(payload, "input", max_bytes=1024)


@pytest.mark.parametrize(
    "body",
    [
        b'{"status":"allow","extra":"\\ud800"}',
        b'{"status":"allow","extra":{"\\udfff":"value"}}',
    ],
)
def test_unpaired_surrogates_rejected_in_response(body: bytes) -> None:
    with pytest.raises(TrustGuardProtocolError):
        parse_verdict(body)


def test_surrogate_pair_and_json_null_boolean_integer_are_supported() -> None:
    verdict = parse_verdict(
        b'{"status":"transform","transformed_payload":{"input":"\\ud83c\\udf0d","values":[null,true,7]}}'
    )
    assert verdict.transformed_payload == {"input": "🌍", "values": [None, True, 7]}


def test_integer_timeout_overflow_is_configuration_error() -> None:
    with pytest.raises(TrustGuardConfigurationError):
        TrustGuardConfig(api_key="key", base_url="https://guard.example", timeout=10**1000)


def test_config_is_immutable_explicit_and_credential_safe() -> None:
    config = TrustGuardConfig(api_key="PRIVATE-KEY", base_url="https://guard.example/prefix/")
    assert config.evaluate_url == "https://guard.example/prefix/v1/evaluate"
    assert "PRIVATE" not in repr(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.api_key = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": ""},
        {"api_key": None},
        {"api_key": "key\nheader"},
        {"api_key": "key space"},
        {"api_key": "sécret"},
        {"api_key": "x" * 4097},
        {"timeout": True},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": "3"},
        {"max_request_bytes": True},
        {"max_request_bytes": 0},
        {"max_response_bytes": -1},
        {"max_response_bytes": 1.5},
        {"allow_local_http": 1},
    ],
)
def test_invalid_config_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TrustGuardConfigurationError) as caught:
        TrustGuardConfig(**{"api_key": "PRIVATE", "base_url": "https://guard.example", **kwargs})
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "",
        None,
        "guard.example",
        "/relative",
        "ftp://guard.example",
        "http://guard.example",
        "https://user:PRIVATE@guard.example",
        "https://PRIVATE@guard.example",
        "https://guard.example?PRIVATE",
        "https://guard.example?",
        "https://guard.example#",
        "https://guard.example#PRIVATE",
        "https://guard.example:0",
        "https://guard.example:65536",
        "https://guard.example:bad",
        "https://guard.example\n",
        " https://guard.example",
        "https://guard.example\\evil",
        "https:///missing",
        "https://[broken",
        "https://guard.example/a/../b",
        "https://guard.example/./b",
        "https://guard.example/%2fsecret",
        "https://guard.example/%2e%2e",
        "https://%65xample.com",
        "https://bad^host",
        "https://-bad.example",
        "https://bad..example",
        "https://[::1%25eth0]",
    ],
)
def test_unsafe_or_malformed_deployment_urls(url: Any) -> None:
    with pytest.raises(TrustGuardConfigurationError) as caught:
        TrustGuardConfig(api_key="PRIVATE", base_url=url)
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize("host", ["127.0.0.1", "127.1.2.3", "[::1]", "localhost"])
def test_http_requires_explicit_loopback_opt_in(host: str) -> None:
    url = f"http://{host}:8080"
    with pytest.raises(TrustGuardConfigurationError):
        TrustGuardConfig(api_key="key", base_url=url)
    assert TrustGuardConfig(api_key="key", base_url=url, allow_local_http=True).base_url == url


@pytest.mark.parametrize(
    "host",
    [
        "192.168.1.1",
        "0.0.0.0",
        "169.254.169.254",
        "example.com",
        "127.0.0.1.example.com",
        "2130706433",
        "127.1",
        "[::ffff:127.0.0.1]",
    ],
)
def test_loopback_opt_in_cannot_target_other_http_hosts(host: str) -> None:
    with pytest.raises(TrustGuardConfigurationError):
        TrustGuardConfig(api_key="key", base_url=f"http://{host}", allow_local_http=True)
