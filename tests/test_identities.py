"""Structural wire aliases protect routing without relaxing content validation."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strands_neuraltrust._content import (
    apply_model_transform,
    apply_tool_input_transform,
    model_payload,
    tool_input_payload,
)
from strands_neuraltrust._identities import bind_tool_identities
from strands_neuraltrust._projection import project_payload
from strands_neuraltrust.exceptions import TrustGuardTransformError, TrustGuardUnsupportedContentError


def use(identifier: str, value: Any = "original") -> dict[str, Any]:
    return {"type": "tool_use", "id": identifier, "name": "echo", "input": {"value": value}}


def result(identifier: str, text: str = "original result") -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": identifier,
        "is_error": False,
        "content": [{"type": "text", "text": text}],
    }


def payload(*identifiers: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "original question"}]},
            {"role": "assistant", "content": [use(identifier) for identifier in identifiers]},
            {"role": "user", "content": [result(identifier) for identifier in identifiers]},
        ]
    }


def uses(value: dict[str, Any]) -> list[dict[str, Any]]:
    return value["messages"][1]["content"]  # type: ignore[no-any-return]


def results(value: dict[str, Any]) -> list[dict[str, Any]]:
    return value["messages"][2]["content"]  # type: ignore[no-any-return]


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def test_same_logical_identity_retains_links_without_exposing_digit_ids() -> None:
    original = payload("tooluse_echo_612345678", "tooluse_echo_698765432", "tooluse_echo_612345678")
    binding = bind_tool_identities(original)
    wire = binding.payload
    assert [item["id"] for item in uses(wire)] == [
        "strands_tool_call_a",
        "strands_tool_call_b",
        "strands_tool_call_a",
    ]
    assert [item["tool_use_id"] for item in results(wire)] == [
        "strands_tool_call_a",
        "strands_tool_call_b",
        "strands_tool_call_a",
    ]
    assert b"612345678" not in encoded(wire) and b"698765432" not in encoded(wire)
    assert encoded(binding.restore(wire)) == encoded(original)


def test_alphabetic_aliases_remain_unique_for_many_calls_and_alias_looking_ids() -> None:
    identifiers = ["strands_tool_call_a", "original", "strands_tool_call_b"]
    identifiers += [f"sdk-{index}" for index in range(750)]
    binding = bind_tool_identities(payload(*identifiers))
    aliases = [item["id"] for item in uses(binding.payload)]
    assert len(set(aliases)) == len(identifiers)
    assert all(re.fullmatch("strands_tool_call_[a-z]+", alias) for alias in aliases)
    assert aliases[:3] == ["strands_tool_call_a", "strands_tool_call_b", "strands_tool_call_c"]
    assert aliases[25:28] == ["strands_tool_call_z", "strands_tool_call_aa", "strands_tool_call_ab"]
    assert aliases[701:704] == ["strands_tool_call_zz", "strands_tool_call_aaa", "strands_tool_call_aab"]
    assert binding.restore(binding.payload) == payload(*identifiers)


def test_result_before_use_and_result_only_are_bound_by_logical_identity() -> None:
    original = {
        "messages": [
            {"role": "user", "content": [result("later-use"), result("result-only")]},
            {"role": "assistant", "content": [use("later-use"), use("use-only")]},
        ]
    }
    binding = bind_tool_identities(original)
    wire = binding.payload
    assert wire["messages"][0]["content"][0]["tool_use_id"] == "strands_tool_call_a"
    assert wire["messages"][0]["content"][1]["tool_use_id"] == "strands_tool_call_b"
    assert wire["messages"][1]["content"][0]["id"] == "strands_tool_call_a"
    assert wire["messages"][1]["content"][1]["id"] == "strands_tool_call_c"
    assert binding.restore(wire) == original


def test_content_names_and_declarations_equal_to_ids_are_never_substituted() -> None:
    identifier = "tooluse_echo_612345678"
    original = payload(identifier)
    original["system"] = identifier
    original["tools"] = [{"name": identifier, "description": identifier, "input_schema": {}}]
    original["messages"][0]["content"][0]["text"] = identifier
    uses(original)[0]["name"] = identifier
    uses(original)[0]["input"] = {identifier: [identifier, {"id": identifier, "tool_use_id": identifier}]}
    results(original)[0]["content"][0]["text"] = identifier
    before = encoded(original)
    binding = bind_tool_identities(original)
    expected = copy.deepcopy(original)
    uses(expected)[0]["id"] = "strands_tool_call_a"
    results(expected)[0]["tool_use_id"] = "strands_tool_call_a"
    assert encoded(binding.payload) == encoded(expected)
    assert encoded(binding.restore(binding.payload)) == before
    assert encoded(original) == before


def test_bind_and_restore_never_share_mutable_state_with_callers() -> None:
    original = payload("sdk-123")
    snapshot = copy.deepcopy(original)
    binding = bind_tool_identities(original)
    original["messages"].clear()
    first_wire = binding.payload
    first_wire["messages"].clear()
    assert binding.restore(binding.payload) == snapshot
    changed = binding.payload
    uses(changed)[0]["input"]["value"] = "redacted"
    restored = binding.restore(changed)
    changed["messages"].clear()
    restored["messages"].clear()
    assert binding.restore(binding.payload) == snapshot


def test_repr_does_not_include_content_or_original_identities() -> None:
    binding = bind_tool_identities(payload("private-sdk-identifier"))
    assert repr(binding) == "ToolIdentityBinding()"


def test_plain_content_without_tool_ids_remains_defensively_transformable() -> None:
    original = {"messages": [{"role": "user", "content": [{"type": "text", "text": "original"}]}]}
    binding = bind_tool_identities(original)
    wire = binding.payload
    assert encoded(wire) == encoded(original)
    wire["messages"][0]["content"][0]["text"] = "changed"
    restored = binding.restore(wire)
    assert restored == wire
    restored["messages"].clear()
    assert binding.payload == original


@pytest.mark.parametrize("path", [(1, "id"), (2, "tool_use_id")])
@pytest.mark.parametrize("replacement", ["strands_tool_call_b", "sdk-123", "[MASKED_PHONE]", "", None, 123])
def test_changed_wire_alias_always_fails_even_if_another_valid_bound_alias(
    path: tuple[int, str], replacement: Any
) -> None:
    binding = bind_tool_identities(payload("sdk-123", "sdk-456"))
    changed = binding.payload
    message, key = path
    changed["messages"][message]["content"][0][key] = replacement
    with pytest.raises(TrustGuardTransformError, match="unsupported or malformed"):
        binding.restore(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["messages"].reverse(),
        lambda value: value["messages"].pop(),
        lambda value: value["messages"][0].update(role="assistant"),
        lambda value: uses(value).reverse(),
        lambda value: uses(value)[0].update(name="other"),
        lambda value: uses(value)[0].update(type="text"),
        lambda value: uses(value)[0].update(extra="unknown"),
        lambda value: uses(value)[0]["input"].update(another="new key"),
        lambda value: uses(value)[0]["input"].update(value=1),
        lambda value: results(value)[0].update(is_error=True),
        lambda value: results(value)[0]["content"].append({"type": "text", "text": "added"}),
        lambda value: value.update(system="added metadata"),
    ],
)
def test_structural_and_content_type_corruption_remains_rejected(mutation: Any) -> None:
    binding = bind_tool_identities(payload("sdk-a", "sdk-b"))
    changed = binding.payload
    mutation(changed)
    with pytest.raises(TrustGuardTransformError):
        binding.restore(changed)


def test_metadata_changes_and_schema_violating_transforms_still_fail() -> None:
    original = payload("sdk-123")
    original["system"] = [{"type": "text", "text": "immutable"}]
    original["tools"] = [
        {
            "name": "echo",
            "description": "immutable",
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": "^original$"}},
            },
        }
    ]
    binding = bind_tool_identities(original)
    candidates = []
    changed = binding.payload
    changed["system"][0]["text"] = "changed"
    candidates.append(changed)
    changed = binding.payload
    changed["tools"][0]["description"] = "changed"
    candidates.append(changed)
    changed = binding.payload
    changed["tools"][0]["input_schema"].clear()
    candidates.append(changed)
    changed = binding.payload
    uses(changed)[0]["input"]["value"] = "redacted"
    candidates.append(changed)
    for changed in candidates:
        with pytest.raises(TrustGuardTransformError):
            binding.restore(changed)


def test_direct_tool_redaction_restores_exact_sdk_id_before_stage_transform() -> None:
    tool_use = {
        "toolUseId": "tooluse_echo_612345678",
        "name": "echo",
        "input": {"value": "123-45-6789", "count": 3, "flag": True},
    }
    tool_spec = {
        "name": "echo",
        "description": "Echo synthetic input",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "count": {"type": "integer"},
                    "flag": {"type": "boolean"},
                },
            }
        },
    }
    original = tool_input_payload(tool_use, tool_spec)
    binding = bind_tool_identities(original)
    response = binding.payload
    block = response["messages"][0]["content"][0]
    assert block["id"] == "strands_tool_call_a"
    block["input"]["value"] = "[MASKED_SSN]"
    restored = binding.restore(response)
    final = apply_tool_input_transform(tool_use, original, restored, tool_spec=tool_spec)
    assert final == {**tool_use, "input": {**tool_use["input"], "value": "[MASKED_SSN]"}}
    assert type(final["input"]["count"]) is int and type(final["input"]["flag"]) is bool
    assert tool_use["input"]["value"] == "123-45-6789"


def test_nested_and_repeated_content_transforms_are_restored_by_path() -> None:
    original = payload("sdk-a", "sdk-b")
    uses(original)[0]["input"]["value"] = {"nested": ["repeat", 1, True, None]}
    uses(original)[1]["input"]["value"] = "repeat"
    binding = bind_tool_identities(original)
    changed = binding.payload
    changed["messages"][0]["content"][0]["text"] = "redacted question"
    uses(changed)[0]["input"]["value"]["nested"][0] = "first"
    uses(changed)[1]["input"]["value"] = "second"
    results(changed)[0]["content"][0]["text"] = "redacted first result"
    restored = binding.restore(changed)
    expected = copy.deepcopy(changed)
    for index, identifier in enumerate(("sdk-a", "sdk-b")):
        uses(expected)[index]["id"] = identifier
        results(expected)[index]["tool_use_id"] = identifier
    assert restored == expected
    assert uses(original)[0]["input"]["value"]["nested"] == ["repeat", 1, True, None]


def test_projection_excludes_structural_ids_before_and_after_binding() -> None:
    original = payload("sdk-612345678")
    wire = bind_tool_identities(original).payload
    assert project_payload(original, "input") == project_payload(wire, "input")
    assert b"strands_tool_call_a" not in encoded(project_payload(wire, "input"))
    assert b"sdk-612345678" not in encoded(project_payload(original, "input"))


def test_sdk_history_round_trip_preserves_tool_result_types_and_ids() -> None:
    original_messages = [
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "sdk-612345678", "name": "echo", "input": {"value": "original"}}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "sdk-612345678",
                        "status": "success",
                        "content": [{"json": {"value": "original", "count": 3, "flag": True}}],
                    }
                }
            ],
        },
    ]
    original = model_payload(original_messages)
    binding = bind_tool_identities(original)
    changed = binding.payload
    changed["messages"][1]["content"][0]["content"][0]["text"] = '{"value":"redacted","count":3,"flag":true}'
    final = apply_model_transform(original_messages, original, binding.restore(changed))
    assert final[0] == original_messages[0]
    final_result = final[1]["content"][0]["toolResult"]
    assert final_result["toolUseId"] == "sdk-612345678"
    assert final_result["content"] == [{"json": {"value": "redacted", "count": 3, "flag": True}}]


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        {},
        {"messages": "wrong"},
        {"messages": [], "extra": True},
        {"messages": [{"role": "assistant", "content": [use("")]}]},
        {"messages": [{"role": "user", "content": [use("wrong-role")]}]},
        {"messages": [{"role": "assistant", "content": [result("wrong-role")]}]},
        {"messages": [{"role": "assistant", "content": [use("invalid-surrogate-\ud800")]}]},
        {"messages": [{"role": "assistant", "content": [use("sdk", float("nan"))]}]},
        {"messages": [{"role": "assistant", "content": [use("sdk", object())]}]},
    ],
)
def test_malformed_original_is_rejected_with_sanitized_typed_error(invalid: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError, match="^Unsupported or malformed") as caught:
        bind_tool_identities(invalid)
    assert "surrogate" not in str(caught.value)


@pytest.mark.parametrize("invalid", [None, [], {}, {"messages": []}, float("nan"), object()])
def test_malformed_response_never_reconstructs_identities(invalid: Any) -> None:
    binding = bind_tool_identities(payload("private-sdk-id"))
    with pytest.raises(TrustGuardTransformError) as caught:
        binding.restore(invalid)
    assert "private-sdk-id" not in str(caught.value)


def test_cyclic_original_and_response_fail_closed() -> None:
    original = payload("sdk")
    uses(original)[0]["input"]["value"] = original
    with pytest.raises(TrustGuardUnsupportedContentError):
        bind_tool_identities(original)
    binding = bind_tool_identities(payload("sdk"))
    changed = binding.payload
    uses(changed)[0]["input"]["value"] = changed
    with pytest.raises(TrustGuardTransformError):
        binding.restore(changed)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=30),
        min_size=1,
        max_size=12,
    )
)
def test_arbitrary_unicode_identities_round_trip_exactly_and_preserve_equivalence(ids: list[str]) -> None:
    original = payload(*ids)
    before = encoded(original)
    binding = bind_tool_identities(original)
    wire = binding.payload
    aliases = [item["id"] for item in uses(wire)]
    assert len(set(aliases)) == len(set(ids))
    assert all(re.fullmatch("strands_tool_call_[a-z]+", alias) for alias in aliases)
    for left in range(len(ids)):
        for right in range(len(ids)):
            assert (ids[left] == ids[right]) == (aliases[left] == aliases[right])
    assert aliases == [item["tool_use_id"] for item in results(wire)]
    assert encoded(binding.restore(wire)) == before
    assert encoded(original) == before
