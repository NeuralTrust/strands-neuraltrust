"""Behavioral contracts for reversible, fail-closed content normalization."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strands_neuraltrust._content import (
    apply_invocation_transform,
    apply_model_transform,
    apply_tool_input_transform,
    apply_tool_result_transform,
    invocation_payload,
    model_payload,
    tool_input_payload,
    tool_result_payload,
)
from strands_neuraltrust.exceptions import TrustGuardTransformError, TrustGuardUnsupportedContentError


def tool_spec() -> dict[str, Any]:
    return {
        "name": "search",
        "description": "Search safe synthetic documents.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
                "required": ["query", "limit"],
                "additionalProperties": False,
            }
        },
    }


def tool_use() -> dict[str, Any]:
    return {"name": "search", "toolUseId": "call-1", "input": {"query": "private", "limit": 2}}


def tool_result() -> dict[str, Any]:
    return {
        "toolUseId": "call-1",
        "status": "success",
        "content": [{"text": "private"}, {"json": {"hits": [{"title": "private", "rank": 1}]}}],
    }


def conversation() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"text": "private"}, {"text": "second"}],
            "tracking_id": "tracking-1",
            "metadata": {"custom": {"not-provider-content": ["retained"]}},
        },
        {"role": "assistant", "content": [{"text": "Searching"}, {"toolUse": tool_use()}]},
        {"role": "user", "content": [{"toolResult": tool_result()}, {"text": "continuation"}]},
    ]


def test_normalization_uses_documented_anthropic_payload_and_all_tool_metadata() -> None:
    messages = conversation()
    spec = tool_spec()
    spec["annotations"] = {"readOnlyHint": True}
    original = copy.deepcopy(messages)
    payload = model_payload(messages, [{"text": "system"}, {"text": "second system"}], [spec])
    assert payload == {
        "system": [{"type": "text", "text": "system"}, {"type": "text", "text": "second system"}],
        "tools": [
            {
                "name": "search",
                "description": spec["description"],
                "input_schema": spec["inputSchema"]["json"],
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "private"}, {"type": "text", "text": "second"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Searching"},
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "search",
                        "input": {"query": "private", "limit": 2},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "is_error": False,
                        "content": [
                            {"type": "text", "text": "private"},
                            {"type": "text", "text": '{"hits":[{"title":"private","rank":1}]}'},
                        ],
                    },
                    {"type": "text", "text": "continuation"},
                ],
            },
        ],
    }
    assert messages == original
    payload["messages"][1]["content"][1]["input"]["query"] = "changed"
    payload["tools"][0]["input_schema"]["required"].clear()
    assert messages == original
    assert spec["inputSchema"]["json"]["required"] == ["query", "limit"]


def test_stage_helpers_and_system_string() -> None:
    assert invocation_payload([]) == {"messages": []}
    assert model_payload([], "") == {"messages": [], "system": ""}
    assert tool_input_payload(tool_use())["messages"][0]["role"] == "assistant"
    result = tool_result()
    result["status"] = "error"
    assert tool_result_payload(result)["messages"][0]["content"][0]["is_error"] is True


def test_transform_updates_each_typed_surface_without_mutating_originals() -> None:
    original = conversation()
    original_snapshot = copy.deepcopy(original)
    payload = model_payload(original, "frozen system", [tool_spec()])
    replacement = copy.deepcopy(payload)
    replacement["messages"][0]["content"][0]["text"] = "redacted user"
    replacement["messages"][1]["content"][0]["text"] = "redacted assistant"
    replacement["messages"][1]["content"][1]["input"]["query"] = "redacted query"
    result_blocks = replacement["messages"][2]["content"][0]["content"]
    result_blocks[0]["text"] = "redacted tool text"
    result_blocks[1]["text"] = '{"hits":[{"title":"redacted title","rank":2}]}'
    transformed = apply_model_transform(original, payload, replacement)
    assert transformed[0]["content"][0] == {"text": "redacted user"}
    assert transformed[1]["content"][0] == {"text": "redacted assistant"}
    assert transformed[1]["content"][1]["toolUse"]["input"]["query"] == "redacted query"
    assert transformed[2]["content"][0]["toolResult"]["content"] == [
        {"text": "redacted tool text"},
        {"json": {"hits": [{"title": "redacted title", "rank": 2}]}},
    ]
    assert transformed[0]["metadata"] == original[0]["metadata"]
    assert transformed[0]["tracking_id"] == "tracking-1"
    assert original == original_snapshot
    transformed[0]["metadata"]["custom"]["not-provider-content"].append("new")
    assert original == original_snapshot
    assert payload != replacement


def test_individual_transforms_and_explicit_tool_specs() -> None:
    use = tool_use()
    payload = tool_input_payload(use)
    replacement = copy.deepcopy(payload)
    replacement["messages"][0]["content"][0]["input"]["query"] = "safe"
    transformed = apply_tool_input_transform(use, payload, replacement, tool_spec())
    assert transformed == {**use, "input": {"query": "safe", "limit": 2}}
    result = tool_result()
    payload = tool_result_payload(result)
    replacement = copy.deepcopy(payload)
    replacement["messages"][0]["content"][0]["content"][0]["text"] = "safe"
    assert apply_tool_result_transform(result, payload, replacement)["content"][0] == {"text": "safe"}
    user = [{"role": "user", "content": [{"text": "private"}]}]
    payload = invocation_payload(user)
    replacement = copy.deepcopy(payload)
    replacement["messages"][0]["content"][0]["text"] = "safe"
    assert apply_invocation_transform(user, payload, replacement) == [
        {"role": "user", "content": [{"text": "safe"}]}
    ]


@pytest.mark.parametrize(
    "message",
    [
        None,
        "text",
        {},
        {"role": "system", "content": []},
        {"role": True, "content": []},
        {"role": "user", "content": "text"},
        {"role": "user", "content": [{"text": 42}]},
        {"role": "user", "content": [{}]},
        {"role": "user", "content": [{"text": "x", "image": {}}]},
        {"role": "user", "content": [{"toolUse": tool_use()}]},
        {"role": "assistant", "content": [{"toolResult": tool_result()}]},
        {"role": "user", "content": [], "unknown": "private"},
        {"role": "user", "content": [], "tracking_id": 1},
        {"role": "user", "content": [], "metadata": []},
    ],
)
def test_rejects_malformed_messages(message: Any) -> None:
    with pytest.raises(
        TrustGuardUnsupportedContentError, match="^Unsupported or malformed Strands content\\.$"
    ):
        model_payload([message])


@pytest.mark.parametrize(
    "kind",
    [
        "image",
        "audio",
        "video",
        "document",
        "cachePoint",
        "reasoningContent",
        "guardContent",
        "citationsContent",
    ],
)
def test_rejects_every_unqualified_content_kind(kind: str) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        invocation_payload([{"role": "user", "content": [{kind: {}}]}])


@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf, b"bytes", {1: "value"}, ("tuple",), {"set"}, object()]
)
def test_json_requires_exact_finite_json_values(value: Any) -> None:
    result = tool_result()
    result["content"] = [{"json": value}]
    with pytest.raises(TrustGuardUnsupportedContentError):
        tool_result_payload(result)


def test_never_stringifies_arbitrary_objects_or_accepts_subclasses() -> None:
    class Unsafe:
        def __str__(self) -> str:
            pytest.fail("The adapter must not stringify arbitrary objects")

    class PretendText(str):
        pass

    for value in (Unsafe(), PretendText("text")):
        with pytest.raises(TrustGuardUnsupportedContentError):
            tool_input_payload({**tool_use(), "input": value})


@pytest.mark.parametrize("kind", ["cycle", "deep", "large"])
def test_content_resource_limits_raise_sanitized_typed_error(kind: str) -> None:
    data: list[Any] = []
    if kind == "cycle":
        data.append(data)
    elif kind == "deep":
        for _ in range(70):
            data = [data]
    else:
        data = [0] * 100_001
    with pytest.raises(TrustGuardUnsupportedContentError):
        tool_input_payload({**tool_use(), "input": data})


@pytest.mark.parametrize("kind", ["missing", "unknown", "signature", "bad-id", "bad-name"])
def test_tool_identity_and_keys_are_validated(kind: str) -> None:
    use = tool_use()
    if kind == "missing":
        del use["input"]
    elif kind == "unknown":
        use["unknown"] = "private"
    elif kind == "signature":
        use["reasoningSignature"] = "signature"
    elif kind == "bad-id":
        use["toolUseId"] = ""
    else:
        use["name"] = False
    with pytest.raises(TrustGuardUnsupportedContentError):
        tool_input_payload(use)


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "pending"},
        {"status": False},
        {"content": [{}]},
        {"content": [{"text": "x", "json": {}}]},
        {"content": [{"image": {}}]},
        {"content": [{"text": 4}]},
        {"toolUseId": None},
    ],
)
def test_tool_results_are_strict(patch: dict[str, Any]) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        tool_result_payload({**tool_result(), **patch})


@pytest.mark.parametrize(
    "system", [1, {}, [{"cachePoint": {"type": "default"}}], [{"text": "x", "extra": 1}], [{"text": None}]]
)
def test_unsupported_system_configuration_is_rejected(system: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        model_payload([], system)


@pytest.mark.parametrize(
    "schema",
    [
        True,
        [],
        {"type": "not-a-type"},
        {"$ref": "https://invalid.example/schema"},
        {"$ref": "#/$defs/local"},
        {"$dynamicRef": "#root"},
        {"properties": {"x": {"$recursiveRef": "#"}}},
        {"$schema": "http://json-schema.org/draft-07/schema#"},
    ],
)
def test_unqualified_or_invalid_schema_is_rejected_without_resolution(schema: Any) -> None:
    spec = tool_spec()
    spec["inputSchema"]["json"] = schema
    with pytest.raises(TrustGuardUnsupportedContentError):
        model_payload([], tool_specs=[spec])


def test_duplicate_tools_and_unsupported_output_schema_are_rejected() -> None:
    for specs in (
        [tool_spec(), tool_spec()],
        [{**tool_spec(), "outputSchema": {"json": {"type": "object"}}}],
    ):
        with pytest.raises(TrustGuardUnsupportedContentError):
            model_payload([], tool_specs=specs)


@pytest.mark.parametrize(
    "mutation",
    [
        "messages-add",
        "messages-remove",
        "role",
        "block-add",
        "block-remove",
        "block-type",
        "tool-id",
        "tool-name",
        "result-id",
        "result-status",
        "result-block-add",
        "result-block-type",
        "system",
        "tools",
        "extra-key",
        "drop-key",
    ],
)
def test_transforms_cannot_change_structure_or_immutable_metadata(mutation: str) -> None:
    original = conversation()
    payload = model_payload(original, "frozen", [tool_spec()])
    changed = copy.deepcopy(payload)
    messages = changed["messages"]
    if mutation == "messages-add":
        messages.append(copy.deepcopy(messages[0]))
    elif mutation == "messages-remove":
        messages.pop()
    elif mutation == "role":
        messages[0]["role"] = "assistant"
    elif mutation == "block-add":
        messages[0]["content"].append({"type": "text", "text": "extra"})
    elif mutation == "block-remove":
        messages[0]["content"].pop()
    elif mutation == "block-type":
        messages[0]["content"][0]["type"] = "image"
    elif mutation == "tool-id":
        messages[1]["content"][1]["id"] = "other"
    elif mutation == "tool-name":
        messages[1]["content"][1]["name"] = "other"
    elif mutation == "result-id":
        messages[2]["content"][0]["tool_use_id"] = "other"
    elif mutation == "result-status":
        messages[2]["content"][0]["is_error"] = 0  # False == 0 must not pass validation.
    elif mutation == "result-block-add":
        messages[2]["content"][0]["content"].append({"type": "text", "text": "extra"})
    elif mutation == "result-block-type":
        messages[2]["content"][0]["content"][0]["type"] = "image"
    elif mutation == "system":
        changed["system"] = "other"
    elif mutation == "tools":
        changed["tools"][0]["description"] = "other"
    elif mutation == "extra-key":
        changed["unknown"] = "private"
    else:
        del changed["messages"]
    with pytest.raises(
        TrustGuardTransformError, match="^TrustGuard returned an unsupported or malformed transformation\\.$"
    ):
        apply_model_transform(original, payload, changed)


@pytest.mark.parametrize(
    "replacement",
    [None, "redacted", {}, {"messages": "wrong"}, {"messages": [{"role": "user", "content": None}]}],
)
def test_missing_or_ambiguous_replacement_is_never_allow(replacement: Any) -> None:
    messages = [{"role": "user", "content": [{"text": "private"}]}]
    with pytest.raises(TrustGuardTransformError):
        apply_invocation_transform(messages, invocation_payload(messages), replacement)


def test_stale_original_and_mismatched_schema_are_rejected() -> None:
    messages = conversation()
    payload = model_payload(messages)
    messages[0]["content"][0]["text"] = "changed after evaluation"
    with pytest.raises(TrustGuardTransformError):
        apply_model_transform(messages, payload, payload)
    use = tool_use()
    payload = tool_input_payload(use, tool_spec())
    different_spec = tool_spec()
    different_spec["description"] = "changed after evaluation"
    with pytest.raises(TrustGuardTransformError):
        apply_tool_input_transform(use, payload, payload, different_spec)


@pytest.mark.parametrize(
    "new_input",
    [
        {"query": "safe", "limit": 0},
        {"query": 1, "limit": 2},
        {"query": "safe", "limit": True},
        {"query": "safe"},
        {"query": "safe", "limit": 2, "new": "value"},
        ["safe"],
        None,
    ],
)
def test_tool_input_replacements_must_preserve_structure_and_satisfy_schema(new_input: Any) -> None:
    use = tool_use()
    payload = tool_input_payload(use, tool_spec())
    replacement = copy.deepcopy(payload)
    replacement["messages"][0]["content"][0]["input"] = new_input
    with pytest.raises(TrustGuardTransformError):
        apply_tool_input_transform(use, payload, replacement)


def test_changed_tool_input_without_schema_fails_closed() -> None:
    use = tool_use()
    payload = tool_input_payload(use)
    changed = copy.deepcopy(payload)
    changed["messages"][0]["content"][0]["input"]["query"] = "safe"
    with pytest.raises(TrustGuardTransformError):
        apply_tool_input_transform(use, payload, changed)
    assert apply_tool_input_transform(use, payload, payload) == use


@pytest.mark.parametrize(
    "use",
    [
        {**tool_use(), "name": "different-tool"},
        {**tool_use(), "input": {"query": "safe", "limit": 0}},
        {**tool_use(), "input": {"query": "safe", "limit": True}},
        {**tool_use(), "input": {"query": "safe"}},
    ],
)
def test_original_tool_arguments_require_matching_declaration_and_valid_schema(use: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        tool_input_payload(use, tool_spec())


def test_schema_arithmetic_overflow_is_a_sanitized_content_failure() -> None:
    spec = {
        "name": "numeric",
        "description": "Validate a numeric tool input.",
        "inputSchema": {"json": {"type": "number", "multipleOf": 0.1}},
    }
    with pytest.raises(
        TrustGuardUnsupportedContentError, match="^Unsupported or malformed Strands content\\.$"
    ):
        tool_input_payload({"name": "numeric", "toolUseId": "call", "input": 10**1000}, spec)


@pytest.mark.parametrize(
    "encoded",
    [
        '{"hits":[{"title":"a","title":"b","rank":1}]}',
        '{"hits":[{"title":"x","rank":NaN}]}',
        '{"hits":[{"title":"x","rank":Infinity}]}',
        '{"hits":[]}',
        '{"hits":[{"title":"x","rank":true}]}',
        '{"hits":[{"title":"x","rank":1,"extra":"x"}]}',
        '"masked"',
        "{bad json",
    ],
)
def test_json_result_transforms_reject_corruption_and_ambiguous_json(encoded: str) -> None:
    result = tool_result()
    payload = tool_result_payload(result)
    replacement = copy.deepcopy(payload)
    replacement["messages"][0]["content"][0]["content"][1]["text"] = encoded
    with pytest.raises(TrustGuardTransformError):
        apply_tool_result_transform(result, payload, replacement)


json_scalar = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**63), max_value=2**63 - 1)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
json_values = st.recursive(
    json_scalar,
    lambda child: st.lists(child, max_size=5) | st.dictionaries(st.text(), child, max_size=5),
    max_leaves=30,
)


@given(json_values)
@settings(max_examples=200)
def test_arbitrary_finite_json_tool_result_round_trips_without_type_loss(value: Any) -> None:
    result = {"toolUseId": "call", "status": "success", "content": [{"json": value}]}
    payload = tool_result_payload(result)
    transformed = apply_tool_result_transform(result, payload, copy.deepcopy(payload))
    assert transformed == result
    assert json.dumps(transformed, sort_keys=True) == json.dumps(result, sort_keys=True)
    assert transformed is not result


@given(st.lists(st.text(), max_size=10))
def test_arbitrary_text_blocks_preserve_count_order_and_unicode(texts: list[str]) -> None:
    messages = [{"role": "user", "content": [{"text": text} for text in texts]}]
    payload = invocation_payload(messages)
    changed = copy.deepcopy(payload)
    for block in changed["messages"][0]["content"]:
        block["text"] = "safe:" + block["text"]
    transformed = apply_invocation_transform(messages, payload, changed)
    assert transformed == [{"role": "user", "content": [{"text": "safe:" + text} for text in texts]}]
    assert messages == [{"role": "user", "content": [{"text": text} for text in texts]}]
