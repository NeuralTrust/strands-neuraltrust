"""Complete content selection and strict path-based projection reconstruction."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from strands_neuraltrust._content import (
    apply_model_transform,
    apply_tool_input_transform,
    apply_tool_result_transform,
    model_payload,
    tool_input_payload,
    tool_result_payload,
)
from strands_neuraltrust._projection import (
    apply_projection_transform,
    project_payload,
    validate_payload_transform,
)
from strands_neuraltrust.exceptions import TrustGuardTransformError, TrustGuardUnsupportedContentError


def spec() -> dict[str, Any]:
    return {
        "name": "echo",
        "description": "Frozen declaration containing repeated-value.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                    "flag": {"type": "boolean"},
                },
                "required": ["value", "count", "flag"],
                "additionalProperties": False,
            }
        },
    }


def use() -> dict[str, Any]:
    return {
        "name": "echo",
        "toolUseId": "call-1",
        "input": {"value": "repeated-value", "count": 3, "flag": True},
    }


def result() -> dict[str, Any]:
    return {
        "toolUseId": "call-1",
        "status": "success",
        "content": [
            {"text": "tool-result-text"},
            {"json": {"value": "repeated-value", "count": 3, "flag": True}},
        ],
    }


def messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"text": "historical-keyword"}]},
        {"role": "assistant", "content": [{"text": "repeated-value"}, {"toolUse": use()}]},
        {"role": "user", "content": [{"toolResult": result()}, {"text": "latest-benign"}]},
    ]


def payload() -> dict[str, Any]:
    return model_payload(messages(), [{"text": "system-one"}, {"text": "system-two"}], [spec()])


def plain(text: str = "original") -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}


def blocks(projection: dict[str, Any]) -> list[dict[str, str]]:
    return projection["messages"][0]["content"]  # type: ignore[no-any-return]


def index_of(projection: dict[str, Any], text: str, occurrence: int = 0) -> int:
    return [index for index, block in enumerate(blocks(projection)) if block["text"] == text][occurrence]


@pytest.mark.parametrize("direction,role", [("input", "user"), ("output", "assistant")])
def test_all_content_visible_in_one_correctly_scoped_message(direction: Any, role: str) -> None:
    original = payload()
    snapshot = copy.deepcopy(original)
    projection = project_payload(original, direction)
    assert list(projection) == ["messages"] and len(projection["messages"]) == 1
    assert projection["messages"][0]["role"] == role
    units = blocks(projection)
    assert all(set(unit) == {"type", "text"} and unit["type"] == "text" for unit in units)
    assert [unit["text"] for unit in units[1:3]] == ["system-one", "system-two"]
    assert index_of(projection, "description") < index_of(projection, spec()["description"])
    assert index_of(projection, "echo") < index_of(projection, "historical-keyword")
    texts = [unit["text"] for unit in units]
    history_index = index_of(projection, "historical-keyword")
    assert texts[history_index:] == [
        "historical-keyword",
        "repeated-value",
        "count",
        "3",
        "flag",
        "true",
        "value",
        "repeated-value",
        "tool-result-text",
        "count",
        "3",
        "flag",
        "true",
        "value",
        "repeated-value",
        "latest-benign",
    ]
    assert original == snapshot
    units[4]["text"] = "mutated independent projection"
    assert original == snapshot


@pytest.mark.parametrize("direction", ["input", "output"])
def test_cursor_envelope_cannot_hide_earlier_protected_units(direction: Any) -> None:
    cursor = "<user_info>synthetic</user_info><user_query>benign</user_query>"
    original = model_payload(
        [
            {"role": "user", "content": [{"text": "historical-keyword"}]},
            {"role": "user", "content": [{"text": cursor}]},
        ]
    )
    projection = project_payload(original, direction)
    combined = " ".join(block["text"] for block in blocks(projection)).strip()
    assert not combined.startswith("<user_info>")
    assert "historical-keyword" in combined and cursor in combined
    assert len(projection["messages"]) == 1
    assert blocks(projection)[0] == blocks(project_payload(plain("other"), direction))[0]


def test_projection_reconstruction_targets_paths_not_equal_strings() -> None:
    original = payload()
    before = copy.deepcopy(original)
    projection = project_payload(original, "input")
    transformed = copy.deepcopy(projection)
    blocks(transformed)[index_of(projection, "repeated-value", 0)]["text"] = "changed assistant only"
    blocks(transformed)[index_of(projection, "repeated-value", 1)]["text"] = "changed argument only"
    blocks(transformed)[index_of(projection, "repeated-value", 2)]["text"] = "changed result only"
    restored = apply_projection_transform(original, projection, transformed)
    final = apply_model_transform(messages(), original, restored)
    assert final[1]["content"][0] == {"text": "changed assistant only"}
    assert final[1]["content"][1]["toolUse"]["input"] == {**use()["input"], "value": "changed argument only"}
    assert final[2]["content"][0]["toolResult"]["content"][1]["json"] == {
        "value": "changed result only",
        "count": 3,
        "flag": True,
    }
    assert restored["tools"] == before["tools"] and restored["system"] == before["system"]
    assert original == before and projection != transformed
    restored["tools"][0]["input_schema"].clear()
    assert original == before


def test_canonical_transform_returns_copies_and_preserves_json_number_category() -> None:
    original = payload()
    changed = copy.deepcopy(original)
    changed["messages"][1]["content"][1]["input"]["value"] = "safe"
    changed["messages"][1]["content"][1]["input"]["count"] = 3.0
    accepted = validate_payload_transform(original, changed)
    assert accepted == changed and accepted is not changed
    accepted["messages"][1]["content"][1]["input"]["value"] = "separate"
    assert changed["messages"][1]["content"][1]["input"]["value"] == "safe"
    assert original["messages"][1]["content"][1]["input"]["value"] == "repeated-value"


def test_prefix_and_every_system_tool_metadata_unit_are_immutable() -> None:
    original = payload()
    projection = project_payload(original, "input")
    for index in range(index_of(projection, "historical-keyword")):
        changed = copy.deepcopy(projection)
        blocks(changed)[index]["text"] += " modified"
        with pytest.raises(TrustGuardTransformError):
            apply_projection_transform(original, projection, changed)


@pytest.mark.parametrize("field", ["system", "tools"])
def test_canonical_metadata_cannot_transform(field: str) -> None:
    original = payload()
    changed = copy.deepcopy(original)
    changed[field] = []
    with pytest.raises(TrustGuardTransformError):
        validate_payload_transform(original, changed)


@pytest.mark.parametrize(
    "fault",
    [
        "top-key",
        "role",
        "message-count",
        "block-count",
        "block-type",
        "tool-id",
        "tool-name",
        "result-id",
        "result-status",
        "result-block-count",
        "result-block-type",
        "result-text-type",
        "argument-keys",
        "argument-bool-as-number",
        "argument-number-as-bool",
        "schema-violation",
    ],
)
def test_canonical_transform_rejects_structural_identity_and_schema_corruption(fault: str) -> None:
    original = payload()
    changed = copy.deepcopy(original)
    chat = changed["messages"]
    arguments = chat[1]["content"][1]["input"]
    tool_result = chat[2]["content"][0]
    if fault == "top-key":
        changed["extra"] = "ignored?"
    elif fault == "role":
        chat[0]["role"] = "assistant"
    elif fault == "message-count":
        chat.pop()
    elif fault == "block-count":
        chat[1]["content"].pop(0)
    elif fault == "block-type":
        chat[0]["content"][0]["type"] = "image"
    elif fault == "tool-id":
        chat[1]["content"][1]["id"] = "changed"
    elif fault == "tool-name":
        chat[1]["content"][1]["name"] = "changed"
    elif fault == "result-id":
        tool_result["tool_use_id"] = "changed"
    elif fault == "result-status":
        tool_result["is_error"] = 0
    elif fault == "result-block-count":
        tool_result["content"].pop()
    elif fault == "result-block-type":
        tool_result["content"][0]["type"] = "json"
    elif fault == "result-text-type":
        tool_result["content"][0]["text"] = False
    elif fault == "argument-keys":
        arguments["extra"] = "changed"
    elif fault == "argument-bool-as-number":
        arguments["flag"] = 1
    elif fault == "argument-number-as-bool":
        arguments["count"] = True
    elif fault == "schema-violation":
        arguments["count"] = -1
    with pytest.raises(TrustGuardTransformError):
        validate_payload_transform(original, changed)


@pytest.mark.parametrize(
    "fault",
    [
        "top-key",
        "role",
        "extra-message",
        "missing-message",
        "extra-block",
        "missing-block",
        "extra-block-key",
        "block-type",
        "text-type",
        "empty",
        "not-object",
    ],
)
def test_projection_transform_requires_complete_exact_block_structure(fault: str) -> None:
    original = plain()
    projection = project_payload(original, "input")
    changed: Any = copy.deepcopy(projection)
    if fault == "top-key":
        changed["extra"] = True
    elif fault == "role":
        changed["messages"][0]["role"] = "assistant"
    elif fault == "extra-message":
        changed["messages"].append(copy.deepcopy(changed["messages"][0]))
    elif fault == "missing-message":
        changed["messages"].clear()
    elif fault == "extra-block":
        blocks(changed).append({"type": "text", "text": "extra"})
    elif fault == "missing-block":
        blocks(changed).pop()
    elif fault == "extra-block-key":
        blocks(changed)[1]["path"] = "redirect"
    elif fault == "block-type":
        blocks(changed)[1]["type"] = "tool_use"
    elif fault == "text-type":
        blocks(changed)[1]["text"] = None
    elif fault == "empty":
        changed = {}
    elif fault == "not-object":
        changed = []
    with pytest.raises(TrustGuardTransformError):
        apply_projection_transform(original, projection, changed)


@pytest.mark.parametrize("wrong_projection", [None, [], {}, {"messages": []}, plain("unbound")])
def test_projection_must_be_bound_to_the_actual_source(wrong_projection: Any) -> None:
    original = plain()
    correct = project_payload(original, "input")
    with pytest.raises(TrustGuardTransformError):
        apply_projection_transform(original, wrong_projection, correct)
    different = plain("different original")
    with pytest.raises(TrustGuardTransformError):
        apply_projection_transform(different, correct, correct)


@pytest.mark.parametrize(
    "json_text",
    [
        '{"value":"safe","value":"duplicate","count":3,"flag":true}',
        '{"value":"safe","count":NaN,"flag":true}',
        '{"value":"safe","count":1e9999,"flag":true}',
        '{"value":"safe","count":3,"flag":true} trailing',
        '{"value":"safe","count":3}',
        '{"value":null,"count":3,"flag":true}',
        '["safe",3,true]',
        '"safe"',
        '{"value":"safe","count":false,"flag":true}',
        '{"value":"safe","count":3,"flag":1}',
        '{"value":"safe","count":0,"flag":true}',
    ],
)
def test_projected_arguments_require_strict_json_structure_and_schema(json_text: str) -> None:
    original = tool_input_payload(use(), spec())
    projection = project_payload(original, "input")
    changed = copy.deepcopy(projection)
    blocks(changed)[index_of(projection, "3")]["text"] = json_text
    with pytest.raises(TrustGuardTransformError):
        apply_projection_transform(original, projection, changed)


def test_json_argument_key_order_is_canonical_but_original_order_is_restored() -> None:
    original_use = use()
    original = tool_input_payload(original_use, spec())
    projection = project_payload(original, "input")
    assert [block["text"] for block in blocks(projection)[-6:]] == [
        "count",
        "3",
        "flag",
        "true",
        "value",
        "repeated-value",
    ]
    changed = copy.deepcopy(projection)
    blocks(changed)[index_of(projection, "repeated-value")]["text"] = "safe"
    blocks(changed)[index_of(projection, "3")]["text"] = "4"
    restored = apply_projection_transform(original, projection, changed)
    final = apply_tool_input_transform(original_use, original, restored, spec())
    assert list(final["input"]) == ["value", "count", "flag"]
    assert final["input"] == {"value": "safe", "count": 4, "flag": True}


def test_originally_typed_json_result_keeps_its_structure_during_projection() -> None:
    original_result = result()
    original = tool_result_payload(original_result)
    projection = project_payload(original, "input")
    changed = copy.deepcopy(projection)
    blocks(changed)[index_of(projection, "repeated-value")]["text"] = "safe"
    restored = apply_projection_transform(original, projection, changed)
    final = apply_tool_result_transform(original_result, original, restored)
    assert final["content"][1]["json"] == {"value": "safe", "count": 3, "flag": True}
    assert original_result == result()


def test_external_tool_schema_is_required_at_final_model_stage() -> None:
    original = model_payload([{"role": "assistant", "content": [{"toolUse": use()}]}])
    projection = project_payload(original, "output")
    changed = copy.deepcopy(projection)
    blocks(changed)[index_of(projection, "repeated-value")]["text"] = "safe"
    restored = apply_projection_transform(original, projection, changed)
    with pytest.raises(TrustGuardTransformError):
        apply_model_transform([{"role": "assistant", "content": [{"toolUse": use()}]}], original, restored)
    final = apply_model_transform(
        [{"role": "assistant", "content": [{"toolUse": use()}]}], original, restored, tool_specs=[spec()]
    )
    assert final[0]["content"][0]["toolUse"]["input"]["value"] == "safe"


@pytest.mark.parametrize(
    "original_text",
    [
        ' { "key" : "forbidden", "count": 3.0, "flag": true, "none": null } \n',
        '[ "forbidden", {"nested":"second"} ]',
        '"forbidden"',
        '"{\\"nested\\":\\"forbidden\\"}"',
        '{"unicode":"\\u0066orbidden", "slash":"a\\/b"}',
        "{}",
        "[]",
        '""',
    ],
)
def test_json_text_exposes_raw_leaves_and_preserves_spelling_on_no_change(original_text: str) -> None:
    original = plain(original_text)
    projected = project_payload(original, "input")
    assert apply_projection_transform(original, projected, projected) == original
    if "forbidden" in original_text or "u0066orbidden" in original_text:
        position = index_of(projected, "forbidden")
        changed = copy.deepcopy(projected)
        blocks(changed)[position]["text"] = "REDACTED"
        restored = apply_projection_transform(original, projected, changed)
        text = restored["messages"][0]["content"][0]["text"]
        assert "forbidden" not in text and "REDACTED" in text
        assert json.loads(text) is not None


@pytest.mark.parametrize("original_text", ["1234567890", "true", "false", "null", "1e3", '"1234567890"'])
def test_scalar_looking_strings_can_receive_string_redactions(original_text: str) -> None:
    original = plain(original_text)
    projected = project_payload(original, "input")
    changed = copy.deepcopy(projected)
    blocks(changed)[1]["text"] = "MASKED"
    restored = apply_projection_transform(original, projected, changed)
    expected = '"MASKED"' if original_text.startswith('"') else "MASKED"
    assert restored["messages"][0]["content"][0]["text"] == expected


@pytest.mark.parametrize(
    "value,replacement", [(3, "false"), (True, "1"), (None, '"changed"'), ([], "[1]"), ({}, '{"a":1}')]
)
def test_actual_json_scalars_and_empty_containers_keep_categories(value: Any, replacement: str) -> None:
    original = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "id", "name": "tool", "input": value}],
            }
        ]
    }
    projected = project_payload(original, "input")
    changed = copy.deepcopy(projected)
    blocks(changed)[1]["text"] = replacement
    with pytest.raises(TrustGuardTransformError):
        apply_projection_transform(original, projected, changed)


def test_json_keys_are_immutable_plain_units_in_arguments_and_encoded_text() -> None:
    for original in (
        plain('{"forbidden":"benign"}'),
        tool_input_payload({"name": "tool", "toolUseId": "id", "input": {"forbidden": "benign"}}),
    ):
        projected = project_payload(original, "input")
        changed = copy.deepcopy(projected)
        blocks(changed)[index_of(projected, "forbidden")]["text"] = "changed key"
        with pytest.raises(TrustGuardTransformError):
            apply_projection_transform(original, projected, changed)


@pytest.mark.parametrize(
    "encoded",
    [
        '{"key":"first","key":"second"}',
        '{"key":NaN}',
        '{"key":1e9999}',
        "[" * 64 + '"forbidden"' + "]" * 64,
        "[" * 70 + '"forbidden"' + "]" * 70,
        "[" * 2000 + '"forbidden"' + "]" * 2000,
        "[" + "0," * 100_000 + "0]",
    ],
)
def test_unsupported_encoded_json_does_not_fall_back_to_opaque_text(encoded: str) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        project_payload(plain(encoded), "input")


def test_non_json_prose_and_partial_code_remain_plain_text() -> None:
    for text in ('{"unfinished":', "[a code sample]", '"unclosed', "NaN", "ordinary prose"):
        projected = project_payload(plain(text), "input")
        assert blocks(projected)[1]["text"] == text


@pytest.mark.parametrize(
    "fault", ["duplicate-tool", "non-object-block", "result-status", "result-block-type", "projection-role"]
)
def test_additional_shape_guards_are_exercised(fault: str) -> None:
    original = payload()
    if fault == "projection-role":
        projected = project_payload(original, "input")
        projected["messages"][0]["role"] = "system"
        with pytest.raises(TrustGuardTransformError):
            apply_projection_transform(original, projected, projected)
        return
    if fault == "duplicate-tool":
        original["tools"].append(copy.deepcopy(original["tools"][0]))
    elif fault == "non-object-block":
        original["messages"][0]["content"][0] = []
    elif fault == "result-status":
        original["messages"][2]["content"][0]["is_error"] = 0
    elif fault == "result-block-type":
        original["messages"][2]["content"][0]["content"][0]["type"] = "json"
    with pytest.raises(TrustGuardUnsupportedContentError):
        project_payload(original, "input")


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        {},
        {"messages": ()},
        {"messages": [], "unknown": True},
        {"messages": [{"role": "system", "content": []}]},
        {"messages": [{"role": "user", "content": [{"type": "image", "data": "x"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": 1}]}]},
        {"messages": [], "system": [{"type": "thinking", "text": "x"}]},
        {"messages": [], "system": None},
        {
            "messages": [],
            "tools": [{"name": "x", "description": "x", "input_schema": {"$ref": "https://invalid"}}],
        },
        {"messages": [], "system": "\ud800"},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": float("nan")}]}]},
    ],
)
def test_invalid_source_content_fails_closed(invalid: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        project_payload(invalid, "input")
    with pytest.raises(TrustGuardTransformError):
        validate_payload_transform(invalid, invalid)


@pytest.mark.parametrize("direction", [None, True, 1, "", "INPUT", "user", []])
def test_projection_direction_is_exact(direction: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        project_payload(plain(), direction)


def test_system_string_empty_messages_and_empty_blocks_are_stable() -> None:
    for original in (
        {"messages": []},
        {"messages": [], "system": ""},
        {"messages": [], "system": [], "tools": []},
    ):
        projected = project_payload(original, "input")
        assert len(projected["messages"]) == 1 and blocks(projected)
        assert apply_projection_transform(original, projected, projected) == original


def test_deep_cyclic_and_arbitrary_values_never_recurse_or_stringify_unboundedly() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)
    nested: Any = "leaf"
    for _ in range(70):
        nested = [nested]

    class NeverStringify:
        def __str__(self) -> str:
            raise AssertionError("arbitrary string conversion")

    for value in (cycle, nested, NeverStringify(), float("inf"), 10**5000):
        original = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "id", "name": "tool", "input": value}],
                }
            ]
        }
        with pytest.raises(TrustGuardUnsupportedContentError):
            project_payload(original, "input")


TEXT = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=50)
JSON = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**15), max_value=10**15)
    | st.floats(allow_nan=False, allow_infinity=False)
    | TEXT,
    lambda children: st.lists(children, max_size=4) | st.dictionaries(TEXT, children, max_size=4),
    max_leaves=25,
)


@given(JSON)
@settings(max_examples=200)
def test_arbitrary_nested_json_roundtrips_without_type_or_key_changes(value: Any) -> None:
    original = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "id", "name": "tool", "input": value}],
            }
        ]
    }
    projected = project_payload(original, "input")
    restored = apply_projection_transform(original, projected, projected)
    assert type(restored["messages"][0]["content"][0]["input"]) is type(value)
    assert restored == original and restored is not original


@given(TEXT, TEXT)
@settings(max_examples=150)
def test_unicode_escapes_and_duplicate_values_transform_only_selected_path(before: str, after: str) -> None:
    try:
        json.loads(before)
    except ValueError:
        pass
    else:
        assume(False)
    original = model_payload(
        [
            {"role": "user", "content": [{"text": before}, {"text": before}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "name": "tool",
                            "toolUseId": "id",
                            "input": {"first": before, "second": before},
                        }
                    }
                ],
            },
        ]
    )
    projected = project_payload(original, "input")
    changed = copy.deepcopy(projected)
    blocks(changed)[1]["text"] = after
    blocks(changed)[4]["text"] = after
    restored = apply_projection_transform(original, projected, changed)
    assert [block["text"] for block in restored["messages"][0]["content"]] == [after, before]
    assert restored["messages"][1]["content"][0]["input"] == {"first": after, "second": before}
    assert [block["text"] for block in original["messages"][0]["content"]] == [before, before]
