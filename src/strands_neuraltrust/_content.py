"""Strict, reversible normalization of the supported Strands content subset.

The evaluator receives documented Anthropic-style LLM payloads, never raw
Bedrock/Strands content. One Strands message/block maps to one normalized
message/block, including the individual text/JSON blocks of tool results.
JSON results are encoded as JSON text because Anthropic has no JSON result
block. Their original block types are retained for reconstruction.

Transforms may replace text and JSON scalar values. Message/block ordering,
roles, tool identities, status, object keys, array lengths, scalar categories,
system instructions, and tool declarations are immutable. Message metadata and
tracking IDs are preserved locally and omitted from the evaluation payload,
matching Strands' provider boundary. Unsupported content fails closed.
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from strands.types.content import Messages
from strands.types.tools import ToolResult, ToolUse

from .exceptions import TrustGuardTransformError, TrustGuardUnsupportedContentError

_MAX_DEPTH = 64
_MAX_NODES = 100_000
_CONTENT_ERROR = "Unsupported or malformed Strands content."
_TRANSFORM_ERROR = "TrustGuard returned an unsupported or malformed transformation."


def _invalid() -> TrustGuardUnsupportedContentError:
    return TrustGuardUnsupportedContentError(_CONTENT_ERROR)


def _json_copy(value: Any) -> Any:
    """Validate and copy JSON without coercion, custom-object hooks, or cycles."""
    count = 0
    ancestors: set[int] = set()

    def visit(item: Any, depth: int) -> Any:
        nonlocal count
        count += 1
        if depth > _MAX_DEPTH or count > _MAX_NODES:
            raise _invalid()
        kind = type(item)
        if item is None or kind in (str, bool, int):
            return item
        if kind is float:
            if not math.isfinite(item):
                raise _invalid()
            return item
        if kind not in (dict, list) or id(item) in ancestors:
            raise _invalid()
        ancestors.add(id(item))
        try:
            if kind is list:
                return [visit(child, depth + 1) for child in item]
            if any(type(key) is not str for key in item):
                raise _invalid()
            return {key: visit(child, depth + 1) for key, child in item.items()}
        finally:
            ancestors.remove(id(item))

    return visit(value, 0)


def _object(value: Any, required: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if (
        type(value) is not dict
        or not required <= value.keys()
        or value.keys() - required - (optional or set())
    ):
        raise _invalid()
    return cast(dict[str, Any], value)


def _text(value: Any, *, nonempty: bool = False) -> str:
    if type(value) is not str or (nonempty and not value):
        raise _invalid()
    return value


def _list(value: Any) -> list[Any]:
    if type(value) is not list:
        raise _invalid()
    return value


def _schema(value: Any) -> dict[str, Any]:
    schema = _object(value, set(), set(value) if type(value) is dict else set())

    def inspect(item: Any) -> None:
        if type(item) is dict:
            # No automatic reference retrieval, including through newer drafts.
            # Reference-bearing schemas are outside the qualified initial scope.
            if any(key in item for key in ("$ref", "$dynamicRef", "$recursiveRef")):
                raise _invalid()
            for child in item.values():
                inspect(child)
        elif type(item) is list:
            for child in item:
                inspect(child)

    inspect(schema)
    declared = schema.get("$schema")
    if declared is not None and declared != "https://json-schema.org/draft/2020-12/schema":
        raise _invalid()
    try:
        Draft202012Validator.check_schema(schema)
    except (SchemaError, RecursionError, ValueError, TypeError):
        raise _invalid() from None
    return schema


def _tool_spec(value: Any) -> dict[str, Any]:
    spec = _object(value, {"name", "description", "inputSchema"}, {"annotations"})
    wrapped = _object(spec["inputSchema"], {"json"})
    return {
        "name": _text(spec["name"], nonempty=True),
        "description": _text(spec["description"]),
        "input_schema": _schema(wrapped["json"]),
    }


def _tool_use(value: Any) -> dict[str, Any]:
    use = _object(value, {"name", "toolUseId", "input"})
    return {
        "type": "tool_use",
        "id": _text(use["toolUseId"], nonempty=True),
        "name": _text(use["name"], nonempty=True),
        "input": use["input"],
    }


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, RecursionError):
        raise _invalid() from None


def _tool_result(value: Any) -> dict[str, Any]:
    result = _object(value, {"toolUseId", "status", "content"})
    if type(result["status"]) is not str or result["status"] not in ("success", "error"):
        raise _invalid()
    content = []
    for block in _list(result["content"]):
        block = _object(block, set(), {"text", "json"})
        if len(block) != 1:
            raise _invalid()
        text = _text(block["text"]) if "text" in block else _json_text(block["json"])
        content.append({"type": "text", "text": text})
    return {
        "type": "tool_result",
        "tool_use_id": _text(result["toolUseId"], nonempty=True),
        "is_error": result["status"] == "error",
        "content": content,
    }


def _messages(value: Any) -> list[dict[str, Any]]:
    messages = []
    for message in _list(value):
        message = _object(message, {"role", "content"}, {"metadata", "tracking_id"})
        role = message["role"]
        if type(role) is not str or role not in ("user", "assistant"):
            raise _invalid()
        if "tracking_id" in message:
            _text(message["tracking_id"], nonempty=True)
        if "metadata" in message and type(message["metadata"]) is not dict:
            raise _invalid()
        blocks = []
        for block in _list(message["content"]):
            block = _object(block, set(), {"text", "toolUse", "toolResult"})
            if len(block) != 1:
                raise _invalid()
            if "text" in block:
                blocks.append({"type": "text", "text": _text(block["text"])})
            elif "toolUse" in block and role == "assistant":
                blocks.append(_tool_use(block["toolUse"]))
            elif "toolResult" in block and role == "user":
                blocks.append(_tool_result(block["toolResult"]))
            else:
                raise _invalid()
        messages.append({"role": role, "content": blocks})
    return messages


def model_payload(messages: Any, system_prompt: Any = None, tool_specs: Any = None) -> dict[str, Any]:
    """Normalize text/tool messages, system text, and registered tool schemas.

    Only reference-free JSON Schema 2020-12 is qualified. Output schemas,
    reasoning signatures, cache points, and multimodal blocks are unsupported.
    """
    source = _json_copy({"messages": messages, "system": system_prompt, "tools": tool_specs})
    payload: dict[str, Any] = {"messages": _messages(source["messages"])}
    system = source["system"]
    if system is not None:
        if type(system) is str:
            payload["system"] = system
        else:
            payload["system"] = [
                {"type": "text", "text": _text(_object(block, {"text"})["text"])} for block in _list(system)
            ]
    if source["tools"] is not None:
        payload["tools"] = [_tool_spec(spec) for spec in _list(source["tools"])]
        names = [spec["name"] for spec in payload["tools"]]
        if len(names) != len(set(names)):
            raise _invalid()
    return payload


def invocation_payload(messages: Any) -> dict[str, Any]:
    """Normalize invocation messages using the same model-boundary contract."""
    return model_payload(messages)


def tool_input_payload(tool_use: Any, tool_spec: Any = None) -> dict[str, Any]:
    """Normalize one model tool request and, when known, its declaration."""
    payload = model_payload(
        [{"role": "assistant", "content": [{"toolUse": tool_use}]}],
        tool_specs=None if tool_spec is None else [tool_spec],
    )
    if tool_spec is not None:
        declaration = payload["tools"][0]
        use = payload["messages"][0]["content"][0]
        if use["name"] != declaration["name"]:
            raise _invalid()
        _validate_arguments(use["input"], declaration["input_schema"])
    return payload


def tool_result_payload(result: Any) -> dict[str, Any]:
    """Normalize one completed tool result, including all text/JSON blocks."""
    return model_payload([{"role": "user", "content": [{"toolResult": result}]}])


def _same(left: Any, right: Any) -> bool:
    """JSON equality that does not equate booleans with numeric values."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _json_transform(original: Any, changed: Any) -> Any:
    """Permit scalar substitutions while preserving JSON structure and types."""
    if type(original) is dict:
        if type(changed) is not dict or original.keys() != changed.keys():
            raise _invalid()
        return {key: _json_transform(original[key], changed[key]) for key in original}
    if type(original) is list:
        if type(changed) is not list or len(original) != len(changed):
            raise _invalid()
        return [_json_transform(a, b) for a, b in zip(original, changed, strict=True)]
    if type(original) is not type(changed):
        # JSON has one number category, but booleans must never enter it.
        if type(original) not in (int, float) or type(changed) not in (int, float):
            raise _invalid()
    return changed


def _load_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise _invalid()
            output[key] = value
        return output

    def reject_constant(_: str) -> Any:
        raise _invalid()

    try:
        return _json_copy(json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant))
    except (ValueError, TypeError, RecursionError):
        raise _invalid() from None


def _replace_result(original: dict[str, Any], changed: dict[str, Any]) -> None:
    for source, target in zip(original["content"], changed["content"], strict=True):
        if "text" in source:
            source["text"] = _text(target["text"])
        else:
            source["json"] = _json_transform(source["json"], _load_json(_text(target["text"])))


def _validate_arguments(value: Any, schema: dict[str, Any]) -> None:
    try:
        Draft202012Validator(schema).validate(value)
    except (ValidationError, ArithmeticError, RecursionError, ValueError, TypeError):
        raise _invalid() from None


def _replace_messages(source: list[Any], old: list[Any], new: list[Any], schemas: dict[str, Any]) -> None:
    if len(old) != len(new):
        raise _invalid()
    for message, previous, changed in zip(source, old, new, strict=True):
        changed = _object(changed, {"role", "content"})
        blocks = _list(changed["content"])
        if not _same(previous["role"], changed["role"]) or len(blocks) != len(previous["content"]):
            raise _invalid()
        for block, prior, replacement in zip(message["content"], previous["content"], blocks, strict=True):
            replacement = _object(replacement, set(prior))
            mutable = {"text"} if "text" in block else {"input"} if "toolUse" in block else {"content"}
            if any(not _same(value, replacement[key]) for key, value in prior.items() if key not in mutable):
                raise _invalid()
            if "text" in block:
                block["text"] = _text(replacement["text"])
            elif "toolUse" in block:
                use = block["toolUse"]
                value = _json_transform(use["input"], replacement["input"])
                if not _same(use["input"], value):
                    schema = schemas.get(use["name"])
                    if schema is None:
                        # A changed tool argument must be qualified against its declaration.
                        raise _invalid()
                    _validate_arguments(value, schema)
                use["input"] = value
            else:
                results = _list(replacement["content"])
                if len(results) != len(prior["content"]):
                    raise _invalid()
                for before, after in zip(prior["content"], results, strict=True):
                    after = _object(after, {"type", "text"})
                    if not _same(before["type"], after["type"]):
                        raise _invalid()
                    _text(after["text"])
                _replace_result(block["toolResult"], replacement)


def apply_model_transform(
    messages: Any, original_payload: Any, transformed_payload: Any, *, tool_specs: Any = None
) -> Messages:
    """Return new Strands messages after validating a complete replacement.

    A tool-input replacement needs a matching frozen declaration. System and
    tool metadata may be inspected but cannot be transformed in this release.
    """
    try:
        source = _json_copy(messages)
        original = _object(_json_copy(original_payload), {"messages"}, {"system", "tools"})
        changed = _object(_json_copy(transformed_payload), set(original))
        if not _same(_messages(source), original["messages"]):
            raise _invalid()
        for key in original.keys() - {"messages"}:
            if not _same(original[key], changed[key]):
                raise _invalid()
        schemas: dict[str, Any] = {}
        for declaration in _list(original.get("tools", [])):
            declaration = _object(declaration, {"name", "description", "input_schema"})
            name = _text(declaration["name"], nonempty=True)
            _text(declaration["description"])
            if name in schemas:
                raise _invalid()
            schemas[name] = _schema(declaration["input_schema"])
        if tool_specs is not None:
            expected_tools = model_payload([], tool_specs=tool_specs)["tools"]
            if "tools" in original and not _same(original["tools"], expected_tools):
                raise _invalid()
            schemas = {spec["name"]: spec["input_schema"] for spec in expected_tools}
        _replace_messages(source, _list(original["messages"]), _list(changed["messages"]), schemas)
        return cast(Messages, source)
    except (TrustGuardUnsupportedContentError, KeyError, ValueError, TypeError, RecursionError):
        raise TrustGuardTransformError(_TRANSFORM_ERROR) from None


def apply_invocation_transform(messages: Any, original_payload: Any, transformed_payload: Any) -> Messages:
    """Apply a complete invocation payload replacement to copied messages."""
    return apply_model_transform(messages, original_payload, transformed_payload)


def apply_tool_input_transform(
    tool_use: Any, original_payload: Any, transformed_payload: Any, tool_spec: Any = None
) -> ToolUse:
    """Apply a schema-checked tool-input transform without changing identity."""
    messages = apply_model_transform(
        [{"role": "assistant", "content": [{"toolUse": tool_use}]}],
        original_payload,
        transformed_payload,
        tool_specs=None if tool_spec is None else [tool_spec],
    )
    return messages[0]["content"][0]["toolUse"]


def apply_tool_result_transform(result: Any, original_payload: Any, transformed_payload: Any) -> ToolResult:
    """Apply a complete result replacement preserving text/JSON block types."""
    messages = apply_model_transform(
        [{"role": "user", "content": [{"toolResult": result}]}], original_payload, transformed_payload
    )
    return messages[0]["content"][0]["toolResult"]
