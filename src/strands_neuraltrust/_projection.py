"""Complete text assessment with reversible, immutable path-based mapping.

Structured messages remain the source of truth. This additional assessment puts
all supported content into one message so a latest-message or role-scoped text
detector sees history, tool arguments/results, and frozen declarations. A fixed
first block prevents protected content from impersonating a Cursor envelope.

Only original text and tool-input JSON values can be restored from transforms.
System/tool declarations and structural identities are immutable. Callers must
still apply the stage-specific content transformer before committing to Strands:
only that layer knows which tool-result text originally represented typed JSON
and which external tool declarations apply to a completed model response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from ._content import (
    _json_copy,
    _json_transform,
    _list,
    _load_json,
    _object,
    _same,
    _schema,
    _text,
    _validate_arguments,
)
from ._contracts import Direction
from .exceptions import TrustGuardTransformError, TrustGuardUnsupportedContentError

_PREFIX = "TrustGuard Strands content assessment. Assess every following protected content block."
_CONTENT_ERROR = "Unsupported or malformed TrustGuard assessment content."
_TRANSFORM_ERROR = "TrustGuard returned an unsupported or malformed transformation."
_Path = tuple[str | int, ...]
_Kind = Literal["text", "json", "immutable_text", "immutable_json"]
_NodeKind = Literal["text", "scalar", "encoded", "dict", "list"]


@dataclass(frozen=True, slots=True)
class _Unit:
    path: _Path
    kind: _Kind


@dataclass(frozen=True, slots=True)
class _Node:
    kind: _NodeKind
    original: Any
    index: int = -1
    immutable: bool = False
    children: tuple[_Node, ...] = ()
    keys: tuple[str, ...] = ()


def _invalid() -> TrustGuardUnsupportedContentError:
    return TrustGuardUnsupportedContentError(_CONTENT_ERROR)


def _serialize(value: Any) -> str:
    """Canonical JSON text, rejecting invalid Unicode and nonfinite numbers."""
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    text.encode("utf-8")
    return text


def _get(value: Any, path: _Path) -> Any:
    for key in path:
        value = value[key]
    return value


def _put(value: Any, path: _Path, replacement: Any) -> None:
    _get(value, path[:-1])[path[-1]] = replacement


def _canonical(payload: Any) -> tuple[dict[str, Any], list[_Unit]]:
    """Copy the exact supported normalized shape and enumerate protected paths."""
    source = _object(_json_copy(payload), {"messages"}, {"system", "tools"})
    # Validate Unicode even in JSON keys and immutable metadata before assessing.
    _serialize(source)
    units: list[_Unit] = []
    if "system" in source:
        system = source["system"]
        if type(system) is str:
            units.append(_Unit(("system",), "immutable_text"))
        else:
            for index, block in enumerate(_list(system)):
                block = _object(block, {"type", "text"})
                if block["type"] != "text":
                    raise _invalid()
                _text(block["text"])
                units.append(_Unit(("system", index, "text"), "immutable_text"))
    schemas: dict[str, Any] = {}
    for index, declaration in enumerate(_list(source.get("tools", []))):
        declaration = _object(declaration, {"name", "description", "input_schema"})
        name = _text(declaration["name"], nonempty=True)
        _text(declaration["description"])
        if name in schemas:
            raise _invalid()
        schemas[name] = _schema(declaration["input_schema"])
        units.append(_Unit(("tools", index), "immutable_json"))
    for message_index, message in enumerate(_list(source["messages"])):
        message = _object(message, {"role", "content"})
        role = _text(message["role"])
        if role not in ("user", "assistant"):
            raise _invalid()
        for block_index, block in enumerate(_list(message["content"])):
            path: _Path = ("messages", message_index, "content", block_index)
            if type(block) is not dict:
                raise _invalid()
            kind = block.get("type")
            if kind == "text":
                block = _object(block, {"type", "text"})
                _text(block["text"])
                units.append(_Unit((*path, "text"), "text"))
            elif kind == "tool_use" and role == "assistant":
                block = _object(block, {"type", "id", "name", "input"})
                _text(block["id"], nonempty=True)
                name = _text(block["name"], nonempty=True)
                if name in schemas:
                    _validate_arguments(block["input"], schemas[name])
                units.append(_Unit((*path, "input"), "json"))
            elif kind == "tool_result" and role == "user":
                block = _object(block, {"type", "tool_use_id", "is_error", "content"})
                _text(block["tool_use_id"], nonempty=True)
                if type(block["is_error"]) is not bool:
                    raise _invalid()
                for result_index, result in enumerate(_list(block["content"])):
                    result = _object(result, {"type", "text"})
                    if result["type"] != "text":
                        raise _invalid()
                    _text(result["text"])
                    units.append(_Unit((*path, "content", result_index, "text"), "text"))
            else:
                raise _invalid()
    return source, units


def validate_payload_transform(original: Any, changed: Any) -> dict[str, Any]:
    """Return a defensive canonical transform with only supported values changed.

    Roles, message/block counts/order/types, object keys/array lengths, JSON
    scalar categories, tool IDs/names/status, and metadata remain unchanged.
    Tool schemas included in the payload validate original and resulting input.
    Missing external schemas and originally typed JSON results are subsequently
    checked by the caller's stage-specific content transformation.
    """
    try:
        restored, units = _canonical(original)
        candidate = _object(_json_copy(changed), set(restored))
        _serialize(candidate)
        for unit in units:
            if unit.kind == "text":
                _put(restored, unit.path, _text(_get(candidate, unit.path)))
            elif unit.kind == "json":
                _put(
                    restored,
                    unit.path,
                    _json_transform(_get(restored, unit.path), _get(candidate, unit.path)),
                )
        if not _same(restored, candidate):
            raise _invalid()
        result, _ = _canonical(restored)
        return result
    except (
        TrustGuardUnsupportedContentError,
        KeyError,
        IndexError,
        ValueError,
        TypeError,
        RecursionError,
        ArithmeticError,
    ):
        raise TrustGuardTransformError(_TRANSFORM_ERROR) from None


def _view(
    value: Any, immutable: bool, content: list[dict[str, str]], budget: list[int], depth: int = 0
) -> _Node:
    """Expose JSON values/keys as separate text; keep reconstruction only locally.

    A prefix plus serialized JSON is not valid JSON for text-only detectors that
    first decode complete input strings. Raw leaves avoid punctuation hiding a
    keyword. Decode JSON inside strings too, including normalized JSON results.
    """
    budget[0] -= 1
    if depth > 64 or budget[0] < 0:
        raise _invalid()
    if type(value) is str:
        try:
            parsed = _load_json(value)
        except TrustGuardUnsupportedContentError:
            # Malformed ordinary prose stays text. Recognizable JSON that only
            # failed our depth/node/duplicate/nonfinite checks must not silently
            # fall back to an opaque string and hide its protected leaves.
            if value.lstrip().startswith(("{", "[", '"')):
                try:
                    json.loads(value)
                except json.JSONDecodeError:
                    pass
                except (ValueError, RecursionError, ArithmeticError):
                    raise _invalid() from None
                else:
                    raise _invalid()
            parsed = value
        # Bare numeric/bool/null-looking SDK strings remain strings. Inferring a
        # scalar category would reject a legitimate string DLP replacement.
        if type(parsed) in (dict, list, str) and not _same(value, parsed):
            child = _view(parsed, immutable, content, budget, depth + 1)
            return _Node("encoded", value, children=(child,))
        content.append({"type": "text", "text": value})
        return _Node("text", value, index=len(content) - 1, immutable=immutable)
    if type(value) is dict and value:
        keys = tuple(sorted(value))
        children = []
        for key in keys:
            children.append(_view(key, True, content, budget, depth + 1))
            children.append(_view(value[key], immutable, content, budget, depth + 1))
        return _Node("dict", value, children=tuple(children), keys=keys)
    if type(value) is list and value:
        children = [_view(item, immutable, content, budget, depth + 1) for item in value]
        return _Node("list", value, children=tuple(children))
    # Empty containers have no values, but retain an immutable visible unit.
    frozen = immutable or type(value) in (dict, list)
    content.append({"type": "text", "text": _serialize(value)})
    return _Node("scalar", value, index=len(content) - 1, immutable=frozen)


def _restore(node: _Node, content: list[dict[str, Any]]) -> Any:
    if node.kind == "encoded":
        child = node.children[0]
        value = _restore(child, content)
        return node.original if _same(child.original, value) else _serialize(value)
    if node.kind == "dict":
        value = {}
        for index, key in enumerate(node.keys):
            _restore(node.children[index * 2], content)  # Keys are immutable assessment units.
            value[key] = _restore(node.children[index * 2 + 1], content)
        return value
    if node.kind == "list":
        return [_restore(child, content) for child in node.children]
    text = _text(content[node.index]["text"])
    expected = node.original if node.kind == "text" else _serialize(node.original)
    if node.immutable:
        if text != expected:
            raise _invalid()
        return _json_copy(node.original)
    if node.kind == "text":
        return text
    return _json_transform(node.original, _load_json(text))


def _project(
    source: dict[str, Any], units: list[_Unit], direction: Direction
) -> tuple[dict[str, Any], list[_Node]]:
    if type(direction) is not str or direction not in ("input", "output"):
        raise _invalid()
    content = [{"type": "text", "text": _PREFIX}]
    nodes = []
    budget = [100_000]
    for unit in units:
        value = _get(source, unit.path)
        immutable = unit.kind in ("immutable_json", "immutable_text")
        nodes.append(_view(value, immutable, content, budget))
    return {
        "messages": [{"role": "user" if direction == "input" else "assistant", "content": content}]
    }, nodes


def project_payload(payload: Any, direction: Direction) -> dict[str, Any]:
    """Assess every supported unit in one user/input or assistant/output message.

    The order is system text, complete tool declarations, then conversation
    content in original message/block order. JSON values, including JSON inside
    text, become raw string/scalar leaves with immutable keys. Original paths
    are local mapping data; caller/server labels cannot redirect reconstruction.
    """
    try:
        source, units = _canonical(payload)
        projection, _ = _project(source, units, direction)
        return projection
    except (
        TrustGuardUnsupportedContentError,
        KeyError,
        IndexError,
        ValueError,
        TypeError,
        RecursionError,
        ArithmeticError,
    ):
        raise _invalid() from None


def apply_projection_transform(payload: Any, projection: Any, changed: Any) -> dict[str, Any]:
    """Restore a complete projected transform by immutable position and path.

    Require that the supplied original projection exactly matches this payload.
    JSON leaves are rebuilt without key/shape/type changes. Original JSON text
    formatting is retained when decoded values do not change. Prefix/metadata
    units cannot change. No substring or global replacement is performed, even
    when identical strings occur many times.
    """
    try:
        source, units = _canonical(payload)
        original_projection = _json_copy(projection)
        role = _get(original_projection, ("messages", 0, "role"))
        if role not in ("user", "assistant"):
            raise _invalid()
        expected, nodes = _project(source, units, "input" if role == "user" else "output")
        if not _same(original_projection, expected):
            raise _invalid()
        assessed = validate_payload_transform(expected, changed)
        old_blocks = expected["messages"][0]["content"]
        new_blocks = assessed["messages"][0]["content"]
        if not _same(old_blocks[0], new_blocks[0]):
            raise _invalid()
        for unit, node in zip(units, nodes, strict=True):
            value = _restore(node, new_blocks)
            _put(source, unit.path, value)
        return validate_payload_transform(payload, source)
    except (
        TrustGuardUnsupportedContentError,
        KeyError,
        IndexError,
        ValueError,
        TypeError,
        RecursionError,
        ArithmeticError,
    ):
        raise TrustGuardTransformError(_TRANSFORM_ERROR) from None
