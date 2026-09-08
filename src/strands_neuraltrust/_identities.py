"""Keep SDK tool identities local while assessing an equivalent wire payload.

Tool-call identifiers are routing metadata, not content. Some content detectors
recursively redact every JSON string, including digit-bearing SDK identifiers.
Use alphabetic aliases only at the structural identifier paths, validate any
returned transformation against that wire snapshot, then restore the exact SDK
identities. Tool names, content, schemas, and declarations are never substituted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._projection import _canonical, _put, validate_payload_transform
from .exceptions import TrustGuardUnsupportedContentError

_CONTENT_ERROR = "Unsupported or malformed TrustGuard assessment content."
_Path = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class _Identity:
    path: _Path
    original: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ToolIdentityBinding:
    """One assessment's private snapshots and reversible structural mapping."""

    _source: dict[str, Any] = field(repr=False)
    _wire: dict[str, Any] = field(repr=False)
    _identities: tuple[_Identity, ...] = field(repr=False)

    @property
    def payload(self) -> dict[str, Any]:
        """Return a defensive wire copy without exposing the bound snapshots."""
        return _canonical(self._wire)[0]

    def restore(self, transformed_payload: Any) -> dict[str, Any]:
        """Validate aliases first, then return a transform with exact SDK IDs.

        No returned alias is interpreted as a replacement identity. The complete
        wire shape, including every alias at its original path, must still match
        before any local ID is restored. The normal stage-specific transformer
        remains responsible for external schemas and typed SDK result blocks.
        """
        restored = validate_payload_transform(self._wire, transformed_payload)
        for identity in self._identities:
            _put(restored, identity.path, identity.original)
        return validate_payload_transform(self._source, restored)


def _alias(index: int) -> str:
    # Bijective base 26: a ... z, aa ... az, ba ... . No digit-bearing tokens.
    suffix = ""
    while index:
        index, digit = divmod(index - 1, 26)
        suffix = chr(ord("a") + digit) + suffix
    return "strands_tool_call_" + suffix


def bind_tool_identities(payload: Any) -> ToolIdentityBinding:
    """Canonicalize and replace only structural tool-use/result identifiers.

    The first appearance of each distinct logical ID receives a unique alias;
    all references to that ID use the same alias, even when a result appears
    before its use. Mapping by original identity and exact path prevents alias-
    looking original IDs or equal strings in content from causing collisions.
    """
    try:
        source, _ = _canonical(payload)
        wire, _ = _canonical(source)
        aliases: dict[str, str] = {}
        identities = []
        for message_index, message in enumerate(wire["messages"]):
            for block_index, block in enumerate(message["content"]):
                kind = block["type"]
                if kind not in ("tool_use", "tool_result"):
                    continue
                key = "id" if kind == "tool_use" else "tool_use_id"
                original = block[key]
                if original not in aliases:
                    aliases[original] = _alias(len(aliases) + 1)
                path: _Path = ("messages", message_index, "content", block_index, key)
                identities.append(_Identity(path, original))
                block[key] = aliases[original]
        return ToolIdentityBinding(source, wire, tuple(identities))
    except (
        TrustGuardUnsupportedContentError,
        KeyError,
        IndexError,
        ValueError,
        TypeError,
        RecursionError,
        ArithmeticError,
    ):
        raise TrustGuardUnsupportedContentError(_CONTENT_ERROR) from None
