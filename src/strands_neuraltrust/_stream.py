"""Validate public raw model events before the SDK parses or logs their content."""

from __future__ import annotations

import json
import math
from typing import Any

from ._contracts import _constant, _float, _integer, _pairs, _validate_json
from .exceptions import TrustGuardUnavailable, TrustGuardUnsupportedContentError


class StreamMonitor:
    """Bound serialized events and accept complete sequential text/tool streams."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.reset()

    def reset(self) -> None:
        self.bytes = 0
        self.begin_model()

    def begin_model(self) -> None:
        self.started = False
        self.stopped = False
        self.block: str | None = None
        self.index: int | None = None
        self.arguments: list[str] = []
        self.tool_ids: set[str] = set()
        self.tools = 0

    def count(self, value: Any) -> None:
        failure = False
        try:
            _validate_json(value)
            for part in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(value):
                self.bytes += len(part.encode("utf-8"))
                if self.bytes > self.max_bytes:
                    raise TrustGuardUnavailable("The invocation exceeded its stream byte budget.")
        except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
            failure = True
        if failure:
            raise TrustGuardUnsupportedContentError("The stream contains unsupported content.")

    def accept(self, chunk: Any) -> None:
        self.count(chunk)
        failure = False
        try:
            self._accept(chunk)
        except (TypeError, ValueError, KeyError, RecursionError, UnicodeError, OverflowError):
            failure = True
        if failure:
            raise TrustGuardUnsupportedContentError("The model stream is malformed or unsupported.")

    def _accept(self, chunk: Any) -> None:
        if type(chunk) is not dict or len(chunk) != 1:
            raise ValueError
        kind, body = next(iter(chunk.items()))
        if type(body) is not dict:
            raise ValueError
        if kind == "messageStart":
            if self.started or body != {"role": "assistant"}:
                raise ValueError
            self.started = True
            return
        if not self.started:
            raise ValueError
        if kind == "metadata":
            if set(body) - {"usage", "metrics"}:
                raise ValueError
            for key, values in body.items():
                allowed = (
                    {
                        "inputTokens",
                        "outputTokens",
                        "totalTokens",
                        "cacheReadInputTokens",
                        "cacheWriteInputTokens",
                    }
                    if key == "usage"
                    else {"latencyMs", "timeToFirstByteMs"}
                )
                if type(values) is not dict or set(values) - allowed:
                    raise ValueError
                for value in values.values():
                    if key == "usage":
                        if type(value) is not int or not 0 <= value <= 2**63 - 1:
                            raise ValueError
                    elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError
            return
        if self.stopped:
            raise ValueError
        if kind == "contentBlockStart":
            if self.block is not None or set(body) - {"start", "contentBlockIndex"}:
                raise ValueError
            self.index = self._index(body)
            start = body["start"]
            if start == {}:
                self.block = "text"
            elif type(start) is dict and set(start) == {"toolUse"}:
                use = start["toolUse"]
                if type(use) is not dict or set(use) != {"toolUseId", "name"}:
                    raise ValueError
                if any(type(v) is not str or not v for v in use.values()):
                    raise ValueError
                if use["toolUseId"] in self.tool_ids:
                    raise ValueError
                self.tool_ids.add(use["toolUseId"])
                self.block = "tool"
                self.tools += 1
                self.arguments = []
            else:
                raise ValueError
        elif kind == "contentBlockDelta":
            if set(body) - {"delta", "contentBlockIndex"}:
                raise ValueError
            delta = body["delta"]
            if type(delta) is not dict or len(delta) != 1:
                raise ValueError
            # Bedrock and some providers omit the explicit start for text blocks.
            if self.block is None and set(delta) == {"text"}:
                self.block = "text"
                self.index = self._index(body)
            self._same_index(body)
            if self.block == "text" and set(delta) == {"text"} and type(delta["text"]) is str:
                return
            if self.block == "tool" and set(delta) == {"toolUse"}:
                use = delta["toolUse"]
                if type(use) is dict and set(use) == {"input"} and type(use["input"]) is str:
                    self.arguments.append(use["input"])
                    return
            raise ValueError
        elif kind == "contentBlockStop":
            if self.block is None or set(body) - {"contentBlockIndex"}:
                raise ValueError
            self._same_index(body)
            if self.block == "tool":
                args = json.loads(
                    "".join(self.arguments),
                    object_pairs_hook=_pairs,
                    parse_int=_integer,
                    parse_float=_float,
                    parse_constant=_constant,
                )
                _validate_json(args)
                if type(args) is not dict:
                    raise ValueError
            self.block = None
            self.arguments = []
        elif kind == "messageStop":
            if self.block is not None or set(body) != {"stopReason"}:
                raise ValueError
            reason = body["stopReason"]
            if reason not in ("end_turn", "stop_sequence", "tool_use"):
                raise ValueError
            if (reason == "tool_use") != bool(self.tools):
                raise ValueError
            self.stopped = True
        else:
            raise ValueError

    @staticmethod
    def _index(body: dict[str, Any]) -> int | None:
        index = body.get("contentBlockIndex")
        if "contentBlockIndex" in body and index is None:
            raise ValueError
        if index is not None and (type(index) is not int or index < 0):
            raise ValueError
        return index

    def _same_index(self, body: dict[str, Any]) -> None:
        index = self._index(body)
        if self.index is None:
            self.index = index
        if index is not None and self.index is not None and index != self.index:
            raise ValueError

    def finish_model(self) -> None:
        if not self.started or not self.stopped or self.block is not None:
            raise TrustGuardUnsupportedContentError("The model stream did not finish a complete response.")
