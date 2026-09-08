"""Adversarial validation of raw public model events before SDK decoding."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strands_neuraltrust._stream import StreamMonitor
from strands_neuraltrust.exceptions import TrustGuardUnavailable, TrustGuardUnsupportedContentError

START = {"messageStart": {"role": "assistant"}}
TEXT_START = {"contentBlockStart": {"start": {}}}
TEXT_DELTA = {"contentBlockDelta": {"delta": {"text": "synthetic"}}}
BLOCK_STOP = {"contentBlockStop": {}}
STOP = {"messageStop": {"stopReason": "end_turn"}}
TOOL_START = {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "call-1", "name": "search"}}}}
TOOL_DELTA = {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"q":"synthetic"}'}}}}
TOOL_STOP = {"messageStop": {"stopReason": "tool_use"}}


def feed(events: list[Any], monitor: StreamMonitor | None = None) -> StreamMonitor:
    monitor = monitor or StreamMonitor(1_000_000)
    for event in events:
        monitor.accept(copy.deepcopy(event))
    return monitor


@pytest.mark.parametrize("explicit_start", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
def test_valid_text_streams_with_optional_public_start_and_index(explicit_start: bool, indexed: bool) -> None:
    events = [START]
    if explicit_start:
        events.append(TEXT_START)
    events.extend([TEXT_DELTA, BLOCK_STOP, STOP])
    events = copy.deepcopy(events)
    if indexed:
        for event in events:
            kind, body = next(iter(event.items()))
            if kind.startswith("contentBlock"):
                body["contentBlockIndex"] = 0
    monitor = feed(events)
    monitor.finish_model()
    assert monitor.stopped is True and monitor.block is None


def test_valid_multi_block_mixed_tool_stream_and_metadata() -> None:
    metadata = {
        "metadata": {
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "totalTokens": 15,
                "cacheReadInputTokens": 0,
                "cacheWriteInputTokens": 0,
            },
            "metrics": {"latencyMs": 1, "timeToFirstByteMs": 0},
        }
    }
    monitor = feed(
        [START, TEXT_START, TEXT_DELTA, BLOCK_STOP, TOOL_START, TOOL_DELTA, BLOCK_STOP, TOOL_STOP, metadata]
    )
    monitor.finish_model()
    assert monitor.tools == 1 and monitor.tool_ids == {"call-1"}


@pytest.mark.parametrize(
    "metadata", [{"metadata": {}}, {"metadata": {"usage": {}}}, {"metadata": {"metrics": {}}}]
)
def test_optional_empty_metadata_is_valid(metadata: Any) -> None:
    feed([START, metadata, TEXT_DELTA, BLOCK_STOP, STOP, metadata]).finish_model()


def test_model_boundary_resets_stream_state_but_preserves_invocation_byte_budget() -> None:
    monitor = feed([START, TOOL_START, TOOL_DELTA, BLOCK_STOP, TOOL_STOP])
    monitor.finish_model()
    counted = monitor.bytes
    monitor.begin_model()
    assert monitor.bytes == counted
    assert monitor.tool_ids == set() and monitor.tools == 0
    assert monitor.started is False and monitor.stopped is False
    feed([START, TEXT_DELTA, BLOCK_STOP, STOP], monitor).finish_model()
    assert monitor.bytes > counted
    monitor.reset()
    assert monitor.bytes == 0 and monitor.started is False


@pytest.mark.parametrize(
    "events",
    [
        [],
        [TEXT_START],
        [TEXT_DELTA],
        [BLOCK_STOP],
        [STOP],
        [START],
        [START, TEXT_START],
        [START, TEXT_DELTA],
        [START, TEXT_DELTA, BLOCK_STOP],
    ],
)
def test_incomplete_or_out_of_order_streams_never_finish(events: list[Any]) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed(events).finish_model()


@pytest.mark.parametrize(
    "chunk",
    [
        None,
        [],
        "private",
        {},
        {"messageStart": {}, "metadata": {}},
        {"messageStart": []},
        {"messageStart": {"role": "user"}},
        {"messageStart": {"role": True}},
        {"messageStart": {"role": "assistant", "extra": "private"}},
    ],
)
def test_invalid_start_and_envelopes_fail_closed(chunk: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([chunk])


@pytest.mark.parametrize(
    "chunk",
    [
        START,
        {"unknown": {}},
        {"redactContent": {"redactAssistantContentMessage": "private"}},
        {"contentBlockStart": {}},
        {"contentBlockStart": {"start": {}, "extra": True}},
        {"contentBlockStart": {"start": []}},
        {"contentBlockStart": {"start": {"reasoningContent": {}}}},
        {"metadata": {"unknown": "private"}},
    ],
)
def test_duplicate_start_unknown_events_and_blocks_fail_closed(chunk: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([START, chunk])


@pytest.mark.parametrize(
    "chunk",
    [
        TEXT_START,
        {"contentBlockDelta": {}},
        {"contentBlockDelta": {"delta": []}},
        {"contentBlockDelta": {"delta": {}}},
        {"contentBlockDelta": {"delta": {"text": "safe", "reasoningContent": "private"}}},
        {"contentBlockDelta": {"delta": {"text": 1}}},
        {"contentBlockDelta": {"delta": {"text": "safe"}, "extra": "private"}},
        TOOL_DELTA,
        {"contentBlockDelta": {"delta": {"citation": {"text": "private"}}}},
        {"contentBlockStop": {"extra": "private"}},
        STOP,
    ],
)
def test_text_block_rejects_nested_start_wrong_deltas_and_unfinished_stop(chunk: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([START, TEXT_START, chunk])


@pytest.mark.parametrize("chunk", [TEXT_DELTA, TEXT_START, BLOCK_STOP, STOP])
def test_no_content_or_duplicate_stop_after_message_stop(chunk: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([START, TEXT_DELTA, BLOCK_STOP, STOP, chunk])


@pytest.mark.parametrize("index", [-1, True, False, "0", 0.0, None, [], {}])
def test_indexes_must_be_present_as_nonnegative_integers(index: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([START, {"contentBlockStart": {"start": {}, "contentBlockIndex": index}}])


@pytest.mark.parametrize("explicit_start_index", [True, False])
def test_a_block_cannot_switch_indexes_between_deltas(explicit_start_index: bool) -> None:
    start = copy.deepcopy(TEXT_START)
    if explicit_start_index:
        start["contentBlockStart"]["contentBlockIndex"] = 0
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed(
            [
                START,
                start,
                {"contentBlockDelta": {"delta": {"text": "a"}, "contentBlockIndex": 0}},
                {"contentBlockDelta": {"delta": {"text": "b"}, "contentBlockIndex": 1}},
            ]
        )


@pytest.mark.parametrize(
    "use",
    [
        {},
        [],
        {"toolUseId": "call"},
        {"toolUseId": "call", "name": "search", "reasoningSignature": "private"},
        {"toolUseId": "", "name": "search"},
        {"toolUseId": "call", "name": ""},
        {"toolUseId": False, "name": "search"},
        {"toolUseId": "call", "name": []},
    ],
)
def test_tool_start_requires_valid_identity_and_no_hidden_fields(use: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([START, {"contentBlockStart": {"start": {"toolUse": use}}}])


def test_duplicate_tool_ids_in_one_model_response_are_rejected() -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([START, TOOL_START, TOOL_DELTA, BLOCK_STOP, TOOL_START])


@pytest.mark.parametrize(
    "delta",
    [
        {"toolUse": {}},
        {"toolUse": []},
        {"toolUse": {"input": {}}},
        {"toolUse": {"input": "{}", "name": "different"}},
        {"toolUse": {"input": "{}", "toolUseId": "different"}},
        {"text": "private"},
    ],
)
def test_tool_delta_cannot_change_identity_or_supply_nontext_json_fragments(delta: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([START, TOOL_START, {"contentBlockDelta": {"delta": delta}}])


@pytest.mark.parametrize(
    "argument",
    [
        "",
        "{bad",
        "[]",
        "null",
        '"string"',
        "true",
        '{"q":"a","q":"b"}',
        '{"q":NaN}',
        '{"q":Infinity}',
        '{"q":1e309}',
        '{"q":' + "9" * 129 + "}",
        '{"q":"\\ud800"}',
    ],
)
def test_tool_argument_json_is_complete_unambiguous_finite_object(argument: str) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed(
            [
                START,
                TOOL_START,
                {"contentBlockDelta": {"delta": {"toolUse": {"input": argument}}}},
                BLOCK_STOP,
            ]
        )


@pytest.mark.parametrize(
    "reason",
    ["cancelled", "max_tokens", "guardrail_intervened", "content_filtered", "unknown", None, True, {}],
)
def test_truncation_cancellation_and_unknown_stops_are_not_complete_safe_results(reason: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([START, TEXT_DELTA, BLOCK_STOP, {"messageStop": {"stopReason": reason}}])


def test_stop_reason_matches_presence_of_tool_requests() -> None:
    for events in (
        [START, TEXT_DELTA, BLOCK_STOP, TOOL_STOP],
        [START, TOOL_START, TOOL_DELTA, BLOCK_STOP, STOP],
    ):
        with pytest.raises(TrustGuardUnsupportedContentError):
            feed(events)
    feed([START, TEXT_DELTA, BLOCK_STOP, {"messageStop": {"stopReason": "stop_sequence"}}]).finish_model()


@pytest.mark.parametrize(
    "body",
    [
        {"usage": "private"},
        {"usage": {"inputTokens": "private"}},
        {"usage": {"inputTokens": True}},
        {"usage": {"inputTokens": -1}},
        {"usage": {"inputTokens": 1.5}},
        {"usage": {"extra": "private"}},
        {"metrics": []},
        {"metrics": {"latencyMs": "private"}},
        {"metrics": {"latencyMs": True}},
        {"metrics": {"latencyMs": -1}},
        {"metrics": {"extra": "private"}},
    ],
)
def test_metadata_cannot_carry_unreviewed_text_or_malformed_usage(body: Any) -> None:
    with pytest.raises(TrustGuardUnsupportedContentError):
        feed([START, {"metadata": body}])


@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf, b"private", {1: "private"}, object(), {"text": "\ud800"}]
)
def test_count_rejects_non_json_content_without_disclosing_values(value: Any) -> None:
    with pytest.raises(
        TrustGuardUnsupportedContentError, match="^The stream contains unsupported content\\.$"
    ):
        StreamMonitor(1_000_000).count(value)


def test_count_handles_cycles_depth_and_exact_unicode_byte_budget() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)
    deep: Any = "private"
    for _ in range(70):
        deep = [deep]
    for value in (cycle, deep):
        with pytest.raises(TrustGuardUnsupportedContentError):
            StreamMonitor(1_000_000).count(value)
    event = {"contentBlockDelta": {"delta": {"text": "é🦆"}}}
    size = len(json.dumps(event, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    monitor = StreamMonitor(size)
    monitor.count(event)
    assert monitor.bytes == size
    with pytest.raises(TrustGuardUnavailable, match="^The invocation exceeded its stream byte budget\\.$"):
        monitor.count("")


@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",))), min_size=1, max_size=10))
def test_arbitrary_unicode_text_fragments_form_one_complete_response(fragments: list[str]) -> None:
    events = [START, TEXT_START]
    events.extend({"contentBlockDelta": {"delta": {"text": fragment}}} for fragment in fragments)
    events.extend([BLOCK_STOP, STOP])
    feed(events).finish_model()


json_scalar = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**63), max_value=2**63 - 1)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(alphabet=st.characters(blacklist_categories=("Cs",)))
)
json_values = st.recursive(
    json_scalar,
    lambda child: (
        st.lists(child, max_size=3)
        | st.dictionaries(st.text(alphabet=st.characters(blacklist_categories=("Cs",))), child, max_size=3)
    ),
    max_leaves=10,
)


@given(
    st.dictionaries(st.text(alphabet=st.characters(blacklist_categories=("Cs",))), json_values, max_size=5),
    st.integers(min_value=1, max_value=20),
)
@settings(max_examples=150)
def test_fragmented_nested_json_arguments_are_validated_after_reassembly(
    arguments: dict[str, Any], width: int
) -> None:
    serialized = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
    monitor = feed([START, TOOL_START])
    for offset in range(0, len(serialized), width):
        monitor.accept(
            {"contentBlockDelta": {"delta": {"toolUse": {"input": serialized[offset : offset + width]}}}}
        )
    feed([BLOCK_STOP, TOOL_STOP], monitor).finish_model()


@given(json_values, st.integers(min_value=0, max_value=6), st.booleans())
@settings(max_examples=1_000, deadline=None)
def test_arbitrary_event_body_mutations_never_escape_as_untyped_or_content_bearing_errors(
    replacement: Any, position: int, tool_stream: bool
) -> None:
    events = copy.deepcopy(
        [START, TOOL_START, TOOL_DELTA, BLOCK_STOP, TOOL_STOP, {"metadata": {}}]
        if tool_stream
        else [START, TEXT_START, TEXT_DELTA, BLOCK_STOP, STOP, {"metadata": {}}]
    )
    if position == 6:
        events.insert(3, replacement)
    else:
        kind = next(iter(events[position]))
        events[position] = {kind: replacement}
    try:
        feed(events).finish_model()
    except (TrustGuardUnsupportedContentError, TrustGuardUnavailable) as error:
        assert str(error) in {
            "The stream contains unsupported content.",
            "The model stream is malformed or unsupported.",
            "The model stream did not finish a complete response.",
            "The invocation exceeded its stream byte budget.",
        }
