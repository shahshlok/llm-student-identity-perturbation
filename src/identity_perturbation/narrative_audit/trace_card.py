from __future__ import annotations

from collections import Counter

from identity_perturbation.codebench_support.codemirror import (
    CodeMirrorParseError,
    apply_change,
    replay_trace,
    touched_lines_for_change,
)
from identity_perturbation.codebench_support.models import CmEvent

from .models import AttemptTraceCard


class V6TraceCardError(ValueError):
    pass


def _code_to_lines(code: str) -> list[str]:
    return [""] if code == "" else code.split("\n")


def _events_grouped_by_submit(events: tuple[CmEvent, ...]) -> tuple[tuple[CmEvent, ...], ...]:
    grouped: list[tuple[CmEvent, ...]] = []
    current: list[CmEvent] = []

    for event in events:
        current.append(event)
        if event.raw_type == "submit":
            grouped.append(tuple(current))
            current = []

    if not grouped:
        raise V6TraceCardError("CodeMirror events contain no submit-bounded intervals")
    return tuple(grouped)


def _selected_text(lines: list[str], change: dict) -> str:
    from_line = int(change["from"]["line"])
    from_ch = int(change["from"]["ch"])
    to_line = int(change["to"]["line"])
    to_ch = int(change["to"]["ch"])

    if from_line == to_line:
        return lines[from_line][from_ch:to_ch]

    selected_lines = [lines[from_line][from_ch:]]
    selected_lines.extend(lines[from_line + 1 : to_line])
    selected_lines.append(lines[to_line][:to_ch])
    return "\n".join(selected_lines)


def _local_test_event_count(events: tuple[CmEvent, ...]) -> int:
    count = 0
    for event in events:
        raw_type = event.raw_type.lower()
        if "test" in raw_type:
            count += 1
    return count


def _build_attempt_trace_card(
    *,
    interval_events: tuple[CmEvent, ...],
    start_code: str,
    submit_index: int,
    previous_submit_timestamp: str | None,
) -> AttemptTraceCard:
    event_type_counts = Counter(event.raw_type for event in interval_events)
    change_events = [event for event in interval_events if event.raw_type == "change"]
    submit_events = [event for event in interval_events if event.raw_type == "submit"]

    if len(submit_events) != 1:
        raise V6TraceCardError(
            f"Expected exactly one submit in interval for submit_index={submit_index}"
        )

    if not change_events:
        return AttemptTraceCard(
            submit_index=submit_index,
            previous_submit_index=None if submit_index == 0 else submit_index - 1,
            submit_timestamp=submit_events[0].timestamp.isoformat(),
            previous_submit_timestamp=previous_submit_timestamp,
            n_interval_events=len(interval_events),
            n_change_events=0,
            n_non_change_events=len(interval_events),
            event_type_counts=tuple(sorted(event_type_counts.items())),
            change_origin_counts=(),
            local_test_event_count=_local_test_event_count(interval_events),
            unique_lines_touched_0idx=(),
            ordered_first_touched_lines_0idx=(),
            first_changed_line_0idx=None,
            last_changed_line_0idx=None,
            line_span_0idx=None,
            insert_only_event_count=0,
            delete_only_event_count=0,
            replace_event_count=0,
            multiline_change_event_count=0,
            revisited_line_count=0,
            inserted_char_count=0,
            deleted_char_count=0,
            net_char_delta=0,
        )

    lines = _code_to_lines(start_code)
    change_origin_counts: Counter[str] = Counter()
    line_touch_counts: Counter[int] = Counter()
    ordered_lines: list[int] = []
    insert_only_event_count = 0
    delete_only_event_count = 0
    replace_event_count = 0
    multiline_change_event_count = 0
    inserted_char_count = 0
    deleted_char_count = 0

    for event in change_events:
        if not isinstance(event.payload, dict):
            raise CodeMirrorParseError("Parsed change payload is not a dict")
        change = event.payload
        touched_lines = sorted(touched_lines_for_change(change))
        deleted_text = _selected_text(lines, change)
        inserted_text = "\n".join(change["text"])

        inserted_char_count += len(inserted_text)
        deleted_char_count += len(deleted_text)

        if inserted_text and deleted_text:
            replace_event_count += 1
        elif inserted_text:
            insert_only_event_count += 1
        elif deleted_text:
            delete_only_event_count += 1

        if len(touched_lines) > 1:
            multiline_change_event_count += 1

        for line in touched_lines:
            line_touch_counts[line] += 1
            if line not in ordered_lines:
                ordered_lines.append(line)

        origin = change.get("origin")
        if not isinstance(origin, str) or not origin:
            raise V6TraceCardError(
                f"Missing or invalid change origin in submit_index={submit_index}"
            )
        change_origin_counts[origin] += 1

        lines = apply_change(lines, change)

    unique_lines_touched = tuple(sorted(line_touch_counts))
    revisited_line_count = sum(1 for count in line_touch_counts.values() if count > 1)

    return AttemptTraceCard(
        submit_index=submit_index,
        previous_submit_index=None if submit_index == 0 else submit_index - 1,
        submit_timestamp=submit_events[0].timestamp.isoformat(),
        previous_submit_timestamp=previous_submit_timestamp,
        n_interval_events=len(interval_events),
        n_change_events=len(change_events),
        n_non_change_events=len(interval_events) - len(change_events),
        event_type_counts=tuple(sorted(event_type_counts.items())),
        change_origin_counts=tuple(sorted(change_origin_counts.items())),
        local_test_event_count=_local_test_event_count(interval_events),
        unique_lines_touched_0idx=unique_lines_touched,
        ordered_first_touched_lines_0idx=tuple(ordered_lines),
        first_changed_line_0idx=unique_lines_touched[0],
        last_changed_line_0idx=unique_lines_touched[-1],
        line_span_0idx=(unique_lines_touched[0], unique_lines_touched[-1]),
        insert_only_event_count=insert_only_event_count,
        delete_only_event_count=delete_only_event_count,
        replace_event_count=replace_event_count,
        multiline_change_event_count=multiline_change_event_count,
        revisited_line_count=revisited_line_count,
        inserted_char_count=inserted_char_count,
        deleted_char_count=deleted_char_count,
        net_char_delta=inserted_char_count - deleted_char_count,
    )


def build_attempt_trace_cards(
    events: tuple[CmEvent, ...],
    initial_code: str = "",
) -> tuple[AttemptTraceCard, ...]:
    intervals = _events_grouped_by_submit(events)
    trace = replay_trace(events, initial_code=initial_code)

    if len(intervals) != len(trace.snapshots):
        raise V6TraceCardError("Submit-bounded interval count does not match replay snapshots")

    cards: list[AttemptTraceCard] = []
    previous_submit_timestamp: str | None = None

    for submit_index, (interval_events, snapshot) in enumerate(
        zip(intervals, trace.snapshots, strict=False)
    ):
        if snapshot.submit_index != submit_index:
            raise V6TraceCardError("Replay snapshot submit indices are not sequential")
        start_code = initial_code if submit_index == 0 else trace.snapshots[submit_index - 1].code
        cards.append(
            _build_attempt_trace_card(
                interval_events=interval_events,
                start_code=start_code,
                submit_index=submit_index,
                previous_submit_timestamp=previous_submit_timestamp,
            )
        )
        previous_submit_timestamp = snapshot.submit_event.timestamp.isoformat()

    return tuple(cards)


def trace_card_to_prompt_dict(card: AttemptTraceCard) -> dict:
    return {
        "submit_index": card.submit_index,
        "previous_submit_index": card.previous_submit_index,
        "submit_timestamp": card.submit_timestamp,
        "previous_submit_timestamp": card.previous_submit_timestamp,
        "n_interval_events": card.n_interval_events,
        "n_change_events": card.n_change_events,
        "n_non_change_events": card.n_non_change_events,
        "event_type_counts": dict(card.event_type_counts),
        "change_origin_counts": dict(card.change_origin_counts),
        "local_test_event_count": card.local_test_event_count,
        "unique_lines_touched_0idx": list(card.unique_lines_touched_0idx),
        "ordered_first_touched_lines_0idx": list(card.ordered_first_touched_lines_0idx),
        "first_changed_line_0idx": card.first_changed_line_0idx,
        "last_changed_line_0idx": card.last_changed_line_0idx,
        "line_span_0idx": list(card.line_span_0idx) if card.line_span_0idx is not None else None,
        "insert_only_event_count": card.insert_only_event_count,
        "delete_only_event_count": card.delete_only_event_count,
        "replace_event_count": card.replace_event_count,
        "multiline_change_event_count": card.multiline_change_event_count,
        "revisited_line_count": card.revisited_line_count,
        "inserted_char_count": card.inserted_char_count,
        "deleted_char_count": card.deleted_char_count,
        "net_char_delta": card.net_char_delta,
    }
