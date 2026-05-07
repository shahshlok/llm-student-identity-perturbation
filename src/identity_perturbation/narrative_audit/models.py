from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttemptTraceCard:
    submit_index: int
    previous_submit_index: int | None
    submit_timestamp: str
    previous_submit_timestamp: str | None
    n_interval_events: int
    n_change_events: int
    n_non_change_events: int
    event_type_counts: tuple[tuple[str, int], ...]
    change_origin_counts: tuple[tuple[str, int], ...]
    local_test_event_count: int
    unique_lines_touched_0idx: tuple[int, ...]
    ordered_first_touched_lines_0idx: tuple[int, ...]
    first_changed_line_0idx: int | None
    last_changed_line_0idx: int | None
    line_span_0idx: tuple[int, int] | None
    insert_only_event_count: int
    delete_only_event_count: int
    replace_event_count: int
    multiline_change_event_count: int
    revisited_line_count: int
    inserted_char_count: int
    deleted_char_count: int
    net_char_delta: int
