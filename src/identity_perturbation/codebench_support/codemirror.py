from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import CmEvent, ReplayTrace, SubmitSnapshot


class CodeMirrorParseError(ValueError):
    pass


def _code_to_lines(code: str) -> list[str]:
    return [""] if code == "" else code.split("\n")


def _first_replay_relevant_event(events: tuple[CmEvent, ...]) -> CmEvent | None:
    for event in events:
        if event.raw_type in {"change", "submit"}:
            return event
    return None


def parse_cm_timestamp(ts_str: str) -> datetime:
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")


def parse_codemirror_log_text(log_text: str) -> tuple[CmEvent, ...]:
    events: list[CmEvent] = []
    in_saida_testar_output = False
    for line_number, raw_line in enumerate(log_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("#", 2)
        if len(parts) < 2:
            if in_saida_testar_output:
                continue
            raise CodeMirrorParseError(f"Malformed CodeMirror line at {line_number}")
        try:
            timestamp = parse_cm_timestamp(parts[0])
        except ValueError as exc:
            if in_saida_testar_output:
                continue
            raise CodeMirrorParseError(
                f"Malformed timestamp at log line {line_number}: {parts[0]!r}"
            ) from exc
        raw_type = parts[1]
        payload_raw = parts[2] if len(parts) > 2 else ""
        payload: object = payload_raw
        if raw_type == "change":
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError as exc:
                raise CodeMirrorParseError(
                    f"Malformed change payload at log line {line_number}"
                ) from exc
        events.append(
            CmEvent(
                timestamp=timestamp,
                raw_type=raw_type,
                payload=payload,
                line_number=line_number,
            )
        )
        in_saida_testar_output = raw_type == "saida_testar"

    if not events:
        raise CodeMirrorParseError("CodeMirror log produced no parseable events")
    return tuple(events)


def infer_initial_code(events: tuple[CmEvent, ...], seed_code: str) -> str:
    first_event = _first_replay_relevant_event(events)
    if first_event is None:
        raise CodeMirrorParseError("CodeMirror log contains neither change nor submit events")
    if first_event.raw_type == "submit":
        return seed_code
    if not isinstance(first_event.payload, dict):
        raise CodeMirrorParseError("First change payload is not a dict")

    from_line = int(first_event.payload["from"]["line"])
    from_ch = int(first_event.payload["from"]["ch"])
    to_line = int(first_event.payload["to"]["line"])
    to_ch = int(first_event.payload["to"]["ch"])
    if from_line == 0 and to_line == 0 and from_ch == 0 and to_ch == 0:
        return ""
    return seed_code


def parse_codemirror_log(path: Path) -> tuple[CmEvent, ...]:
    return parse_codemirror_log_text(path.read_text(encoding="utf-8", errors="replace"))


def apply_change(lines: list[str], change: dict) -> list[str]:
    if not isinstance(change, dict):
        raise CodeMirrorParseError("Change payload must be a dict")
    from_line = int(change["from"]["line"])
    from_ch = int(change["from"]["ch"])
    to_line = int(change["to"]["line"])
    to_ch = int(change["to"]["ch"])
    text = change["text"]

    if from_line < 0 or to_line < 0:
        raise CodeMirrorParseError("Change coordinates must be 0-indexed and non-negative")
    if from_line > to_line:
        raise CodeMirrorParseError("Change from.line cannot exceed to.line")
    if from_line >= len(lines) or to_line >= len(lines):
        raise CodeMirrorParseError(
            f"Change coordinates out of bounds for current buffer: {from_line=} {to_line=} {len(lines)=}"
        )
    if from_ch < 0 or to_ch < 0:
        raise CodeMirrorParseError("Change character positions must be non-negative")
    if from_ch > len(lines[from_line]) or to_ch > len(lines[to_line]):
        raise CodeMirrorParseError("Change character position exceeds current line length")
    if from_line == to_line and from_ch > to_ch:
        raise CodeMirrorParseError("Change from.ch cannot exceed to.ch on the same line")
    if not isinstance(text, list) or not text or not all(isinstance(item, str) for item in text):
        raise CodeMirrorParseError("Change text must be a non-empty list[str]")

    prefix = lines[from_line][:from_ch]
    suffix = lines[to_line][to_ch:]

    if len(text) == 1:
        new_lines = [prefix + text[0] + suffix]
    else:
        new_lines = [prefix + text[0], *text[1:-1], text[-1] + suffix]

    lines[from_line : to_line + 1] = new_lines
    return lines


def replay_changes(start_code: str, changes: tuple[dict, ...]) -> str:
    lines = _code_to_lines(start_code)
    for change in changes:
        lines = apply_change(lines, change)
    return "\n".join(lines)


def normalize_code_for_match(code: str) -> str:
    lines = [line.rstrip() for line in code.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def replay_trace(events: tuple[CmEvent, ...], initial_code: str = "") -> ReplayTrace:
    lines = _code_to_lines(initial_code)
    changes_since_prev: list[dict] = []
    snapshots: list[SubmitSnapshot] = []
    submit_index = 0

    for event in events:
        if event.raw_type == "change":
            if not isinstance(event.payload, dict):
                raise CodeMirrorParseError("Parsed change payload is not a dict")
            lines = apply_change(lines, event.payload)
            changes_since_prev.append(event.payload)
        elif event.raw_type == "submit":
            snapshots.append(
                SubmitSnapshot(
                    submit_event=event,
                    code="\n".join(lines),
                    changes_since_prev=tuple(changes_since_prev),
                    submit_index=submit_index,
                )
            )
            submit_index += 1
            changes_since_prev = []

    if not snapshots:
        raise CodeMirrorParseError("CodeMirror log contains no submit events")
    return ReplayTrace(
        snapshots=tuple(snapshots),
        final_code="\n".join(lines),
        trailing_changes=tuple(changes_since_prev),
    )


def full_replay(events: tuple[CmEvent, ...]) -> tuple[SubmitSnapshot, ...]:
    return replay_trace(events).snapshots


def touched_lines_for_change(change: dict) -> set[int]:
    from_line = int(change["from"]["line"])
    to_line = int(change["to"]["line"])
    inserted = change["text"]
    inserted_end = from_line + len(inserted) - 1 if inserted else from_line
    end_line = max(from_line, to_line, inserted_end)
    return set(range(from_line, end_line + 1))


def lines_touched_0idx(changes: tuple[dict, ...]) -> tuple[int, ...]:
    touched: set[int] = set()
    for change in changes:
        touched.update(touched_lines_for_change(change))
    if not touched:
        raise CodeMirrorParseError("Aligned change set produced no touched lines")
    return tuple(sorted(touched))


def first_change_line_0idx(changes: tuple[dict, ...]) -> int:
    if not changes:
        raise CodeMirrorParseError("Aligned change set is empty")
    return int(changes[0]["from"]["line"])


def lines_touched_bucket_3way(changes: tuple[dict, ...]) -> str:
    count = len(lines_touched_0idx(changes))
    if count <= 2:
        return "local_1_to_2_lines"
    if count <= 5:
        return "regional_3_to_5_lines"
    return "broad_6_plus_lines"
