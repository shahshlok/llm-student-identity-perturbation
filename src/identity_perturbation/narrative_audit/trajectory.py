from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from identity_perturbation.codebench_support.codemirror import parse_cm_timestamp, parse_codemirror_log
from identity_perturbation.codebench_support.models import CmEvent

from .semantic_schema import AttemptNSemanticTape


class V6TrajectoryError(ValueError):
    pass


@dataclass(frozen=True)
class ChangeMicroEvent:
    ordinal_in_interval_1idx: int
    seconds_since_previous_event: float | None
    timestamp: str
    from_line_0idx: int
    from_ch_0idx: int
    to_line_0idx: int
    to_ch_0idx: int
    inserted_text: str
    removed_text: str
    origin: str


@dataclass(frozen=True)
class IntervalMarker:
    ordinal_in_interval_1idx: int
    seconds_since_previous_event: float | None
    timestamp: str
    raw_type: str
    payload_preview: str


@dataclass(frozen=True)
class RawTraceEvent:
    timestamp: datetime
    raw_type: str
    payload: object
    line_number: int
    output_lines: tuple[str, ...] = ()


def _submit_bounded_interval(
    events: tuple[CmEvent, ...], submit_index_0idx: int
) -> tuple[CmEvent, ...]:
    current: list[CmEvent] = []
    seen_submit_index = 0

    for event in events:
        current.append(event)
        if event.raw_type != "submit":
            continue
        if seen_submit_index == submit_index_0idx:
            return tuple(current)
        seen_submit_index += 1
        current = []

    raise V6TrajectoryError(f"Submit index not found in CodeMirror log: {submit_index_0idx}")


def load_submit_bounded_interval(
    *,
    codemirror_log_path: Path,
    submit_index_0idx: int,
) -> tuple[CmEvent, ...]:
    return _submit_bounded_interval(parse_codemirror_log(codemirror_log_path), submit_index_0idx)


def _seconds_since_previous(current: datetime, previous: datetime | None) -> float | None:
    if previous is None:
        return None
    return round((current - previous).total_seconds(), 3)


def build_recent_change_microtrace(
    *,
    interval_events: tuple[CmEvent, ...],
    limit: int = 20,
) -> tuple[ChangeMicroEvent, ...]:
    if limit <= 0:
        raise V6TrajectoryError(f"Recent microtrace limit must be positive, got {limit}")

    change_events = [event for event in interval_events if event.raw_type == "change"]
    recent = change_events[-limit:]
    rendered: list[ChangeMicroEvent] = []
    previous_timestamp: datetime | None = None

    for ordinal_1idx, event in enumerate(recent, start=len(change_events) - len(recent) + 1):
        if not isinstance(event.payload, dict):
            raise V6TrajectoryError("Change payload is not a dict")
        origin = event.payload.get("origin")
        if not isinstance(origin, str) or not origin:
            raise V6TrajectoryError("Change event is missing a valid origin")
        rendered.append(
            ChangeMicroEvent(
                ordinal_in_interval_1idx=ordinal_1idx,
                seconds_since_previous_event=_seconds_since_previous(
                    event.timestamp, previous_timestamp
                ),
                timestamp=event.timestamp.isoformat(),
                from_line_0idx=int(event.payload["from"]["line"]),
                from_ch_0idx=int(event.payload["from"]["ch"]),
                to_line_0idx=int(event.payload["to"]["line"]),
                to_ch_0idx=int(event.payload["to"]["ch"]),
                inserted_text="\n".join(event.payload["text"]),
                removed_text="\n".join(event.payload["removed"]),
                origin=origin,
            )
        )
        previous_timestamp = event.timestamp

    return tuple(rendered)


def build_interval_markers(interval_events: tuple[CmEvent, ...]) -> tuple[IntervalMarker, ...]:
    markers: list[IntervalMarker] = []
    previous_timestamp: datetime | None = None

    for ordinal_1idx, event in enumerate(interval_events, start=1):
        if event.raw_type not in {"saida_testar", "submit"} and not event.raw_type.startswith(
            "tab-click"
        ):
            previous_timestamp = event.timestamp
            continue
        payload_preview = str(event.payload)
        markers.append(
            IntervalMarker(
                ordinal_in_interval_1idx=ordinal_1idx,
                seconds_since_previous_event=_seconds_since_previous(
                    event.timestamp, previous_timestamp
                ),
                timestamp=event.timestamp.isoformat(),
                raw_type=event.raw_type,
                payload_preview=payload_preview[:160],
            )
        )
        previous_timestamp = event.timestamp

    return tuple(markers)


def microtrace_to_prompt_dict(events: tuple[ChangeMicroEvent, ...]) -> list[dict]:
    return [
        {
            "ordinal_in_interval_1idx": event.ordinal_in_interval_1idx,
            "seconds_since_previous_event": event.seconds_since_previous_event,
            "timestamp": event.timestamp,
            "from": {"line": event.from_line_0idx, "ch": event.from_ch_0idx},
            "to": {"line": event.to_line_0idx, "ch": event.to_ch_0idx},
            "inserted_text": event.inserted_text,
            "removed_text": event.removed_text,
            "origin": event.origin,
        }
        for event in events
    ]


def markers_to_prompt_dict(markers: tuple[IntervalMarker, ...]) -> list[dict]:
    return [
        {
            "ordinal_in_interval_1idx": marker.ordinal_in_interval_1idx,
            "seconds_since_previous_event": marker.seconds_since_previous_event,
            "timestamp": marker.timestamp,
            "raw_type": marker.raw_type,
            "payload_preview": marker.payload_preview,
        }
        for marker in markers
    ]


def _parse_raw_trace_line(raw_line: str, line_number: int) -> tuple[datetime, str, str] | None:
    stripped = raw_line.strip()
    if not stripped:
        return None
    parts = stripped.split("#", 2)
    if len(parts) < 2:
        return None
    try:
        timestamp = parse_cm_timestamp(parts[0])
    except ValueError:
        return None
    raw_type = parts[1]
    payload_raw = parts[2] if len(parts) > 2 else ""
    return (timestamp, raw_type, payload_raw)


def parse_raw_trace_events_with_output(path: Path) -> tuple[RawTraceEvent, ...]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    events: list[RawTraceEvent] = []
    line_index = 0

    while line_index < len(lines):
        raw_line = lines[line_index]
        parsed = _parse_raw_trace_line(raw_line, line_index + 1)
        if parsed is None:
            if raw_line.strip() == "":
                line_index += 1
                continue
            raise V6TrajectoryError(
                f"Malformed CodeMirror raw line outside saida_testar output at {path}:{line_index + 1}"
            )

        timestamp, raw_type, payload_raw = parsed
        payload: object = payload_raw
        if raw_type == "change":
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError as exc:
                raise V6TrajectoryError(
                    f"Malformed change payload at {path}:{line_index + 1}"
                ) from exc

        output_lines: list[str] = []
        if raw_type == "saida_testar":
            lookahead = line_index + 1
            while lookahead < len(lines):
                maybe_event = _parse_raw_trace_line(lines[lookahead], lookahead + 1)
                if maybe_event is not None:
                    break
                output_lines.append(lines[lookahead])
                lookahead += 1
            line_index = lookahead - 1

        events.append(
            RawTraceEvent(
                timestamp=timestamp,
                raw_type=raw_type,
                payload=payload,
                line_number=line_index + 1,
                output_lines=tuple(output_lines),
            )
        )
        line_index += 1

    if not events:
        raise V6TrajectoryError(f"CodeMirror log produced no parseable raw events: {path}")
    return tuple(events)


def load_raw_submit_bounded_interval(
    *,
    codemirror_log_path: Path,
    submit_index_0idx: int,
) -> tuple[RawTraceEvent, ...]:
    current: list[RawTraceEvent] = []
    seen_submit_index = 0

    for event in parse_raw_trace_events_with_output(codemirror_log_path):
        current.append(event)
        if event.raw_type != "submit":
            continue
        if seen_submit_index == submit_index_0idx:
            return tuple(current)
        seen_submit_index += 1
        current = []

    raise V6TrajectoryError(
        f"Submit index not found in raw CodeMirror interval parser: {submit_index_0idx}"
    )


def _normalize_keyhandled(payload: object) -> str:
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return payload
        if not isinstance(decoded, str):
            raise V6TrajectoryError("Decoded keyHandled payload is not a string")
        return decoded
    raise V6TrajectoryError("keyHandled payload is not a string")


def _semantic_event_dict(
    *,
    event_index_1idx: int,
    raw_ordinal_1idx: int,
    previous_timestamp: datetime | None,
    event_timestamp: datetime,
    event_type: str,
    fields: dict,
) -> dict:
    return {
        "event_index_1idx": event_index_1idx,
        "raw_ordinal_in_interval_1idx": raw_ordinal_1idx,
        "seconds_since_previous_semantic_event": _seconds_since_previous(
            event_timestamp, previous_timestamp
        ),
        "timestamp": event_timestamp.isoformat(),
        "event_type": event_type,
        **fields,
    }


def _truncate_output_lines(
    output_lines: list[str],
    output_line_limit: int | None,
) -> tuple[list[str], int]:
    if output_line_limit is None or len(output_lines) <= output_line_limit:
        return output_lines, 0
    if output_line_limit <= 2:
        return output_lines[:output_line_limit], len(output_lines) - output_line_limit
    head_count = max(1, (output_line_limit - 1) // 2)
    tail_count = output_line_limit - head_count - 1
    truncated_count = len(output_lines) - (head_count + tail_count)
    return (
        [
            *output_lines[:head_count],
            f"... truncated {truncated_count} stdout lines ...",
            *output_lines[-tail_count:],
        ],
        truncated_count,
    )


def build_semantic_event_tape(
    *,
    interval_events: tuple[RawTraceEvent, ...],
    idle_gap_seconds: float = 30.0,
    include_keyhandled: bool = False,
    include_navigation: bool = True,
    output_line_limit: int | None = None,
) -> list[dict]:
    if idle_gap_seconds <= 0:
        raise V6TrajectoryError(f"Idle-gap threshold must be positive, got {idle_gap_seconds}")

    semantic_events: list[dict] = []
    previous_semantic_timestamp: datetime | None = None
    previous_semantic_type: str | None = None
    event_index = 1
    meaningful_keyhandled = {"Enter", "Backspace", "Tab", "Delete", "Insert"}

    def _append_idle_gap_if_needed(
        *,
        raw_ordinal_1idx: int,
        current_timestamp: datetime,
        next_event_type: str,
    ) -> None:
        nonlocal event_index
        if previous_semantic_timestamp is None:
            return
        gap_seconds = round((current_timestamp - previous_semantic_timestamp).total_seconds(), 3)
        if gap_seconds < idle_gap_seconds:
            return
        semantic_events.append(
            {
                "event_index_1idx": event_index,
                "raw_ordinal_in_interval_1idx": raw_ordinal_1idx,
                "event_type": "idle_gap",
                "seconds_since_previous_semantic_event": gap_seconds,
                "starts_at": previous_semantic_timestamp.isoformat(),
                "ends_at": current_timestamp.isoformat(),
                "previous_event_type": previous_semantic_type,
                "next_event_type": next_event_type,
            }
        )
        event_index += 1

    for raw_ordinal_1idx, event in enumerate(interval_events, start=1):
        if event.raw_type == "change":
            if not isinstance(event.payload, dict):
                raise V6TrajectoryError("Change payload is not a dict in semantic tape builder")
            origin = event.payload.get("origin")
            if not isinstance(origin, str) or not origin:
                raise V6TrajectoryError("Change event is missing a valid origin")
            _append_idle_gap_if_needed(
                raw_ordinal_1idx=raw_ordinal_1idx,
                current_timestamp=event.timestamp,
                next_event_type="change",
            )
            semantic_events.append(
                _semantic_event_dict(
                    event_index_1idx=event_index,
                    raw_ordinal_1idx=raw_ordinal_1idx,
                    previous_timestamp=previous_semantic_timestamp,
                    event_timestamp=event.timestamp,
                    event_type="change",
                    fields={
                        "from": {
                            "line": int(event.payload["from"]["line"]),
                            "ch": int(event.payload["from"]["ch"]),
                        },
                        "to": {
                            "line": int(event.payload["to"]["line"]),
                            "ch": int(event.payload["to"]["ch"]),
                        },
                        "inserted_text": "\n".join(event.payload["text"]),
                        "removed_text": "\n".join(event.payload["removed"]),
                        "origin": origin,
                    },
                )
            )
            previous_semantic_timestamp = event.timestamp
            previous_semantic_type = "change"
            event_index += 1
            continue

        if event.raw_type == "saida_testar":
            if output_line_limit is not None and output_line_limit <= 0:
                raise V6TrajectoryError(
                    f"Output-line limit must be positive when provided, got {output_line_limit}"
                )
            output_lines, truncated_count = _truncate_output_lines(
                list(event.output_lines), output_line_limit
            )
            _append_idle_gap_if_needed(
                raw_ordinal_1idx=raw_ordinal_1idx,
                current_timestamp=event.timestamp,
                next_event_type="saida_testar",
            )
            semantic_events.append(
                _semantic_event_dict(
                    event_index_1idx=event_index,
                    raw_ordinal_1idx=raw_ordinal_1idx,
                    previous_timestamp=previous_semantic_timestamp,
                    event_timestamp=event.timestamp,
                    event_type="saida_testar",
                    fields={
                        "command": str(event.payload),
                        "output_lines": output_lines,
                        "truncated_output_line_count": truncated_count,
                        "output_line_limit": output_line_limit,
                    },
                )
            )
            previous_semantic_timestamp = event.timestamp
            previous_semantic_type = "saida_testar"
            event_index += 1
            continue

        if event.raw_type == "submit":
            _append_idle_gap_if_needed(
                raw_ordinal_1idx=raw_ordinal_1idx,
                current_timestamp=event.timestamp,
                next_event_type="submit",
            )
            semantic_events.append(
                _semantic_event_dict(
                    event_index_1idx=event_index,
                    raw_ordinal_1idx=raw_ordinal_1idx,
                    previous_timestamp=previous_semantic_timestamp,
                    event_timestamp=event.timestamp,
                    event_type="submit",
                    fields={"feedback": str(event.payload)},
                )
            )
            previous_semantic_timestamp = event.timestamp
            previous_semantic_type = "submit"
            event_index += 1
            continue

        if event.raw_type == "kill_program":
            raw_value = str(event.payload)
            lowered = raw_value.strip().lower()
            if lowered not in {"true", "false", ""}:
                raise V6TrajectoryError(f"Unexpected kill_program payload: {event.payload!r}")
            _append_idle_gap_if_needed(
                raw_ordinal_1idx=raw_ordinal_1idx,
                current_timestamp=event.timestamp,
                next_event_type="kill_program",
            )
            semantic_events.append(
                _semantic_event_dict(
                    event_index_1idx=event_index,
                    raw_ordinal_1idx=raw_ordinal_1idx,
                    previous_timestamp=previous_semantic_timestamp,
                    event_timestamp=event.timestamp,
                    event_type="kill_program",
                    fields={
                        "value": lowered == "true",
                        "raw_value": raw_value,
                    },
                )
            )
            previous_semantic_timestamp = event.timestamp
            previous_semantic_type = "kill_program"
            event_index += 1
            continue

        if include_keyhandled and event.raw_type == "keyHandled":
            key = _normalize_keyhandled(event.payload)
            if key in meaningful_keyhandled:
                _append_idle_gap_if_needed(
                    raw_ordinal_1idx=raw_ordinal_1idx,
                    current_timestamp=event.timestamp,
                    next_event_type="keyHandled",
                )
                semantic_events.append(
                    _semantic_event_dict(
                        event_index_1idx=event_index,
                        raw_ordinal_1idx=raw_ordinal_1idx,
                        previous_timestamp=previous_semantic_timestamp,
                        event_timestamp=event.timestamp,
                        event_type="keyHandled",
                        fields={"key": key},
                    )
                )
                previous_semantic_timestamp = event.timestamp
                previous_semantic_type = "keyHandled"
                event_index += 1
            continue

        if include_navigation and event.raw_type.startswith("tab-click:"):
            _append_idle_gap_if_needed(
                raw_ordinal_1idx=raw_ordinal_1idx,
                current_timestamp=event.timestamp,
                next_event_type="tab_click",
            )
            semantic_events.append(
                _semantic_event_dict(
                    event_index_1idx=event_index,
                    raw_ordinal_1idx=raw_ordinal_1idx,
                    previous_timestamp=previous_semantic_timestamp,
                    event_timestamp=event.timestamp,
                    event_type="tab_click",
                    fields={"target": event.raw_type.split(":", 1)[1]},
                )
            )
            previous_semantic_timestamp = event.timestamp
            previous_semantic_type = "tab_click"
            event_index += 1

    return semantic_events


def render_transition_semantic_tape(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index_0idx: int,
    idle_gap_seconds: float = 30.0,
    include_keyhandled: bool = False,
    include_navigation: bool = True,
    output_line_limit: int | None = None,
) -> str:
    payload_path = Path(
        f"data/v6/hq144_transition_payloads/{class_id}/{assessment_id}/{exercise_id}/{student_id}/{transition_index_0idx}/payload.json"
    )
    if not payload_path.exists():
        raise V6TrajectoryError(f"Payload not found: {payload_path}")

    payload = json.loads(payload_path.read_text())
    log_path = Path(
        f"../tracer/2024-1/{class_id}/users/{student_id}/codemirror/{assessment_id}_{exercise_id}.log"
    )
    raw_interval = load_raw_submit_bounded_interval(
        codemirror_log_path=log_path,
        submit_index_0idx=transition_index_0idx,
    )
    semantic_tape = build_semantic_event_tape(
        interval_events=raw_interval,
        idle_gap_seconds=idle_gap_seconds,
        include_keyhandled=include_keyhandled,
        include_navigation=include_navigation,
        output_line_limit=output_line_limit,
    )
    tape_payload = build_attempt_semantic_tape_payload(
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        student_id=student_id,
        transition_index_0idx=transition_index_0idx,
        current_trace_card=payload["attempt_n"]["trace_card"],
        raw_interval=raw_interval,
        semantic_tape=semantic_tape,
    )
    return json.dumps(tape_payload.model_dump(by_alias=True), ensure_ascii=True, indent=2)


def build_attempt_semantic_tape_payload(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index_0idx: int,
    current_trace_card: dict,
    raw_interval: tuple[RawTraceEvent, ...],
    semantic_tape: list[dict],
) -> AttemptNSemanticTape:
    return AttemptNSemanticTape.model_validate(
        {
            "schema_version": "v6_1_attempt_semantic_tape_v1",
            "transition": {
                "class_id": class_id,
                "assessment_id": assessment_id,
                "exercise_id": exercise_id,
                "student_id": student_id,
                "transition_index_0idx": transition_index_0idx,
            },
            "current_trace_card": current_trace_card,
            "semantic_tape_summary": {
                "raw_interval_event_count": len(raw_interval),
                "semantic_event_count": len(semantic_tape),
                "change_event_count": sum(
                    1 for event in semantic_tape if event["event_type"] == "change"
                ),
                "saida_testar_count": sum(
                    1 for event in semantic_tape if event["event_type"] == "saida_testar"
                ),
                "submit_count": sum(
                    1 for event in semantic_tape if event["event_type"] == "submit"
                ),
                "kill_program_count": sum(
                    1 for event in semantic_tape if event["event_type"] == "kill_program"
                ),
                "idle_gap_count": sum(
                    1 for event in semantic_tape if event["event_type"] == "idle_gap"
                ),
            },
            "semantic_event_tape": semantic_tape,
        }
    )


def render_transition_trace_example(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index_0idx: int,
    limit: int = 12,
) -> str:
    payload_path = Path(
        f"data/v6/hq144_transition_payloads/{class_id}/{assessment_id}/{exercise_id}/{student_id}/{transition_index_0idx}/payload.json"
    )
    if not payload_path.exists():
        raise V6TrajectoryError(f"Payload not found: {payload_path}")

    payload = json.loads(payload_path.read_text())
    log_path = Path(
        f"../tracer/2024-1/{class_id}/users/{student_id}/codemirror/{assessment_id}_{exercise_id}.log"
    )
    interval_events = load_submit_bounded_interval(
        codemirror_log_path=log_path,
        submit_index_0idx=transition_index_0idx,
    )

    recent_microtrace = build_recent_change_microtrace(interval_events=interval_events, limit=limit)
    markers = build_interval_markers(interval_events)

    block = {
        "transition": {
            "class_id": class_id,
            "assessment_id": assessment_id,
            "exercise_id": exercise_id,
            "student_id": student_id,
            "transition_index_0idx": transition_index_0idx,
        },
        "current_trace_card": payload["attempt_n"]["trace_card"],
        "recent_change_microtrace": microtrace_to_prompt_dict(recent_microtrace),
        "interval_markers": markers_to_prompt_dict(markers),
    }
    return json.dumps(block, ensure_ascii=True, indent=2)
