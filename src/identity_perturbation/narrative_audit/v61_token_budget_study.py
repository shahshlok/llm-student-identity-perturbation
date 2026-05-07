from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from identity_perturbation.narrative_audit.trajectory import (
    RawTraceEvent,
    V6TrajectoryError,
    parse_raw_trace_events_with_output,
)

DEFAULT_DATA_ROOT = Path("2024-1")
DEFAULT_OUT_ROOT = Path("data/v61/token_budget_study/two_level_raw_segments")
DEFAULT_MODEL = "gpt-5.4"
PERCENTILES = (90, 95, 99)
ANCHOR_TYPES = {"saida_testar", "submit"}


class TokenBudgetStudyError(RuntimeError):
    pass


@dataclass(frozen=True)
class SegmentMetadata:
    log_path: str
    class_id: str
    assessment_id: str
    exercise_id: str
    student_id: str
    global_segment_index_0idx: int
    attempt_index_0idx: int
    segment_index_in_attempt_0idx: int
    start_boundary_kind: str
    start_boundary_timestamp: str | None
    start_boundary_line_number: int | None
    end_anchor_type: str
    end_anchor_timestamp: str
    end_anchor_line_number: int
    change_count: int
    end_anchor_output_line_count: int
    end_anchor_payload_char_count: int


@dataclass(frozen=True)
class RealizedSegment:
    metadata: SegmentMetadata
    change_events: tuple[RawTraceEvent, ...]
    end_anchor_event: RawTraceEvent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def _nearest_rank(sorted_values: list[int], percentile: int) -> int:
    if not sorted_values:
        raise TokenBudgetStudyError("Cannot compute percentile on an empty value list")
    rank = math.ceil((percentile / 100.0) * len(sorted_values))
    index = max(0, min(len(sorted_values) - 1, rank - 1))
    return sorted_values[index]


def _iter_codemirror_logs(data_root: Path) -> Iterable[Path]:
    if not data_root.exists():
        raise TokenBudgetStudyError(f"Data root does not exist: {data_root}")
    yield from sorted(data_root.glob("*/users/*/codemirror/*.log"))


def _parse_log_path(log_path: Path, data_root: Path) -> tuple[str, str, str, str]:
    relative = log_path.relative_to(data_root)
    parts = relative.parts
    if len(parts) != 5:
        raise TokenBudgetStudyError(f"Unexpected CodeMirror path shape: {log_path}")
    class_id = parts[0]
    student_id = parts[2]
    stem = log_path.stem
    if "_" not in stem:
        raise TokenBudgetStudyError(f"Unexpected CodeMirror filename: {log_path.name}")
    assessment_id, exercise_id = stem.split("_", 1)
    return class_id, assessment_id, exercise_id, student_id


def _iter_segment_metadata(log_path: Path, data_root: Path) -> Iterable[SegmentMetadata]:
    class_id, assessment_id, exercise_id, student_id = _parse_log_path(log_path, data_root)
    events = parse_raw_trace_events_with_output(log_path)

    attempt_index = 0
    segment_index_in_attempt = 0
    global_segment_index = 0
    start_boundary_kind = "attempt_start"
    start_boundary_timestamp: str | None = None
    start_boundary_line_number: int | None = None
    pending_changes: list[RawTraceEvent] = []

    for event in events:
        if event.raw_type == "change":
            pending_changes.append(event)
            continue

        if event.raw_type not in ANCHOR_TYPES:
            continue

        yield SegmentMetadata(
            log_path=str(log_path),
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            global_segment_index_0idx=global_segment_index,
            attempt_index_0idx=attempt_index,
            segment_index_in_attempt_0idx=segment_index_in_attempt,
            start_boundary_kind=start_boundary_kind,
            start_boundary_timestamp=start_boundary_timestamp,
            start_boundary_line_number=start_boundary_line_number,
            end_anchor_type=event.raw_type,
            end_anchor_timestamp=event.timestamp.isoformat(),
            end_anchor_line_number=event.line_number,
            change_count=len(pending_changes),
            end_anchor_output_line_count=len(event.output_lines),
            end_anchor_payload_char_count=len(str(event.payload)),
        )

        global_segment_index += 1
        pending_changes = []
        start_boundary_kind = event.raw_type
        start_boundary_timestamp = event.timestamp.isoformat()
        start_boundary_line_number = event.line_number

        if event.raw_type == "submit":
            attempt_index += 1
            segment_index_in_attempt = 0
        else:
            segment_index_in_attempt += 1


def _realize_segment(metadata: SegmentMetadata) -> RealizedSegment:
    log_path = Path(metadata.log_path)
    events = parse_raw_trace_events_with_output(log_path)

    attempt_index = 0
    segment_index_in_attempt = 0
    global_segment_index = 0
    start_boundary_kind = "attempt_start"
    start_boundary_timestamp: str | None = None
    start_boundary_line_number: int | None = None
    pending_changes: list[RawTraceEvent] = []

    for event in events:
        if event.raw_type == "change":
            pending_changes.append(event)
            continue

        if event.raw_type not in ANCHOR_TYPES:
            continue

        candidate = SegmentMetadata(
            log_path=metadata.log_path,
            class_id=metadata.class_id,
            assessment_id=metadata.assessment_id,
            exercise_id=metadata.exercise_id,
            student_id=metadata.student_id,
            global_segment_index_0idx=global_segment_index,
            attempt_index_0idx=attempt_index,
            segment_index_in_attempt_0idx=segment_index_in_attempt,
            start_boundary_kind=start_boundary_kind,
            start_boundary_timestamp=start_boundary_timestamp,
            start_boundary_line_number=start_boundary_line_number,
            end_anchor_type=event.raw_type,
            end_anchor_timestamp=event.timestamp.isoformat(),
            end_anchor_line_number=event.line_number,
            change_count=len(pending_changes),
            end_anchor_output_line_count=len(event.output_lines),
            end_anchor_payload_char_count=len(str(event.payload)),
        )

        if candidate == metadata:
            return RealizedSegment(
                metadata=metadata,
                change_events=tuple(pending_changes),
                end_anchor_event=event,
            )

        global_segment_index += 1
        pending_changes = []
        start_boundary_kind = event.raw_type
        start_boundary_timestamp = event.timestamp.isoformat()
        start_boundary_line_number = event.line_number

        if event.raw_type == "submit":
            attempt_index += 1
            segment_index_in_attempt = 0
        else:
            segment_index_in_attempt += 1

    raise TokenBudgetStudyError(f"Failed to realize segment metadata: {metadata}")


def _change_event_to_dict(event: RawTraceEvent) -> dict:
    if not isinstance(event.payload, dict):
        raise TokenBudgetStudyError(
            f"Change event payload is not a dict at {event.line_number}: {event.payload!r}"
        )
    payload = event.payload
    origin = payload.get("origin")
    if not isinstance(origin, str) or not origin:
        raise TokenBudgetStudyError(
            f"Change event origin is missing or invalid at {event.line_number}: {payload!r}"
        )
    text = payload.get("text")
    removed = payload.get("removed")
    if not isinstance(text, list) or not all(isinstance(item, str) for item in text):
        raise TokenBudgetStudyError(
            f"Change event text is not list[str] at {event.line_number}: {payload!r}"
        )
    if not isinstance(removed, list) or not all(isinstance(item, str) for item in removed):
        raise TokenBudgetStudyError(
            f"Change event removed is not list[str] at {event.line_number}: {payload!r}"
        )
    return {
        "event_type": "change",
        "timestamp": event.timestamp.isoformat(),
        "line_number": event.line_number,
        "from": {
            "line": int(payload["from"]["line"]),
            "ch": int(payload["from"]["ch"]),
        },
        "to": {
            "line": int(payload["to"]["line"]),
            "ch": int(payload["to"]["ch"]),
        },
        "text": text,
        "removed": removed,
        "origin": origin,
    }


def _anchor_event_to_dict(event: RawTraceEvent) -> dict:
    if event.raw_type == "saida_testar":
        return {
            "item_type": "saida_testar",
            "timestamp": event.timestamp.isoformat(),
            "line_number": event.line_number,
            "command": str(event.payload),
            "output_lines": list(event.output_lines),
        }
    if event.raw_type == "submit":
        return {
            "item_type": "submit",
            "timestamp": event.timestamp.isoformat(),
            "line_number": event.line_number,
            "feedback": str(event.payload),
        }
    raise TokenBudgetStudyError(f"Unsupported anchor event type: {event.raw_type}")


def _serialize_attempt_block_excerpt(segment: RealizedSegment) -> dict:
    metadata = segment.metadata
    return {
        "schema_version": "v6_1_two_level_attempt_block_excerpt_v0",
        "source": {
            "codemirror_log_path": metadata.log_path,
            "class_id": metadata.class_id,
            "assessment_id": metadata.assessment_id,
            "exercise_id": metadata.exercise_id,
            "student_id": metadata.student_id,
            "global_segment_index_0idx": metadata.global_segment_index_0idx,
        },
        "attempt_block_excerpt": {
            "attempt_index_0idx": metadata.attempt_index_0idx,
            "segment_index_in_attempt_0idx": metadata.segment_index_in_attempt_0idx,
            "start_boundary": {
                "kind": metadata.start_boundary_kind,
                "timestamp": metadata.start_boundary_timestamp,
                "line_number": metadata.start_boundary_line_number,
            },
            "items": [
                {
                    "item_type": "edit_segment",
                    "change_event_count": metadata.change_count,
                    "changes": [_change_event_to_dict(event) for event in segment.change_events],
                },
                _anchor_event_to_dict(segment.end_anchor_event),
            ],
        },
    }


def _select_representative_segment(
    segments: list[SegmentMetadata], target_change_count: int
) -> SegmentMetadata:
    if not segments:
        raise TokenBudgetStudyError("No non-empty segments available for representative selection")
    return min(
        segments,
        key=lambda item: (
            abs(item.change_count - target_change_count),
            abs(item.end_anchor_payload_char_count),
            item.log_path,
            item.global_segment_index_0idx,
        ),
    )


def _token_counts_for_examples(example_texts: dict[str, str], model: str) -> dict[str, int]:
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        raise TokenBudgetStudyError(
            "tiktoken is not installed in the current uv environment. "
            "Install it explicitly before rerunning this study."
        ) from exc

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError as exc:
        raise TokenBudgetStudyError(
            f"tiktoken does not know how to encode model {model!r}. "
            "No fallback encoding is permitted."
        ) from exc

    return {label: len(encoding.encode(text)) for label, text in example_texts.items()}


def _classify_parse_error(exc: V6TrajectoryError) -> str:
    message = str(exc)
    if "no parseable raw events" in message:
        return "empty_log"
    if "Malformed change payload" in message:
        return "malformed_change_payload"
    if "Malformed CodeMirror raw line outside saida_testar output" in message:
        return "malformed_raw_line"
    return "other_parse_error"


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    all_segments: list[SegmentMetadata] = []
    positive_segments: list[SegmentMetadata] = []
    log_count = 0
    parseable_log_count = 0
    no_anchor_log_count = 0
    invalid_log_counts: dict[str, int] = {}
    invalid_log_examples: dict[str, list[str]] = {}

    for log_path in _iter_codemirror_logs(args.data_root):
        log_count += 1
        try:
            segment_count_before = len(all_segments)
            for segment in _iter_segment_metadata(log_path, args.data_root):
                all_segments.append(segment)
                if segment.change_count > 0:
                    positive_segments.append(segment)
            parseable_log_count += 1
            if len(all_segments) == segment_count_before:
                no_anchor_log_count += 1
        except V6TrajectoryError as exc:
            error_key = _classify_parse_error(exc)
            invalid_log_counts[error_key] = invalid_log_counts.get(error_key, 0) + 1
            invalid_log_examples.setdefault(error_key, [])
            if len(invalid_log_examples[error_key]) < 10:
                invalid_log_examples[error_key].append(f"{log_path} :: {exc}")

    if not all_segments:
        raise TokenBudgetStudyError(f"No anchor-bounded segments found under {args.data_root}")
    if not positive_segments:
        raise TokenBudgetStudyError(
            f"No non-empty anchor-bounded segments found under {args.data_root}"
        )

    sorted_positive_counts = sorted(segment.change_count for segment in positive_segments)
    thresholds = {
        f"p{percentile}": _nearest_rank(sorted_positive_counts, percentile)
        for percentile in PERCENTILES
    }

    example_payloads: dict[str, dict] = {}
    example_texts: dict[str, str] = {}
    example_manifest_entries: dict[str, dict] = {}

    for label, threshold in thresholds.items():
        selected = _select_representative_segment(positive_segments, threshold)
        realized = _realize_segment(selected)
        payload = _serialize_attempt_block_excerpt(realized)
        serialized = json.dumps(payload, ensure_ascii=True, indent=2)
        example_path = args.out_root / f"{label}_segment.json"
        example_path.write_text(serialized + "\n", encoding="utf-8")

        example_payloads[label] = payload
        example_texts[label] = serialized
        example_manifest_entries[label] = {
            "threshold_change_count": threshold,
            "selected_change_count": selected.change_count,
            "selected_segment": {
                "log_path": selected.log_path,
                "class_id": selected.class_id,
                "assessment_id": selected.assessment_id,
                "exercise_id": selected.exercise_id,
                "student_id": selected.student_id,
                "global_segment_index_0idx": selected.global_segment_index_0idx,
                "attempt_index_0idx": selected.attempt_index_0idx,
                "segment_index_in_attempt_0idx": selected.segment_index_in_attempt_0idx,
                "start_boundary_kind": selected.start_boundary_kind,
                "end_anchor_type": selected.end_anchor_type,
                "end_anchor_line_number": selected.end_anchor_line_number,
                "end_anchor_output_line_count": selected.end_anchor_output_line_count,
                "end_anchor_payload_char_count": selected.end_anchor_payload_char_count,
            },
            "serialization_metrics": {
                "char_count": len(serialized),
                "byte_count": len(serialized.encode("utf-8")),
                "line_count": serialized.count("\n") + 1,
            },
            "example_path": str(example_path),
        }

    manifest = {
        "study": {
            "schema_version": "v6_1_two_level_token_budget_study_v0",
            "data_root": str(args.data_root),
            "out_root": str(args.out_root),
            "model": args.model,
            "segment_definition": (
                "Non-empty raw change segments bounded by saida_testar or submit anchors, "
                "serialized as a two-level attempt-block excerpt with one edit_segment "
                "followed by its closing anchor."
            ),
            "percentile_method": "nearest_rank_on_non_empty_segments",
            "log_count": log_count,
            "parseable_log_count": parseable_log_count,
            "no_anchor_log_count": no_anchor_log_count,
            "invalid_log_counts": invalid_log_counts,
            "invalid_log_examples": invalid_log_examples,
            "all_segment_count": len(all_segments),
            "non_empty_segment_count": len(positive_segments),
            "thresholds": thresholds,
        },
        "examples": example_manifest_entries,
    }

    manifest_path = args.out_root / "manifest.json"

    try:
        token_counts = _token_counts_for_examples(example_texts, args.model)
    except TokenBudgetStudyError as exc:
        manifest["token_count_error"] = str(exc)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=True, indent=2))
        raise SystemExit(str(exc)) from exc

    for label, token_count in token_counts.items():
        example_manifest_entries[label]["serialization_metrics"]["token_count"] = token_count

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
