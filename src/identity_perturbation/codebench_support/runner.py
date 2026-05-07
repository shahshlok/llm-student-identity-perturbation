from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .alignment import AlignmentError, align_transition
from .codemirror import (
    infer_initial_code,
    lines_touched_bucket_3way,
    parse_codemirror_log,
    replay_trace,
)
from .executions import (
    ExecutionParseError,
    build_transitions,
    next_test_outcome,
    parse_execution_log,
)
from .focus_region import FocusRegionError, first_focus_region_3way
from .models import CohortWindow, ExecutionTransition, ReplayTrace
from .payload import build_payload, failed_test_indices_0idx, pass_count
from .prompting import build_system_prompt, build_user_prompt

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT.parent / "tracer" / "2024-1"
DEFAULT_OUT_ROOT = ROOT / "data" / "v5" / "slices"
HEAD_LABELS = {
    "first_focus_region_3way": (
        "output_region",
        "conditional_region",
        "loop_region",
    ),
    "lines_touched_bucket_3way": (
        "local_1_to_2_lines",
        "regional_3_to_5_lines",
        "broad_6_plus_lines",
    ),
    "next_test_outcome": (
        "all_fail",
        "mixed",
        "all_pass",
    ),
}


class SliceBuildError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one strict v5 cohort slice artifact bundle."
    )
    parser.add_argument("--class-id", required=True, help="CodeBench class id")
    parser.add_argument("--assessment-id", required=True, help="Assessment id")
    parser.add_argument("--exercise-id", required=True, help="Exercise id")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Raw dataset root containing 2024-1/<class_id>/...",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Artifact root for v5 slice outputs",
    )
    parser.add_argument(
        "--max-students",
        type=int,
        default=None,
        help="Optional cap for debugging on the first N candidate students",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=10,
        help="Representative cards to include in the payload",
    )
    parser.add_argument(
        "--student-id",
        action="append",
        default=[],
        help="Optional student filter for strict debugging runs; may be repeated",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial alignments for debugging only. Primary v5 analysis should leave this off.",
    )
    return parser.parse_args()


def parse_assessment_file(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SliceBuildError(f"Assessment file not found: {path}")

    title = None
    exercise_ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("---- assessment title:"):
            title = line.split(":", 1)[1].strip()
        if line.startswith("---- exercise "):
            exercise_ids.append(line.split(":", 1)[1].strip())

    if title is None:
        raise SliceBuildError(f"Assessment title missing in: {path}")
    if not exercise_ids:
        raise SliceBuildError(f"Assessment exercise list missing in: {path}")

    return {
        "assessment_id": path.stem,
        "title": title,
        "exercise_ids": tuple(exercise_ids),
    }


def write_json(path: Path, payload: object) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(text, encoding="utf-8")


def _candidate_logs(
    class_root: Path,
    assessment_id: str,
    exercise_id: str,
    allowed_student_ids: frozenset[str] | None = None,
) -> tuple[tuple[str, Path, Path], ...]:
    users_root = class_root / "users"
    if not users_root.exists():
        raise SliceBuildError(f"Users directory not found: {users_root}")

    candidates: list[tuple[str, Path, Path]] = []
    for user_dir in sorted(users_root.iterdir(), key=lambda path: path.name):
        if not user_dir.is_dir():
            continue
        if allowed_student_ids is not None and user_dir.name not in allowed_student_ids:
            continue
        exec_path = user_dir / "executions" / f"{assessment_id}_{exercise_id}.log"
        if not exec_path.exists():
            continue
        cm_path = user_dir / "codemirror" / f"{assessment_id}_{exercise_id}.log"
        candidates.append((user_dir.name, exec_path, cm_path))

    if not candidates:
        raise SliceBuildError(
            f"No execution logs found for class={class_root.name} assessment={assessment_id} "
            f"exercise={exercise_id}"
        )
    return tuple(candidates)


def build_observed_distributions(window_records: list[dict[str, object]]) -> dict[str, object]:
    if not window_records:
        raise SliceBuildError("Cannot build observed distributions for an empty slice")

    by_head: dict[str, object] = {}
    n_windows = len(window_records)
    for head, labels in HEAD_LABELS.items():
        counts = Counter(str(record[head]) for record in window_records)
        by_head[head] = {
            "counts": {label: counts.get(label, 0) for label in labels},
            "distribution": {label: counts.get(label, 0) / n_windows for label in labels},
        }
    return {
        "n_windows": n_windows,
        "by_head": by_head,
    }


def build_student_windows(
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    exec_path: Path,
    cm_path: Path,
    allow_partial: bool = False,
) -> tuple[list[CohortWindow], list[dict[str, object]]]:
    attempts, trace, transitions = load_student_transition_context(
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        student_id=student_id,
        exec_path=exec_path,
        cm_path=cm_path,
    )
    if len(attempts) < 2:
        return [], []

    windows: list[CohortWindow] = []
    records: list[dict[str, object]] = []
    for transition_index, transition in enumerate(transitions):
        window, record = build_transition_window_record(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition_index=transition_index,
            transition=transition,
            trace=trace,
            allow_partial=allow_partial,
        )
        windows.append(window)
        records.append(record)
    return windows, records


def load_student_transition_context(
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    exec_path: Path,
    cm_path: Path,
) -> tuple[tuple, ReplayTrace | None, tuple[ExecutionTransition, ...]]:
    try:
        attempts = parse_execution_log(exec_path)
    except ExecutionParseError as exc:
        raise SliceBuildError(
            f"Execution parsing failed for student={student_id} class={class_id} "
            f"assessment={assessment_id} exercise={exercise_id}: {exc}"
        ) from exc
    if len(attempts) < 2:
        return attempts, None, ()
    if not cm_path.exists():
        raise SliceBuildError(f"Missing CodeMirror log for student {student_id}: {cm_path}")

    try:
        cm_events = parse_codemirror_log(cm_path)
        initial_code = infer_initial_code(cm_events, attempts[0].code)
        trace = replay_trace(cm_events, initial_code=initial_code)
    except Exception as exc:
        raise SliceBuildError(
            f"CodeMirror parsing/replay failed for student={student_id} class={class_id} "
            f"assessment={assessment_id} exercise={exercise_id}: {exc}"
        ) from exc
    transitions = build_transitions(attempts)
    return attempts, trace, transitions


def build_transition_window_record(
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index: int,
    transition: ExecutionTransition,
    trace: ReplayTrace | None,
    allow_partial: bool = False,
) -> tuple[CohortWindow, dict[str, object]]:
    if trace is None:
        raise SliceBuildError(
            f"Missing replay trace for student={student_id} class={class_id} "
            f"assessment={assessment_id} exercise={exercise_id}"
        )

    try:
        alignment = align_transition(
            transition,
            trace.snapshots,
            trailing_code=trace.final_code,
            trailing_changes=trace.trailing_changes,
            allow_partial=allow_partial,
        )
    except AlignmentError as exc:
        raise SliceBuildError(
            f"Alignment failed for student={student_id} class={class_id} "
            f"assessment={assessment_id} exercise={exercise_id} "
            f"transition_index_0idx={transition_index}: {exc}"
        ) from exc
    try:
        focus_region = first_focus_region_3way(
            transition.attempt_n.code,
            alignment.first_change_line_0idx,
        )
    except FocusRegionError as exc:
        raise SliceBuildError(
            f"Focus-region mapping failed for student={student_id} class={class_id} "
            f"assessment={assessment_id} exercise={exercise_id} "
            f"transition_index_0idx={transition_index}: {exc}"
        ) from exc

    window = CohortWindow(
        student_id=student_id,
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        transition=transition,
    )
    record = {
        "student_id": student_id,
        "class_id": class_id,
        "assessment_id": assessment_id,
        "exercise_id": exercise_id,
        "transition_index_0idx": transition_index,
        "attempt_n_index_0idx": transition.attempt_n.attempt_index_0idx,
        "attempt_n1_index_0idx": transition.attempt_n1.attempt_index_0idx,
        "attempt_n_timestamp": transition.attempt_n.timestamp,
        "attempt_n1_timestamp": transition.attempt_n1.timestamp,
        "alignment_status": alignment.status,
        "snap_n_index_0idx": alignment.snap_n_index,
        "snap_n1_index_0idx": alignment.snap_n1_index,
        "submit_n_timestamp": alignment.submit_n_timestamp,
        "submit_n1_timestamp": alignment.submit_n1_timestamp,
        "first_change_line_0idx": alignment.first_change_line_0idx,
        "lines_touched_0idx": list(alignment.lines_touched_0idx),
        "first_focus_region_3way": focus_region,
        "lines_touched_bucket_3way": lines_touched_bucket_3way(alignment.changes_between),
        "next_test_outcome": next_test_outcome(transition.attempt_n1),
        "attempt_n_pass_count": pass_count(transition.attempt_n),
        "attempt_n1_pass_count": pass_count(transition.attempt_n1),
        "attempt_n_failed_test_indices_0idx": list(failed_test_indices_0idx(transition.attempt_n)),
        "attempt_n1_failed_test_indices_0idx": list(
            failed_test_indices_0idx(transition.attempt_n1)
        ),
    }
    return window, record


def build_slice_artifacts(
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    data_root: Path,
    out_root: Path,
    max_students: int | None = None,
    max_cards: int = 10,
    student_ids: tuple[str, ...] = (),
    allow_partial: bool = False,
) -> dict[str, object]:
    class_root = data_root / class_id
    if not class_root.exists():
        raise SliceBuildError(f"Class directory not found: {class_root}")

    assessment_meta = parse_assessment_file(class_root / "assessments" / f"{assessment_id}.data")
    if exercise_id not in assessment_meta["exercise_ids"]:
        raise SliceBuildError(
            f"Exercise {exercise_id} not listed in assessment {assessment_id} for class {class_id}"
        )

    allowed_student_ids = frozenset(student_ids) if student_ids else None
    candidates = list(
        _candidate_logs(
            class_root,
            assessment_id,
            exercise_id,
            allowed_student_ids=allowed_student_ids,
        )
    )
    if max_students is not None:
        candidates = candidates[:max_students]
    if not candidates:
        raise SliceBuildError("Candidate list is empty after applying --max-students")

    windows: list[CohortWindow] = []
    window_records: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for student_id, exec_path, cm_path in candidates:
        student_windows, student_records = build_student_windows(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            exec_path=exec_path,
            cm_path=cm_path,
            allow_partial=allow_partial,
        )
        if not student_windows:
            exclusions.append(
                {
                    "student_id": student_id,
                    "reason": "insufficient_attempts_for_transition",
                }
            )
            continue
        windows.extend(student_windows)
        window_records.extend(student_records)

    if not windows:
        raise SliceBuildError("Slice produced zero aligned windows")

    payload = build_payload(tuple(windows), max_cards=max_cards)
    payload["slice_header"]["assessment_title"] = assessment_meta["title"]
    observed = build_observed_distributions(window_records)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(payload)

    slice_dir = out_root / class_id / assessment_id / exercise_id
    slice_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "v5_version": "v5",
        "class_id": class_id,
        "assessment_id": assessment_id,
        "exercise_id": exercise_id,
        "assessment_title": assessment_meta["title"],
        "alignment_policy": "allow_partial_debug" if allow_partial else "full_match_only",
        "data_root": str(data_root),
        "candidate_student_count": len(candidates),
        "included_student_count": len({window.student_id for window in windows}),
        "excluded_student_count": len(exclusions),
        "window_count": len(windows),
        "artifact_dir": str(slice_dir),
    }

    write_json(slice_dir / "manifest.json", manifest)
    write_json(slice_dir / "exclusions.json", exclusions)
    write_json(slice_dir / "observed_labels.json", observed)
    write_json(slice_dir / "window_records.json", window_records)
    write_json(slice_dir / "payload.json", payload)
    write_text(slice_dir / "system_prompt.txt", system_prompt)
    write_text(slice_dir / "user_prompt.txt", user_prompt)

    return {
        "manifest": manifest,
        "payload": payload,
        "observed_labels": observed,
        "window_records": window_records,
        "exclusions": exclusions,
        "paths": {
            "slice_dir": str(slice_dir),
            "manifest": str(slice_dir / "manifest.json"),
            "payload": str(slice_dir / "payload.json"),
            "observed_labels": str(slice_dir / "observed_labels.json"),
            "window_records": str(slice_dir / "window_records.json"),
            "system_prompt": str(slice_dir / "system_prompt.txt"),
            "user_prompt": str(slice_dir / "user_prompt.txt"),
        },
    }


def main() -> int:
    args = parse_args()
    artifacts = build_slice_artifacts(
        class_id=args.class_id,
        assessment_id=args.assessment_id,
        exercise_id=args.exercise_id,
        data_root=args.data_root,
        out_root=args.out_root,
        max_students=args.max_students,
        max_cards=args.max_cards,
        student_ids=tuple(args.student_id),
        allow_partial=args.allow_partial,
    )
    print(json.dumps(artifacts["manifest"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
