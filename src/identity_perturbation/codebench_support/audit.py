from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .runner import (
    ROOT,
    SliceBuildError,
    _candidate_logs,
    build_transition_window_record,
    load_student_transition_context,
    parse_assessment_file,
)

DEFAULT_DATA_ROOT = ROOT.parent / "tracer" / "2024-1"
DEFAULT_OUT_ROOT = ROOT / "data" / "v5" / "audits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one strict v5 cohort slice before prompt building."
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
        help="Artifact root for v5 audit outputs",
    )
    parser.add_argument(
        "--max-students",
        type=int,
        default=None,
        help="Optional cap for debugging on the first N candidate students",
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


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown"


def _failure_parts(exc: SliceBuildError) -> tuple[str, str, str]:
    message = str(exc)
    stage_map = {
        "Execution parsing failed": "execution",
        "CodeMirror parsing/replay failed": "codemirror",
        "Alignment failed": "alignment",
        "Focus-region mapping failed": "focus_region",
        "Missing replay trace": "alignment",
    }
    for prefix, stage in stage_map.items():
        if message.startswith(prefix):
            detail = message.split(": ", 1)[1] if ": " in message else message
            return stage, _slug(detail), detail
    return "unknown", _slug(message), message


def _write_json(path: Path, payload: object) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def audit_slice(
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    data_root: Path,
    out_root: Path,
    max_students: int | None = None,
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
        raise SliceBuildError("Candidate list is empty after applying filters")

    student_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    candidate_transition_count = 0
    included_transition_count = 0

    for student_id, exec_path, cm_path in candidates:
        try:
            attempts, trace, transitions = load_student_transition_context(
                class_id=class_id,
                assessment_id=assessment_id,
                exercise_id=exercise_id,
                student_id=student_id,
                exec_path=exec_path,
                cm_path=cm_path,
            )
        except SliceBuildError as exc:
            stage, reason_code, reason_message = _failure_parts(exc)
            reason_counts[reason_code] += 1
            student_rows.append(
                {
                    "student_id": student_id,
                    "status": "excluded",
                    "reason_code": reason_code,
                    "reason_message": reason_message,
                    "reason_stage": stage,
                    "attempt_count": None,
                    "transition_count": None,
                    "included_transition_count": 0,
                    "excluded_transition_count": None,
                }
            )
            continue

        transition_count = len(transitions)
        candidate_transition_count += transition_count

        if transition_count == 0:
            reason_code = "insufficient_attempts_for_transition"
            reason_counts[reason_code] += 1
            student_rows.append(
                {
                    "student_id": student_id,
                    "status": "excluded",
                    "reason_code": reason_code,
                    "reason_message": reason_code,
                    "reason_stage": "execution",
                    "attempt_count": len(attempts),
                    "transition_count": 0,
                    "included_transition_count": 0,
                    "excluded_transition_count": 0,
                }
            )
            continue

        student_included = 0
        student_excluded = 0
        student_reason_code: str | None = None
        student_reason_message: str | None = None
        student_reason_stage: str | None = None

        for transition_index, transition in enumerate(transitions):
            try:
                _window, record = build_transition_window_record(
                    class_id=class_id,
                    assessment_id=assessment_id,
                    exercise_id=exercise_id,
                    student_id=student_id,
                    transition_index=transition_index,
                    transition=transition,
                    trace=trace,
                    allow_partial=allow_partial,
                )
            except SliceBuildError as exc:
                stage, reason_code, reason_message = _failure_parts(exc)
                reason_counts[reason_code] += 1
                student_excluded += 1
                if student_reason_code is None:
                    student_reason_code = reason_code
                    student_reason_message = reason_message
                    student_reason_stage = stage
                transition_rows.append(
                    {
                        "student_id": student_id,
                        "transition_index_0idx": transition_index,
                        "status": "excluded",
                        "reason_stage": stage,
                        "reason_code": reason_code,
                        "reason_message": reason_message,
                        "attempt_n_index_0idx": transition.attempt_n.attempt_index_0idx,
                        "attempt_n1_index_0idx": transition.attempt_n1.attempt_index_0idx,
                        "alignment_status": "none",
                        "first_change_line_0idx": None,
                        "first_focus_region_3way": None,
                        "lines_touched_bucket_3way": None,
                        "next_test_outcome": None,
                    }
                )
                continue

            student_included += 1
            included_transition_count += 1
            transition_rows.append(
                {
                    "student_id": student_id,
                    "transition_index_0idx": record["transition_index_0idx"],
                    "status": "included",
                    "reason_stage": None,
                    "reason_code": None,
                    "reason_message": None,
                    "attempt_n_index_0idx": record["attempt_n_index_0idx"],
                    "attempt_n1_index_0idx": record["attempt_n1_index_0idx"],
                    "alignment_status": record["alignment_status"],
                    "first_change_line_0idx": record["first_change_line_0idx"],
                    "first_focus_region_3way": record["first_focus_region_3way"],
                    "lines_touched_bucket_3way": record["lines_touched_bucket_3way"],
                    "next_test_outcome": record["next_test_outcome"],
                }
            )

        student_status = "included"
        if student_included == 0:
            student_status = "excluded"
        elif student_excluded > 0:
            student_status = "partial"

        student_rows.append(
            {
                "student_id": student_id,
                "status": student_status,
                "reason_code": student_reason_code,
                "reason_message": student_reason_message,
                "reason_stage": student_reason_stage,
                "attempt_count": len(attempts),
                "transition_count": transition_count,
                "included_transition_count": student_included,
                "excluded_transition_count": student_excluded,
            }
        )

    included_student_count = sum(1 for row in student_rows if row["status"] != "excluded")
    excluded_student_count = sum(1 for row in student_rows if row["status"] == "excluded")
    bundle = {
        "schema_version": "v5_slice_audit_v1",
        "slice_summary": {
            "class_id": class_id,
            "assessment_id": assessment_id,
            "exercise_id": exercise_id,
            "assessment_title": assessment_meta["title"],
            "allow_partial": allow_partial,
            "candidate_student_count": len(candidates),
            "included_student_count": included_student_count,
            "excluded_student_count": excluded_student_count,
            "candidate_transition_count": candidate_transition_count,
            "included_transition_count": included_transition_count,
            "excluded_transition_count": candidate_transition_count - included_transition_count,
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "students": student_rows,
        "transitions": transition_rows,
    }

    slice_dir = out_root / class_id / assessment_id / exercise_id
    slice_dir.mkdir(parents=True, exist_ok=False)
    _write_json(slice_dir / "audit.json", bundle)
    return {
        "bundle": bundle,
        "paths": {
            "slice_dir": str(slice_dir),
            "audit": str(slice_dir / "audit.json"),
        },
    }


def main() -> int:
    args = parse_args()
    result = audit_slice(
        class_id=args.class_id,
        assessment_id=args.assessment_id,
        exercise_id=args.exercise_id,
        data_root=args.data_root,
        out_root=args.out_root,
        max_students=args.max_students,
        student_ids=tuple(args.student_id),
        allow_partial=args.allow_partial,
    )
    print(json.dumps(result["bundle"]["slice_summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
