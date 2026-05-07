from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .audit import DEFAULT_DATA_ROOT, audit_slice
from .runner import ROOT, SliceBuildError, parse_assessment_file

DEFAULT_ASSESSMENT_OUT_ROOT = ROOT / "data" / "v5" / "assessment_audits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit every exercise in one assessment.")
    parser.add_argument("--class-id", required=True, help="CodeBench class id")
    parser.add_argument("--assessment-id", required=True, help="Assessment id")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Raw dataset root containing 2024-1/<class_id>/...",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_ASSESSMENT_OUT_ROOT,
        help="Artifact root for v5 assessment audit outputs",
    )
    parser.add_argument(
        "--max-students",
        type=int,
        default=None,
        help="Optional cap for debugging on the first N candidate students per exercise",
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


def _write_json(path: Path, payload: object) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def audit_assessment(
    class_id: str,
    assessment_id: str,
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
    exercise_ids = tuple(str(exercise_id) for exercise_id in assessment_meta["exercise_ids"])
    assessment_dir = out_root / class_id / assessment_id
    assessment_dir.mkdir(parents=True, exist_ok=False)

    by_exercise: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    candidate_students: set[str] = set()
    included_students: set[str] = set()
    candidate_transition_count = 0
    included_transition_count = 0
    included_exercise_count = 0

    exercise_out_root = assessment_dir / "exercises"
    exercise_out_root.mkdir(parents=False, exist_ok=False)

    for exercise_id in exercise_ids:
        result = audit_slice(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            data_root=data_root,
            out_root=exercise_out_root,
            max_students=max_students,
            student_ids=student_ids,
            allow_partial=allow_partial,
        )
        bundle = result["bundle"]
        slice_summary = bundle["slice_summary"]

        for row in bundle["students"]:
            candidate_students.add(str(row["student_id"]))
            if row["status"] != "excluded":
                included_students.add(str(row["student_id"]))

        reason_counts.update(slice_summary["reason_counts"])
        candidate_transition_count += int(slice_summary["candidate_transition_count"])
        included_transition_count += int(slice_summary["included_transition_count"])

        status = "included" if int(slice_summary["included_transition_count"]) > 0 else "excluded"
        if status == "included":
            included_exercise_count += 1

        by_exercise.append(
            {
                "exercise_id": exercise_id,
                "status": status,
                "slice_summary": slice_summary,
                "audit_path": str(
                    Path("exercises") / class_id / assessment_id / exercise_id / "audit.json"
                ),
            }
        )

    summary = {
        "class_id": class_id,
        "assessment_id": assessment_id,
        "assessment_title": assessment_meta["title"],
        "exercise_count": len(exercise_ids),
        "included_exercise_count": included_exercise_count,
        "excluded_exercise_count": len(exercise_ids) - included_exercise_count,
        "allow_partial": allow_partial,
        "candidate_student_count": len(candidate_students),
        "included_student_count": len(included_students),
        "candidate_transition_count": candidate_transition_count,
        "included_transition_count": included_transition_count,
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    bundle = {
        "schema_version": "v5_assessment_audit_v1",
        "assessment_summary": summary,
        "by_exercise": by_exercise,
    }
    _write_json(assessment_dir / "assessment_audit.json", bundle)
    return {
        "bundle": bundle,
        "paths": {
            "assessment_dir": str(assessment_dir),
            "assessment_audit": str(assessment_dir / "assessment_audit.json"),
        },
    }


def main() -> int:
    args = parse_args()
    result = audit_assessment(
        class_id=args.class_id,
        assessment_id=args.assessment_id,
        data_root=args.data_root,
        out_root=args.out_root,
        max_students=args.max_students,
        student_ids=tuple(args.student_id),
        allow_partial=args.allow_partial,
    )
    print(json.dumps(result["bundle"]["assessment_summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
