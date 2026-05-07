from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import ROOT, SliceBuildError

DEFAULT_OUT_ROOT = ROOT / "data" / "v5" / "slice_size_reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank benchmark-eligible v5 slices by included size using assessment audit outputs."
    )
    parser.add_argument(
        "--assessment-audit",
        action="append",
        default=[],
        type=Path,
        help="Path to an assessment_audit.json file; may be repeated",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=None,
        help="Optional root directory to search recursively for assessment_audit.json files",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Artifact root for slice-size report outputs",
    )
    parser.add_argument(
        "--report-name",
        default="latest",
        help="Subdirectory name for the output report bundle",
    )
    parser.add_argument(
        "--min-included-transitions",
        type=int,
        default=1,
        help="Minimum included full-match transitions required for eligibility",
    )
    parser.add_argument(
        "--min-included-students",
        type=int,
        default=1,
        help="Minimum included students required for eligibility",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Maximum number of ranked slices to emit",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _assessment_audit_paths(
    explicit_paths: tuple[Path, ...],
    audit_root: Path | None,
) -> tuple[Path, ...]:
    discovered: list[Path] = []
    for path in explicit_paths:
        if not path.exists():
            raise SliceBuildError(f"Assessment audit file not found: {path}")
        discovered.append(path)
    if audit_root is not None:
        if not audit_root.exists():
            raise SliceBuildError(f"Audit root not found: {audit_root}")
        discovered.extend(sorted(audit_root.rglob("assessment_audit.json")))
    unique_paths = tuple(dict.fromkeys(path.resolve() for path in discovered))
    if not unique_paths:
        raise SliceBuildError("No assessment audit files provided or discovered")
    return unique_paths


def build_slice_size_report(
    assessment_audit_paths: tuple[Path, ...],
    min_included_transitions: int,
    min_included_students: int,
    top_n: int,
    out_root: Path,
    report_name: str,
) -> dict[str, object]:
    ranked_rows: list[dict[str, object]] = []
    assessments: list[dict[str, object]] = []

    for path in assessment_audit_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["assessment_summary"]
        assessments.append(summary)

        for row in payload["by_exercise"]:
            slice_summary = row["slice_summary"]
            included_transitions = int(slice_summary["included_transition_count"])
            included_students = int(slice_summary["included_student_count"])
            candidate_transitions = int(slice_summary["candidate_transition_count"])
            coverage_rate = (
                included_transitions / candidate_transitions if candidate_transitions else 0.0
            )
            eligible = (
                included_transitions >= min_included_transitions
                and included_students >= min_included_students
            )
            ranked_rows.append(
                {
                    "class_id": summary["class_id"],
                    "assessment_id": summary["assessment_id"],
                    "assessment_title": summary["assessment_title"],
                    "exercise_id": row["exercise_id"],
                    "included_transitions": included_transitions,
                    "included_students": included_students,
                    "candidate_transitions": candidate_transitions,
                    "coverage_rate": coverage_rate,
                    "eligible": eligible,
                    "status": row["status"],
                    "audit_path": row["audit_path"],
                }
            )

    ranked_rows.sort(
        key=lambda row: (
            not row["eligible"],
            -row["included_transitions"],
            -row["included_students"],
            -row["coverage_rate"],
            row["class_id"],
            row["assessment_id"],
            row["exercise_id"],
        )
    )

    eligible_rows = [row for row in ranked_rows if row["eligible"]]
    report = {
        "schema_version": "v5_slice_size_report_v1",
        "thresholds": {
            "min_included_transitions": min_included_transitions,
            "min_included_students": min_included_students,
            "top_n": top_n,
        },
        "summary": {
            "assessment_count": len(assessments),
            "slice_count": len(ranked_rows),
            "eligible_slice_count": len(eligible_rows),
        },
        "ranked_slices": ranked_rows[:top_n],
    }

    report_dir = out_root / report_name
    report_dir.mkdir(parents=True, exist_ok=False)
    _write_json(report_dir / "slice_size_report.json", report)
    return {
        "report": report,
        "paths": {
            "report_dir": str(report_dir),
            "report": str(report_dir / "slice_size_report.json"),
        },
    }


def main() -> int:
    args = parse_args()
    result = build_slice_size_report(
        assessment_audit_paths=_assessment_audit_paths(
            explicit_paths=tuple(args.assessment_audit),
            audit_root=args.audit_root,
        ),
        min_included_transitions=args.min_included_transitions,
        min_included_students=args.min_included_students,
        top_n=args.top_n,
        out_root=args.out_root,
        report_name=args.report_name,
    )
    print(json.dumps(result["report"]["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
