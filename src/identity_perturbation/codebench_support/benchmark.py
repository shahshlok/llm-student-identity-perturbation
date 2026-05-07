from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import ROOT, SliceBuildError
from .slice_sizes import _assessment_audit_paths

DEFAULT_OUT_ROOT = ROOT / "data" / "v5" / "benchmark_manifests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a v5 benchmark manifest from assessment audit outputs."
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
        help="Artifact root for benchmark manifests",
    )
    parser.add_argument(
        "--manifest-name",
        default="latest",
        help="Subdirectory name for the output manifest bundle",
    )
    parser.add_argument(
        "--min-included-transitions",
        type=int,
        default=3,
        help="Minimum included full-match transitions required for selection",
    )
    parser.add_argument(
        "--min-included-students",
        type=int,
        default=2,
        help="Minimum included students required for selection",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=None,
        help="Optional cap on selected slices after ranking",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(text, encoding="utf-8")


def _rank_key(row: dict[str, object]) -> tuple:
    return (
        -int(row["included_transitions"]),
        -int(row["included_students"]),
        -float(row["coverage_rate"]),
        str(row["class_id"]),
        str(row["assessment_id"]),
        str(row["exercise_id"]),
    )


def build_benchmark_manifest(
    assessment_audit_paths: tuple[Path, ...],
    out_root: Path,
    manifest_name: str,
    min_included_transitions: int,
    min_included_students: int,
    max_slices: int | None,
) -> dict[str, object]:
    ranked_rows: list[dict[str, object]] = []

    for path in assessment_audit_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assessment_summary = payload["assessment_summary"]
        for row in payload["by_exercise"]:
            slice_summary = row["slice_summary"]
            included_transitions = int(slice_summary["included_transition_count"])
            included_students = int(slice_summary["included_student_count"])
            candidate_transitions = int(slice_summary["candidate_transition_count"])
            coverage_rate = (
                included_transitions / candidate_transitions if candidate_transitions else 0.0
            )
            ranked_rows.append(
                {
                    "class_id": assessment_summary["class_id"],
                    "assessment_id": assessment_summary["assessment_id"],
                    "assessment_title": assessment_summary["assessment_title"],
                    "exercise_id": row["exercise_id"],
                    "included_transitions": included_transitions,
                    "included_students": included_students,
                    "candidate_transitions": candidate_transitions,
                    "coverage_rate": coverage_rate,
                    "audit_path": str(
                        Path("exercises")
                        / str(assessment_summary["class_id"])
                        / str(assessment_summary["assessment_id"])
                        / str(row["exercise_id"])
                        / "audit.json"
                    ),
                }
            )

    ranked_rows.sort(key=_rank_key)
    selected_rows = [
        row
        for row in ranked_rows
        if row["included_transitions"] >= min_included_transitions
        and row["included_students"] >= min_included_students
    ]
    if max_slices is not None:
        selected_rows = selected_rows[:max_slices]

    manifest = {
        "schema_version": "v5_benchmark_manifest_v1",
        "source_assessment_audit_paths": [str(path.resolve()) for path in assessment_audit_paths],
        "selection_policy": {
            "min_included_transitions": min_included_transitions,
            "min_included_students": min_included_students,
            "max_slices": max_slices,
        },
        "summary": {
            "assessment_count": len(assessment_audit_paths),
            "candidate_slice_count": len(ranked_rows),
            "selected_slice_count": len(selected_rows),
        },
        "selected_slices": selected_rows,
    }

    lines = [
        "# V5 Benchmark Manifest",
        "",
        f"- Candidate slices: {manifest['summary']['candidate_slice_count']}",
        f"- Selected slices: {manifest['summary']['selected_slice_count']}",
        f"- Min included transitions: {min_included_transitions}",
        f"- Min included students: {min_included_students}",
        "",
        "## Selected Slices",
        "",
        "| Rank | Class | Assessment | Exercise | Included transitions | Included students | Coverage |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(selected_rows, start=1):
        lines.append(
            f"| {index} | {row['class_id']} | {row['assessment_id']} | {row['exercise_id']} | "
            f"{row['included_transitions']} | {row['included_students']} | {row['coverage_rate']:.3f} |"
        )
    if not selected_rows:
        lines.append("| - | - | - | - | - | - | - |")

    manifest_dir = out_root / manifest_name
    manifest_dir.mkdir(parents=True, exist_ok=False)
    _write_json(manifest_dir / "benchmark_manifest.json", manifest)
    _write_text(manifest_dir / "benchmark_manifest.md", "\n".join(lines) + "\n")
    return {
        "manifest": manifest,
        "paths": {
            "manifest_dir": str(manifest_dir),
            "manifest_json": str(manifest_dir / "benchmark_manifest.json"),
            "manifest_md": str(manifest_dir / "benchmark_manifest.md"),
        },
    }


def main() -> int:
    args = parse_args()
    result = build_benchmark_manifest(
        assessment_audit_paths=_assessment_audit_paths(
            explicit_paths=tuple(args.assessment_audit),
            audit_root=args.audit_root,
        ),
        out_root=args.out_root,
        manifest_name=args.manifest_name,
        min_included_transitions=args.min_included_transitions,
        min_included_students=args.min_included_students,
        max_slices=args.max_slices,
    )
    print(json.dumps(result["manifest"]["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
