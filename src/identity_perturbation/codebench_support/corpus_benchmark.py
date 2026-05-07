from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assessment_audit import audit_assessment
from .benchmark import build_benchmark_manifest
from .runner import ROOT, SliceBuildError
from .slice_sizes import build_slice_size_report

DEFAULT_OUT_ROOT = ROOT / "data" / "v5" / "corpus_benchmarks"
PRESETS = {
    "strong_lab56": (
        ("594", "5835"),
        ("594", "5889"),
        ("589", "5846"),
        ("589", "5897"),
        ("590", "5843"),
        ("590", "5896"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict multi-assessment v5 audits and build one combined benchmark bundle."
    )
    parser.add_argument(
        "--assessment",
        action="append",
        default=[],
        help="Assessment spec as class_id:assessment_id; may be repeated",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(sorted(PRESETS)),
        default=None,
        help="Optional named preset of class:assessment pairs",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT.parent / "tracer" / "2024-1",
        help="Raw dataset root containing 2024-1/<class_id>/...",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Artifact root for combined corpus benchmark bundles",
    )
    parser.add_argument(
        "--bundle-name",
        default="latest",
        help="Subdirectory name for the corpus benchmark bundle",
    )
    parser.add_argument(
        "--max-students",
        type=int,
        default=None,
        help="Optional cap for debugging on the first N candidate students per exercise",
    )
    parser.add_argument(
        "--min-included-transitions",
        type=int,
        default=3,
        help="Minimum included transitions required for benchmark selection",
    )
    parser.add_argument(
        "--min-included-students",
        type=int,
        default=2,
        help="Minimum included students required for benchmark selection",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=None,
        help="Optional cap on selected slices after ranking",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Maximum number of ranked slices to emit in the slice-size report",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _parse_assessment_spec(raw: str) -> tuple[str, str]:
    parts = raw.split(":")
    if len(parts) != 2 or not all(parts):
        raise SliceBuildError(
            f"Invalid --assessment value {raw!r}; expected class_id:assessment_id"
        )
    return parts[0], parts[1]


def _assessment_specs(args: argparse.Namespace) -> tuple[tuple[str, str], ...]:
    specs: list[tuple[str, str]] = []
    if args.preset is not None:
        specs.extend(PRESETS[args.preset])
    specs.extend(_parse_assessment_spec(raw) for raw in args.assessment)
    unique = tuple(dict.fromkeys(specs))
    if not unique:
        raise SliceBuildError("Provide at least one --assessment or a --preset")
    return unique


def build_corpus_benchmark(
    *,
    assessment_specs: tuple[tuple[str, str], ...],
    data_root: Path,
    out_root: Path,
    bundle_name: str,
    max_students: int | None,
    min_included_transitions: int,
    min_included_students: int,
    max_slices: int | None,
    top_n: int,
) -> dict[str, object]:
    bundle_dir = out_root / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=False)

    assessment_audit_out_root = bundle_dir / "assessment_audits"
    slice_size_out_root = bundle_dir / "slice_size_reports"
    benchmark_out_root = bundle_dir / "benchmark_manifests"
    assessment_audit_out_root.mkdir(parents=False, exist_ok=False)
    slice_size_out_root.mkdir(parents=False, exist_ok=False)
    benchmark_out_root.mkdir(parents=False, exist_ok=False)

    assessment_audit_paths: list[Path] = []
    assessment_rows: list[dict[str, object]] = []
    for class_id, assessment_id in assessment_specs:
        result = audit_assessment(
            class_id=class_id,
            assessment_id=assessment_id,
            data_root=data_root,
            out_root=assessment_audit_out_root,
            max_students=max_students,
            student_ids=(),
            allow_partial=False,
        )
        assessment_path = Path(result["paths"]["assessment_audit"])
        assessment_audit_paths.append(assessment_path)
        assessment_rows.append(
            {
                "class_id": class_id,
                "assessment_id": assessment_id,
                "assessment_audit_path": str(assessment_path.resolve()),
            }
        )

    slice_size_result = build_slice_size_report(
        assessment_audit_paths=tuple(assessment_audit_paths),
        min_included_transitions=min_included_transitions,
        min_included_students=min_included_students,
        top_n=top_n,
        out_root=slice_size_out_root,
        report_name="default",
    )
    benchmark_result = build_benchmark_manifest(
        assessment_audit_paths=tuple(assessment_audit_paths),
        out_root=benchmark_out_root,
        manifest_name="default",
        min_included_transitions=min_included_transitions,
        min_included_students=min_included_students,
        max_slices=max_slices,
    )

    bundle = {
        "schema_version": "v5_corpus_benchmark_v1",
        "assessment_specs": [
            {"class_id": class_id, "assessment_id": assessment_id}
            for class_id, assessment_id in assessment_specs
        ],
        "selection_policy": {
            "min_included_transitions": min_included_transitions,
            "min_included_students": min_included_students,
            "max_slices": max_slices,
            "top_n": top_n,
            "allow_partial": False,
        },
        "summary": {
            "assessment_count": len(assessment_specs),
            "selected_slice_count": benchmark_result["manifest"]["summary"]["selected_slice_count"],
            "eligible_slice_count": slice_size_result["report"]["summary"]["eligible_slice_count"],
        },
        "artifacts": {
            "assessment_rows": assessment_rows,
            "slice_size_report": str(Path(slice_size_result["paths"]["report"]).resolve()),
            "benchmark_manifest": str(Path(benchmark_result["paths"]["manifest_json"]).resolve()),
        },
    }
    _write_json(bundle_dir / "corpus_benchmark.json", bundle)
    return {
        "bundle": bundle,
        "paths": {
            "bundle_dir": str(bundle_dir.resolve()),
            "corpus_benchmark": str((bundle_dir / "corpus_benchmark.json").resolve()),
            "slice_size_report": slice_size_result["paths"]["report"],
            "benchmark_manifest": benchmark_result["paths"]["manifest_json"],
        },
    }


def main() -> int:
    args = parse_args()
    result = build_corpus_benchmark(
        assessment_specs=_assessment_specs(args),
        data_root=args.data_root,
        out_root=args.out_root,
        bundle_name=args.bundle_name,
        max_students=args.max_students,
        min_included_transitions=args.min_included_transitions,
        min_included_students=args.min_included_students,
        max_slices=args.max_slices,
        top_n=args.top_n,
    )
    print(json.dumps(result["bundle"]["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
