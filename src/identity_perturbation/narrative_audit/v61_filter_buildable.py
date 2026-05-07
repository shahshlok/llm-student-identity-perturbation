from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from identity_perturbation.codebench_support.runner import SliceBuildError, write_json, write_text

from .v61_manifest_payloads import V61ManifestPayloadError, _load_transition_manifest
from .v61_runner import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUT_ROOT,
    DEFAULT_REASONING_EFFORT,
    V61BuildError,
    build_transition_payload_bundle,
)


class V61FilterBuildableError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a v6.1 transition manifest down to rows that build cleanly."
    )
    parser.add_argument("--transition-manifest", required=True, type=Path)
    parser.add_argument("--manifest-name", required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--idle-gap-seconds", type=float, default=30.0)
    parser.add_argument("--include-keyhandled", action="store_true")
    parser.add_argument("--exclude-navigation", action="store_true")
    parser.add_argument("--manifest-root", type=Path, default=Path("data/v61/transition_manifests"))
    parser.add_argument("--probe-root", type=Path, default=Path("data/v61/_buildability_probe"))
    return parser.parse_args()


def _transition_key(row: dict[str, object]) -> str:
    return (
        f"{row['class_id']}:{row['assessment_id']}:{row['exercise_id']}:"
        f"{row['student_id']}:{row['transition_index_0idx']}"
    )


def filter_buildable_transitions(
    *,
    transition_manifest_path: Path,
    manifest_name: str,
    out_root: Path,
    data_root: Path,
    model: str,
    reasoning_effort: str,
    idle_gap_seconds: float,
    include_keyhandled: bool,
    include_navigation: bool,
    manifest_root: Path,
    probe_root: Path,
) -> dict[str, object]:
    manifest_payload = _load_transition_manifest(transition_manifest_path)
    selected_transitions = manifest_payload["selected_transitions"]
    if not isinstance(selected_transitions, list) or not selected_transitions:
        raise V61FilterBuildableError(
            f"Transition manifest must contain a non-empty selected_transitions list: {transition_manifest_path}"
        )

    probe_root = probe_root / manifest_name
    if probe_root.exists():
        raise V61FilterBuildableError(f"Probe root already exists: {probe_root}")
    probe_root.mkdir(parents=True, exist_ok=False)

    kept_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []
    try:
        for row in selected_transitions:
            if not isinstance(row, dict):
                raise V61FilterBuildableError(f"Transition row must be an object: {row!r}")
            key = _transition_key(row)
            try:
                artifacts = build_transition_payload_bundle(
                    class_id=str(row["class_id"]),
                    assessment_id=str(row["assessment_id"]),
                    exercise_id=str(row["exercise_id"]),
                    student_id=str(row["student_id"]),
                    transition_index=int(row["transition_index_0idx"]),
                    data_root=data_root,
                    out_root=probe_root,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    idle_gap_seconds=idle_gap_seconds,
                    include_keyhandled=include_keyhandled,
                    include_navigation=include_navigation,
                )
            except (V61BuildError, SliceBuildError) as exc:
                dropped_rows.append(
                    {
                        "transition": key,
                        "class_id": str(row["class_id"]),
                        "assessment_id": str(row["assessment_id"]),
                        "exercise_id": str(row["exercise_id"]),
                        "student_id": str(row["student_id"]),
                        "transition_index_0idx": int(row["transition_index_0idx"]),
                        "error": str(exc),
                    }
                )
                continue

            kept_rows.append(
                {
                    "class_id": str(row["class_id"]),
                    "assessment_id": str(row["assessment_id"]),
                    "exercise_id": str(row["exercise_id"]),
                    "student_id": str(row["student_id"]),
                    "transition_index_0idx": int(row["transition_index_0idx"]),
                }
            )
            bundle_dir = Path(artifacts["paths"]["manifest"]).parent
            shutil.rmtree(bundle_dir)
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)

    manifest_dir = manifest_root / manifest_name
    manifest_dir.mkdir(parents=True, exist_ok=False)

    output_manifest = {
        "schema_version": "v6_1_clean_transition_manifest_v1",
        "source_transition_manifest_path": str(transition_manifest_path.resolve()),
        "buildability_filter": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "idle_gap_seconds": idle_gap_seconds,
            "include_keyhandled": include_keyhandled,
            "include_navigation": include_navigation,
        },
        "summary": {
            "input_count": len(selected_transitions),
            "kept_count": len(kept_rows),
            "dropped_count": len(dropped_rows),
        },
        "selected_transitions": kept_rows,
    }
    failure_report = {
        "schema_version": "v6_1_buildability_report_v1",
        "source_transition_manifest_path": str(transition_manifest_path.resolve()),
        "summary": output_manifest["summary"],
        "dropped_rows": dropped_rows,
    }

    lines = [
        "# V6.1 Buildable Transition Filter",
        "",
        f"- Source transition manifest: `{transition_manifest_path.resolve()}`",
        f"- Input count: {len(selected_transitions)}",
        f"- Kept count: {len(kept_rows)}",
        f"- Dropped count: {len(dropped_rows)}",
        "",
    ]
    if dropped_rows:
        lines.extend(
            [
                "## Dropped Rows",
                "",
                "| Transition | Error |",
                "| --- | --- |",
            ]
        )
        for row in dropped_rows:
            lines.append(f"| {row['transition']} | {row['error']} |")

    write_json(manifest_dir / "transition_manifest.json", output_manifest)
    write_json(manifest_dir / "buildability_report.json", failure_report)
    write_text(manifest_dir / "buildability_report.md", "\n".join(lines) + "\n")
    return {
        "manifest": output_manifest,
        "report": failure_report,
        "paths": {
            "manifest_dir": str(manifest_dir.resolve()),
            "transition_manifest_json": str((manifest_dir / "transition_manifest.json").resolve()),
            "buildability_report_json": str((manifest_dir / "buildability_report.json").resolve()),
            "buildability_report_md": str((manifest_dir / "buildability_report.md").resolve()),
        },
    }


def main() -> int:
    args = parse_args()
    try:
        result = filter_buildable_transitions(
            transition_manifest_path=args.transition_manifest,
            manifest_name=args.manifest_name,
            out_root=args.out_root,
            data_root=args.data_root,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            idle_gap_seconds=args.idle_gap_seconds,
            include_keyhandled=args.include_keyhandled,
            include_navigation=not args.exclude_navigation,
            manifest_root=args.manifest_root,
            probe_root=args.probe_root,
        )
    except (
        V61FilterBuildableError,
        V61ManifestPayloadError,
        V61BuildError,
        SliceBuildError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result["manifest"]["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
