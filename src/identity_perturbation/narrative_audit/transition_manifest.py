from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_perturbation.codebench_support.runner import ROOT, SliceBuildError, write_json, write_text

DEFAULT_OUT_ROOT = ROOT / "data" / "v6" / "transition_manifests"


class V6TransitionManifestError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit v6 transition manifest for batch preparation."
    )
    parser.add_argument(
        "--transition",
        action="append",
        required=True,
        help=(
            "Transition spec formatted as "
            "class_id:assessment_id:exercise_id:student_id:transition_index_0idx; may be repeated"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Artifact root for v6 transition manifests",
    )
    parser.add_argument(
        "--manifest-name",
        default="latest",
        help="Subdirectory name for the output manifest bundle",
    )
    return parser.parse_args()


def _parse_transition_spec(spec: str) -> dict[str, object]:
    parts = spec.split(":")
    if len(parts) != 5:
        raise V6TransitionManifestError(
            "Transition spec must have exactly 5 colon-separated fields: "
            "class_id:assessment_id:exercise_id:student_id:transition_index_0idx"
        )
    class_id, assessment_id, exercise_id, student_id, transition_index_raw = parts
    try:
        transition_index = int(transition_index_raw)
    except ValueError as exc:
        raise V6TransitionManifestError(
            f"Transition index must be an integer in spec: {spec}"
        ) from exc
    if transition_index < 0:
        raise V6TransitionManifestError(f"Transition index must be non-negative in spec: {spec}")
    return {
        "class_id": class_id,
        "assessment_id": assessment_id,
        "exercise_id": exercise_id,
        "student_id": student_id,
        "transition_index_0idx": transition_index,
    }


def build_transition_manifest(
    *,
    transition_specs: tuple[str, ...],
    out_root: Path,
    manifest_name: str,
) -> dict[str, object]:
    parsed_rows: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str, str, int]] = set()
    for spec in transition_specs:
        row = _parse_transition_spec(spec)
        key = (
            str(row["class_id"]),
            str(row["assessment_id"]),
            str(row["exercise_id"]),
            str(row["student_id"]),
            int(row["transition_index_0idx"]),
        )
        if key in seen_keys:
            raise V6TransitionManifestError(f"Duplicate transition spec detected: {spec}")
        seen_keys.add(key)
        parsed_rows.append(row)

    if not parsed_rows:
        raise V6TransitionManifestError("Transition manifest cannot be empty")

    manifest = {
        "schema_version": "v6_transition_manifest_v1",
        "selection_policy": {
            "mode": "explicit_transition_list",
            "count": len(parsed_rows),
        },
        "summary": {
            "transition_count": len(parsed_rows),
            "class_count": len({str(row["class_id"]) for row in parsed_rows}),
            "assessment_count": len(
                {(str(row["class_id"]), str(row["assessment_id"])) for row in parsed_rows}
            ),
            "exercise_count": len(
                {
                    (
                        str(row["class_id"]),
                        str(row["assessment_id"]),
                        str(row["exercise_id"]),
                    )
                    for row in parsed_rows
                }
            ),
            "student_count": len(
                {
                    (
                        str(row["class_id"]),
                        str(row["assessment_id"]),
                        str(row["exercise_id"]),
                        str(row["student_id"]),
                    )
                    for row in parsed_rows
                }
            ),
        },
        "selected_transitions": parsed_rows,
    }

    lines = [
        "# V6 Transition Manifest",
        "",
        f"- Transition count: {manifest['summary']['transition_count']}",
        f"- Class count: {manifest['summary']['class_count']}",
        f"- Assessment count: {manifest['summary']['assessment_count']}",
        f"- Exercise count: {manifest['summary']['exercise_count']}",
        f"- Student count: {manifest['summary']['student_count']}",
        "",
        "## Selected Transitions",
        "",
        "| Rank | Class | Assessment | Exercise | Student | Transition |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(parsed_rows, start=1):
        lines.append(
            f"| {index} | {row['class_id']} | {row['assessment_id']} | {row['exercise_id']} | "
            f"{row['student_id']} | {row['transition_index_0idx']} |"
        )

    manifest_dir = out_root / manifest_name
    manifest_dir.mkdir(parents=True, exist_ok=False)
    write_json(manifest_dir / "transition_manifest.json", manifest)
    write_text(manifest_dir / "transition_manifest.md", "\n".join(lines) + "\n")
    return {
        "manifest": manifest,
        "paths": {
            "manifest_dir": str(manifest_dir.resolve()),
            "manifest_json": str((manifest_dir / "transition_manifest.json").resolve()),
            "manifest_md": str((manifest_dir / "transition_manifest.md").resolve()),
        },
    }


def main() -> int:
    args = parse_args()
    try:
        result = build_transition_manifest(
            transition_specs=tuple(args.transition),
            out_root=args.out_root,
            manifest_name=args.manifest_name,
        )
    except (V6TransitionManifestError, SliceBuildError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result["manifest"]["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
