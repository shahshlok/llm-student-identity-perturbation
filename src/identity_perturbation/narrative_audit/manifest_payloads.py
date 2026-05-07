from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_perturbation.codebench_support.runner import SliceBuildError, write_json, write_text

from .conditions import FULL_V6_CONDITION, V6ConditionError, validate_condition
from .runner import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUT_ROOT,
    DEFAULT_REASONING_EFFORT,
    V6BuildError,
    build_transition_payload_bundle,
)


class V6ManifestPayloadError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v6 transition bundles from an explicit transition manifest."
    )
    parser.add_argument(
        "--transition-manifest",
        required=True,
        type=Path,
        help="Path to transition_manifest.json",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Artifact root for generated v6 transition bundles",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Raw dataset root containing 2024-1/<class_id>/...",
    )
    parser.add_argument(
        "--condition",
        default=FULL_V6_CONDITION,
        help="v6 condition to build, e.g. full_v6 or no_trace",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    return parser.parse_args()


def _load_transition_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise V6ManifestPayloadError(f"Transition manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v6_transition_manifest_v1":
        raise V6ManifestPayloadError(
            f"Unexpected transition manifest schema in {path}: {payload.get('schema_version')}"
        )
    selected_transitions = payload.get("selected_transitions")
    if not isinstance(selected_transitions, list) or not selected_transitions:
        raise V6ManifestPayloadError(
            f"Transition manifest must contain a non-empty selected_transitions list: {path}"
        )
    return payload


def build_manifest_payloads(
    *,
    transition_manifest_path: Path,
    out_root: Path,
    data_root: Path,
    model: str,
    reasoning_effort: str,
    condition: str,
) -> dict[str, object]:
    validated_condition = validate_condition(condition)
    manifest_payload = _load_transition_manifest(transition_manifest_path)
    selected_transitions = manifest_payload["selected_transitions"]

    bundle_rows: list[dict[str, object]] = []
    for row in selected_transitions:
        required_keys = {
            "class_id",
            "assessment_id",
            "exercise_id",
            "student_id",
            "transition_index_0idx",
        }
        missing = required_keys - set(row)
        if missing:
            raise V6ManifestPayloadError(
                f"Transition manifest row missing required keys {sorted(missing)}: {row}"
            )
        artifacts = build_transition_payload_bundle(
            class_id=str(row["class_id"]),
            assessment_id=str(row["assessment_id"]),
            exercise_id=str(row["exercise_id"]),
            student_id=str(row["student_id"]),
            transition_index=int(row["transition_index_0idx"]),
            data_root=data_root,
            out_root=out_root,
            model=model,
            reasoning_effort=reasoning_effort,
            condition=validated_condition,
        )
        bundle_rows.append(
            {
                "condition": validated_condition,
                "class_id": str(row["class_id"]),
                "assessment_id": str(row["assessment_id"]),
                "exercise_id": str(row["exercise_id"]),
                "student_id": str(row["student_id"]),
                "transition_index_0idx": int(row["transition_index_0idx"]),
                "bundle_dir": str(Path(artifacts["paths"]["manifest"]).parent.resolve()),
                "bundle_manifest_path": str(Path(artifacts["paths"]["manifest"]).resolve()),
            }
        )

    manifest_dir = (
        out_root / "_manifests" / transition_manifest_path.parent.name / validated_condition
    )
    manifest_dir.mkdir(parents=True, exist_ok=False)
    bundle_manifest = {
        "schema_version": "v6_transition_bundle_manifest_v1",
        "source_transition_manifest_path": str(transition_manifest_path.resolve()),
        "data_root": str(data_root.resolve()),
        "condition": validated_condition,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "summary": {
            "bundle_count": len(bundle_rows),
        },
        "bundles": bundle_rows,
    }
    lines = [
        "# V6 Transition Bundle Manifest",
        "",
        f"- Source transition manifest: `{transition_manifest_path.resolve()}`",
        f"- Bundle count: {len(bundle_rows)}",
        f"- Condition: `{validated_condition}`",
        f"- Model: `{model}`",
        f"- Reasoning effort: `{reasoning_effort}`",
        "",
        "## Bundles",
        "",
        "| Rank | Class | Assessment | Exercise | Student | Transition | Bundle dir |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(bundle_rows, start=1):
        lines.append(
            f"| {index} | {row['class_id']} | {row['assessment_id']} | {row['exercise_id']} | "
            f"{row['student_id']} | {row['transition_index_0idx']} | {row['bundle_dir']} |"
        )
    write_json(manifest_dir / "bundle_manifest.json", bundle_manifest)
    write_text(manifest_dir / "bundle_manifest.md", "\n".join(lines) + "\n")
    return {
        "manifest": bundle_manifest,
        "paths": {
            "manifest_dir": str(manifest_dir.resolve()),
            "manifest_json": str((manifest_dir / "bundle_manifest.json").resolve()),
            "manifest_md": str((manifest_dir / "bundle_manifest.md").resolve()),
        },
    }


def main() -> int:
    args = parse_args()
    try:
        result = build_manifest_payloads(
            transition_manifest_path=args.transition_manifest,
            out_root=args.out_root,
            data_root=args.data_root,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            condition=args.condition,
        )
    except (V6ManifestPayloadError, V6BuildError, SliceBuildError, V6ConditionError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result["manifest"]["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
