from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_perturbation.codebench_support.runner import SliceBuildError, write_json, write_text

from .v61_conditions import FULL_V61_CONDITION, condition_output_dirname
from .v61_runner import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    DEFAULT_OUT_ROOT,
    DEFAULT_REASONING_EFFORT,
    V61BuildError,
    build_transition_payload_bundle,
)


class V61ManifestPayloadError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v6.1 transition bundles from an explicit transition manifest."
    )
    parser.add_argument("--transition-manifest", required=True, type=Path)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument(
        "--condition",
        default=FULL_V61_CONDITION,
        help=(
            "v6.1 condition to build, e.g. full_v61, no_trace, trace_shuffled_within_exercise, "
            "trace_shuffled_within_class_random, or trace_shuffled_cross_class_random"
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--idle-gap-seconds", type=float, default=30.0)
    parser.add_argument("--include-keyhandled", action="store_true")
    parser.add_argument("--exclude-navigation", action="store_true")
    return parser.parse_args()


def _load_transition_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise V61ManifestPayloadError(f"Transition manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version not in {"v6_transition_manifest_v1", "v6_1_clean_transition_manifest_v1"}:
        raise V61ManifestPayloadError(
            f"Unexpected transition manifest schema in {path}: {schema_version}"
        )
    selected_transitions = payload.get("selected_transitions")
    if not isinstance(selected_transitions, list) or not selected_transitions:
        raise V61ManifestPayloadError(
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
    shuffle_seed: int | None,
    idle_gap_seconds: float,
    include_keyhandled: bool,
    include_navigation: bool,
) -> dict[str, object]:
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
            raise V61ManifestPayloadError(
                f"Transition manifest row missing required keys {sorted(missing)}: {row}"
            )
        class_id = str(row["class_id"])
        assessment_id = str(row["assessment_id"])
        exercise_id = str(row["exercise_id"])
        student_id = str(row["student_id"])
        transition_index = int(row["transition_index_0idx"])
        try:
            artifacts = build_transition_payload_bundle(
                class_id=class_id,
                assessment_id=assessment_id,
                exercise_id=exercise_id,
                student_id=student_id,
                transition_index=transition_index,
                data_root=data_root,
                out_root=out_root,
                model=model,
                reasoning_effort=reasoning_effort,
                condition=condition,
                shuffle_seed=shuffle_seed,
                idle_gap_seconds=idle_gap_seconds,
                include_keyhandled=include_keyhandled,
                include_navigation=include_navigation,
            )
        except (V61BuildError, SliceBuildError) as exc:
            raise V61ManifestPayloadError(
                "Failed building transition "
                f"{class_id}:{assessment_id}:{exercise_id}:{student_id}:{transition_index}: {exc}"
            ) from exc
        bundle_rows.append(
            {
                "class_id": class_id,
                "assessment_id": assessment_id,
                "exercise_id": exercise_id,
                "student_id": student_id,
                "transition_index_0idx": transition_index,
                "bundle_dir": str(Path(artifacts["paths"]["manifest"]).parent.resolve()),
                "bundle_manifest_path": str(Path(artifacts["paths"]["manifest"]).resolve()),
            }
        )

    manifest_dir = out_root / "_manifests" / transition_manifest_path.parent.name
    condition_dirname = condition_output_dirname(condition, shuffle_seed)
    if condition_dirname is not None:
        manifest_dir = manifest_dir / condition_dirname
    manifest_dir.mkdir(parents=True, exist_ok=False)
    bundle_manifest = {
        "schema_version": "v6_1_transition_bundle_manifest_v1",
        "source_transition_manifest_path": str(transition_manifest_path.resolve()),
        "data_root": str(data_root.resolve()),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "condition": condition,
        "shuffle_seed": shuffle_seed,
        "idle_gap_seconds": idle_gap_seconds,
        "include_keyhandled": include_keyhandled,
        "include_navigation": include_navigation,
        "summary": {"bundle_count": len(bundle_rows)},
        "bundles": bundle_rows,
    }
    lines = [
        "# V6.1 Transition Bundle Manifest",
        "",
        f"- Source transition manifest: `{transition_manifest_path.resolve()}`",
        f"- Bundle count: {len(bundle_rows)}",
        f"- Model: `{model}`",
        f"- Reasoning effort: `{reasoning_effort}`",
        f"- Condition: `{condition}`",
        f"- Shuffle seed: `{shuffle_seed}`",
        f"- Idle gap seconds: `{idle_gap_seconds}`",
        f"- Include keyHandled: `{include_keyhandled}`",
        f"- Include navigation: `{include_navigation}`",
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
            shuffle_seed=args.shuffle_seed,
            idle_gap_seconds=args.idle_gap_seconds,
            include_keyhandled=args.include_keyhandled,
            include_navigation=not args.exclude_navigation,
        )
    except (V61ManifestPayloadError, V61BuildError, SliceBuildError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result["manifest"]["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
