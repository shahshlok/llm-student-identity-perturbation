"""CLI wrapper: score one v6.2 bundle using the v2 scorer.

This is the v2 companion to ``score_full_trace_bundle.py``.  It reuses
the bundle-loading / response-extraction helpers from the v1 CLI so the
two scripts accept the exact same inputs; only the scoring function
differs.  Keeping the wrappers separate lets callers pick a scorer
version explicitly and keeps old artifacts reproducible.

Run with uv, e.g.:

    uv run python -m identity_perturbation.prediction_audit.score_full_trace_bundle_v2 \
        --bundle-dir path/to/bundle \
        --batch-output-jsonl path/to/output.jsonl \
        --out path/to/scored_prediction_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from identity_perturbation.prediction_audit.full_trace_scorer_v2 import (
    DEFAULT_EDIT_STEP_WINDOW_RADIUS,
    DEFAULT_FOOTPRINT_WINDOW_RADIUS,
    SCHEMA_VERSION,
    score_full_trace_prediction_v2,
)
from identity_perturbation.prediction_audit.score_full_trace_bundle import (
    FullTraceBundleScoringError,
    _load_bundle_manifest,
    _load_json_strict,
    _prediction_from_batch_output_item_json,
    _prediction_from_batch_output_jsonl,
    _prediction_from_prediction_json,
    _resolve_bundle_artifact_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score one identity-perturbation prompt bundle against one model response "
            "using the v2 scorer (normalized diff, bounded gain, graded "
            "trajectory metrics, rank-weighted aggregation)."
        )
    )
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--footprint-window-radius",
        type=int,
        default=DEFAULT_FOOTPRINT_WINDOW_RADIUS,
        help="Line radius for windowed repair footprint scoring (default 1).",
    )
    parser.add_argument(
        "--edit-step-window-radius",
        type=int,
        default=DEFAULT_EDIT_STEP_WINDOW_RADIUS,
        help="Line radius for trajectory edit-step IoU (default 1).",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction-json", type=Path)
    source.add_argument("--batch-output-item-json", type=Path)
    source.add_argument("--batch-output-jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir: Path = args.bundle_dir
    if not bundle_dir.exists():
        raise SystemExit(f"Bundle directory not found: {bundle_dir}")

    manifest = _load_bundle_manifest(bundle_dir)
    custom_id = str(manifest["custom_id"])
    observed_repair_target_path = _resolve_bundle_artifact_path(
        bundle_dir=bundle_dir,
        manifest_value=manifest["observed_next_repair_target_path"],
        expected_filename="observed_next_repair_target.json",
    )
    observed_coarse_path_path = _resolve_bundle_artifact_path(
        bundle_dir=bundle_dir,
        manifest_value=manifest["observed_next_coarse_path_path"],
        expected_filename="observed_next_coarse_path.json",
    )
    observed_repair_target = _load_json_strict(observed_repair_target_path)
    observed_coarse_path = _load_json_strict(observed_coarse_path_path)

    if args.prediction_json is not None:
        response_payload, source_meta = _prediction_from_prediction_json(args.prediction_json)
    elif args.batch_output_item_json is not None:
        response_payload, source_meta = _prediction_from_batch_output_item_json(
            path=args.batch_output_item_json,
            expected_custom_id=custom_id,
        )
    elif args.batch_output_jsonl is not None:
        response_payload, source_meta = _prediction_from_batch_output_jsonl(
            path=args.batch_output_jsonl,
            expected_custom_id=custom_id,
        )
    else:
        raise FullTraceBundleScoringError("Exactly one response source must be provided")

    scored = score_full_trace_prediction_v2(
        response_payload=response_payload,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
        footprint_window_radius=args.footprint_window_radius,
        edit_step_window_radius=args.edit_step_window_radius,
    )
    artifact: dict[str, Any] = {
        "schema_version": "v6_2_full_trace_scored_bundle_v2",
        "scorer_schema_version": SCHEMA_VERSION,
        "custom_id": custom_id,
        "bundle_dir": str(bundle_dir.resolve()),
        "manifest_path": str((bundle_dir / "manifest.json").resolve()),
        "observed_next_repair_target_path": str(observed_repair_target_path.resolve()),
        "observed_next_coarse_path_path": str(observed_coarse_path_path.resolve()),
        "response_source": source_meta,
        "validated_response": response_payload,
        "scored_prediction": scored,
    }

    out_path: Path = args.out if args.out is not None else bundle_dir / "scored_prediction_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"custom_id": custom_id, "out_path": str(out_path.resolve())}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
