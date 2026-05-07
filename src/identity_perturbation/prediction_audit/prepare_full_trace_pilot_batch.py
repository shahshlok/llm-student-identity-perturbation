from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_SCRIPT = REPO_ROOT / "src/identity_perturbation/prediction_audit/build_full_trace_prototype.py"
DEFAULT_PILOT_PROBE = (
    REPO_ROOT / "data/v62/probes/full_trace_same_task_family_2plus_pilot10_v1/probe.json"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "data/v62/batch_runs/full_trace_same_task_family_2plus_pilot10_v1"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
SCHEMA_VERSION = "v6_2_full_trace_pilot_batch_manifest_v1"
EXPECTED_PILOT_ROW_COUNT = 10
DEFAULT_COMPLETION_WINDOW = "24h"


class PrepareFullTracePilotBatchError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the fixed 10-row v6.2 full_trace pilot bundles and aggregate one batch requests.jsonl."
    )
    parser.add_argument("--pilot-probe", type=Path, default=DEFAULT_PILOT_PROBE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    return parser.parse_args()


def _load_json_strict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PrepareFullTracePilotBatchError(f"JSON file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PrepareFullTracePilotBatchError(f"Top-level JSON is not an object in {path}")
    return payload


def _resolve_repo_relative(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _row_id_from_fields(row: dict[str, Any]) -> str:
    required = ["class_id", "assessment_id", "exercise_id", "student_id", "transition_index_0idx"]
    for key in required:
        if key not in row:
            raise PrepareFullTracePilotBatchError(f"Pilot row missing required key {key!r}: {row}")
    return (
        f"{row['class_id']}:{row['assessment_id']}:{row['exercise_id']}:"
        f"{row['student_id']}:{row['transition_index_0idx']}"
    )


def _build_one_bundle(
    *,
    repo_root: Path,
    bundles_root: Path,
    row: dict[str, Any],
    model: str,
    reasoning_effort: str,
) -> Path:
    required = [
        "class_id",
        "assessment_id",
        "exercise_id",
        "student_id",
        "transition_index_0idx",
        "row_id",
    ]
    for key in required:
        if key not in row:
            raise PrepareFullTracePilotBatchError(f"Pilot row missing required key {key!r}: {row}")
    expected_row_id = _row_id_from_fields(row)
    if row["row_id"] != expected_row_id:
        raise PrepareFullTracePilotBatchError(
            f"Pilot row_id does not match its fields: row_id={row['row_id']!r}, expected={expected_row_id!r}"
        )
    cmd = [
        sys.executable,
        str(BUILDER_SCRIPT),
        "--class-id",
        str(row["class_id"]),
        "--assessment-id",
        str(row["assessment_id"]),
        "--exercise-id",
        str(row["exercise_id"]),
        "--student-id",
        str(row["student_id"]),
        "--transition-index",
        str(row["transition_index_0idx"]),
        "--out-root",
        str(bundles_root),
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort,
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PrepareFullTracePilotBatchError(
            f"Bundle build failed for {row['row_id']} with return code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    bundle_path = Path(completed.stdout.strip().splitlines()[-1])
    if not bundle_path.exists():
        raise PrepareFullTracePilotBatchError(
            f"Builder reported bundle path that does not exist for {row['row_id']}: {bundle_path}"
        )
    return bundle_path


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    pilot_probe_path = _resolve_repo_relative(args.pilot_probe)
    out_root = _resolve_repo_relative(args.out_root)
    pilot_probe = _load_json_strict(pilot_probe_path)
    if pilot_probe.get("schema_version") != "v6_2_full_trace_same_task_family_2plus_pilot10_v1":
        raise SystemExit(
            f"Unexpected pilot probe schema_version: {pilot_probe.get('schema_version')}"
        )
    rows = pilot_probe.get("selected_rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Pilot probe missing selected_rows")
    if len(rows) != EXPECTED_PILOT_ROW_COUNT:
        raise SystemExit(
            f"Pilot probe must contain exactly {EXPECTED_PILOT_ROW_COUNT} rows, found {len(rows)}"
        )
    if pilot_probe.get("selected_row_count") != EXPECTED_PILOT_ROW_COUNT:
        raise SystemExit(
            f"Pilot probe selected_row_count must be {EXPECTED_PILOT_ROW_COUNT}, "
            f"got {pilot_probe.get('selected_row_count')!r}"
        )

    out_root.mkdir(parents=True, exist_ok=False)
    bundles_root = out_root / "bundles"
    requests_path = out_root / "requests.jsonl"
    manifest_path = out_root / "manifest.json"
    output_path = out_root / "output.jsonl"
    error_path = out_root / "errors.jsonl"

    ordered_custom_ids: list[str] = []
    bundle_map: dict[str, dict[str, Any]] = {}
    request_lines: list[str] = []
    seen_row_ids: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            raise PrepareFullTracePilotBatchError("Pilot row is not an object")
        row_id = row.get("row_id")
        if not isinstance(row_id, str):
            raise PrepareFullTracePilotBatchError(f"Pilot row missing row_id: {row}")
        if row_id in seen_row_ids:
            raise PrepareFullTracePilotBatchError(f"Duplicate row_id in pilot probe: {row_id}")
        seen_row_ids.add(row_id)
        bundle_dir = _build_one_bundle(
            repo_root=repo_root,
            bundles_root=bundles_root,
            row=row,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        manifest = _load_json_strict(bundle_dir / "manifest.json")
        custom_id = manifest.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise PrepareFullTracePilotBatchError(
                f"Bundle manifest missing custom_id: {bundle_dir}"
            )
        if custom_id != row_id:
            raise PrepareFullTracePilotBatchError(
                f"Built bundle custom_id does not match pilot row_id: custom_id={custom_id!r}, row_id={row_id!r}"
            )
        if custom_id in bundle_map:
            raise PrepareFullTracePilotBatchError(
                f"Duplicate custom_id while preparing pilot batch: {custom_id}"
            )
        batch_request_rel = manifest.get("batch_request_jsonl_path")
        if not isinstance(batch_request_rel, str):
            raise PrepareFullTracePilotBatchError(
                f"Bundle manifest missing batch_request_jsonl_path: {bundle_dir}"
            )
        batch_request_path = bundle_dir / batch_request_rel
        if not batch_request_path.exists():
            raise PrepareFullTracePilotBatchError(
                f"Bundle batch request JSONL missing: {batch_request_path}"
            )
        bundle_request_lines = [
            line
            for line in batch_request_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(bundle_request_lines) != 1:
            raise PrepareFullTracePilotBatchError(
                f"Bundle request JSONL must contain exactly one nonblank line, found {len(bundle_request_lines)}: {batch_request_path}"
            )
        try:
            request_item = json.loads(bundle_request_lines[0])
        except json.JSONDecodeError as exc:
            raise PrepareFullTracePilotBatchError(
                f"Bundle request JSONL line is not valid JSON in {batch_request_path}: {exc}"
            ) from exc
        if not isinstance(request_item, dict):
            raise PrepareFullTracePilotBatchError(
                f"Bundle request JSONL line is not a JSON object: {batch_request_path}"
            )
        if request_item.get("custom_id") != custom_id:
            raise PrepareFullTracePilotBatchError(
                f"Bundle request custom_id mismatch in {batch_request_path}: expected {custom_id!r}, got {request_item.get('custom_id')!r}"
            )
        request_lines.append(bundle_request_lines[0])
        ordered_custom_ids.append(custom_id)
        bundle_map[custom_id] = {
            "slot_name": row["slot_name"],
            "row_id": row["row_id"],
            "bundle_dir": str(bundle_dir.resolve()),
            "bundle_manifest_path": str((bundle_dir / "manifest.json").resolve()),
            "request_body_path": str((bundle_dir / manifest["request_body_path"]).resolve()),
            "observed_next_repair_target_path": str(
                (bundle_dir / manifest["observed_next_repair_target_path"]).resolve()
            ),
            "observed_next_coarse_path_path": str(
                (bundle_dir / manifest["observed_next_coarse_path_path"]).resolve()
            ),
        }

    if len(request_lines) != EXPECTED_PILOT_ROW_COUNT:
        raise PrepareFullTracePilotBatchError(
            f"Expected exactly {EXPECTED_PILOT_ROW_COUNT} aggregated requests, found {len(request_lines)}"
        )
    requests_path.write_text("\n".join(request_lines) + "\n", encoding="utf-8")
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_name": out_root.name,
        "pilot_probe_path": str(pilot_probe_path.resolve()),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "counts": {
            "rows": EXPECTED_PILOT_ROW_COUNT,
            "requests_created": len(request_lines),
        },
        "paths": {
            "bundles_root": str(bundles_root.resolve()),
            "requests_jsonl": str(requests_path.resolve()),
            "output_jsonl": str(output_path.resolve()),
            "error_jsonl": str(error_path.resolve()),
        },
        "batch": {
            "endpoint": "/v1/responses",
            "completion_window": DEFAULT_COMPLETION_WINDOW,
            "input_file_id": None,
            "batch_id": None,
            "status": "prepared",
            "output_file_id": None,
            "error_file_id": None,
        },
        "ordered_custom_ids": ordered_custom_ids,
        "bundle_map": bundle_map,
    }
    manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"run_root": str(out_root.resolve()), "request_count": len(request_lines)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
