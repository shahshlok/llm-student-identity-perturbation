from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SCAN_REPORT = Path("data/v62/full_trace_same_task_family_scan_narrow_v1/report.json")
DEFAULT_OUT_ROOT = Path("data/v62/probes/full_trace_same_task_family_2plus_smoke_v1")
SCHEMA_VERSION = "v6_2_full_trace_same_task_family_2plus_smoke_v1"


class SelectFullTraceSmokeStratumError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the same-task-family full_trace smoke stratum with visible_attempt_count >= 2."
    )
    parser.add_argument("--scan-report", type=Path, default=DEFAULT_SCAN_REPORT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--min-visible-attempts", type=int, default=2)
    return parser.parse_args()


def _row_id_parts(row_id: str) -> dict[str, str | int]:
    parts = row_id.split(":")
    if len(parts) != 5:
        raise SelectFullTraceSmokeStratumError(f"Unexpected row_id format: {row_id}")
    class_id, assessment_id, exercise_id, student_id, transition_index = parts
    return {
        "class_id": class_id,
        "assessment_id": assessment_id,
        "exercise_id": exercise_id,
        "student_id": student_id,
        "transition_index_0idx": int(transition_index),
    }


def build_probe(*, scan_report: dict[str, Any], min_visible_attempts: int) -> dict[str, Any]:
    rows = scan_report.get("admissible_rows")
    if not isinstance(rows, list):
        raise SelectFullTraceSmokeStratumError("Scan report missing admissible_rows")
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SelectFullTraceSmokeStratumError("Admissible row is not a dict")
        visible_attempt_count = row.get("visible_attempt_count")
        if not isinstance(visible_attempt_count, int):
            raise SelectFullTraceSmokeStratumError("Admissible row missing visible_attempt_count")
        if visible_attempt_count < min_visible_attempts:
            continue
        row_id = row.get("row_id")
        if not isinstance(row_id, str):
            raise SelectFullTraceSmokeStratumError("Admissible row missing row_id")
        selected.append(
            {
                **_row_id_parts(row_id),
                "row_id": row_id,
                "visible_attempt_count": visible_attempt_count,
                "total_attempt_count": int(row["total_attempt_count"]),
                "target_snapshot_index_0idx": int(row["target_snapshot_index_0idx"]),
                "coarse_path_action_sequence": list(row["coarse_path_action_sequence"]),
            }
        )
    selected.sort(
        key=lambda row: (
            int(row["visible_attempt_count"]),
            int(row["total_attempt_count"]),
            str(row["row_id"]),
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_scan_report": scan_report.get("schema_version"),
        "min_visible_attempts": min_visible_attempts,
        "selected_row_count": len(selected),
        "selected_rows": selected,
    }


def main() -> int:
    args = parse_args()
    scan_report = json.loads(args.scan_report.read_text(encoding="utf-8"))
    probe = build_probe(scan_report=scan_report, min_visible_attempts=args.min_visible_attempts)
    args.out_root.mkdir(parents=True, exist_ok=False)
    out_path = args.out_root / "probe.json"
    out_path.write_text(json.dumps(probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path.resolve())
    print(
        json.dumps(
            {"selected_row_count": probe["selected_row_count"]}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
