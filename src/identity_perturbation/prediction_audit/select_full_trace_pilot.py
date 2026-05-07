from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SMOKE_PROBE = (
    REPO_ROOT / "data/v62/probes/full_trace_same_task_family_2plus_smoke_v1/probe.json"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "data/v62/probes/full_trace_same_task_family_2plus_pilot10_v1"
SCHEMA_VERSION = "v6_2_full_trace_same_task_family_2plus_pilot10_v1"

PILOT_SLOTS: tuple[tuple[str, str], ...] = (
    ("slot_1_v2_edit_submit", "590:5843:1187:9880:1"),
    ("slot_2_v2_edit_local_run_submit", "589:5897:6353:9735:1"),
    ("slot_3_v2_local_run_edit_submit", "591:5853:1186:9788:1"),
    ("slot_4_v2_longer_path", "591:5902:6354:9782:1"),
    ("slot_5_v3_edit_submit", "593:5834:6316:9709:2"),
    ("slot_6_v3_nontrivial_path", "590:5843:6656:9806:2"),
    ("slot_7_v4_nontrivial_path", "589:5897:6353:9755:3"),
    ("slot_8_v5_nontrivial_path", "589:5897:3189:9741:4"),
    ("slot_9_v8_local_run_edit_submit", "589:5897:6353:9755:7"),
    ("slot_10_v13_long_history", "589:5846:3232:9746:12"),
)


class SelectFullTracePilotError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the fixed 10-row v6.2 full_trace pilot from the 2+ same-task-family smoke probe."
    )
    parser.add_argument("--smoke-probe", type=Path, default=DEFAULT_SMOKE_PROBE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def _resolve_repo_relative(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_pilot_probe(*, smoke_probe: dict[str, Any], source_probe_path: Path) -> dict[str, Any]:
    rows = smoke_probe.get("selected_rows")
    if not isinstance(rows, list):
        raise SelectFullTracePilotError("Smoke probe missing selected_rows")
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SelectFullTracePilotError("Smoke probe row is not an object")
        row_id = row.get("row_id")
        if not isinstance(row_id, str):
            raise SelectFullTracePilotError("Smoke probe row missing row_id")
        if row_id in row_by_id:
            raise SelectFullTracePilotError(f"Duplicate row_id in smoke probe: {row_id}")
        row_by_id[row_id] = row

    selected_rows: list[dict[str, Any]] = []
    seen_row_ids: set[str] = set()
    for slot_name, row_id in PILOT_SLOTS:
        if row_id in seen_row_ids:
            raise SelectFullTracePilotError(f"Duplicate pilot row_id in slot map: {row_id}")
        row = row_by_id.get(row_id)
        if row is None:
            raise SelectFullTracePilotError(f"Pilot row_id not found in smoke probe: {row_id}")
        selected_rows.append(
            {
                **row,
                "slot_name": slot_name,
            }
        )
        seen_row_ids.add(row_id)
    if len(selected_rows) != len(PILOT_SLOTS):
        raise SelectFullTracePilotError(
            f"Expected exactly {len(PILOT_SLOTS)} pilot rows, built {len(selected_rows)}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_probe_schema_version": smoke_probe.get("schema_version"),
        "source_probe_path": str(source_probe_path.resolve()),
        "selection_method": "fixed_curated_slots_v1",
        "selected_row_count": len(selected_rows),
        "selected_rows": selected_rows,
    }


def main() -> int:
    args = parse_args()
    smoke_probe_path = _resolve_repo_relative(args.smoke_probe)
    out_root = _resolve_repo_relative(args.out_root)
    if not smoke_probe_path.exists():
        raise SystemExit(f"Smoke probe not found: {smoke_probe_path}")
    smoke_probe = json.loads(smoke_probe_path.read_text(encoding="utf-8"))
    pilot_probe = build_pilot_probe(smoke_probe=smoke_probe, source_probe_path=smoke_probe_path)
    out_root.mkdir(parents=True, exist_ok=False)
    out_path = out_root / "probe.json"
    out_path.write_text(
        json.dumps(pilot_probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(out_path.resolve())
    print(
        json.dumps(
            {"selected_row_count": pilot_probe["selected_row_count"]}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
