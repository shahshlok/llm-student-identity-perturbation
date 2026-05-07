from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_perturbation.codebench_support.runner import ROOT, write_json

from .v61_encoding_report import build_report

DEFAULT_DATA_ROOT = ROOT.parent / "tracer" / "2024-1"
DEFAULT_OUT_ROOT = ROOT / "data" / "v61" / "transition_manifests"


class V61SelectionError(ValueError):
    pass


PROFILE_RULES = {
    "clean126": lambda row: row["case_type"] != "large_dense",
    "clean121": lambda row: row["case_type"] not in {"large_dense", "submit_only"},
    "tight111": lambda row: int(row["prompt_chars"]) <= 120000,
    "core90": lambda row: row["case_type"] == "repair_focused" and not bool(row["flagged_case"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a clean v6.1 study set with full logs.")
    parser.add_argument("--transition-manifest", required=True, type=Path)
    parser.add_argument("--manifest-name", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_RULES), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--idle-gap-seconds", type=float, default=30.0)
    parser.add_argument("--include-keyhandled", action="store_true")
    parser.add_argument("--exclude-navigation", action="store_true")
    return parser.parse_args()


def _load_source_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    rows = payload.get("selected_transitions")
    if not isinstance(rows, list) or not rows:
        raise V61SelectionError(f"Source manifest missing selected_transitions: {path}")
    return rows


def main() -> int:
    args = parse_args()
    source_rows = _load_source_rows(args.transition_manifest)
    source_index = {
        f"{row['class_id']}:{row['assessment_id']}:{row['exercise_id']}:{row['student_id']}:{row['transition_index_0idx']}": row
        for row in source_rows
    }
    report = build_report(
        transition_manifest=args.transition_manifest,
        data_root=args.data_root,
        idle_gap_seconds=args.idle_gap_seconds,
        include_keyhandled=args.include_keyhandled,
        include_navigation=not args.exclude_navigation,
    )
    rule = PROFILE_RULES[args.profile]
    selected_rows = [row for row in report["rows"] if rule(row)]
    if not selected_rows:
        raise V61SelectionError(f"Profile {args.profile} selected zero rows")

    selected_transitions = []
    for row in selected_rows:
        source_row = source_index.get(str(row["id"]))
        if source_row is None:
            raise V61SelectionError(f"Selected row missing from source manifest: {row['id']}")
        selected_transitions.append(source_row)

    out_dir = args.out_root / args.manifest_name
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = out_dir / "transition_manifest.json"
    report_path = out_dir / "encoding_report.json"

    manifest = {
        "schema_version": "v6_1_clean_transition_manifest_v1",
        "manifest_name": args.manifest_name,
        "source_transition_manifest": str(args.transition_manifest.resolve()),
        "profile": args.profile,
        "selection_policy": {
            "profile": args.profile,
            "idle_gap_seconds": args.idle_gap_seconds,
            "include_keyhandled": args.include_keyhandled,
            "include_navigation": not args.exclude_navigation,
        },
        "summary": {
            "selected_transition_count": len(selected_transitions),
        },
        "selected_transitions": selected_transitions,
    }
    write_json(manifest_path, manifest)
    write_json(report_path, report)

    print(
        json.dumps(
            {
                "transition_count": len(selected_transitions),
                "profile": args.profile,
                "manifest": str(manifest_path),
                "encoding_report": str(report_path),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
