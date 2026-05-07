from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

from identity_perturbation.codebench_support.runner import ROOT, SliceBuildError

from .transition_manifest import build_transition_manifest

DEFAULT_TRANSITION_SPACE_REPORT = (
    ROOT
    / "data"
    / "v6"
    / "transition_space_reports"
    / "lab56_all7classes_transition_space"
    / "transition_space.json"
)


class V6TransitionSpaceSelectionError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a reproducible v6 transition set from a v6 transition-space report."
    )
    parser.add_argument(
        "--transition-space-report",
        type=Path,
        default=DEFAULT_TRANSITION_SPACE_REPORT,
        help="Path to transition_space.json",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of transitions to select",
    )
    parser.add_argument(
        "--allowed-tier",
        action="append",
        default=[],
        help="Validation tier to allow; may be repeated. Defaults to strict_3head and broad_2head.",
    )
    parser.add_argument(
        "--quality-policy",
        choices=("none", "tier_then_alignment"),
        default="tier_then_alignment",
        help="How to prioritize higher-quality rows before filling the requested count",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "v6" / "transition_manifests",
        help="Artifact root for v6 transition manifests",
    )
    parser.add_argument(
        "--manifest-name",
        required=True,
        help="Subdirectory name for the output manifest bundle",
    )
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, object]:
    if not path.exists():
        raise V6TransitionSpaceSelectionError(f"Transition-space report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v6_transition_space_report_v1":
        raise V6TransitionSpaceSelectionError(
            f"Unexpected transition-space report schema in {path}: {payload.get('schema_version')}"
        )
    return payload


def _allowed_tiers(args: argparse.Namespace) -> tuple[str, ...]:
    if args.allowed_tier:
        allowed = tuple(dict.fromkeys(str(value) for value in args.allowed_tier))
    else:
        allowed = ("strict_3head", "broad_2head")
    for value in allowed:
        if value not in {"strict_3head", "broad_2head"}:
            raise V6TransitionSpaceSelectionError(
                f"Unsupported allowed tier {value!r}; expected strict_3head or broad_2head"
            )
    return allowed


def build_selected_transition_specs(
    *,
    transition_space_report_path: Path,
    count: int,
    allowed_tiers: tuple[str, ...],
    quality_policy: str,
) -> tuple[str, ...]:
    if count < 1:
        raise V6TransitionSpaceSelectionError(f"--count must be positive; got {count}")

    report = _load_report(transition_space_report_path)
    rows = report.get("transitions")
    if not isinstance(rows, list) or not rows:
        raise V6TransitionSpaceSelectionError(
            f"Transition-space report must contain a non-empty transitions list: {transition_space_report_path}"
        )

    eligible_rows: list[dict[str, object]] = []
    for row in rows:
        if not bool(row.get("promptable")):
            continue
        validation_tier = str(row.get("validation_tier"))
        if validation_tier not in allowed_tiers:
            continue
        eligible_rows.append(row)

    if not eligible_rows:
        raise V6TransitionSpaceSelectionError(
            f"No promptable transitions matched allowed tiers {list(allowed_tiers)}"
        )

    def bucket_key(row: dict[str, object]) -> tuple[str, str]:
        return (str(row["validation_tier"]), str(row.get("alignment_status") or "none"))

    def ordered_buckets(rows_in: list[dict[str, object]]) -> list[list[dict[str, object]]]:
        if quality_policy == "none":
            return [rows_in]
        ordered_keys = [
            ("strict_3head", "matched"),
            ("strict_3head", "partial"),
            ("broad_2head", "matched"),
            ("broad_2head", "partial"),
        ]
        bucketed: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in rows_in:
            bucketed[bucket_key(row)].append(row)
        ordered: list[list[dict[str, object]]] = []
        for key in ordered_keys:
            rows_for_key = bucketed.get(key, [])
            if rows_for_key:
                ordered.append(rows_for_key)
        leftovers = [
            rows_for_key
            for key, rows_for_key in bucketed.items()
            if key not in ordered_keys and rows_for_key
        ]
        ordered.extend(leftovers)
        return ordered

    def per_slice_queues_for_rows(
        rows_in: list[dict[str, object]],
    ) -> list[deque[dict[str, object]]]:
        grouped: dict[tuple[str, str, str], dict[str, list[dict[str, object]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows_in:
            slice_key = (
                str(row["class_id"]),
                str(row["assessment_id"]),
                str(row["exercise_id"]),
            )
            grouped[slice_key][str(row["student_id"])].append(row)

        per_slice_queues: list[deque[dict[str, object]]] = []
        for slice_key in sorted(grouped):
            rows_by_student = grouped[slice_key]
            student_queues: dict[str, deque[dict[str, object]]] = {}
            for student_id in sorted(rows_by_student):
                ordered_rows = sorted(
                    rows_by_student[student_id],
                    key=lambda row: int(row["transition_index_0idx"]),
                )
                student_queues[student_id] = deque(ordered_rows)

            slice_ordered: list[dict[str, object]] = []
            while True:
                advanced = False
                for student_id in sorted(student_queues):
                    queue = student_queues[student_id]
                    if not queue:
                        continue
                    advanced = True
                    slice_ordered.append(queue.popleft())
                if not advanced:
                    break
            per_slice_queues.append(deque(slice_ordered))
        return per_slice_queues

    buckets = ordered_buckets(eligible_rows)
    total_available = sum(len(bucket) for bucket in buckets)
    if count > total_available:
        raise V6TransitionSpaceSelectionError(
            f"Requested {count} transitions but only {total_available} are available "
            f"for tiers {list(allowed_tiers)}"
        )

    selected_rows: list[dict[str, object]] = []
    for bucket in buckets:
        per_slice_queues = per_slice_queues_for_rows(bucket)
        while len(selected_rows) < count:
            advanced = False
            for queue in per_slice_queues:
                if not queue:
                    continue
                advanced = True
                selected_rows.append(queue.popleft())
                if len(selected_rows) == count:
                    break
            if not advanced:
                break
        if len(selected_rows) == count:
            break

    if len(selected_rows) != count:
        raise V6TransitionSpaceSelectionError(
            f"Selection terminated early: expected {count}, got {len(selected_rows)}"
        )

    return tuple(
        (
            f"{row['class_id']}:{row['assessment_id']}:{row['exercise_id']}:"
            f"{row['student_id']}:{row['transition_index_0idx']}"
        )
        for row in selected_rows
    )


def main() -> int:
    args = parse_args()
    allowed_tiers = _allowed_tiers(args)
    try:
        transition_specs = build_selected_transition_specs(
            transition_space_report_path=args.transition_space_report,
            count=args.count,
            allowed_tiers=allowed_tiers,
            quality_policy=args.quality_policy,
        )
        result = build_transition_manifest(
            transition_specs=transition_specs,
            out_root=args.out_root,
            manifest_name=args.manifest_name,
        )
    except (V6TransitionSpaceSelectionError, SliceBuildError) as exc:
        raise SystemExit(str(exc)) from exc

    manifest = result["manifest"]
    manifest["selection_policy"] = {
        "mode": "v6_transition_space_round_robin",
        "source_transition_space_report": str(args.transition_space_report.resolve()),
        "requested_count": args.count,
        "allowed_tiers": list(allowed_tiers),
        "quality_policy": args.quality_policy,
        "slice_order": "sorted_class_assessment_exercise_round_robin",
        "student_order_within_slice": "sorted_student_round_robin",
        "transition_order_within_student": "ascending_transition_index_0idx",
    }
    manifest_path = Path(result["paths"]["manifest_json"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
