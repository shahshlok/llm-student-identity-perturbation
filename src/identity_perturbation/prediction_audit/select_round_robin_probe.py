from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean, median

DEFAULT_AUDIT_REPORT = Path("data/v62/raw_same_task_family_audit/report.json")
DEFAULT_OUT_ROOT = Path("data/v62/probes")


class V62SelectionError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a conservative round-robin probe from a v6.2 raw audit report."
    )
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--probe-name", required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--exclude-transition-id",
        action="append",
        default=[],
        help=(
            "Transition id to exclude before selection, formatted as "
            "class_id:assessment_id:exercise_id:student_id:transition_index_0idx."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=(
            "raw_round_robin",
            "mixed_fair_deepest_per_log",
            "history_first_then_true_two_attempt_backfill",
        ),
        default="raw_round_robin",
    )
    return parser.parse_args()


def load_audit_rows(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not path.exists():
        raise V62SelectionError(f"Audit report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v6_2_raw_same_task_family_audit_v1":
        raise V62SelectionError(
            f"Unexpected audit schema in {path}: {payload.get('schema_version')}"
        )
    rows = payload.get("candidate_transitions")
    if not isinstance(rows, list) or not rows:
        raise V62SelectionError(f"Audit report has no candidate_transitions: {path}")
    return payload, rows


def transition_id(row: dict[str, object]) -> str:
    return (
        f"{row['class_id']}:{row['assessment_id']}:{row['exercise_id']}:"
        f"{row['student_id']}:{row['transition_index_0idx']}"
    )


def filter_excluded_rows(
    rows: list[dict[str, object]],
    *,
    excluded_transition_ids: list[str],
) -> list[dict[str, object]]:
    if not excluded_transition_ids:
        return rows
    excluded = set(excluded_transition_ids)
    available = {transition_id(row) for row in rows}
    missing = sorted(excluded - available)
    if missing:
        raise V62SelectionError(
            f"Excluded transition ids were not found in the audit rows: {missing}"
        )
    return [row for row in rows if transition_id(row) not in excluded]


def ordered_slice_queues(rows: list[dict[str, object]]) -> list[deque[dict[str, object]]]:
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        slice_key = (str(row["class_id"]), str(row["assessment_id"]), str(row["exercise_id"]))
        student_id = str(row["student_id"])
        grouped[slice_key][student_id].append(row)

    queues: list[deque[dict[str, object]]] = []
    for slice_key in sorted(grouped):
        student_groups = grouped[slice_key]
        for student_rows in student_groups.values():
            student_rows.sort(key=lambda row: int(row["transition_index_0idx"]))
        student_queues = {
            student_id: deque(student_rows)
            for student_id, student_rows in sorted(student_groups.items())
        }
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
        queues.append(deque(slice_ordered))
    return queues


def preprocess_rows(
    rows: list[dict[str, object]],
    *,
    selection_mode: str,
) -> list[dict[str, object]]:
    if selection_mode == "raw_round_robin":
        return rows
    if selection_mode == "mixed_fair_deepest_per_log":
        grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            key = (
                str(row["class_id"]),
                str(row["assessment_id"]),
                str(row["exercise_id"]),
                str(row["student_id"]),
            )
            grouped[key].append(row)
        chosen: list[dict[str, object]] = []
        for key in sorted(grouped):
            items = grouped[key]
            items.sort(key=lambda row: int(row["transition_index_0idx"]))
            total_attempt_count = int(items[0]["total_attempt_count"])
            if total_attempt_count < 2:
                raise V62SelectionError(f"Encountered invalid total_attempt_count<2 for {key}")
            chosen.append(items[0] if total_attempt_count == 2 else items[-1])
        return chosen
    if selection_mode == "history_first_then_true_two_attempt_backfill":
        grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            key = (
                str(row["class_id"]),
                str(row["assessment_id"]),
                str(row["exercise_id"]),
                str(row["student_id"]),
            )
            grouped[key].append(row)
        richer_logs: list[dict[str, object]] = []
        two_attempt_logs: list[dict[str, object]] = []
        for key in sorted(grouped):
            items = grouped[key]
            items.sort(key=lambda row: int(row["transition_index_0idx"]))
            total_attempt_count = int(items[0]["total_attempt_count"])
            if total_attempt_count < 2:
                raise V62SelectionError(f"Encountered invalid total_attempt_count<2 for {key}")
            if total_attempt_count == 2:
                two_attempt_logs.append(items[0])
                continue
            deepest = items[-1]
            if int(deepest["transition_index_0idx"]) < 1:
                continue
            richer_logs.append(deepest)
        return richer_logs + two_attempt_logs
    raise V62SelectionError(f"Unsupported selection mode: {selection_mode}")


def select_rows(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count < 1:
        raise V62SelectionError(f"Selection count must be positive, got {count}")
    if count > len(rows):
        raise V62SelectionError(f"Requested {count} rows but only {len(rows)} are available")

    queues = ordered_slice_queues(rows)
    selected: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str, str, int]] = set()

    while len(selected) < count:
        advanced = False
        for queue in queues:
            if not queue:
                continue
            advanced = True
            row = queue.popleft()
            key = (
                str(row["class_id"]),
                str(row["assessment_id"]),
                str(row["exercise_id"]),
                str(row["student_id"]),
                int(row["transition_index_0idx"]),
            )
            if key in seen_keys:
                raise V62SelectionError(f"Duplicate transition selected: {key}")
            seen_keys.add(key)
            selected.append(row)
            if len(selected) == count:
                break
        if not advanced:
            raise V62SelectionError(
                f"Selection terminated early at {len(selected)} rows while targeting {count}"
            )
    return selected


def select_rows_history_first_backfill(
    rows: list[dict[str, object]], count: int
) -> list[dict[str, object]]:
    richer_rows = [row for row in rows if int(row["total_attempt_count"]) >= 3]
    true_two_attempt_rows = [row for row in rows if int(row["total_attempt_count"]) == 2]
    selected = select_rows(richer_rows, min(count, len(richer_rows)))
    if len(selected) == count:
        return selected
    selected_keys = {
        (
            str(row["class_id"]),
            str(row["assessment_id"]),
            str(row["exercise_id"]),
            str(row["student_id"]),
            int(row["transition_index_0idx"]),
        )
        for row in selected
    }
    remaining_two_attempt_rows = []
    for row in true_two_attempt_rows:
        key = (
            str(row["class_id"]),
            str(row["assessment_id"]),
            str(row["exercise_id"]),
            str(row["student_id"]),
            int(row["transition_index_0idx"]),
        )
        if key in selected_keys:
            raise V62SelectionError(f"Duplicate row encountered during backfill: {key}")
        remaining_two_attempt_rows.append(row)
    backfill_count = count - len(selected)
    selected.extend(select_rows(remaining_two_attempt_rows, backfill_count))
    return selected


def metric_summary(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    values = [int(row[field]) for row in rows]
    if not values:
        raise V62SelectionError(f"No values found for metric field {field!r}")
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": median(values),
        "mean": round(mean(values), 3),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def write_probe(
    *,
    audit_report_path: Path,
    audit_payload: dict[str, object],
    selected_rows: list[dict[str, object]],
    count: int,
    probe_name: str,
    out_root: Path,
    selection_mode: str,
    excluded_transition_ids: list[str],
) -> None:
    out_dir = out_root / probe_name
    out_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "schema_version": "v6_2_round_robin_probe_v1",
        "source_audit_report": str(audit_report_path.resolve()),
        "selection_policy": {
            "mode": selection_mode,
            "requested_count": count,
            "excluded_transition_ids": excluded_transition_ids,
        },
        "summary": {
            "selected_count": len(selected_rows),
            "class_count": len({str(row["class_id"]) for row in selected_rows}),
            "assessment_count": len(
                {(str(row["class_id"]), str(row["assessment_id"])) for row in selected_rows}
            ),
            "exercise_count": len(
                {
                    (str(row["class_id"]), str(row["assessment_id"]), str(row["exercise_id"]))
                    for row in selected_rows
                }
            ),
            "student_count": len({str(row["student_id"]) for row in selected_rows}),
        },
        "metric_summary": {
            field: metric_summary(selected_rows, field)
            for field in (
                "visible_attempt_count",
                "visible_run_count",
                "visible_anchor_count",
                "visible_change_event_count",
            )
        },
        "scope": audit_payload["scope"],
        "selected_transitions": selected_rows,
    }
    (out_dir / "probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# V6.2 Round-Robin Probe",
        "",
        f"- Source audit: `{audit_report_path.resolve()}`",
        f"- Selected count: `{report['summary']['selected_count']}`",
        f"- Class count: `{report['summary']['class_count']}`",
        f"- Assessment count: `{report['summary']['assessment_count']}`",
        f"- Exercise count: `{report['summary']['exercise_count']}`",
        f"- Student count: `{report['summary']['student_count']}`",
        "",
        "## Metric Summary",
        "",
        "| Metric | Min | Median | Mean | P90 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, summary in report["metric_summary"].items():
        lines.append(
            f"| {field} | {summary['min']} | {summary['median']} | {summary['mean']} | "
            f"{summary['p90']} | {summary['max']} |"
        )
    (out_dir / "probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    audit_payload, rows = load_audit_rows(args.audit_report)
    filtered_rows = filter_excluded_rows(
        rows,
        excluded_transition_ids=[str(item) for item in args.exclude_transition_id],
    )
    preprocessed_rows = preprocess_rows(filtered_rows, selection_mode=args.selection_mode)
    if args.selection_mode == "history_first_then_true_two_attempt_backfill":
        selected_rows = select_rows_history_first_backfill(preprocessed_rows, args.count)
    else:
        selected_rows = select_rows(preprocessed_rows, args.count)
    write_probe(
        audit_report_path=args.audit_report,
        audit_payload=audit_payload,
        selected_rows=selected_rows,
        count=args.count,
        probe_name=args.probe_name,
        out_root=args.out_root,
        selection_mode=args.selection_mode,
        excluded_transition_ids=[str(item) for item in args.exclude_transition_id],
    )
    print(
        json.dumps(
            {
                "selected_count": len(selected_rows),
                "probe_name": args.probe_name,
                "out_dir": str((args.out_root / args.probe_name).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
