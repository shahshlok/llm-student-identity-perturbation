from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from identity_perturbation.codebench_support.alignment import AlignmentError, align_transition
from identity_perturbation.codebench_support.codemirror import infer_initial_code, parse_codemirror_log
from identity_perturbation.codebench_support.focus_region import FocusRegionError, first_focus_region_3way
from identity_perturbation.codebench_support.runner import (
    ROOT,
    SliceBuildError,
    load_student_transition_context,
    write_json,
    write_text,
)

from .payload import V6PayloadError, build_payload
from .runner import DEFAULT_DATA_ROOT
from .trace_card import V6TraceCardError, build_attempt_trace_cards

DEFAULT_AUDIT_ROOT = (
    ROOT / "data" / "v5" / "corpus_benchmarks" / "strong_lab56_min2s3t" / "assessment_audits"
)
DEFAULT_OUT_ROOT = ROOT / "data" / "v6" / "transition_space_reports"


class V6TransitionSpaceError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a v6 transition space from raw logs, separating promptability from validation tiers."
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=DEFAULT_AUDIT_ROOT,
        help="Root containing per-exercise audit.json files to enumerate candidate transitions",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Raw dataset root containing 2024-1/<class_id>/...",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Artifact root for v6 transition-space reports",
    )
    parser.add_argument(
        "--report-name",
        required=True,
        help="Subdirectory name for the output report bundle",
    )
    return parser.parse_args()


def _exercise_audit_paths(audit_root: Path) -> tuple[Path, ...]:
    if not audit_root.exists():
        raise V6TransitionSpaceError(f"Audit root not found: {audit_root}")
    paths = tuple(sorted(audit_root.rglob("audit.json")))
    if not paths:
        raise V6TransitionSpaceError(f"No exercise audit.json files found under {audit_root}")
    return paths


def _parse_exercise_audit_path(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    if len(parts) < 4:
        raise V6TransitionSpaceError(f"Unexpected audit path structure: {path}")
    return parts[-4], parts[-3], parts[-2]


def _load_transition_rows(audit_root: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for path in _exercise_audit_paths(audit_root):
        class_id, assessment_id, exercise_id = _parse_exercise_audit_path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        transitions = payload.get("transitions")
        if not isinstance(transitions, list):
            raise V6TransitionSpaceError(f"Exercise audit missing transitions list: {path}")
        for row in transitions:
            rows.append(
                {
                    "class_id": class_id,
                    "assessment_id": assessment_id,
                    "exercise_id": exercise_id,
                    "student_id": str(row["student_id"]),
                    "transition_index_0idx": int(row["transition_index_0idx"]),
                }
            )
    return tuple(rows)


def _exec_cm_paths(
    *,
    data_root: Path,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
) -> tuple[Path, Path]:
    student_root = data_root / class_id / "users" / student_id
    exec_path = student_root / "executions" / f"{assessment_id}_{exercise_id}.log"
    cm_path = student_root / "codemirror" / f"{assessment_id}_{exercise_id}.log"
    if not exec_path.exists():
        raise V6TransitionSpaceError(f"Execution log not found: {exec_path}")
    if not cm_path.exists():
        raise V6TransitionSpaceError(f"CodeMirror log not found: {cm_path}")
    return exec_path, cm_path


def _focus_region_reason(exc: FocusRegionError) -> str:
    message = str(exc)
    if message:
        return message.replace(" ", "_").replace("/", "_").lower()
    return "focus_region_error"


def build_transition_space_report(
    *,
    audit_root: Path,
    data_root: Path,
    out_root: Path,
    report_name: str,
) -> dict[str, object]:
    rows = _load_transition_rows(audit_root)
    report_dir = out_root / report_name
    report_dir.mkdir(parents=True, exist_ok=False)

    context_cache: dict[
        tuple[str, str, str, str],
        tuple[object, object, object, object] | dict[str, str],
    ] = {}
    transition_rows: list[dict[str, object]] = []
    summary_counts: Counter[str] = Counter()
    per_slice_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    for row in rows:
        class_id = str(row["class_id"])
        assessment_id = str(row["assessment_id"])
        exercise_id = str(row["exercise_id"])
        student_id = str(row["student_id"])
        transition_index = int(row["transition_index_0idx"])
        slice_key = (class_id, assessment_id, exercise_id)
        cache_key = (class_id, assessment_id, exercise_id, student_id)

        if cache_key not in context_cache:
            try:
                exec_path, cm_path = _exec_cm_paths(
                    data_root=data_root,
                    class_id=class_id,
                    assessment_id=assessment_id,
                    exercise_id=exercise_id,
                    student_id=student_id,
                )
                attempts, trace, transitions = load_student_transition_context(
                    class_id=class_id,
                    assessment_id=assessment_id,
                    exercise_id=exercise_id,
                    student_id=student_id,
                    exec_path=exec_path,
                    cm_path=cm_path,
                )
                cm_events = parse_codemirror_log(cm_path)
                initial_code = infer_initial_code(cm_events, attempts[0].code)
                trace_cards = build_attempt_trace_cards(cm_events, initial_code=initial_code)
                context_cache[cache_key] = (attempts, trace, transitions, trace_cards)
            except (SliceBuildError, V6TraceCardError) as exc:
                context_cache[cache_key] = {
                    "failure_stage": "prompt",
                    "failure_reason_code": str(exc).replace(" ", "_").lower(),
                }

        cached = context_cache[cache_key]
        if isinstance(cached, dict):
            record = {
                "class_id": class_id,
                "assessment_id": assessment_id,
                "exercise_id": exercise_id,
                "student_id": student_id,
                "transition_index_0idx": transition_index,
                "attempt_n_index_0idx": None,
                "attempt_n1_index_0idx": None,
                "promptable": False,
                "validation_tier": "none",
                "alignment_status": None,
                "failure_stage": cached["failure_stage"],
                "failure_reason_code": cached["failure_reason_code"],
            }
            transition_rows.append(record)
            summary_counts["total_transitions"] += 1
            summary_counts["promptable_false"] += 1
            summary_counts["validation_none"] += 1
            per_slice_counts[slice_key]["total_transitions"] += 1
            per_slice_counts[slice_key]["promptable_false"] += 1
            per_slice_counts[slice_key]["validation_none"] += 1
            continue

        attempts, trace, transitions, trace_cards = cached
        if transition_index < 0 or transition_index >= len(transitions):
            raise V6TransitionSpaceError(
                f"Transition index {transition_index} out of bounds for "
                f"{class_id}:{assessment_id}:{exercise_id}:{student_id}"
            )
        transition = transitions[transition_index]

        record = {
            "class_id": class_id,
            "assessment_id": assessment_id,
            "exercise_id": exercise_id,
            "student_id": student_id,
            "transition_index_0idx": transition_index,
            "attempt_n_index_0idx": transition.attempt_n.attempt_index_0idx,
            "attempt_n1_index_0idx": transition.attempt_n1.attempt_index_0idx,
            "promptable": False,
            "validation_tier": "none",
            "alignment_status": None,
            "failure_stage": None,
            "failure_reason_code": None,
        }

        try:
            build_payload(
                class_id=class_id,
                assessment_id=assessment_id,
                exercise_id=exercise_id,
                student_id=student_id,
                transition=transition,
                trace=trace,
                trace_cards=trace_cards,
            )
            record["promptable"] = True
        except (V6PayloadError, V6TraceCardError) as exc:
            record["failure_stage"] = "prompt"
            record["failure_reason_code"] = str(exc).replace(" ", "_").lower()
            transition_rows.append(record)
            summary_counts["total_transitions"] += 1
            summary_counts["promptable_false"] += 1
            summary_counts["validation_none"] += 1
            per_slice_counts[slice_key]["total_transitions"] += 1
            per_slice_counts[slice_key]["promptable_false"] += 1
            per_slice_counts[slice_key]["validation_none"] += 1
            continue

        summary_counts["promptable_true"] += 1
        per_slice_counts[slice_key]["promptable_true"] += 1

        try:
            alignment = align_transition(
                transition,
                trace.snapshots,
                trailing_code=trace.final_code,
                trailing_changes=trace.trailing_changes,
                allow_partial=True,
            )
            record["alignment_status"] = alignment.status
            record["validation_tier"] = "broad_2head"
            summary_counts["validation_broad_2head"] += 1
            per_slice_counts[slice_key]["validation_broad_2head"] += 1
            try:
                first_focus_region_3way(
                    transition.attempt_n.code,
                    alignment.first_change_line_0idx,
                )
                record["validation_tier"] = "strict_3head"
                summary_counts["validation_broad_2head"] -= 1
                summary_counts["validation_strict_3head"] += 1
                per_slice_counts[slice_key]["validation_broad_2head"] -= 1
                per_slice_counts[slice_key]["validation_strict_3head"] += 1
            except FocusRegionError as exc:
                record["failure_stage"] = "focus_region"
                record["failure_reason_code"] = _focus_region_reason(exc)
        except AlignmentError as exc:
            record["validation_tier"] = "none"
            record["failure_stage"] = "alignment"
            record["failure_reason_code"] = str(exc).replace(" ", "_").lower()
            summary_counts["validation_none"] += 1
            per_slice_counts[slice_key]["validation_none"] += 1

        transition_rows.append(record)
        summary_counts["total_transitions"] += 1
        per_slice_counts[slice_key]["total_transitions"] += 1

    slice_rows = []
    for (class_id, assessment_id, exercise_id), counts in sorted(per_slice_counts.items()):
        slice_rows.append(
            {
                "class_id": class_id,
                "assessment_id": assessment_id,
                "exercise_id": exercise_id,
                "total_transitions": counts["total_transitions"],
                "promptable_true": counts["promptable_true"],
                "validation_strict_3head": counts["validation_strict_3head"],
                "validation_broad_2head": counts["validation_broad_2head"],
                "validation_none": counts["validation_none"],
            }
        )

    summary = {
        "audit_root": str(audit_root.resolve()),
        "data_root": str(data_root.resolve()),
        "total_transitions": summary_counts["total_transitions"],
        "promptable_true": summary_counts["promptable_true"],
        "promptable_false": summary_counts["promptable_false"],
        "validation_strict_3head": summary_counts["validation_strict_3head"],
        "validation_broad_2head": summary_counts["validation_broad_2head"],
        "validation_none": summary_counts["validation_none"],
        "distinct_slice_count": len(slice_rows),
    }
    report = {
        "schema_version": "v6_transition_space_report_v1",
        "summary": summary,
        "by_slice": slice_rows,
        "transitions": transition_rows,
    }
    write_json(report_dir / "transition_space.json", report)

    lines = [
        "# V6 Transition Space Report",
        "",
        f"- Audit root: `{audit_root.resolve()}`",
        f"- Total transitions: `{summary['total_transitions']}`",
        f"- Promptable: `{summary['promptable_true']}`",
        f"- Strict 3-head validatable: `{summary['validation_strict_3head']}`",
        f"- Broad 2-head validatable: `{summary['validation_broad_2head']}`",
        f"- Unusable for current validation tiers: `{summary['validation_none']}`",
        "",
        "## By Slice",
        "",
        "| Class | Assessment | Exercise | Total | Promptable | Strict 3-head | Broad 2-head | None |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in slice_rows:
        lines.append(
            f"| {row['class_id']} | {row['assessment_id']} | {row['exercise_id']} | "
            f"{row['total_transitions']} | {row['promptable_true']} | "
            f"{row['validation_strict_3head']} | {row['validation_broad_2head']} | "
            f"{row['validation_none']} |"
        )
    write_text(report_dir / "transition_space.md", "\n".join(lines) + "\n")
    return {
        "report": report,
        "paths": {
            "report_dir": str(report_dir.resolve()),
            "report_json": str((report_dir / "transition_space.json").resolve()),
            "report_md": str((report_dir / "transition_space.md").resolve()),
        },
    }


def main() -> int:
    args = parse_args()
    try:
        result = build_transition_space_report(
            audit_root=args.audit_root,
            data_root=args.data_root,
            out_root=args.out_root,
            report_name=args.report_name,
        )
    except (V6TransitionSpaceError, SliceBuildError, V6TraceCardError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result["report"]["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
