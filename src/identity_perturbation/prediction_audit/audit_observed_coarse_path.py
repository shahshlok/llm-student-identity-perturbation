from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.codemirror import infer_initial_code
from identity_perturbation.codebench_support.executions import parse_execution_log_text
from identity_perturbation.prediction_audit.build_full_trace_prototype import (
    _candidate_snapshot_indices_exact,
    _count_monotonic_solutions,
    _reconstruct_submit_bounded_blocks,
    _resolve_paths,
    _resolve_unique_alignment,
)
from identity_perturbation.prediction_audit.coarse_path import build_observed_coarse_path_artifact
from identity_perturbation.prediction_audit.raw_same_task_family_audit import (
    cm_events_from_raw_events,
    parse_raw_trace_events_with_output,
    read_text_strict,
)

DEFAULT_PROBE_PATH = Path("data/v62/probes/raw_same_task_family_rich_history_core143_v2/probe.json")
DEFAULT_DATA_ROOT = Path("2024-1")
DEFAULT_OUT_ROOT = Path("data/v62/observed_coarse_path_audit/core143_v3_current_schema")


class ObservedCoarsePathAuditError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit observed coarse path buildability across a v6.2 probe set."
    )
    parser.add_argument("--probe-path", type=Path, default=DEFAULT_PROBE_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def _read_probe_rows(path: Path) -> list[dict[str, Any]]:
    probe = json.loads(read_text_strict(path))
    rows = probe.get("selected_transitions")
    if not isinstance(rows, list) or not rows:
        raise ObservedCoarsePathAuditError(
            f"Probe {path} does not contain a non-empty selected_transitions list"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise ObservedCoarsePathAuditError("Probe selected_transitions must all be dict objects")
    return rows


def _custom_id(row: dict[str, Any]) -> str:
    return (
        f"{row['class_id']}:{row['assessment_id']}:{row['exercise_id']}:"
        f"{row['student_id']}:{row['transition_index_0idx']}"
    )


def _normalize_reason(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"


def _audit_row(row: dict[str, Any], *, data_root: Path) -> dict[str, Any]:
    class_id = str(row["class_id"])
    assessment_id = str(row["assessment_id"])
    exercise_id = str(row["exercise_id"])
    student_id = str(row["student_id"])
    transition_index = int(row["transition_index_0idx"])
    custom_id = _custom_id(row)

    try:
        _, execution_path, codemirror_path = _resolve_paths(
            data_root=data_root,
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
        )
    except Exception as exc:
        return {
            "custom_id": custom_id,
            "status": "failure",
            "stage": "resolve_paths",
            "reason": _normalize_reason(exc),
        }

    try:
        attempts = parse_execution_log_text(read_text_strict(execution_path))
        if transition_index < 0 or transition_index >= len(attempts) - 1:
            raise ObservedCoarsePathAuditError(
                f"transition-index must be between 0 and {len(attempts) - 2}, got {transition_index}"
            )
        raw_events = parse_raw_trace_events_with_output(
            read_text_strict(codemirror_path),
            codemirror_path,
        )
        cm_events = cm_events_from_raw_events(raw_events)
        initial_code = infer_initial_code(cm_events, attempts[0].code)
        blocks = _reconstruct_submit_bounded_blocks(raw_events, initial_code=initial_code)
    except Exception as exc:
        return {
            "custom_id": custom_id,
            "status": "failure",
            "stage": "parse_and_reconstruct",
            "reason": _normalize_reason(exc),
        }

    try:
        visible_attempts = attempts[: transition_index + 1]
        snapshot_codes = tuple(
            block["trace"]["items"][-1]["code_after_anchor"]  # type: ignore[index]
            for block in blocks
        )
        candidates = _candidate_snapshot_indices_exact(
            attempt_codes=tuple(attempt.code for attempt in visible_attempts),
            snapshot_codes=snapshot_codes,
        )
        if not all(candidates):
            raise ObservedCoarsePathAuditError(
                "At least one visible attempt had no exact matching submit snapshot in the CodeMirror replay"
            )
        if _count_monotonic_solutions(candidates) != 1:
            raise ObservedCoarsePathAuditError(
                "Visible attempts did not have a unique exact monotonic alignment to submit snapshots"
            )
        alignment = _resolve_unique_alignment(candidates)
    except Exception as exc:
        return {
            "custom_id": custom_id,
            "status": "failure",
            "stage": "visible_alignment",
            "reason": _normalize_reason(exc),
        }

    try:
        target_attempt = attempts[transition_index + 1]
        target_snapshot_index = alignment[-1] + 1
        if target_snapshot_index >= len(blocks):
            raise ObservedCoarsePathAuditError(
                "Target attempt n+1 had no following submit-bounded CodeMirror block"
            )
        target_block = blocks[target_snapshot_index]
        target_submit_item = target_block["trace"]["items"][-1]  # type: ignore[index]
        if target_submit_item["item_type"] != "submit":
            raise ObservedCoarsePathAuditError(
                f"Last item for target submit-bounded block {target_snapshot_index} is not a submit item"
            )
        if target_submit_item["code_after_anchor"] != target_attempt.code:
            raise ObservedCoarsePathAuditError(
                f"Exact aligned submit snapshot did not match execution code for target attempt {target_attempt.attempt_index_0idx}"
            )
    except Exception as exc:
        return {
            "custom_id": custom_id,
            "status": "failure",
            "stage": "target_alignment",
            "reason": _normalize_reason(exc),
        }

    try:
        artifact = build_observed_coarse_path_artifact(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition_index_0idx=transition_index,
            attempt_index_0idx=target_attempt.attempt_index_0idx,
            aligned_submit_index_0idx=target_snapshot_index,
            trace=target_block["trace"],  # type: ignore[arg-type]
        )
    except Exception as exc:
        return {
            "custom_id": custom_id,
            "status": "failure",
            "stage": "observed_coarse_path",
            "reason": _normalize_reason(exc),
        }

    steps = artifact["attempt_n1"]["coarse_path_steps"]  # type: ignore[index]
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        raise ObservedCoarsePathAuditError(
            "Observed coarse path artifact did not contain a valid steps list"
        )
    action_sequence = [str(step["action_type"]) for step in steps]
    step_count = len(steps)

    return {
        "custom_id": custom_id,
        "status": "success",
        "stage": "success",
        "step_count": step_count,
        "action_sequence": action_sequence,
        "schema_path_length_ok": 1 <= step_count <= 8,
        "attempt_n1_grade": target_attempt.grade,
        "attempt_n1_failed_test_indices_0idx": [
            result.test_index_0idx for result in target_attempt.test_results if not result.passed
        ],
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    status_counter = Counter(result["status"] for result in results)
    stage_counter = Counter(result["stage"] for result in results)
    reason_counter = Counter(
        result["reason"] for result in results if result["status"] == "failure"
    )
    success_rows = [result for result in results if result["status"] == "success"]
    step_count_counter = Counter(result["step_count"] for result in success_rows)
    action_sequence_counter = Counter(
        " -> ".join(result["action_sequence"]) for result in success_rows
    )
    schema_path_length_counter = Counter(
        "compatible" if result["schema_path_length_ok"] else "incompatible"
        for result in success_rows
    )

    top_failure_examples: list[dict[str, Any]] = []
    seen_reasons: set[str] = set()
    for result in results:
        if result["status"] != "failure":
            continue
        reason = str(result["reason"])
        if reason in seen_reasons:
            continue
        top_failure_examples.append(
            {
                "custom_id": result["custom_id"],
                "stage": result["stage"],
                "reason": reason,
            }
        )
        seen_reasons.add(reason)
        if len(top_failure_examples) >= 10:
            break

    return {
        "total_rows": total,
        "status_counts": dict(status_counter),
        "stage_counts": dict(stage_counter),
        "failure_reason_counts": dict(reason_counter),
        "success_count": len(success_rows),
        "step_count_distribution": dict(step_count_counter),
        "action_sequence_distribution": dict(action_sequence_counter),
        "schema_path_length_distribution_current_1_to_8": dict(schema_path_length_counter),
        "top_failure_examples": top_failure_examples,
    }


def main() -> int:
    args = parse_args()
    rows = _read_probe_rows(args.probe_path)
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=False)

    results = [_audit_row(row, data_root=args.data_root) for row in rows]
    summary = _build_summary(results)
    report = {
        "schema_version": "v6_2_observed_coarse_path_audit_v1",
        "probe_path": str(args.probe_path.resolve()),
        "data_root": str(args.data_root.resolve()),
        "summary": summary,
        "results": results,
    }

    report_path = out_root / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(str(report_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
