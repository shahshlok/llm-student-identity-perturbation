from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from identity_perturbation.codebench_support.codemirror import infer_initial_code
from identity_perturbation.codebench_support.executions import parse_execution_log_text
from identity_perturbation.codebench_support.payload import pass_count
from identity_perturbation.prediction_audit.build_full_trace_prototype import (
    _attempt_execution_result,
    _candidate_snapshot_indices_exact,
    _count_monotonic_solutions,
    _reconstruct_submit_bounded_blocks,
    _resolve_paths,
    _resolve_unique_alignment,
)
from identity_perturbation.prediction_audit.coarse_path import build_observed_coarse_path_artifact
from identity_perturbation.prediction_audit.full_trace_prompting import build_user_prompt
from identity_perturbation.prediction_audit.full_trace_targets import build_observed_repair_target
from identity_perturbation.prediction_audit.raw_same_task_family_audit import (
    cm_events_from_raw_events,
    parse_assessment_data,
    parse_raw_trace_events_with_output,
    read_text_strict,
)

DEFAULT_PROBE_PATH = Path("data/v62/probes/raw_same_task_family_rich_history_core143_v2/probe.json")
DEFAULT_DATA_ROOT = Path("2024-1")
DEFAULT_OUT_ROOT = Path("data/v62/full_trace_target_audit/core143_v1")


class FullTraceTargetAuditError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit v6.2 full-trace target coverage across a probe set."
    )
    parser.add_argument("--probe-path", type=Path, default=DEFAULT_PROBE_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def _read_probe_rows(path: Path) -> list[dict[str, Any]]:
    probe = json.loads(read_text_strict(path))
    rows = probe.get("selected_transitions")
    if not isinstance(rows, list) or not rows:
        raise FullTraceTargetAuditError(
            f"Probe {path} does not contain a non-empty selected_transitions list"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise FullTraceTargetAuditError("Probe selected_transitions must all be dict objects")
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


def _prompt_char_stats(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    p90_index = math.ceil(0.9 * len(ordered)) - 1
    return {
        "min": ordered[0],
        "median": int(median(ordered)),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


def _numeric_stats(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "median": float(median(ordered)),
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def _coverage_entry(
    *, count: int, total: int, denominator: int | None = None
) -> dict[str, float | int]:
    entry: dict[str, float | int] = {
        "count": count,
        "fraction_of_total": count / total if total else 0.0,
    }
    if denominator is not None:
        entry["fraction_of_denominator"] = count / denominator if denominator else 0.0
    return entry


def _build_visible_prompt_payload(
    *,
    assessment_title: str,
    execution_path: Path,
    codemirror_path: Path,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index: int,
    visible_attempts: list[Any],
    blocks: tuple[dict[str, object], ...],
    alignment: tuple[int, ...],
    target_attempt_index_0idx: int,
) -> dict[str, object]:
    prompt_attempts: list[dict[str, object]] = []
    for attempt, snapshot_index in zip(visible_attempts, alignment, strict=True):
        block = blocks[snapshot_index]
        submit_item = block["trace"]["items"][-1]  # type: ignore[index]
        if submit_item["item_type"] != "submit":
            raise FullTraceTargetAuditError(
                f"Last item for submit-bounded block {snapshot_index} is not a submit item"
            )
        if submit_item["code_after_anchor"] != attempt.code:
            raise FullTraceTargetAuditError(
                f"Exact aligned submit snapshot did not match execution code for attempt {attempt.attempt_index_0idx}"
            )
        prompt_attempts.append(
            {
                "attempt_index_0idx": attempt.attempt_index_0idx,
                "aligned_submit_index_0idx": snapshot_index,
                "execution_result": _attempt_execution_result(attempt),
                "trace": block["trace"],
            }
        )
    return {
        "schema_version": "v6_2_full_trace_prompt_payload_v2",
        "source": {
            "class_id": class_id,
            "assessment_id": assessment_id,
            "assessment_title": assessment_title,
            "exercise_id": exercise_id,
            "student_id": student_id,
            "condition": "full",
            "visible_transition_index_0idx": transition_index,
            "visible_attempt_indices_0idx": [
                attempt.attempt_index_0idx for attempt in visible_attempts
            ],
            "prediction_target_attempt_index_0idx": target_attempt_index_0idx,
            "alignment_policy": "exact_submit_code_match_only",
            "execution_log_path": str(execution_path.resolve()),
            "codemirror_log_path": str(codemirror_path.resolve()),
        },
        "visible_attempts": prompt_attempts,
        "prediction_target": {
            "attempt_index_0idx": target_attempt_index_0idx,
            "task": "Predict the next submitted attempt after the visible history.",
        },
    }


def _audit_row(row: dict[str, Any], *, data_root: Path) -> dict[str, Any]:
    class_id = str(row["class_id"])
    assessment_id = str(row["assessment_id"])
    exercise_id = str(row["exercise_id"])
    student_id = str(row["student_id"])
    transition_index = int(row["transition_index_0idx"])
    custom_id = _custom_id(row)

    result: dict[str, Any] = {
        "custom_id": custom_id,
        "status": "failure",
        "parse_and_reconstruct_ok": False,
        "visible_alignment_ok": False,
        "prompt_payload_build_ok": False,
        "repair_target_ok": False,
        "target_exact_coverage_ok": False,
        "exploratory_path_ok": False,
    }

    try:
        assessment_path, execution_path, codemirror_path = _resolve_paths(
            data_root=data_root,
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
        )
        assessment_spec = parse_assessment_data(assessment_path)
        if exercise_id not in assessment_spec.exercise_ids:
            raise FullTraceTargetAuditError(
                f"Exercise {exercise_id} is not declared in assessment {assessment_id}"
            )
        attempts = parse_execution_log_text(read_text_strict(execution_path))
        if transition_index < 0 or transition_index >= len(attempts) - 1:
            raise FullTraceTargetAuditError(
                f"transition-index must be between 0 and {len(attempts) - 2}, got {transition_index}"
            )
        raw_events = parse_raw_trace_events_with_output(
            read_text_strict(codemirror_path),
            codemirror_path,
        )
        cm_events = cm_events_from_raw_events(raw_events)
        initial_code = infer_initial_code(cm_events, attempts[0].code)
        blocks = _reconstruct_submit_bounded_blocks(raw_events, initial_code=initial_code)
        result["parse_and_reconstruct_ok"] = True
    except Exception as exc:
        result["stage"] = "parse_and_reconstruct"
        result["reason"] = _normalize_reason(exc)
        return result

    visible_attempts = attempts[: transition_index + 1]
    target_attempt = attempts[transition_index + 1]
    result["visible_attempt_count"] = len(visible_attempts)
    result["target_attempt_index_0idx"] = target_attempt.attempt_index_0idx

    try:
        snapshot_codes = tuple(
            block["trace"]["items"][-1]["code_after_anchor"]  # type: ignore[index]
            for block in blocks
        )
        candidates = _candidate_snapshot_indices_exact(
            attempt_codes=tuple(attempt.code for attempt in visible_attempts),
            snapshot_codes=snapshot_codes,
        )
        if not all(candidates):
            raise FullTraceTargetAuditError(
                "At least one visible attempt had no exact matching submit snapshot in the CodeMirror replay"
            )
        if _count_monotonic_solutions(candidates) != 1:
            raise FullTraceTargetAuditError(
                "Visible attempts did not have a unique exact monotonic alignment to submit snapshots"
            )
        alignment = _resolve_unique_alignment(candidates)
        prompt_payload = _build_visible_prompt_payload(
            assessment_title=assessment_spec.title,
            execution_path=execution_path,
            codemirror_path=codemirror_path,
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition_index=transition_index,
            visible_attempts=visible_attempts,
            blocks=blocks,
            alignment=alignment,
            target_attempt_index_0idx=target_attempt.attempt_index_0idx,
        )
        user_prompt = build_user_prompt(prompt_payload)
        result["visible_alignment_ok"] = True
        result["prompt_payload_build_ok"] = True
        result["alignment"] = list(alignment)
        result["prompt_char_count"] = len(user_prompt)
        result["visible_trace_change_event_count"] = sum(
            int(item["change_event_count"])
            for attempt_payload in prompt_payload["visible_attempts"]  # type: ignore[index]
            for item in attempt_payload["trace"]["items"]  # type: ignore[index]
            if item["item_type"] == "edit_segment"
        )
        result["visible_trace_local_run_count"] = sum(
            1
            for attempt_payload in prompt_payload["visible_attempts"]  # type: ignore[index]
            for item in attempt_payload["trace"]["items"]  # type: ignore[index]
            if item["item_type"] == "saida_testar"
        )
    except Exception as exc:
        result["stage"] = "visible_alignment"
        result["reason"] = _normalize_reason(exc)
        return result

    try:
        repair_target = build_observed_repair_target(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition_index_0idx=transition_index,
            attempt_n=visible_attempts[-1],
            attempt_n1=target_attempt,
        )
        repair_payload = repair_target["repair_target"]  # type: ignore[index]
        result["repair_target_ok"] = True
        result["attempt_n_grade"] = float(visible_attempts[-1].grade)
        result["attempt_n_pass_count"] = pass_count(visible_attempts[-1])
        result["attempt_n1_grade"] = float(target_attempt.grade)
        result["attempt_n1_pass_count"] = pass_count(target_attempt)
        result["repair_outcome_bucket"] = repair_payload["outcome_bucket"]
        result["repair_span_count"] = len(repair_payload["repair_spans_in_next_code"])
        result["repair_changed_line_count"] = repair_payload["changed_line_count_in_next_code"]
    except Exception as exc:
        result["stage"] = "repair_target"
        result["reason"] = _normalize_reason(exc)
        return result

    try:
        target_snapshot_index = alignment[-1] + 1
        if target_snapshot_index >= len(blocks):
            raise FullTraceTargetAuditError(
                "Target attempt n+1 had no following submit-bounded CodeMirror block"
            )
        target_block = blocks[target_snapshot_index]
        target_submit_item = target_block["trace"]["items"][-1]  # type: ignore[index]
        if target_submit_item["item_type"] != "submit":
            raise FullTraceTargetAuditError(
                f"Last item for target submit-bounded block {target_snapshot_index} is not a submit item"
            )
        if target_submit_item["code_after_anchor"] != target_attempt.code:
            raise FullTraceTargetAuditError(
                f"Exact aligned submit snapshot did not match execution code for target attempt {target_attempt.attempt_index_0idx}"
            )
        result["target_exact_coverage_ok"] = True
        result["target_aligned_submit_index_0idx"] = target_snapshot_index
    except Exception as exc:
        result["stage"] = "target_alignment"
        result["reason"] = _normalize_reason(exc)
        return result

    try:
        coarse_path = build_observed_coarse_path_artifact(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition_index_0idx=transition_index,
            attempt_index_0idx=target_attempt.attempt_index_0idx,
            aligned_submit_index_0idx=target_snapshot_index,
            trace=target_block["trace"],  # type: ignore[arg-type]
        )
        steps = coarse_path["attempt_n1"]["coarse_path_steps"]  # type: ignore[index]
        if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
            raise FullTraceTargetAuditError(
                "Observed coarse path artifact did not contain a valid steps list"
            )
        result["exploratory_path_ok"] = True
        result["exploratory_step_count"] = len(steps)
        result["exploratory_action_sequence"] = [str(step["action_type"]) for step in steps]
    except Exception as exc:
        result["stage"] = "exploratory_path"
        result["reason"] = _normalize_reason(exc)
        return result

    result["stage"] = "success"
    result["status"] = "success"
    return result


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    parse_rows = [row for row in results if row["parse_and_reconstruct_ok"]]
    visible_rows = [row for row in results if row["visible_alignment_ok"]]
    prompt_rows = [row for row in results if row["prompt_payload_build_ok"]]
    repair_rows = [row for row in results if row["repair_target_ok"]]
    exact_rows = [row for row in results if row["target_exact_coverage_ok"]]
    exploratory_rows = [row for row in results if row["exploratory_path_ok"]]
    all_target_rows = [
        row for row in results if row["repair_target_ok"] and row["exploratory_path_ok"]
    ]

    failure_reason_counts = Counter(row["reason"] for row in results if row["status"] == "failure")
    stage_counts = Counter(row["stage"] for row in results)

    top_failure_examples: list[dict[str, Any]] = []
    seen_reasons: set[str] = set()
    for row in results:
        if row["status"] != "failure":
            continue
        reason = str(row["reason"])
        if reason in seen_reasons:
            continue
        top_failure_examples.append(
            {
                "custom_id": row["custom_id"],
                "stage": row["stage"],
                "reason": reason,
            }
        )
        seen_reasons.add(reason)
        if len(top_failure_examples) >= 10:
            break

    return {
        "total_rows": total,
        "stage_counts": dict(stage_counts),
        "failure_reason_counts": dict(failure_reason_counts),
        "coverage": {
            "parse_and_reconstruct_ok": _coverage_entry(count=len(parse_rows), total=total),
            "visible_alignment_ok": _coverage_entry(count=len(visible_rows), total=total),
            "prompt_payload_build_ok": _coverage_entry(count=len(prompt_rows), total=total),
            "observed_next_repair_target_ok": _coverage_entry(
                count=len(repair_rows),
                total=total,
                denominator=len(visible_rows),
            ),
            "target_alignment_ok": _coverage_entry(
                count=len(exact_rows),
                total=total,
                denominator=len(visible_rows),
            ),
            "observed_next_coarse_path_ok": _coverage_entry(
                count=len(exploratory_rows),
                total=total,
                denominator=len(exact_rows),
            ),
            "all_targets_ok": _coverage_entry(
                count=len(all_target_rows),
                total=total,
                denominator=len(exact_rows),
            ),
        },
        "visible_prompt_char_stats": _prompt_char_stats(
            [int(row["prompt_char_count"]) for row in prompt_rows]
        ),
        "visible_attempt_count_distribution": dict(
            Counter(int(row["visible_attempt_count"]) for row in prompt_rows)
        ),
        "visible_attempt_count_stats": _numeric_stats(
            [int(row["visible_attempt_count"]) for row in prompt_rows]
        ),
        "visible_trace_change_event_count_stats": _prompt_char_stats(
            [int(row["visible_trace_change_event_count"]) for row in prompt_rows]
        ),
        "visible_trace_local_run_count_stats": _prompt_char_stats(
            [int(row["visible_trace_local_run_count"]) for row in prompt_rows]
        ),
        "repair_outcome_distribution": dict(
            Counter(str(row["repair_outcome_bucket"]) for row in repair_rows)
        ),
        "repair_span_count_stats": _numeric_stats(
            [int(row["repair_span_count"]) for row in repair_rows]
        ),
        "repair_changed_line_count_stats": _numeric_stats(
            [int(row["repair_changed_line_count"]) for row in repair_rows]
        ),
        "target_grade_distribution": dict(
            Counter(float(row["attempt_n1_grade"]) for row in repair_rows)
        ),
        "target_pass_count_distribution": dict(
            Counter(int(row["attempt_n1_pass_count"]) for row in repair_rows)
        ),
        "exact_covered_target_grade_distribution": dict(
            Counter(float(row["attempt_n1_grade"]) for row in exact_rows)
        ),
        "exact_covered_target_pass_count_distribution": dict(
            Counter(int(row["attempt_n1_pass_count"]) for row in exact_rows)
        ),
        "pass_delta_distribution": dict(
            Counter(
                int(row["attempt_n1_pass_count"]) - int(row["attempt_n_pass_count"])
                for row in repair_rows
            )
        ),
        "exploratory_step_count_distribution": dict(
            Counter(int(row["exploratory_step_count"]) for row in exploratory_rows)
        ),
        "exploratory_step_count_stats": _numeric_stats(
            [int(row["exploratory_step_count"]) for row in exploratory_rows]
        ),
        "exploratory_action_sequence_distribution": dict(
            Counter(" -> ".join(row["exploratory_action_sequence"]) for row in exploratory_rows)
        ),
        "exploratory_schema_compatibility_current_1_to_8": _coverage_entry(
            count=sum(
                1 for row in exploratory_rows if 1 <= int(row["exploratory_step_count"]) <= 8
            ),
            total=total,
            denominator=len(exploratory_rows),
        ),
        "top_failure_examples": top_failure_examples,
    }


def main() -> None:
    args = parse_args()
    rows = _read_probe_rows(args.probe_path)
    results = [_audit_row(row, data_root=args.data_root) for row in rows]
    summary = _build_summary(results)

    args.out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "v6_2_full_trace_target_audit_v1",
        "probe_path": str(args.probe_path.resolve()),
        "data_root": str(args.data_root.resolve()),
        "summary": summary,
        "rows": results,
    }
    report_path = args.out_root / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path.resolve())


if __name__ == "__main__":
    main()
