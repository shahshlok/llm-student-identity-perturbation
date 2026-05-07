from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.codemirror import CodeMirrorParseError, infer_initial_code
from identity_perturbation.codebench_support.executions import ExecutionParseError, parse_execution_log_text
from identity_perturbation.prediction_audit.build_full_trace_prototype import _reconstruct_submit_bounded_blocks
from identity_perturbation.prediction_audit.coarse_path import ObservedCoarsePathError, build_observed_coarse_path_artifact
from identity_perturbation.prediction_audit.match_policy import (
    MatchPolicyError,
    candidate_snapshot_indices_narrow_normalized,
    count_monotonic_solutions,
    matching_indices_after,
    narrow_normalize_code_for_match,
    resolve_unique_monotonic_alignment,
)
from identity_perturbation.prediction_audit.raw_same_task_family_audit import (
    AssessmentSpec,
    cm_events_from_raw_events,
    collect_target_assessments,
    parse_raw_trace_events_with_output,
    read_text_strict,
)

DEFAULT_DATA_ROOT = Path("2024-1")
DEFAULT_OUT_ROOT = Path("data/v62/full_trace_same_task_family_scan_narrow_v1")
SCHEMA_VERSION = "v6_2_full_trace_same_task_family_scan_narrow_v1"


class FullTraceSameTaskFamilyScanError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Authoritative raw-only v6.2 full_trace scan for the Lab 5 / Lab 6 task family "
            "using narrow-normalized visible and target matching."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def _normalize_reason(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"


def _candidate_row_id(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index_0idx: int,
) -> str:
    return f"{class_id}:{assessment_id}:{exercise_id}:{student_id}:{transition_index_0idx}"


def _iter_student_exercise_pairs(
    *,
    data_root: Path,
    assessments: tuple[AssessmentSpec, ...],
) -> list[tuple[AssessmentSpec, str, str, Path, Path]]:
    pairs: list[tuple[AssessmentSpec, str, str, Path, Path]] = []
    for assessment in assessments:
        users_root = data_root / assessment.class_id / "users"
        if not users_root.exists():
            raise FullTraceSameTaskFamilyScanError(
                f"Users directory missing for class {assessment.class_id}"
            )
        for user_dir in sorted(users_root.iterdir()):
            if not user_dir.is_dir():
                continue
            student_id = user_dir.name
            for exercise_id in assessment.exercise_ids:
                execution_path = (
                    user_dir / "executions" / f"{assessment.assessment_id}_{exercise_id}.log"
                )
                codemirror_path = (
                    user_dir / "codemirror" / f"{assessment.assessment_id}_{exercise_id}.log"
                )
                pairs.append((assessment, student_id, exercise_id, execution_path, codemirror_path))
    return pairs


def _scan_row(
    *,
    assessment: AssessmentSpec,
    exercise_id: str,
    student_id: str,
    transition_index_0idx: int,
    attempts: tuple[Any, ...],
    blocks: tuple[dict[str, object], ...],
) -> dict[str, Any]:
    visible_attempts = attempts[: transition_index_0idx + 1]
    target_attempt = attempts[transition_index_0idx + 1]
    snapshot_codes = tuple(
        block["trace"]["items"][-1]["code_after_anchor"]  # type: ignore[index]
        for block in blocks
    )
    row_id = _candidate_row_id(
        class_id=assessment.class_id,
        assessment_id=assessment.assessment_id,
        exercise_id=exercise_id,
        student_id=student_id,
        transition_index_0idx=transition_index_0idx,
    )

    attempt_codes = tuple(attempt.code for attempt in visible_attempts)
    candidates = candidate_snapshot_indices_narrow_normalized(
        attempt_codes=attempt_codes,
        snapshot_codes=snapshot_codes,
    )
    if not all(candidates):
        return {
            "row_id": row_id,
            "status": "failure",
            "stage": "visible_alignment",
            "reason": "visible_no_match",
        }
    if count_monotonic_solutions(candidates) != 1:
        return {
            "row_id": row_id,
            "status": "failure",
            "stage": "visible_alignment",
            "reason": "visible_ambiguous_alignment",
        }
    alignment = resolve_unique_monotonic_alignment(candidates)
    target_snapshot_index = alignment[-1] + 1
    if target_snapshot_index >= len(blocks):
        return {
            "row_id": row_id,
            "status": "failure",
            "stage": "target_alignment",
            "reason": "target_no_following_submit_block",
        }

    normalized_target_code = narrow_normalize_code_for_match(target_attempt.code)
    target_submit_code = str(snapshot_codes[target_snapshot_index])
    if narrow_normalize_code_for_match(target_submit_code) != normalized_target_code:
        return {
            "row_id": row_id,
            "status": "failure",
            "stage": "target_alignment",
            "reason": "target_code_mismatch_narrow",
        }

    matching_after_prefix = matching_indices_after(
        snapshot_codes=snapshot_codes,
        normalized_target_code=normalized_target_code,
        start_index=alignment[-1],
    )
    if matching_after_prefix != (target_snapshot_index,):
        return {
            "row_id": row_id,
            "status": "failure",
            "stage": "target_alignment",
            "reason": "target_ambiguous_narrow_match",
        }

    target_block = blocks[target_snapshot_index]
    try:
        coarse_path = build_observed_coarse_path_artifact(
            class_id=assessment.class_id,
            assessment_id=assessment.assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition_index_0idx=transition_index_0idx,
            attempt_index_0idx=target_attempt.attempt_index_0idx,
            aligned_submit_index_0idx=target_snapshot_index,
            trace=target_block["trace"],  # type: ignore[arg-type]
        )
    except ObservedCoarsePathError as exc:
        return {
            "row_id": row_id,
            "status": "failure",
            "stage": "coarse_path",
            "reason": _normalize_reason(exc),
        }

    steps = coarse_path["attempt_n1"]["coarse_path_steps"]  # type: ignore[index]
    if not isinstance(steps, list) or not steps:
        raise FullTraceSameTaskFamilyScanError(f"Observed coarse path steps missing for {row_id}")
    if steps[-1]["action_type"] != "submit":
        raise FullTraceSameTaskFamilyScanError(
            f"Observed coarse path does not end in submit for {row_id}"
        )
    if not any(step["action_type"] == "edit" for step in steps):
        return {
            "row_id": row_id,
            "status": "failure",
            "stage": "coarse_path",
            "reason": "coarse_path_no_real_edit",
        }

    return {
        "row_id": row_id,
        "status": "success",
        "stage": "success",
        "reason": "admissible",
        "target_snapshot_index_0idx": target_snapshot_index,
        "visible_attempt_count": len(visible_attempts),
        "total_attempt_count": len(attempts),
        "coarse_path_action_sequence": [step["action_type"] for step in steps],
    }


def build_report(data_root: Path) -> dict[str, Any]:
    assessments = collect_target_assessments(data_root)
    pairs = _iter_student_exercise_pairs(data_root=data_root, assessments=assessments)

    pair_counts: Counter[str] = Counter()
    transition_failure_counts: Counter[str] = Counter()
    transition_stage_counts: Counter[str] = Counter()
    pair_failure_examples: dict[str, list[str]] = {}
    transition_failure_examples: dict[str, list[str]] = {}

    replay_clean_transition_rows = 0
    admissible_rows: list[dict[str, Any]] = []

    for assessment, student_id, exercise_id, execution_path, codemirror_path in pairs:
        pair_id = f"{assessment.class_id}:{assessment.assessment_id}:{exercise_id}:{student_id}"
        pair_counts["pair_count"] += 1
        if not execution_path.exists() or not codemirror_path.exists():
            pair_counts["missing_log"] += 1
            pair_failure_examples.setdefault("missing_log", [])
            if len(pair_failure_examples["missing_log"]) < 10:
                pair_failure_examples["missing_log"].append(pair_id)
            continue
        pair_counts["with_exec_and_cm"] += 1

        try:
            attempts = parse_execution_log_text(read_text_strict(execution_path))
        except ExecutionParseError as exc:
            key = f"execution_parse_error:{type(exc).__name__}"
            pair_counts[key] += 1
            pair_failure_examples.setdefault(key, [])
            if len(pair_failure_examples[key]) < 10:
                pair_failure_examples[key].append(f"{pair_id} :: {exc}")
            continue
        if len(attempts) < 2:
            pair_counts["lt2_attempts"] += 1
            pair_failure_examples.setdefault("lt2_attempts", [])
            if len(pair_failure_examples["lt2_attempts"]) < 10:
                pair_failure_examples["lt2_attempts"].append(pair_id)
            continue
        pair_counts["min_two_attempt_pairs"] += 1

        try:
            raw_events = parse_raw_trace_events_with_output(
                read_text_strict(codemirror_path), codemirror_path
            )
            cm_events = cm_events_from_raw_events(raw_events)
            initial_code = infer_initial_code(cm_events, attempts[0].code)
            blocks = _reconstruct_submit_bounded_blocks(raw_events, initial_code=initial_code)
        except (
            CodeMirrorParseError,
            MatchPolicyError,
            FullTraceSameTaskFamilyScanError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            key = _normalize_reason(exc)
            pair_counts[key] += 1
            pair_failure_examples.setdefault(key, [])
            if len(pair_failure_examples[key]) < 10:
                pair_failure_examples[key].append(pair_id)
            continue
        pair_counts["replay_clean_pairs"] += 1

        for transition_index_0idx in range(len(attempts) - 1):
            replay_clean_transition_rows += 1
            row_result = _scan_row(
                assessment=assessment,
                exercise_id=exercise_id,
                student_id=student_id,
                transition_index_0idx=transition_index_0idx,
                attempts=attempts,
                blocks=blocks,
            )
            transition_stage_counts[row_result["stage"]] += 1
            if row_result["status"] == "success":
                admissible_rows.append(row_result)
                continue
            transition_failure_counts[row_result["reason"]] += 1
            transition_failure_examples.setdefault(row_result["reason"], [])
            if len(transition_failure_examples[row_result["reason"]]) < 10:
                transition_failure_examples[row_result["reason"]].append(row_result["row_id"])

    class_ids = sorted({assessment.class_id for assessment in assessments})
    assessment_ids = sorted({assessment.assessment_id for assessment in assessments})
    exercise_keys = {
        (assessment.class_id, assessment.assessment_id, exercise_id)
        for assessment in assessments
        for exercise_id in assessment.exercise_ids
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "policy": {
            "visible_alignment": "narrow_normalized_unique_monotonic",
            "target_match": "immediate_next_submit_block_narrow_normalized_unique",
            "narrow_normalization": {
                "line_endings": "CRLF/CR normalized to LF",
                "terminal_newline": "ignore one terminal newline only",
                "trailing_horizontal_whitespace": "ignore spaces/tabs at line ends",
            },
            "ambiguity_rule": "fail_loudly",
            "target_requires_real_edit": True,
            "target_requires_final_submit": True,
            "local_run_required": False,
        },
        "scope": {
            "assessment_titles": sorted({assessment.title for assessment in assessments}),
            "assessment_count": len(assessment_ids),
            "class_count": len(class_ids),
            "exercise_count": len(exercise_keys),
        },
        "summary": {
            "pair_counts": dict(pair_counts),
            "replay_clean_transition_rows": replay_clean_transition_rows,
            "admissible_row_count": len(admissible_rows),
            "admissible_fraction_of_replay_clean_rows": (
                len(admissible_rows) / replay_clean_transition_rows
                if replay_clean_transition_rows
                else 0.0
            ),
            "transition_stage_counts": dict(transition_stage_counts),
            "transition_failure_counts": dict(transition_failure_counts),
        },
        "pair_failure_examples": pair_failure_examples,
        "transition_failure_examples": transition_failure_examples,
        "admissible_rows": admissible_rows,
    }


def write_report(*, report: dict[str, Any], out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=False)
    report_path = out_root / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report_path


def main() -> int:
    args = parse_args()
    report = build_report(args.data_root)
    report_path = write_report(report=report, out_root=args.out_root)
    print(report_path.resolve())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
