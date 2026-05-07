from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from identity_perturbation.codebench_support.executions import ExecutionParseError, parse_execution_log_text
from identity_perturbation.prediction_audit.build_full_trace_prototype import (
    ALIGNMENT_POLICY,
    _build_prompt_payload,
    _build_request_body,
)
from identity_perturbation.prediction_audit.full_trace_prompting import (
    SUPPORTED_CONDITIONS,
    build_system_prompt,
    build_user_prompt,
    validate_condition,
)
from identity_perturbation.prediction_audit.full_trace_target_schema import FullTracePredictionResponse
from identity_perturbation.prediction_audit.pair_matching import (
    build_match_state,
    pass_vector_signature,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCOPES_JSON = REPO_ROOT / "data/v62/diversity_audit_full/scopes.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "2024-1"
DEFAULT_OUT_DIR = REPO_ROOT / "data/v62/probes/branching_v1"
DEFAULT_TARGET_SCOPES = 50
DEFAULT_SEED = 42
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
MATCHED_COHORT_PREFLIGHT_CONDITION = "full"
RUN_MANIFEST_SCHEMA_VERSION = "v6_2_full_trace_pilot_batch_manifest_v1"
REPORT_SCHEMA_VERSION = "v6_2_branching_probe_selection_report_v1"
BUNDLE_SCHEMA_VERSION = "v6_2_full_trace_prototype_bundle_v6"
DEFAULT_COMPLETION_WINDOW = "24h"
MANAGED_OUT_BASENAMES = {
    "bundles",
    "manifest.json",
    "requests.jsonl",
    "output.jsonl",
    "errors.jsonl",
    "selection_report.json",
    "selection_report.md",
}
TERTILES = ("T1", "T2", "T3")
FIXED_DIVERSITY_THRESHOLDS = (0.15, 0.30, 0.45)
FIXED_INTERIOR_N = 1
FIXED_VISIBLE_ATTEMPT_COUNT = FIXED_INTERIOR_N + 1
MIN_TOTAL_ATTEMPTS_FOR_FIXED_SELECTION = FIXED_VISIBLE_ATTEMPT_COUNT + 1


class BranchingProbeSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class ScopeRecord:
    class_id: str
    assessment_id: str
    exercise_id: str
    n_students: int
    mean_pairwise_dist: float
    distinct_codes: int
    canonical_80: bool

    @property
    def scope_id(self) -> str:
        return f"{self.class_id}:{self.assessment_id}:{self.exercise_id}"

    @property
    def exercise_scope(self) -> str:
        return self.scope_id


@dataclass(frozen=True)
class ScopeWithTertile:
    scope: ScopeRecord
    tertile: str

    @property
    def scope_id(self) -> str:
        return self.scope.scope_id


@dataclass(frozen=True)
class StudentLogCandidate:
    class_id: str
    assessment_id: str
    exercise_id: str
    student_id: str
    execution_path: Path
    codemirror_path: Path

    @property
    def scope_id(self) -> str:
        return f"{self.class_id}:{self.assessment_id}:{self.exercise_id}"


@dataclass(frozen=True)
class ValidatedStudentCandidate:
    class_id: str
    assessment_id: str
    exercise_id: str
    student_id: str
    execution_path: Path
    codemirror_path: Path
    total_attempt_count: int
    interior_n: int
    attempt_n_code: str
    attempt_n1_code: str
    attempt_n_normalized_code: str | None = None
    attempt_n_pass_fail_vector: tuple[bool, ...] = ()
    attempt_n_failed_test_indices_0idx: tuple[int, ...] = ()
    attempt_n_test_count: int = 0
    attempt_n_pass_vector_signature: str = ""

    @property
    def scope_id(self) -> str:
        return f"{self.class_id}:{self.assessment_id}:{self.exercise_id}"

    @property
    def custom_id(self) -> str:
        return (
            f"{self.class_id}:{self.student_id}:{self.assessment_id}:"
            f"{self.exercise_id}:{self.interior_n}"
        )

    @property
    def visible_attempt_count(self) -> int:
        return self.interior_n + 1


@dataclass(frozen=True)
class ScopePairSelection:
    scope: ScopeWithTertile
    students: tuple[ValidatedStudentCandidate, ...]

    @property
    def scope_id(self) -> str:
        return self.scope.scope_id

    @property
    def student_count(self) -> int:
        return len(self.students)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic, auditable v6.2 branching-probe batch and emit full-trace bundles."
        )
    )
    parser.add_argument("--scopes-json", type=Path, default=DEFAULT_SCOPES_JSON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--condition", choices=SUPPORTED_CONDITIONS, required=True)
    parser.add_argument("--target-scopes", type=int, default=DEFAULT_TARGET_SCOPES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise BranchingProbeSelectionError(f"Scopes JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise BranchingProbeSelectionError(f"Expected top-level list in {path}")
    if not payload:
        raise BranchingProbeSelectionError(f"Scopes JSON is empty: {path}")
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise BranchingProbeSelectionError(f"Scope row {index} is not an object")
    return payload


def _load_scope_records(path: Path) -> list[ScopeRecord]:
    scopes: list[ScopeRecord] = []
    for index, row in enumerate(_read_json_list(path)):
        try:
            scopes.append(
                ScopeRecord(
                    class_id=str(row["class_id"]),
                    assessment_id=str(row["assessment_id"]),
                    exercise_id=str(row["exercise_id"]),
                    n_students=int(row["n_students"]),
                    mean_pairwise_dist=float(row["mean_pairwise_dist"]),
                    distinct_codes=int(row["distinct_codes"]),
                    canonical_80=bool(row["canonical_80"]),
                )
            )
        except KeyError as exc:
            raise BranchingProbeSelectionError(
                f"Scope row {index} is missing required key {exc.args[0]!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise BranchingProbeSelectionError(
                f"Invalid scope row at index {index}: {row}"
            ) from exc
    return scopes


def _scope_sort_key(scope: ScopeRecord | ScopeWithTertile) -> tuple[str, str, str]:
    raw_scope = scope.scope if isinstance(scope, ScopeWithTertile) else scope
    return (raw_scope.class_id, raw_scope.assessment_id, raw_scope.exercise_id)


def stage1_canonical_exclusion(
    scopes: list[ScopeRecord],
) -> tuple[list[ScopeRecord], dict[str, Any]]:
    retained: list[ScopeRecord] = []
    dropped: list[dict[str, Any]] = []
    for scope in sorted(scopes, key=_scope_sort_key):
        if scope.canonical_80:
            dropped.append(
                {
                    "scope_id": scope.scope_id,
                    "reason": "canonical_80_true",
                    "mean_pairwise_dist": scope.mean_pairwise_dist,
                    "n_students": scope.n_students,
                    "distinct_codes": scope.distinct_codes,
                }
            )
            continue
        retained.append(scope)
    report = {
        "input_count": len(scopes),
        "dropped_count": len(dropped),
        "retained_count": len(retained),
        "dropped_scopes": dropped,
    }
    return retained, report


def _empirical_percentile(values: list[float], q: float) -> float:
    if not values:
        raise BranchingProbeSelectionError("Cannot compute percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise BranchingProbeSelectionError(f"Percentile must be between 0 and 1, got {q}")
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def _fixed_threshold_breakdown(values: list[float]) -> list[dict[str, Any]]:
    if not values:
        return []
    sorted_values = sorted(values)
    bins = [
        ("<= 0.15", lambda value: value <= FIXED_DIVERSITY_THRESHOLDS[0]),
        (
            "(0.15, 0.30]",
            lambda value: FIXED_DIVERSITY_THRESHOLDS[0] < value <= FIXED_DIVERSITY_THRESHOLDS[1],
        ),
        (
            "(0.30, 0.45]",
            lambda value: FIXED_DIVERSITY_THRESHOLDS[1] < value <= FIXED_DIVERSITY_THRESHOLDS[2],
        ),
        ("> 0.45", lambda value: value > FIXED_DIVERSITY_THRESHOLDS[2]),
    ]
    output: list[dict[str, Any]] = []
    for label, predicate in bins:
        output.append(
            {
                "bucket": label,
                "count": sum(1 for value in sorted_values if predicate(value)),
            }
        )
    return output


def stage2_assign_tertiles(
    scopes: list[ScopeRecord],
) -> tuple[list[ScopeWithTertile], dict[str, Any]]:
    if not scopes:
        raise BranchingProbeSelectionError("Stage 2 received no scopes after canonical exclusion")
    distances = [scope.mean_pairwise_dist for scope in scopes]
    p33 = _empirical_percentile(distances, 0.33)
    p67 = _empirical_percentile(distances, 0.67)
    assignments: list[ScopeWithTertile] = []
    tertile_counts: Counter[str] = Counter()
    for scope in sorted(scopes, key=_scope_sort_key):
        if scope.mean_pairwise_dist <= p33:
            tertile = "T1"
        elif scope.mean_pairwise_dist <= p67:
            tertile = "T2"
        else:
            tertile = "T3"
        tertile_counts[tertile] += 1
        assignments.append(ScopeWithTertile(scope=scope, tertile=tertile))
    report = {
        "p33_mean_pairwise_dist": p33,
        "p67_mean_pairwise_dist": p67,
        "retained_scope_count": len(scopes),
        "tertile_counts": {label: tertile_counts.get(label, 0) for label in TERTILES},
        "fixed_threshold_counts": _fixed_threshold_breakdown(distances),
        "distribution": {
            "min": min(distances),
            "median": _empirical_percentile(distances, 0.5),
            "max": max(distances),
        },
        "assigned_scopes": [
            {
                "scope_id": item.scope.scope_id,
                "tertile": item.tertile,
                "mean_pairwise_dist": item.scope.mean_pairwise_dist,
            }
            for item in assignments
        ],
    }
    return assignments, report


def stage3_enumerate_scope_candidates(
    scoped_assignments: list[ScopeWithTertile],
    dataset_root: Path,
) -> tuple[dict[str, list[StudentLogCandidate]], dict[str, Any]]:
    scope_candidates: dict[str, list[StudentLogCandidate]] = {}
    scope_reports: list[dict[str, Any]] = []
    total_with_both_logs = 0
    total_missing = 0

    for scoped in sorted(scoped_assignments, key=_scope_sort_key):
        class_users_root = dataset_root / scoped.scope.class_id / "users"
        if not class_users_root.exists():
            raise BranchingProbeSelectionError(f"Users directory missing: {class_users_root}")
        candidates: list[StudentLogCandidate] = []
        dropped_students: list[dict[str, Any]] = []
        user_dirs = sorted(path for path in class_users_root.iterdir() if path.is_dir())
        for user_dir in user_dirs:
            student_id = user_dir.name
            execution_path = (
                user_dir
                / "executions"
                / f"{scoped.scope.assessment_id}_{scoped.scope.exercise_id}.log"
            )
            codemirror_path = (
                user_dir
                / "codemirror"
                / f"{scoped.scope.assessment_id}_{scoped.scope.exercise_id}.log"
            )
            if execution_path.exists() and codemirror_path.exists():
                candidates.append(
                    StudentLogCandidate(
                        class_id=scoped.scope.class_id,
                        assessment_id=scoped.scope.assessment_id,
                        exercise_id=scoped.scope.exercise_id,
                        student_id=student_id,
                        execution_path=execution_path,
                        codemirror_path=codemirror_path,
                    )
                )
                continue
            missing_reason_parts: list[str] = []
            if not execution_path.exists():
                missing_reason_parts.append("execution_log_missing")
            if not codemirror_path.exists():
                missing_reason_parts.append("codemirror_log_missing")
            dropped_students.append(
                {
                    "student_id": student_id,
                    "reason": "+".join(missing_reason_parts),
                    "execution_path": str(execution_path),
                    "codemirror_path": str(codemirror_path),
                }
            )
        scope_candidates[scoped.scope_id] = candidates
        total_with_both_logs += len(candidates)
        total_missing += len(dropped_students)
        reason_counts = Counter(item["reason"] for item in dropped_students)
        scope_reports.append(
            {
                "scope_id": scoped.scope_id,
                "tertile": scoped.tertile,
                "n_users_in_class": len(user_dirs),
                "n_candidates_with_both_logs": len(candidates),
                "n_missing_logs": len(dropped_students),
                "missing_log_reason_counts": dict(sorted(reason_counts.items())),
                "candidate_student_ids": [candidate.student_id for candidate in candidates],
                "dropped_students": dropped_students,
            }
        )
    report = {
        "scope_count": len(scoped_assignments),
        "total_candidates_with_both_logs": total_with_both_logs,
        "total_missing_logs": total_missing,
        "scope_reports": scope_reports,
    }
    return scope_candidates, report


def _validated_student_to_dict(candidate: ValidatedStudentCandidate) -> dict[str, Any]:
    return {
        "student_id": candidate.student_id,
        "interior_n": candidate.interior_n,
        "visible_attempt_count": candidate.visible_attempt_count,
        "total_attempt_count": candidate.total_attempt_count,
        "attempt_n_code": candidate.attempt_n_code,
        "attempt_n_normalized_code": candidate.attempt_n_normalized_code,
        "attempt_n1_code": candidate.attempt_n1_code,
        "attempt_n_pass_fail_vector": list(candidate.attempt_n_pass_fail_vector),
        "attempt_n_failed_test_indices_0idx": list(candidate.attempt_n_failed_test_indices_0idx),
        "attempt_n_test_count": candidate.attempt_n_test_count,
        "attempt_n_pass_vector_signature": candidate.attempt_n_pass_vector_signature,
        "execution_path": str(candidate.execution_path),
        "codemirror_path": str(candidate.codemirror_path),
        "custom_id": candidate.custom_id,
    }


def _assert_fixed_student_transition(student: ValidatedStudentCandidate, *, context: str) -> None:
    if student.interior_n != FIXED_INTERIOR_N:
        raise BranchingProbeSelectionError(
            f"{context} requires fixed interior_n={FIXED_INTERIOR_N} for {student.custom_id}, got {student.interior_n}"
        )


def stage4_validate_attempt_depth(
    scoped_assignments: list[ScopeWithTertile],
    scope_candidates: dict[str, list[StudentLogCandidate]],
) -> tuple[dict[str, list[ValidatedStudentCandidate]], dict[str, Any]]:
    validated_by_scope: dict[str, list[ValidatedStudentCandidate]] = {}
    scope_reports: list[dict[str, Any]] = []
    parse_failure_messages: Counter[str] = Counter()
    total_candidates = 0
    total_dropped_parse = 0
    total_dropped_insufficient = 0
    total_retained = 0

    for scoped in sorted(scoped_assignments, key=_scope_sort_key):
        candidates = scope_candidates.get(scoped.scope_id, [])
        retained: list[ValidatedStudentCandidate] = []
        dropped_parse: list[dict[str, Any]] = []
        dropped_insufficient: list[dict[str, Any]] = []
        total_candidates += len(candidates)
        for candidate in sorted(candidates, key=lambda item: item.student_id):
            log_text = candidate.execution_path.read_text(encoding="utf-8")
            try:
                attempts = parse_execution_log_text(log_text)
            except ExecutionParseError as exc:
                message = str(exc)
                parse_failure_messages[message] += 1
                dropped_parse.append(
                    {
                        "student_id": candidate.student_id,
                        "reason": "execution_parse_failure",
                        "error_type": type(exc).__name__,
                        "error_message": message,
                        "execution_path": str(candidate.execution_path),
                    }
                )
                continue
            if len(attempts) < MIN_TOTAL_ATTEMPTS_FOR_FIXED_SELECTION:
                dropped_insufficient.append(
                    {
                        "student_id": candidate.student_id,
                        "reason": "insufficient_parseable_submissions",
                        "attempt_count": len(attempts),
                        "required_attempt_count": MIN_TOTAL_ATTEMPTS_FOR_FIXED_SELECTION,
                        "execution_path": str(candidate.execution_path),
                    }
                )
                continue
            chosen_n = FIXED_INTERIOR_N
            attempt_n = attempts[chosen_n]
            attempt_n1 = attempts[chosen_n + 1]
            match_state = build_match_state(attempt_n)
            retained.append(
                ValidatedStudentCandidate(
                    class_id=candidate.class_id,
                    assessment_id=candidate.assessment_id,
                    exercise_id=candidate.exercise_id,
                    student_id=candidate.student_id,
                    execution_path=candidate.execution_path,
                    codemirror_path=candidate.codemirror_path,
                    total_attempt_count=len(attempts),
                    interior_n=chosen_n,
                    attempt_n_code=attempt_n.code,
                    attempt_n_normalized_code=match_state.normalized_code,
                    attempt_n1_code=attempt_n1.code,
                    attempt_n_pass_fail_vector=match_state.pass_vector,
                    attempt_n_failed_test_indices_0idx=match_state.failed_test_indices_0idx,
                    attempt_n_test_count=match_state.test_count,
                    attempt_n_pass_vector_signature=pass_vector_signature(match_state.pass_vector),
                )
            )
        retained = sorted(retained, key=lambda item: item.student_id)
        validated_by_scope[scoped.scope_id] = retained
        total_dropped_parse += len(dropped_parse)
        total_dropped_insufficient += len(dropped_insufficient)
        total_retained += len(retained)
        scope_reports.append(
            {
                "scope_id": scoped.scope_id,
                "tertile": scoped.tertile,
                "n_candidates": len(candidates),
                "n_dropped_parse": len(dropped_parse),
                "n_dropped_insufficient_attempts": len(dropped_insufficient),
                "n_retained": len(retained),
                "retained_students": [_validated_student_to_dict(item) for item in retained],
                "dropped_parse_students": dropped_parse,
                "dropped_insufficient_attempts_students": dropped_insufficient,
            }
        )

    parse_failure_rate = (total_dropped_parse / total_candidates) if total_candidates else 0.0
    report = {
        "fixed_transition_index_0idx": FIXED_INTERIOR_N,
        "fixed_visible_attempt_count": FIXED_VISIBLE_ATTEMPT_COUNT,
        "minimum_total_attempt_count": MIN_TOTAL_ATTEMPTS_FOR_FIXED_SELECTION,
        "scope_reports": scope_reports,
        "total_candidates": total_candidates,
        "total_dropped_parse": total_dropped_parse,
        "total_dropped_insufficient_attempts": total_dropped_insufficient,
        "total_retained_students": total_retained,
        "parse_failure_rate": parse_failure_rate,
        "parse_failure_message_counts": dict(sorted(parse_failure_messages.items())),
    }
    return validated_by_scope, report


def stage5_select_student_pairs(
    scoped_assignments: list[ScopeWithTertile],
    validated_by_scope: dict[str, list[ValidatedStudentCandidate]],
    *,
    seed: int,
    dataset_root: Path,
    condition: str,
) -> tuple[list[ScopePairSelection], dict[str, Any]]:
    validate_condition(condition)
    selected_pairs: list[ScopePairSelection] = []
    selected_scope_reports: list[dict[str, Any]] = []
    dropped_scopes_insufficient: list[dict[str, Any]] = []
    dropped_scopes_preflight: list[dict[str, Any]] = []
    dropped_scopes_l2a: list[dict[str, Any]] = []
    selected_row_count = 0
    selected_l2a_group_count = 0
    for scoped in sorted(scoped_assignments, key=_scope_sort_key):
        validated = validated_by_scope.get(scoped.scope_id, [])
        if len(validated) < 2:
            dropped_scopes_insufficient.append(
                {
                    "scope_id": scoped.scope_id,
                    "tertile": scoped.tertile,
                    "reason": "fewer_than_two_retained_students",
                    "retained_student_ids": [item.student_id for item in validated],
                }
            )
            continue
        buildable_students: list[ValidatedStudentCandidate] = []
        preflight_failures: list[dict[str, Any]] = []
        for student in validated:
            resolved, failure = _resolve_buildable_student_selection(
                student,
                dataset_root,
                MATCHED_COHORT_PREFLIGHT_CONDITION,
            )
            if resolved is None:
                if failure is None:
                    raise BranchingProbeSelectionError(
                        f"Internal error: missing preflight failure details for {student.custom_id}"
                    )
                preflight_failures.append(failure)
                continue
            buildable_students.append(resolved)
        if len(buildable_students) < 2:
            dropped_scopes_preflight.append(
                {
                    "scope_id": scoped.scope_id,
                    "tertile": scoped.tertile,
                    "reason": "fewer_than_two_buildable_students",
                    "buildable_student_ids": [item.student_id for item in buildable_students],
                    "student_failures": preflight_failures,
                }
            )
            continue
        groups: dict[tuple[bool, ...], list[ValidatedStudentCandidate]] = {}
        for student in buildable_students:
            if not student.attempt_n_pass_fail_vector:
                raise BranchingProbeSelectionError(
                    f"Missing attempt_n_pass_fail_vector for buildable student {student.custom_id}"
                )
            groups.setdefault(student.attempt_n_pass_fail_vector, []).append(student)
        kept_groups: list[tuple[tuple[bool, ...], list[ValidatedStudentCandidate]]] = []
        dropped_singletons: list[dict[str, Any]] = []
        for vector, students in sorted(
            groups.items(),
            key=lambda item: (
                pass_vector_signature(item[0]),
                [student.student_id for student in item[1]],
            ),
        ):
            ordered_students = tuple(sorted(students, key=lambda item: item.student_id))
            signature = pass_vector_signature(vector)
            if len(ordered_students) < 2:
                dropped_singletons.append(
                    {
                        "attempt_n_pass_vector_signature": signature,
                        "attempt_n_pass_fail_vector": list(vector),
                        "student_ids": [student.student_id for student in ordered_students],
                    }
                )
                continue
            kept_groups.append((vector, list(ordered_students)))
        if not kept_groups:
            dropped_scopes_l2a.append(
                {
                    "scope_id": scoped.scope_id,
                    "tertile": scoped.tertile,
                    "reason": "no_l2a_group_with_at_least_two_buildable_students",
                    "buildable_student_ids": [item.student_id for item in buildable_students],
                    "preflight_failures": preflight_failures,
                    "dropped_singleton_l2a_groups": dropped_singletons,
                }
            )
            continue
        selected_students = tuple(
            student
            for _vector, students in kept_groups
            for student in sorted(students, key=lambda item: item.student_id)
        )
        excluded_buildable_student_ids = sorted(
            {student_id for group in dropped_singletons for student_id in group["student_ids"]}
        )
        selected_pairs.append(
            ScopePairSelection(
                scope=scoped,
                students=selected_students,
            )
        )
        selected_scope_reports.append(
            {
                "scope_id": scoped.scope_id,
                "tertile": scoped.tertile,
                "buildable_student_count_before_l2a": len(buildable_students),
                "buildable_student_ids_before_l2a": [
                    item.student_id for item in buildable_students
                ],
                "selected_row_count": len(selected_students),
                "excluded_buildable_student_count": len(excluded_buildable_student_ids),
                "excluded_buildable_student_ids": excluded_buildable_student_ids,
                "l2a_group_signature_counts": dict(
                    sorted(
                        Counter(
                            student.attempt_n_pass_vector_signature for student in selected_students
                        ).items()
                    )
                ),
                "preflight_failures": preflight_failures,
                "dropped_singleton_l2a_groups": dropped_singletons,
                "students": [_validated_student_to_dict(item) for item in selected_students],
            }
        )
        selected_row_count += len(selected_students)
        selected_l2a_group_count += len(kept_groups)
    report = {
        "seed": seed,
        "cohort_preflight_condition": MATCHED_COHORT_PREFLIGHT_CONDITION,
        "selected_scope_count": len(selected_pairs),
        "selected_row_count": selected_row_count,
        "selected_l2a_group_count": selected_l2a_group_count,
        "dropped_scope_count": (
            len(dropped_scopes_insufficient)
            + len(dropped_scopes_preflight)
            + len(dropped_scopes_l2a)
        ),
        "dropped_scope_count_insufficient_students": len(dropped_scopes_insufficient),
        "dropped_scope_count_bundle_preflight": len(dropped_scopes_preflight),
        "dropped_scope_count_no_l2a_match": len(dropped_scopes_l2a),
        "dropped_scopes": dropped_scopes_insufficient
        + dropped_scopes_preflight
        + dropped_scopes_l2a,
        "selected_scopes": selected_scope_reports,
    }
    return selected_pairs, report


def _resolve_buildable_student_selection(
    student: ValidatedStudentCandidate,
    dataset_root: Path,
    condition: str,
) -> tuple[ValidatedStudentCandidate | None, dict[str, Any] | None]:
    validated_condition = validate_condition(condition)
    _assert_fixed_student_transition(student, context="Stage 5 preflight")
    log_text = student.execution_path.read_text(encoding="utf-8")
    attempts = parse_execution_log_text(log_text)
    if len(attempts) < MIN_TOTAL_ATTEMPTS_FOR_FIXED_SELECTION:
        raise BranchingProbeSelectionError(
            f"Expected at least {MIN_TOTAL_ATTEMPTS_FOR_FIXED_SELECTION} parseable attempts for {student.custom_id}, "
            f"found {len(attempts)}"
        )
    try:
        _build_prompt_payload(
            class_id=student.class_id,
            assessment_id=student.assessment_id,
            exercise_id=student.exercise_id,
            student_id=student.student_id,
            transition_index=student.interior_n,
            data_root=dataset_root,
            condition=validated_condition,
        )
    except Exception as exc:
        return (
            None,
            {
                "student_id": student.student_id,
                "reason": "fixed_transition_not_buildable",
                "attempted_failures": [
                    {
                        "interior_n": student.interior_n,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                ],
            },
        )
    return (
        replace(
            student,
            attempt_n_code=attempts[student.interior_n].code,
            attempt_n_normalized_code=build_match_state(
                attempts[student.interior_n]
            ).normalized_code,
            attempt_n1_code=attempts[student.interior_n + 1].code,
        ),
        None,
    )


def _compute_scope_quotas(target_scopes: int) -> dict[str, int]:
    if target_scopes < 3:
        raise BranchingProbeSelectionError(
            f"target-scopes must be at least 3 to allocate across tertiles, got {target_scopes}"
        )
    base = target_scopes // 3
    remainder = target_scopes % 3
    quotas = {"T1": base, "T2": base, "T3": base}
    if remainder >= 1:
        quotas["T1"] += 1
    if remainder >= 2:
        quotas["T2"] += 1
    return quotas


def _donor_order_from_remaining(remaining: dict[str, deque[ScopePairSelection]]) -> list[str]:
    return sorted(TERTILES, key=lambda tertile: (-len(remaining[tertile]), tertile))


def stage6_stratified_sampling(
    selected_pairs: list[ScopePairSelection],
    *,
    target_scopes: int,
    seed: int,
) -> tuple[list[ScopePairSelection], dict[str, deque[ScopePairSelection]], dict[str, Any]]:
    quotas = _compute_scope_quotas(target_scopes)
    available: dict[str, list[ScopePairSelection]] = {tertile: [] for tertile in TERTILES}
    for pair in selected_pairs:
        available[pair.scope.tertile].append(pair)
    available_counts = {tertile: len(available[tertile]) for tertile in TERTILES}
    available_row_counts = {
        tertile: sum(item.student_count for item in available[tertile]) for tertile in TERTILES
    }
    rng = random.Random(seed)
    for tertile in TERTILES:
        available[tertile] = sorted(available[tertile], key=lambda item: item.scope_id)
        rng.shuffle(available[tertile])

    selected: list[ScopePairSelection] = []
    remaining: dict[str, deque[ScopePairSelection]] = {tertile: deque() for tertile in TERTILES}
    initial_counts: dict[str, int] = {}
    initial_row_counts: dict[str, int] = {}
    for tertile in TERTILES:
        quota = quotas[tertile]
        take = min(quota, len(available[tertile]))
        initial_counts[tertile] = take
        initially_selected = available[tertile][:take]
        initial_row_counts[tertile] = sum(item.student_count for item in initially_selected)
        selected.extend(initially_selected)
        remaining[tertile].extend(available[tertile][take:])

    rebalance_additions: Counter[str] = Counter()
    while len(selected) < target_scopes:
        donor = next(
            (tertile for tertile in _donor_order_from_remaining(remaining) if remaining[tertile]),
            None,
        )
        if donor is None:
            raise BranchingProbeSelectionError(
                f"Only {len(selected)} scopes available after stage 5; cannot reach target {target_scopes}"
            )
        selected.append(remaining[donor].popleft())
        rebalance_additions[donor] += 1

    actual_counts: Counter[str] = Counter(item.scope.tertile for item in selected)
    actual_row_counts: Counter[str] = Counter()
    for item in selected:
        actual_row_counts[item.scope.tertile] += item.student_count
    reserve_scope_count = sum(len(items) for items in remaining.values())
    reserve_row_count = sum(item.student_count for items in remaining.values() for item in items)
    not_selected_scopes: list[dict[str, Any]] = []
    for tertile in TERTILES:
        for item in remaining[tertile]:
            not_selected_scopes.append(
                {
                    "scope_id": item.scope_id,
                    "tertile": tertile,
                    "row_count": item.student_count,
                    "reason": "not_selected_after_stratified_sampling",
                }
            )
    report = {
        "seed": seed,
        "target_scopes": target_scopes,
        "requested_quotas": quotas,
        "available_scope_counts": available_counts,
        "available_row_counts": available_row_counts,
        "initial_selected_counts": initial_counts,
        "initial_selected_row_counts": initial_row_counts,
        "rebalance_additions": {
            tertile: rebalance_additions.get(tertile, 0) for tertile in TERTILES
        },
        "final_selected_counts": {tertile: actual_counts.get(tertile, 0) for tertile in TERTILES},
        "final_selected_row_counts": {
            tertile: actual_row_counts.get(tertile, 0) for tertile in TERTILES
        },
        "reserve_scope_count": reserve_scope_count,
        "reserve_row_count": reserve_row_count,
        "selected_scope_ids": [item.scope_id for item in selected],
        "not_selected_scopes": sorted(
            not_selected_scopes, key=lambda item: (item["tertile"], item["scope_id"])
        ),
    }
    return selected, remaining, report


def _prepare_out_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
        return
    for child in out_dir.iterdir():
        if child.name not in MANAGED_OUT_BASENAMES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _bundle_dir(out_dir: Path, student: ValidatedStudentCandidate) -> Path:
    return (
        out_dir
        / "bundles"
        / student.class_id
        / student.assessment_id
        / student.exercise_id
        / student.student_id
        / str(student.interior_n)
    )


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    if not path.exists():
        raise BranchingProbeSelectionError(f"{context} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BranchingProbeSelectionError(f"{context} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BranchingProbeSelectionError(f"{context} is not a JSON object: {path}")
    return payload


def _validate_request_body_contract(
    payload: dict[str, Any],
    *,
    custom_id: str,
    expected_model: str,
    expected_reasoning_effort: str,
) -> None:
    model = payload.get("model")
    if model != expected_model:
        raise BranchingProbeSelectionError(
            f"Request body model mismatch for {custom_id}: {model!r} != {expected_model!r}"
        )
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, dict):
        raise BranchingProbeSelectionError(
            f"Request body reasoning metadata is missing for {custom_id}"
        )
    reasoning_effort = reasoning.get("effort")
    if reasoning_effort != expected_reasoning_effort:
        raise BranchingProbeSelectionError(
            f"Request body reasoning effort mismatch for {custom_id}: "
            f"{reasoning_effort!r} != {expected_reasoning_effort!r}"
        )


def _relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _emit_one_bundle(
    *,
    out_dir: Path,
    dataset_root: Path,
    scope: ScopeWithTertile,
    student: ValidatedStudentCandidate,
    condition: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict[str, Any]:
    validated_condition = validate_condition(condition)
    _assert_fixed_student_transition(student, context="Bundle emission")
    if student.attempt_n_normalized_code is None:
        raise BranchingProbeSelectionError(
            f"Missing attempt_n_normalized_code for {student.custom_id}"
        )
    payload, observed_next_repair_target, observed_next_coarse_path = _build_prompt_payload(
        class_id=scope.scope.class_id,
        assessment_id=scope.scope.assessment_id,
        exercise_id=scope.scope.exercise_id,
        student_id=student.student_id,
        transition_index=student.interior_n,
        data_root=dataset_root,
        condition=validated_condition,
    )
    visible_attempt_count = len(payload["visible_attempts"])
    if visible_attempt_count != FIXED_VISIBLE_ATTEMPT_COUNT:
        raise BranchingProbeSelectionError(
            f"Expected exactly {FIXED_VISIBLE_ATTEMPT_COUNT} visible attempts for {student.custom_id}, "
            f"found {visible_attempt_count}"
        )
    system_prompt = build_system_prompt(payload["source"]["condition"])
    user_prompt = build_user_prompt(payload)
    response_schema = FullTracePredictionResponse.model_json_schema()
    request_body = _build_request_body(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    custom_id = student.custom_id
    batch_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": request_body,
    }

    bundle_dir = _bundle_dir(out_dir, student)
    bundle_dir.mkdir(parents=True, exist_ok=False)
    system_prompt_path = bundle_dir / "system_prompt.txt"
    user_payload_path = bundle_dir / "user_payload.json"
    user_prompt_path = bundle_dir / "user_prompt.txt"
    response_schema_path = bundle_dir / "response_schema.json"
    request_body_path = bundle_dir / "request_body.json"
    batch_request_path = bundle_dir / "batch_request.json"
    batch_request_jsonl_path = bundle_dir / "requests.jsonl"
    observed_next_repair_target_path = bundle_dir / "observed_next_repair_target.json"
    observed_next_coarse_path_path = bundle_dir / "observed_next_coarse_path.json"
    manifest_path = bundle_dir / "manifest.json"

    system_prompt_path.write_text(system_prompt, encoding="utf-8")
    _write_json(user_payload_path, payload)
    user_prompt_path.write_text(user_prompt, encoding="utf-8")
    _write_json(response_schema_path, response_schema)
    _write_json(request_body_path, request_body)
    _write_json(batch_request_path, batch_request)
    batch_request_jsonl_path.write_text(
        json.dumps(batch_request, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_json(observed_next_repair_target_path, observed_next_repair_target)
    _write_json(observed_next_coarse_path_path, observed_next_coarse_path)

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "custom_id": custom_id,
        "condition": validated_condition,
        "exercise_scope": scope.scope.exercise_scope,
        "tertile": scope.tertile,
        "diversity_score": scope.scope.mean_pairwise_dist,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "alignment_policy": ALIGNMENT_POLICY,
        "system_prompt_path": _relpath(system_prompt_path, bundle_dir),
        "user_payload_path": _relpath(user_payload_path, bundle_dir),
        "user_prompt_path": _relpath(user_prompt_path, bundle_dir),
        "response_schema_path": _relpath(response_schema_path, bundle_dir),
        "request_body_path": _relpath(request_body_path, bundle_dir),
        "batch_request_path": _relpath(batch_request_path, bundle_dir),
        "batch_request_jsonl_path": _relpath(batch_request_jsonl_path, bundle_dir),
        "observed_next_repair_target_path": _relpath(observed_next_repair_target_path, bundle_dir),
        "observed_next_coarse_path_path": _relpath(observed_next_coarse_path_path, bundle_dir),
        "transition_index_0idx": student.interior_n,
        "prompt_char_count": len(user_prompt),
        "visible_attempt_count": visible_attempt_count,
        "attempt_n_pass_fail_vector": list(student.attempt_n_pass_fail_vector),
        "attempt_n_failed_test_indices_0idx": list(student.attempt_n_failed_test_indices_0idx),
        "attempt_n_test_count": student.attempt_n_test_count,
        "attempt_n_pass_vector_signature": student.attempt_n_pass_vector_signature,
        "attempt_n_normalized_code": student.attempt_n_normalized_code,
    }
    _write_json(manifest_path, manifest)

    return {
        "custom_id": custom_id,
        "request_jsonl_line": json.dumps(batch_request, ensure_ascii=False),
        "bundle_map_entry": {
            "row_id": custom_id,
            "condition": validated_condition,
            "bundle_dir": str(bundle_dir.resolve()),
            "bundle_manifest_path": str(manifest_path.resolve()),
            "request_body_path": str(request_body_path.resolve()),
            "observed_next_repair_target_path": str(observed_next_repair_target_path.resolve()),
            "observed_next_coarse_path_path": str(observed_next_coarse_path_path.resolve()),
            "exercise_scope": scope.scope.exercise_scope,
            "tertile": scope.tertile,
            "diversity_score": scope.scope.mean_pairwise_dist,
            "transition_index_0idx": student.interior_n,
            "visible_attempt_count": student.visible_attempt_count,
            "attempt_n_pass_fail_vector": list(student.attempt_n_pass_fail_vector),
            "attempt_n_failed_test_indices_0idx": list(student.attempt_n_failed_test_indices_0idx),
            "attempt_n_test_count": student.attempt_n_test_count,
            "attempt_n_pass_vector_signature": student.attempt_n_pass_vector_signature,
            "attempt_n_normalized_code": student.attempt_n_normalized_code,
        },
        "bundle_dir": bundle_dir,
    }


def stage7_emit_bundles(
    initial_selected_scopes: list[ScopePairSelection],
    *,
    out_dir: Path,
    dataset_root: Path,
    target_scopes: int,
    condition: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated_condition = validate_condition(condition)
    if len(initial_selected_scopes) != target_scopes:
        raise BranchingProbeSelectionError(
            f"Stage 7 expected exactly {target_scopes} selected scopes, got {len(initial_selected_scopes)}"
        )
    committed_scopes: list[ScopePairSelection] = []
    bundle_map: dict[str, dict[str, Any]] = {}
    ordered_custom_ids: list[str] = []
    request_lines: list[str] = []

    for scope_selection in initial_selected_scopes:
        emitted_for_scope: list[dict[str, Any]] = []
        try:
            for student in scope_selection.students:
                _assert_fixed_student_transition(student, context="Stage 7 bundle emission")
                emitted_for_scope.append(
                    _emit_one_bundle(
                        out_dir=out_dir,
                        dataset_root=dataset_root,
                        scope=scope_selection.scope,
                        student=student,
                        condition=validated_condition,
                        model=model,
                        reasoning_effort=reasoning_effort,
                    )
                )
        except Exception as exc:
            for student in scope_selection.students:
                bundle_dir = _bundle_dir(out_dir, student)
                if bundle_dir.exists():
                    shutil.rmtree(bundle_dir)
            for emitted in emitted_for_scope:
                bundle_dir = emitted["bundle_dir"]
                if bundle_dir.exists():
                    shutil.rmtree(bundle_dir)
            bundles_root = out_dir / "bundles"
            if bundles_root.exists():
                shutil.rmtree(bundles_root)
            raise BranchingProbeSelectionError(
                f"Bundle emission failed for {scope_selection.scope_id}: {exc}"
            ) from exc

        committed_scopes.append(scope_selection)
        for emitted in emitted_for_scope:
            ordered_custom_ids.append(emitted["custom_id"])
            bundle_map[emitted["custom_id"]] = emitted["bundle_map_entry"]
            request_lines.append(emitted["request_jsonl_line"])

    requests_path = out_dir / "requests.jsonl"
    requests_path.write_text("\n".join(request_lines) + "\n", encoding="utf-8")
    actual_tertile_counts = Counter(item.scope.tertile for item in committed_scopes)
    actual_row_tertile_counts: Counter[str] = Counter()
    for item in committed_scopes:
        actual_row_tertile_counts[item.scope.tertile] += item.student_count
    report = {
        "successful_scope_count": len(committed_scopes),
        "successful_bundle_count": len(ordered_custom_ids),
        "emission_failure_count": 0,
        "emission_failures": [],
        "replacement_log": [],
        "final_scope_ids": [item.scope_id for item in committed_scopes],
        "final_scope_tertile_counts": {
            tertile: actual_tertile_counts.get(tertile, 0) for tertile in TERTILES
        },
        "final_row_tertile_counts": {
            tertile: actual_row_tertile_counts.get(tertile, 0) for tertile in TERTILES
        },
    }
    artifacts = {
        "bundle_map": bundle_map,
        "ordered_custom_ids": ordered_custom_ids,
        "committed_scopes": committed_scopes,
        "requests_path": requests_path,
    }
    return artifacts, report


def _build_run_manifest(
    *,
    scopes_json: Path,
    scopes_json_sha256: str,
    dataset_root: Path,
    out_dir: Path,
    condition: str,
    target_scopes: int,
    seed: int,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    requested_quotas: dict[str, int],
    final_scope_counts: dict[str, int],
    final_row_counts: dict[str, int],
    stage5_report: dict[str, Any],
    selection_report_path: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    validated_condition = validate_condition(condition)
    stage5_scope_details = {
        str(item["scope_id"]): item
        for item in stage5_report.get("selected_scopes", [])
        if isinstance(item, dict)
    }
    documented_scope_rows = [
        {
            "scope_id": selection.scope_id,
            "tertile": selection.scope.tertile,
            "row_count": selection.student_count,
            "l2a_group_signature_counts": dict(
                sorted(
                    Counter(
                        student.attempt_n_pass_vector_signature for student in selection.students
                    ).items()
                )
            ),
            "buildable_student_count_before_l2a": int(
                stage5_scope_details[selection.scope_id]["buildable_student_count_before_l2a"]
            ),
            "buildable_student_ids_before_l2a": list(
                stage5_scope_details[selection.scope_id]["buildable_student_ids_before_l2a"]
            ),
            "excluded_buildable_student_count": int(
                stage5_scope_details[selection.scope_id]["excluded_buildable_student_count"]
            ),
            "excluded_buildable_student_ids": list(
                stage5_scope_details[selection.scope_id]["excluded_buildable_student_ids"]
            ),
            "dropped_singleton_l2a_groups": list(
                stage5_scope_details[selection.scope_id]["dropped_singleton_l2a_groups"]
            ),
        }
        for selection in artifacts["committed_scopes"]
    ]
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_name": out_dir.name,
        "probe_path": str(scopes_json.resolve()),
        "probe_sha256": scopes_json_sha256,
        "selection_report_path": str(selection_report_path.resolve()),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "condition": validated_condition,
        "counts": {
            "scopes": target_scopes,
            "rows": len(artifacts["ordered_custom_ids"]),
            "requests_created": len(artifacts["ordered_custom_ids"]),
        },
        "selection": {
            "dataset_root": str(dataset_root.resolve()),
            "condition": validated_condition,
            "target_scopes": target_scopes,
            "seed": seed,
            "fixed_transition_index_0idx": FIXED_INTERIOR_N,
            "fixed_visible_attempt_count": FIXED_VISIBLE_ATTEMPT_COUNT,
            "matched_cohort_preflight_condition": MATCHED_COHORT_PREFLIGHT_CONDITION,
            "requested_scope_quotas": requested_quotas,
            "documented_final_scope_counts": final_scope_counts,
            "documented_final_row_counts": final_row_counts,
            "documented_final_scope_rows": documented_scope_rows,
        },
        "paths": {
            "bundles_root": str((out_dir / "bundles").resolve()),
            "requests_jsonl": str(artifacts["requests_path"].resolve()),
            "output_jsonl": str((out_dir / "output.jsonl").resolve()),
            "error_jsonl": str((out_dir / "errors.jsonl").resolve()),
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
        "ordered_custom_ids": artifacts["ordered_custom_ids"],
        "bundle_map": artifacts["bundle_map"],
    }


def stage8_self_validate_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise BranchingProbeSelectionError(
            f"Manifest not found for self-validation: {manifest_path}"
        )
    manifest = _read_json_object(manifest_path, context="Run manifest")
    manifest_model = manifest.get("model")
    if not isinstance(manifest_model, str) or not manifest_model:
        raise BranchingProbeSelectionError("Manifest model is missing or invalid")
    manifest_reasoning_effort = manifest.get("reasoning_effort")
    if not isinstance(manifest_reasoning_effort, str) or not manifest_reasoning_effort:
        raise BranchingProbeSelectionError("Manifest reasoning_effort is missing or invalid")
    ordered_custom_ids = manifest.get("ordered_custom_ids")
    if not isinstance(ordered_custom_ids, list) or not all(
        isinstance(item, str) and item for item in ordered_custom_ids
    ):
        raise BranchingProbeSelectionError("Manifest ordered_custom_ids is missing or invalid")
    if len(set(ordered_custom_ids)) != len(ordered_custom_ids):
        raise BranchingProbeSelectionError("Manifest contains duplicate custom_id values")
    bundle_map = manifest.get("bundle_map")
    if not isinstance(bundle_map, dict):
        raise BranchingProbeSelectionError("Manifest bundle_map is missing or invalid")
    if set(bundle_map.keys()) != set(ordered_custom_ids):
        raise BranchingProbeSelectionError(
            "Manifest bundle_map keys do not match ordered_custom_ids"
        )
    manifest_condition = manifest.get("condition")
    if manifest_condition not in SUPPORTED_CONDITIONS:
        raise BranchingProbeSelectionError("Manifest condition is missing or invalid")

    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise BranchingProbeSelectionError("Manifest selection metadata is missing or invalid")
    selection_condition = selection.get("condition")
    if selection_condition not in SUPPORTED_CONDITIONS:
        raise BranchingProbeSelectionError("Manifest selection.condition is missing or invalid")
    if selection_condition != manifest_condition:
        raise BranchingProbeSelectionError(
            "Manifest condition and selection.condition do not match"
        )
    matched_cohort_preflight_condition = selection.get("matched_cohort_preflight_condition")
    if matched_cohort_preflight_condition != MATCHED_COHORT_PREFLIGHT_CONDITION:
        raise BranchingProbeSelectionError(
            "Manifest selection.matched_cohort_preflight_condition is missing or invalid"
        )
    try:
        expected_transition_index = int(selection.get("fixed_transition_index_0idx"))
        expected_visible_attempt_count = int(selection.get("fixed_visible_attempt_count"))
    except (TypeError, ValueError) as exc:
        raise BranchingProbeSelectionError(
            "Manifest selection fixed-depth metadata is missing or invalid"
        ) from exc
    if expected_transition_index != FIXED_INTERIOR_N:
        raise BranchingProbeSelectionError(
            f"Manifest selection.fixed_transition_index_0idx must equal {FIXED_INTERIOR_N}, got {expected_transition_index}"
        )
    if expected_visible_attempt_count != FIXED_VISIBLE_ATTEMPT_COUNT:
        raise BranchingProbeSelectionError(
            f"Manifest selection.fixed_visible_attempt_count must equal {FIXED_VISIBLE_ATTEMPT_COUNT}, got {expected_visible_attempt_count}"
        )
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise BranchingProbeSelectionError("Manifest counts metadata is missing or invalid")
    try:
        documented_row_total = int(counts.get("rows"))
        documented_requests_created = int(counts.get("requests_created"))
    except (TypeError, ValueError) as exc:
        raise BranchingProbeSelectionError(
            "Manifest counts metadata is missing or invalid"
        ) from exc
    if documented_row_total != len(ordered_custom_ids):
        raise BranchingProbeSelectionError(
            f"Manifest counts.rows mismatch: {documented_row_total} != {len(ordered_custom_ids)}"
        )
    if documented_requests_created != len(ordered_custom_ids):
        raise BranchingProbeSelectionError(
            f"Manifest counts.requests_created mismatch: {documented_requests_created} != {len(ordered_custom_ids)}"
        )
    paths = manifest.get("paths")
    if not isinstance(paths, dict):
        raise BranchingProbeSelectionError("Manifest paths metadata is missing or invalid")
    bundles_root = paths.get("bundles_root")
    if not isinstance(bundles_root, str) or not Path(bundles_root).exists():
        raise BranchingProbeSelectionError(
            "Manifest paths.bundles_root is missing or does not exist"
        )
    requests_jsonl = paths.get("requests_jsonl")
    if not isinstance(requests_jsonl, str) or not Path(requests_jsonl).exists():
        raise BranchingProbeSelectionError(
            "Manifest paths.requests_jsonl is missing or does not exist"
        )
    request_lines = [
        line
        for line in Path(requests_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(request_lines) != len(ordered_custom_ids):
        raise BranchingProbeSelectionError(
            f"requests.jsonl line count mismatch: {len(request_lines)} != {len(ordered_custom_ids)}"
        )
    request_custom_ids: list[str] = []
    requests_by_custom_id: dict[str, dict[str, Any]] = {}
    for line in request_lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BranchingProbeSelectionError("requests.jsonl contains invalid JSON") from exc
        if not isinstance(payload, dict):
            raise BranchingProbeSelectionError("requests.jsonl contains a non-object entry")
        custom_id = payload.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise BranchingProbeSelectionError(
                "requests.jsonl contains an entry without a valid custom_id"
            )
        if custom_id in requests_by_custom_id:
            raise BranchingProbeSelectionError(
                f"requests.jsonl contains duplicate custom_id {custom_id}"
            )
        method = payload.get("method")
        if method != "POST":
            raise BranchingProbeSelectionError(
                f"requests.jsonl method mismatch for {custom_id}: {method!r} != 'POST'"
            )
        url = payload.get("url")
        if url != "/v1/responses":
            raise BranchingProbeSelectionError(
                f"requests.jsonl url mismatch for {custom_id}: {url!r} != '/v1/responses'"
            )
        request_body = payload.get("body")
        if not isinstance(request_body, dict):
            raise BranchingProbeSelectionError(
                f"requests.jsonl body is missing or invalid for {custom_id}"
            )
        _validate_request_body_contract(
            request_body,
            custom_id=custom_id,
            expected_model=manifest_model,
            expected_reasoning_effort=manifest_reasoning_effort,
        )
        request_custom_ids.append(custom_id)
        requests_by_custom_id[custom_id] = payload
    if request_custom_ids != ordered_custom_ids:
        raise BranchingProbeSelectionError(
            "requests.jsonl custom_id order does not match ordered_custom_ids"
        )
    documented_scope_rows = selection.get("documented_final_scope_rows")
    if not isinstance(documented_scope_rows, list):
        raise BranchingProbeSelectionError(
            "Manifest selection.documented_final_scope_rows is missing"
        )
    documented_scope_map: dict[str, dict[str, Any]] = {}
    for item in documented_scope_rows:
        if not isinstance(item, dict):
            raise BranchingProbeSelectionError(
                "Manifest selection.documented_final_scope_rows contains a non-object"
            )
        scope_id = item.get("scope_id")
        if not isinstance(scope_id, str) or not scope_id:
            raise BranchingProbeSelectionError(
                "Manifest selection.documented_final_scope_rows has an invalid scope_id"
            )
        if scope_id in documented_scope_map:
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows contains duplicate scope_id {scope_id}"
            )
        tertile = item.get("tertile")
        if tertile not in TERTILES:
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows has invalid tertile for {scope_id}"
            )
        row_count = item.get("row_count")
        if not isinstance(row_count, int) or row_count < 2:
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows has invalid row_count for {scope_id}"
            )
        signature_counts = item.get("l2a_group_signature_counts")
        if not isinstance(signature_counts, dict) or not signature_counts:
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows is missing l2a_group_signature_counts for {scope_id}"
            )
        normalized_signature_counts: dict[str, int] = {}
        for signature, count in signature_counts.items():
            if not isinstance(signature, str) or not signature:
                raise BranchingProbeSelectionError(
                    f"Manifest selection.documented_final_scope_rows has invalid L2A signature for {scope_id}"
                )
            if not isinstance(count, int) or count < 2:
                raise BranchingProbeSelectionError(
                    f"Manifest selection.documented_final_scope_rows has invalid L2A signature count for {scope_id}"
                )
            normalized_signature_counts[signature] = count
        if sum(normalized_signature_counts.values()) != row_count:
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows row_count mismatch for {scope_id}"
            )
        buildable_student_count_before_l2a = item.get("buildable_student_count_before_l2a")
        if (
            not isinstance(buildable_student_count_before_l2a, int)
            or buildable_student_count_before_l2a < row_count
        ):
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows has invalid buildable_student_count_before_l2a for {scope_id}"
            )
        buildable_student_ids_before_l2a = item.get("buildable_student_ids_before_l2a")
        if (
            not isinstance(buildable_student_ids_before_l2a, list)
            or len(buildable_student_ids_before_l2a) != buildable_student_count_before_l2a
            or not all(
                isinstance(student_id, str) and student_id
                for student_id in buildable_student_ids_before_l2a
            )
        ):
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows has invalid buildable_student_ids_before_l2a for {scope_id}"
            )
        excluded_buildable_student_count = item.get("excluded_buildable_student_count")
        if (
            not isinstance(excluded_buildable_student_count, int)
            or excluded_buildable_student_count != buildable_student_count_before_l2a - row_count
        ):
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows has invalid excluded_buildable_student_count for {scope_id}"
            )
        excluded_buildable_student_ids = item.get("excluded_buildable_student_ids")
        if (
            not isinstance(excluded_buildable_student_ids, list)
            or len(excluded_buildable_student_ids) != excluded_buildable_student_count
            or not all(
                isinstance(student_id, str) and student_id
                for student_id in excluded_buildable_student_ids
            )
        ):
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows has invalid excluded_buildable_student_ids for {scope_id}"
            )
        dropped_singleton_l2a_groups = item.get("dropped_singleton_l2a_groups")
        if not isinstance(dropped_singleton_l2a_groups, list):
            raise BranchingProbeSelectionError(
                f"Manifest selection.documented_final_scope_rows is missing dropped_singleton_l2a_groups for {scope_id}"
            )
        documented_scope_map[scope_id] = {
            "tertile": tertile,
            "row_count": row_count,
            "l2a_group_signature_counts": dict(sorted(normalized_signature_counts.items())),
            "buildable_student_count_before_l2a": buildable_student_count_before_l2a,
            "buildable_student_ids_before_l2a": list(buildable_student_ids_before_l2a),
            "excluded_buildable_student_count": excluded_buildable_student_count,
            "excluded_buildable_student_ids": list(excluded_buildable_student_ids),
            "dropped_singleton_l2a_groups": list(dropped_singleton_l2a_groups),
        }

    per_scope_counts: Counter[str] = Counter()
    tertile_counts: Counter[str] = Counter()
    per_scope_signature_counts: dict[str, Counter[str]] = {}
    for custom_id in ordered_custom_ids:
        entry = bundle_map[custom_id]
        if not isinstance(entry, dict):
            raise BranchingProbeSelectionError(f"Bundle map entry is not an object for {custom_id}")
        condition = entry.get("condition")
        if condition not in SUPPORTED_CONDITIONS:
            raise BranchingProbeSelectionError(
                f"Bundle entry missing valid condition for {custom_id}"
            )
        if condition != manifest_condition:
            raise BranchingProbeSelectionError(
                f"Bundle entry condition mismatch for {custom_id}: {condition!r} != {manifest_condition!r}"
            )
        exercise_scope = entry.get("exercise_scope")
        if not isinstance(exercise_scope, str) or not exercise_scope:
            raise BranchingProbeSelectionError(
                f"Bundle entry missing exercise_scope for {custom_id}"
            )
        tertile = entry.get("tertile")
        if tertile not in TERTILES:
            raise BranchingProbeSelectionError(
                f"Bundle entry missing valid tertile for {custom_id}"
            )
        transition_index = entry.get("transition_index_0idx")
        if not isinstance(transition_index, int):
            raise BranchingProbeSelectionError(
                f"Bundle entry missing valid transition_index_0idx for {custom_id}"
            )
        if transition_index != expected_transition_index:
            raise BranchingProbeSelectionError(
                f"Bundle entry transition_index_0idx mismatch for {custom_id}: "
                f"{transition_index} != {expected_transition_index}"
            )
        visible_attempt_count = entry.get("visible_attempt_count")
        if not isinstance(visible_attempt_count, int):
            raise BranchingProbeSelectionError(
                f"Bundle entry missing valid visible_attempt_count for {custom_id}"
            )
        if visible_attempt_count != expected_visible_attempt_count:
            raise BranchingProbeSelectionError(
                f"Bundle entry visible_attempt_count mismatch for {custom_id}: "
                f"{visible_attempt_count} != {expected_visible_attempt_count}"
            )
        pass_vector = entry.get("attempt_n_pass_fail_vector")
        if not isinstance(pass_vector, list) or not pass_vector:
            raise BranchingProbeSelectionError(
                f"Bundle entry missing attempt_n_pass_fail_vector for {custom_id}"
            )
        if not all(isinstance(value, bool) for value in pass_vector):
            raise BranchingProbeSelectionError(
                f"Bundle entry attempt_n_pass_fail_vector must be a boolean list for {custom_id}"
            )
        failed_test_indices = entry.get("attempt_n_failed_test_indices_0idx")
        if not isinstance(failed_test_indices, list):
            raise BranchingProbeSelectionError(
                f"Bundle entry missing attempt_n_failed_test_indices_0idx for {custom_id}"
            )
        if not all(isinstance(value, int) and value >= 0 for value in failed_test_indices):
            raise BranchingProbeSelectionError(
                f"Bundle entry attempt_n_failed_test_indices_0idx must be non-negative integers for {custom_id}"
            )
        test_count = entry.get("attempt_n_test_count")
        if not isinstance(test_count, int) or test_count <= 0:
            raise BranchingProbeSelectionError(
                f"Bundle entry missing valid attempt_n_test_count for {custom_id}"
            )
        if len(pass_vector) != test_count:
            raise BranchingProbeSelectionError(
                f"Bundle entry attempt_n_test_count mismatch for {custom_id}: "
                f"{len(pass_vector)} != {test_count}"
            )
        derived_failed_indices = [index for index, passed in enumerate(pass_vector) if not passed]
        if failed_test_indices != derived_failed_indices:
            raise BranchingProbeSelectionError(
                f"Bundle entry attempt_n_failed_test_indices_0idx mismatch for {custom_id}"
            )
        signature = entry.get("attempt_n_pass_vector_signature")
        if not isinstance(signature, str) or not signature:
            raise BranchingProbeSelectionError(
                f"Bundle entry missing attempt_n_pass_vector_signature for {custom_id}"
            )
        derived_signature = pass_vector_signature(tuple(pass_vector))
        if signature != derived_signature:
            raise BranchingProbeSelectionError(
                f"Bundle entry attempt_n_pass_vector_signature mismatch for {custom_id}: "
                f"{signature!r} != {derived_signature!r}"
            )
        normalized_code = entry.get("attempt_n_normalized_code")
        if not isinstance(normalized_code, str):
            raise BranchingProbeSelectionError(
                f"Bundle entry missing attempt_n_normalized_code for {custom_id}"
            )
        per_scope_counts[exercise_scope] += 1
        tertile_counts[tertile] += 1
        per_scope_signature_counts.setdefault(exercise_scope, Counter())[signature] += 1
        for key in ("bundle_dir", "bundle_manifest_path", "request_body_path"):
            value = entry.get(key)
            if not isinstance(value, str) or not value:
                raise BranchingProbeSelectionError(f"Bundle entry missing {key} for {custom_id}")
            if not Path(value).exists():
                raise BranchingProbeSelectionError(
                    f"Referenced file does not exist for {custom_id}: {value}"
                )
        for key in ("observed_next_repair_target_path", "observed_next_coarse_path_path"):
            value = entry.get(key)
            if not isinstance(value, str) or not value:
                raise BranchingProbeSelectionError(f"Bundle entry missing {key} for {custom_id}")
            if not Path(value).exists():
                raise BranchingProbeSelectionError(
                    f"Referenced file does not exist for {custom_id}: {value}"
                )
        bundle_dir = Path(str(entry["bundle_dir"]))
        bundle_manifest_path = Path(str(entry["bundle_manifest_path"]))
        request_body_path = Path(str(entry["request_body_path"]))
        bundle_manifest = _read_json_object(
            bundle_manifest_path, context=f"Bundle manifest for {custom_id}"
        )
        if bundle_manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise BranchingProbeSelectionError(
                f"Bundle manifest schema_version mismatch for {custom_id}: "
                f"{bundle_manifest.get('schema_version')!r} != {BUNDLE_SCHEMA_VERSION!r}"
            )
        if bundle_manifest.get("custom_id") != custom_id:
            raise BranchingProbeSelectionError(
                f"Bundle manifest custom_id mismatch for {custom_id}: "
                f"{bundle_manifest.get('custom_id')!r} != {custom_id!r}"
            )
        if bundle_manifest.get("condition") != manifest_condition:
            raise BranchingProbeSelectionError(
                f"Bundle manifest condition mismatch for {custom_id}: "
                f"{bundle_manifest.get('condition')!r} != {manifest_condition!r}"
            )
        if bundle_manifest.get("model") != manifest_model:
            raise BranchingProbeSelectionError(
                f"Bundle manifest model mismatch for {custom_id}: "
                f"{bundle_manifest.get('model')!r} != {manifest_model!r}"
            )
        if bundle_manifest.get("reasoning_effort") != manifest_reasoning_effort:
            raise BranchingProbeSelectionError(
                f"Bundle manifest reasoning_effort mismatch for {custom_id}: "
                f"{bundle_manifest.get('reasoning_effort')!r} != {manifest_reasoning_effort!r}"
            )
        if bundle_manifest.get("transition_index_0idx") != expected_transition_index:
            raise BranchingProbeSelectionError(
                f"Bundle manifest transition_index_0idx mismatch for {custom_id}: "
                f"{bundle_manifest.get('transition_index_0idx')!r} != {expected_transition_index!r}"
            )
        if bundle_manifest.get("visible_attempt_count") != expected_visible_attempt_count:
            raise BranchingProbeSelectionError(
                f"Bundle manifest visible_attempt_count mismatch for {custom_id}: "
                f"{bundle_manifest.get('visible_attempt_count')!r} != {expected_visible_attempt_count!r}"
            )
        if bundle_manifest.get("attempt_n_pass_fail_vector") != pass_vector:
            raise BranchingProbeSelectionError(
                f"Bundle manifest attempt_n_pass_fail_vector mismatch for {custom_id}"
            )
        if bundle_manifest.get("attempt_n_failed_test_indices_0idx") != failed_test_indices:
            raise BranchingProbeSelectionError(
                f"Bundle manifest attempt_n_failed_test_indices_0idx mismatch for {custom_id}"
            )
        if bundle_manifest.get("attempt_n_test_count") != test_count:
            raise BranchingProbeSelectionError(
                f"Bundle manifest attempt_n_test_count mismatch for {custom_id}: "
                f"{bundle_manifest.get('attempt_n_test_count')!r} != {test_count!r}"
            )
        if bundle_manifest.get("attempt_n_pass_vector_signature") != signature:
            raise BranchingProbeSelectionError(
                f"Bundle manifest attempt_n_pass_vector_signature mismatch for {custom_id}"
            )
        if bundle_manifest.get("attempt_n_normalized_code") != normalized_code:
            raise BranchingProbeSelectionError(
                f"Bundle manifest attempt_n_normalized_code mismatch for {custom_id}"
            )
        request_body = _read_json_object(request_body_path, context=f"Request body for {custom_id}")
        _validate_request_body_contract(
            request_body,
            custom_id=custom_id,
            expected_model=manifest_model,
            expected_reasoning_effort=manifest_reasoning_effort,
        )
        aggregate_request = requests_by_custom_id[custom_id]
        aggregate_request_body = aggregate_request["body"]
        if request_body != aggregate_request_body:
            raise BranchingProbeSelectionError(
                f"Request body file does not match requests.jsonl body for {custom_id}"
            )
        batch_request_path_rel = bundle_manifest.get("batch_request_path")
        if not isinstance(batch_request_path_rel, str) or not batch_request_path_rel:
            raise BranchingProbeSelectionError(
                f"Bundle manifest batch_request_path is missing for {custom_id}"
            )
        batch_request_path = bundle_dir / batch_request_path_rel
        batch_request = _read_json_object(
            batch_request_path, context=f"Bundle batch_request for {custom_id}"
        )
        if batch_request != aggregate_request:
            raise BranchingProbeSelectionError(
                f"Bundle batch_request does not match requests.jsonl entry for {custom_id}"
            )
        batch_request_jsonl_path_rel = bundle_manifest.get("batch_request_jsonl_path")
        if not isinstance(batch_request_jsonl_path_rel, str) or not batch_request_jsonl_path_rel:
            raise BranchingProbeSelectionError(
                f"Bundle manifest batch_request_jsonl_path is missing for {custom_id}"
            )
        batch_request_jsonl_path = bundle_dir / batch_request_jsonl_path_rel
        if not batch_request_jsonl_path.exists():
            raise BranchingProbeSelectionError(
                f"Bundle batch_request_jsonl does not exist for {custom_id}: {batch_request_jsonl_path}"
            )
        bundle_request_lines = [
            line
            for line in batch_request_jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(bundle_request_lines) != 1:
            raise BranchingProbeSelectionError(
                f"Bundle requests.jsonl must contain exactly one line for {custom_id}"
            )
        try:
            bundle_request_jsonl = json.loads(bundle_request_lines[0])
        except json.JSONDecodeError as exc:
            raise BranchingProbeSelectionError(
                f"Bundle requests.jsonl contains invalid JSON for {custom_id}"
            ) from exc
        if bundle_request_jsonl != aggregate_request:
            raise BranchingProbeSelectionError(
                f"Bundle requests.jsonl entry does not match requests.jsonl body for {custom_id}"
            )

    if set(per_scope_counts) != set(documented_scope_map):
        raise BranchingProbeSelectionError(
            "Actual scope set does not match selection.documented_final_scope_rows"
        )
    for scope_id, actual_row_count in sorted(per_scope_counts.items()):
        documented = documented_scope_map[scope_id]
        if actual_row_count != documented["row_count"]:
            raise BranchingProbeSelectionError(
                f"Bundle row count mismatch for {scope_id}: {actual_row_count} != {documented['row_count']}"
            )
        actual_signature_counts = dict(sorted(per_scope_signature_counts[scope_id].items()))
        if actual_signature_counts != documented["l2a_group_signature_counts"]:
            raise BranchingProbeSelectionError(
                f"L2A signature counts do not match documented scope rows for {scope_id}: "
                f"{actual_signature_counts} != {documented['l2a_group_signature_counts']}"
            )
        if any(count < 2 for count in actual_signature_counts.values()):
            raise BranchingProbeSelectionError(
                f"Scope {scope_id} contains a singleton L2A group, which violates the frozen cohort contract"
            )
    expected_counts = selection.get("documented_final_scope_counts")
    if not isinstance(expected_counts, dict):
        raise BranchingProbeSelectionError(
            "Manifest selection.documented_final_scope_counts is missing"
        )
    normalized_expected = {tertile: int(expected_counts.get(tertile, 0)) for tertile in TERTILES}
    normalized_actual = Counter(
        documented["tertile"] for documented in documented_scope_map.values()
    )
    normalized_actual_dict = {
        tertile: int(normalized_actual.get(tertile, 0)) for tertile in TERTILES
    }
    actual_row_counts = {tertile: int(tertile_counts.get(tertile, 0)) for tertile in TERTILES}
    documented_row_counts = {
        tertile: sum(
            documented["row_count"]
            for documented in documented_scope_map.values()
            if documented["tertile"] == tertile
        )
        for tertile in TERTILES
    }
    if actual_row_counts != documented_row_counts:
        raise BranchingProbeSelectionError(
            "Bundle tertile row totals do not match documented_final_scope_rows"
        )
    expected_row_counts = selection.get("documented_final_row_counts")
    if not isinstance(expected_row_counts, dict):
        raise BranchingProbeSelectionError(
            "Manifest selection.documented_final_row_counts is missing"
        )
    normalized_expected_row_counts = {
        tertile: int(expected_row_counts.get(tertile, 0)) for tertile in TERTILES
    }
    if documented_row_counts != normalized_expected_row_counts:
        raise BranchingProbeSelectionError(
            f"Tertile row counts do not match documented final row counts: actual={documented_row_counts}, expected={normalized_expected_row_counts}"
        )
    if normalized_actual_dict != normalized_expected:
        raise BranchingProbeSelectionError(
            f"Tertile scope counts do not match documented final counts: actual={normalized_actual_dict}, expected={normalized_expected}"
        )
    target_scopes = selection.get("target_scopes")
    if int(sum(normalized_actual_dict.values())) != int(target_scopes):
        raise BranchingProbeSelectionError(
            f"Validated scope total {sum(normalized_actual_dict.values())} does not match target_scopes={target_scopes}"
        )

    return {
        "status": "passed",
        "unique_custom_ids": len(ordered_custom_ids),
        "validated_scope_counts": normalized_actual_dict,
        "validated_bundle_count": len(ordered_custom_ids),
    }


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _render_report_markdown(report: dict[str, Any], manifest_path: Path) -> str:
    stage1 = report["stage1_canonical_exclusion"]
    stage2 = report["stage2_diversity_tertiles"]
    stage3 = report["stage3_candidate_enumeration"]
    stage4 = report["stage4_attempt_depth_validation"]
    stage5 = report["stage5_pair_selection"]
    stage6 = report["stage6_stratified_sampling"]
    stage7 = report["stage7_bundle_emission"]
    stage8 = report["stage8_self_validation"]
    final_counts = report["summary"]["final_scope_counts"]

    threshold_lines = [
        f"| {item['bucket']} | {item['count']} |" for item in stage2["fixed_threshold_counts"]
    ]
    rebalance_note = (
        "none"
        if stage6["requested_quotas"] == final_counts
        else json.dumps(final_counts, ensure_ascii=False, sort_keys=True)
    )

    lines = [
        "# V6.2 Branching Probe Selection Report",
        "",
        "Pre-commit diversity distribution (stage-1 retained scopes):",
        f"- Retained scopes before selection: `{stage1['retained_count']}`",
        (
            "- Mean pairwise distance distribution: "
            f"`min={stage2['distribution']['min']:.6f}`, "
            f"`p33={stage2['p33_mean_pairwise_dist']:.6f}`, "
            f"`median={stage2['distribution']['median']:.6f}`, "
            f"`p67={stage2['p67_mean_pairwise_dist']:.6f}`, "
            f"`max={stage2['distribution']['max']:.6f}`"
        ),
        "- Fixed cut-points for sensitivity analysis: `0.15`, `0.30`, `0.45`",
        "",
        "| Diversity bucket | Scope count |",
        "| --- | ---: |",
        *threshold_lines,
        "",
        "## Final Output",
        "",
        f"- Final scope counts: `T1={final_counts['T1']}`, `T2={final_counts['T2']}`, `T3={final_counts['T3']}`",
        f"- Final scope total: `{report['summary']['final_scope_total']}`",
        f"- Final bundle total: `{report['summary']['final_bundle_total']}`",
        f"- Documented rebalance: `{rebalance_note}`",
        f"- Manifest: `{manifest_path}`",
        "",
        "## Stage 1 Canonical Exclusion",
        "",
        f"- Input scopes: `{stage1['input_count']}`",
        f"- Dropped canonical scopes: `{stage1['dropped_count']}`",
        f"- Retained scopes: `{stage1['retained_count']}`",
        "",
        "## Stage 2 Diversity Tertiles",
        "",
        f"- P33 boundary: `{stage2['p33_mean_pairwise_dist']:.6f}`",
        f"- P67 boundary: `{stage2['p67_mean_pairwise_dist']:.6f}`",
        (
            "- Retained-tertile counts: "
            f"`T1={stage2['tertile_counts']['T1']}`, "
            f"`T2={stage2['tertile_counts']['T2']}`, "
            f"`T3={stage2['tertile_counts']['T3']}`"
        ),
        "",
        "## Stage 3 Candidate Enumeration",
        "",
        f"- Students with both logs: `{stage3['total_candidates_with_both_logs']}`",
        f"- Students dropped for missing logs: `{stage3['total_missing_logs']}`",
        "",
        "## Stage 4 Attempt-Depth Validation",
        "",
        f"- Fixed transition index (0-indexed): `{stage4['fixed_transition_index_0idx']}`",
        f"- Fixed visible attempt count: `{stage4['fixed_visible_attempt_count']}`",
        f"- Minimum total attempts required: `{stage4['minimum_total_attempt_count']}`",
        f"- Candidate students checked: `{stage4['total_candidates']}`",
        f"- Dropped parse failures: `{stage4['total_dropped_parse']}`",
        f"- Dropped insufficient attempts: `{stage4['total_dropped_insufficient_attempts']}`",
        f"- Retained students: `{stage4['total_retained_students']}`",
        f"- Parse failure rate over stage-3 candidates: `{_format_percent(stage4['parse_failure_rate'])}`",
        "",
        "## Stage 5 L2A Matching",
        "",
        f"- Scopes with matched rows: `{stage5['selected_scope_count']}`",
        f"- Rows retained in frozen matched cohort: `{stage5['selected_row_count']}`",
        f"- Retained L2A groups: `{stage5['selected_l2a_group_count']}`",
        f"- Scopes dropped for <2 retained students: `{stage5['dropped_scope_count_insufficient_students']}`",
        f"- Scopes dropped by bundle preflight: `{stage5['dropped_scope_count_bundle_preflight']}`",
        f"- Scopes dropped for no non-singleton L2A group: `{stage5['dropped_scope_count_no_l2a_match']}`",
        "",
        "## Stage 6 Stratified Sampling",
        "",
        (
            "- Requested quotas: "
            f"`T1={stage6['requested_quotas']['T1']}`, "
            f"`T2={stage6['requested_quotas']['T2']}`, "
            f"`T3={stage6['requested_quotas']['T3']}`"
        ),
        (
            "- Final sampled counts before emission: "
            f"`T1={stage6['final_selected_counts']['T1']}`, "
            f"`T2={stage6['final_selected_counts']['T2']}`, "
            f"`T3={stage6['final_selected_counts']['T3']}`"
        ),
        (
            "- Final sampled row counts before emission: "
            f"`T1={stage6['final_selected_row_counts']['T1']}`, "
            f"`T2={stage6['final_selected_row_counts']['T2']}`, "
            f"`T3={stage6['final_selected_row_counts']['T3']}`"
        ),
        f"- Reserve scopes after sampling: `{stage6['reserve_scope_count']}`",
        f"- Reserve rows after sampling: `{stage6['reserve_row_count']}`",
        "",
        "## Stage 7 Bundle Emission",
        "",
        f"- Successful scopes emitted: `{stage7['successful_scope_count']}`",
        f"- Successful bundles emitted: `{stage7['successful_bundle_count']}`",
        f"- Emission failures: `{stage7['emission_failure_count']}`",
        "",
        "## Stage 8 Self-Validation",
        "",
        f"- Status: `{stage8['status']}`",
        (
            "- Validated final scope counts: "
            f"`T1={stage8['validated_scope_counts']['T1']}`, "
            f"`T2={stage8['validated_scope_counts']['T2']}`, "
            f"`T3={stage8['validated_scope_counts']['T3']}`"
        ),
        f"- Unique custom_ids: `{stage8['unique_custom_ids']}`",
    ]
    return "\n".join(lines) + "\n"


def run_selector(
    *,
    scopes_json: Path,
    dataset_root: Path,
    out_dir: Path,
    condition: str,
    target_scopes: int,
    seed: int,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    dry_run: bool,
) -> dict[str, Any]:
    validated_condition = validate_condition(condition)
    scopes_json_sha256 = _sha256_of_file(scopes_json)
    scopes = _load_scope_records(scopes_json)
    retained_scopes, stage1 = stage1_canonical_exclusion(scopes)
    scoped_assignments, stage2 = stage2_assign_tertiles(retained_scopes)
    scope_candidates, stage3 = stage3_enumerate_scope_candidates(scoped_assignments, dataset_root)
    validated_by_scope, stage4 = stage4_validate_attempt_depth(scoped_assignments, scope_candidates)
    selected_pairs, stage5 = stage5_select_student_pairs(
        scoped_assignments,
        validated_by_scope,
        seed=seed,
        dataset_root=dataset_root,
        condition=validated_condition,
    )
    sampled_scopes, _reserve_scopes, stage6 = stage6_stratified_sampling(
        selected_pairs,
        target_scopes=target_scopes,
        seed=seed,
    )

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "inputs": {
            "scopes_json": str(scopes_json.resolve()),
            "scopes_json_sha256": scopes_json_sha256,
            "dataset_root": str(dataset_root.resolve()),
            "out_dir": str(out_dir.resolve()),
            "condition": validated_condition,
            "target_scopes": target_scopes,
            "seed": seed,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "fixed_transition_index_0idx": FIXED_INTERIOR_N,
            "fixed_visible_attempt_count": FIXED_VISIBLE_ATTEMPT_COUNT,
            "dry_run": dry_run,
        },
        "stage1_canonical_exclusion": stage1,
        "stage2_diversity_tertiles": stage2,
        "stage3_candidate_enumeration": stage3,
        "stage4_attempt_depth_validation": stage4,
        "stage5_pair_selection": stage5,
        "stage6_stratified_sampling": stage6,
    }

    if dry_run:
        final_scope_counts = stage6["final_selected_counts"]
        final_row_counts = stage6["final_selected_row_counts"]
        final_bundle_total = sum(selection.student_count for selection in sampled_scopes)
        report["stage7_bundle_emission"] = {
            "successful_scope_count": len(sampled_scopes),
            "successful_bundle_count": final_bundle_total,
            "emission_failure_count": 0,
            "emission_failures": [],
            "replacement_log": [],
            "final_scope_tertile_counts": final_scope_counts,
            "final_row_tertile_counts": final_row_counts,
        }
        report["stage8_self_validation"] = {
            "status": "skipped_dry_run",
            "validated_scope_counts": final_scope_counts,
            "unique_custom_ids": final_bundle_total,
            "validated_bundle_count": final_bundle_total,
        }
        report["summary"] = {
            "final_scope_counts": final_scope_counts,
            "final_scope_total": len(sampled_scopes),
            "final_bundle_total": final_bundle_total,
            "stage_drop_counts": {
                "stage1_canonical_exclusion": stage1["dropped_count"],
                "stage3_missing_logs": stage3["total_missing_logs"],
                "stage4_parse_failures": stage4["total_dropped_parse"],
                "stage4_insufficient_attempts": stage4["total_dropped_insufficient_attempts"],
                "stage5_insufficient_scopes": stage5["dropped_scope_count_insufficient_students"],
                "stage5_bundle_preflight_scopes": stage5["dropped_scope_count_bundle_preflight"],
                "stage5_no_l2a_match_scopes": stage5["dropped_scope_count_no_l2a_match"],
                "stage6_unsampled_scopes": len(stage6["not_selected_scopes"]),
                "stage7_emission_failures": 0,
            },
        }
        return report

    _prepare_out_dir(out_dir)
    selection_report_path = out_dir / "selection_report.json"
    artifacts, stage7 = stage7_emit_bundles(
        sampled_scopes,
        out_dir=out_dir,
        dataset_root=dataset_root,
        target_scopes=target_scopes,
        condition=validated_condition,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    run_manifest = _build_run_manifest(
        scopes_json=scopes_json,
        scopes_json_sha256=scopes_json_sha256,
        dataset_root=dataset_root,
        out_dir=out_dir,
        condition=validated_condition,
        target_scopes=target_scopes,
        seed=seed,
        model=model,
        reasoning_effort=reasoning_effort,
        requested_quotas=stage6["requested_quotas"],
        final_scope_counts=stage7["final_scope_tertile_counts"],
        final_row_counts=stage7["final_row_tertile_counts"],
        stage5_report=stage5,
        selection_report_path=selection_report_path,
        artifacts=artifacts,
    )
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, run_manifest)
    try:
        stage8 = stage8_self_validate_manifest(manifest_path)
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise

    report["stage7_bundle_emission"] = stage7
    report["stage8_self_validation"] = stage8
    report["summary"] = {
        "final_scope_counts": stage8["validated_scope_counts"],
        "final_scope_total": sum(stage8["validated_scope_counts"].values()),
        "final_bundle_total": stage8["validated_bundle_count"],
        "stage_drop_counts": {
            "stage1_canonical_exclusion": stage1["dropped_count"],
            "stage3_missing_logs": stage3["total_missing_logs"],
            "stage4_parse_failures": stage4["total_dropped_parse"],
            "stage4_insufficient_attempts": stage4["total_dropped_insufficient_attempts"],
            "stage5_insufficient_scopes": stage5["dropped_scope_count_insufficient_students"],
            "stage5_bundle_preflight_scopes": stage5["dropped_scope_count_bundle_preflight"],
            "stage5_no_l2a_match_scopes": stage5["dropped_scope_count_no_l2a_match"],
            "stage6_unsampled_scopes": len(stage6["not_selected_scopes"]),
            "stage7_emission_failures": stage7["emission_failure_count"],
        },
    }

    _write_json(selection_report_path, report)
    selection_report_md_path = out_dir / "selection_report.md"
    selection_report_md_path.write_text(
        _render_report_markdown(report, manifest_path), encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    report = run_selector(
        scopes_json=args.scopes_json.resolve(),
        dataset_root=args.dataset_root.resolve(),
        out_dir=args.out_dir.resolve(),
        condition=args.condition,
        target_scopes=args.target_scopes,
        seed=args.seed,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
