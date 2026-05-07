from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from identity_perturbation.codebench_support.executions import parse_execution_log_text
from identity_perturbation.prediction_audit.full_trace_prompting import SUPPORTED_CONDITIONS, validate_condition
from identity_perturbation.prediction_audit.pair_matching import (
    MatchState,
    build_match_state,
    l2a_error_match,
    match_state_to_dict,
    normalized_match_state_code_distance,
    pass_vector_signature,
)
from identity_perturbation.prediction_audit.select_branching_probe import (
    DEFAULT_SCOPES_JSON,
    BranchingProbeSelectionError,
    ScopeWithTertile,
    ValidatedStudentCandidate,
    _load_scope_records,
    _resolve_buildable_student_selection,
    _scope_sort_key,
    stage1_canonical_exclusion,
    stage2_assign_tertiles,
    stage3_enumerate_scope_candidates,
    stage4_validate_attempt_depth,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = REPO_ROOT / "2024-1"
DEFAULT_OUT_DIR = REPO_ROOT / "data/v62/match_l2_n2_audit"
REPORT_SCHEMA_VERSION = "v6_2_match_l2_n2_audit_report_v1"
DEFAULT_L2B_THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass(frozen=True)
class BuildableStudentRecord:
    scope_id: str
    tertile: str
    student: ValidatedStudentCandidate
    match_state: MatchState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit fixed-n=2 Match-L2A / Match-L2B yield over the real buildable v6.2 cohort."
        )
    )
    parser.add_argument("--scopes-json", type=Path, default=DEFAULT_SCOPES_JSON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--condition", choices=SUPPORTED_CONDITIONS, default="full")
    parser.add_argument(
        "--l2b-thresholds",
        type=str,
        default=",".join(f"{value:.2f}" for value in DEFAULT_L2B_THRESHOLDS),
        help="Comma-separated normalized-code-distance thresholds for the L2B sweep.",
    )
    return parser.parse_args()


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            value = float(stripped)
        except ValueError as exc:
            raise BranchingProbeSelectionError(
                f"Invalid L2B threshold {stripped!r}; expected a comma-separated float list"
            ) from exc
        if not 0.0 <= value <= 1.0:
            raise BranchingProbeSelectionError(
                f"L2B threshold must be between 0 and 1, got {value}"
            )
        values.append(value)
    if not values:
        raise BranchingProbeSelectionError("At least one L2B threshold is required")
    return tuple(sorted(set(values)))


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10, method="linear")),
        "p25": float(np.quantile(arr, 0.25, method="linear")),
        "median": float(np.quantile(arr, 0.50, method="linear")),
        "p75": float(np.quantile(arr, 0.75, method="linear")),
        "p90": float(np.quantile(arr, 0.90, method="linear")),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _student_record_to_dict(record: BuildableStudentRecord) -> dict[str, Any]:
    return {
        "student_id": record.student.student_id,
        "custom_id": record.student.custom_id,
        "interior_n": record.student.interior_n,
        "visible_attempt_count": record.student.visible_attempt_count,
        "total_attempt_count": record.student.total_attempt_count,
        "match_state": match_state_to_dict(record.match_state),
    }


def _build_matchable_students(
    scoped_assignments: list[ScopeWithTertile],
    validated_by_scope: dict[str, list[ValidatedStudentCandidate]],
    *,
    dataset_root: Path,
    condition: str,
) -> tuple[dict[str, list[BuildableStudentRecord]], dict[str, Any]]:
    validated_condition = validate_condition(condition)
    buildable_by_scope: dict[str, list[BuildableStudentRecord]] = {}
    scope_reports: list[dict[str, Any]] = []
    total_validated_students = 0
    total_buildable_students = 0
    preflight_failure_counts: Counter[str] = Counter()

    for scoped in sorted(scoped_assignments, key=_scope_sort_key):
        validated = validated_by_scope.get(scoped.scope_id, [])
        total_validated_students += len(validated)
        buildable_students: list[BuildableStudentRecord] = []
        dropped_students: list[dict[str, Any]] = []

        for student in validated:
            resolved, failure = _resolve_buildable_student_selection(
                student,
                dataset_root,
                validated_condition,
            )
            if resolved is None:
                if failure is None:
                    raise BranchingProbeSelectionError(
                        f"Internal error: missing preflight failure details for {student.custom_id}"
                    )
                reason = str(failure["reason"])
                preflight_failure_counts[reason] += 1
                dropped_students.append(failure)
                continue

            attempts = parse_execution_log_text(resolved.execution_path.read_text(encoding="utf-8"))
            if resolved.interior_n < 0 or resolved.interior_n >= len(attempts) - 1:
                raise BranchingProbeSelectionError(
                    f"Resolved student has invalid interior_n for parsed attempts: {resolved.custom_id}"
                )
            match_state = build_match_state(attempts[resolved.interior_n])
            buildable_students.append(
                BuildableStudentRecord(
                    scope_id=scoped.scope_id,
                    tertile=scoped.tertile,
                    student=resolved,
                    match_state=match_state,
                )
            )

        buildable_students = sorted(buildable_students, key=lambda item: item.student.student_id)
        buildable_by_scope[scoped.scope_id] = buildable_students
        total_buildable_students += len(buildable_students)
        signature_counts = Counter(
            pass_vector_signature(item.match_state.pass_vector) for item in buildable_students
        )
        scope_reports.append(
            {
                "scope_id": scoped.scope_id,
                "tertile": scoped.tertile,
                "n_validated_students": len(validated),
                "n_buildable_students": len(buildable_students),
                "n_preflight_failures": len(dropped_students),
                "buildable_students": [
                    _student_record_to_dict(item) for item in buildable_students
                ],
                "preflight_failures": dropped_students,
                "pass_vector_signature_counts": dict(sorted(signature_counts.items())),
            }
        )

    report = {
        "condition_used_for_preflight": validated_condition,
        "total_validated_students": total_validated_students,
        "total_buildable_students": total_buildable_students,
        "total_preflight_failures": total_validated_students - total_buildable_students,
        "preflight_failure_reason_counts": dict(sorted(preflight_failure_counts.items())),
        "scope_reports": scope_reports,
    }
    return buildable_by_scope, report


def _threshold_template(thresholds: tuple[float, ...]) -> dict[str, dict[str, int]]:
    return {
        f"{threshold:.2f}": {
            "undirected_pair_count_l2b": 0,
            "directed_pair_count_l2b": 0,
            "sender_row_count_l2b": 0,
            "undirected_pair_count_l2a_and_l2b": 0,
            "directed_pair_count_l2a_and_l2b": 0,
            "sender_row_count_l2a_and_l2b": 0,
        }
        for threshold in thresholds
    }


def audit_l2_pairs(
    buildable_by_scope: dict[str, list[BuildableStudentRecord]],
    *,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    total_buildable_rows = 0
    total_scopes = len(buildable_by_scope)
    scopes_with_at_least_two_buildable_rows = 0
    all_pair_count = 0
    l2a_pair_count = 0
    directed_all_pair_count = 0
    directed_l2a_pair_count = 0
    sender_rows_with_any_peer = 0
    sender_rows_with_l2a_peer = 0
    threshold_totals = _threshold_template(thresholds)
    all_code_distances: list[float] = []
    l2a_code_distances: list[float] = []
    scope_reports: list[dict[str, Any]] = []

    for scope_id in sorted(buildable_by_scope):
        students = buildable_by_scope[scope_id]
        total_buildable_rows += len(students)
        if len(students) >= 2:
            scopes_with_at_least_two_buildable_rows += 1
            directed_all_pair_count += len(students) * (len(students) - 1)
            sender_rows_with_any_peer += len(students)

        l2a_students: set[str] = set()
        threshold_student_sets_l2b = {f"{threshold:.2f}": set() for threshold in thresholds}
        threshold_student_sets_l2a_l2b = {f"{threshold:.2f}": set() for threshold in thresholds}
        threshold_counts = _threshold_template(thresholds)
        per_scope_all_code_distances: list[float] = []
        per_scope_l2a_code_distances: list[float] = []
        pass_vector_signature_counts = Counter(
            pass_vector_signature(item.match_state.pass_vector) for item in students
        )
        pair_count = 0
        l2a_pairs = 0

        for left, right in combinations(students, 2):
            pair_count += 1
            distance = normalized_match_state_code_distance(left.match_state, right.match_state)
            all_code_distances.append(distance)
            per_scope_all_code_distances.append(distance)

            is_l2a = l2a_error_match(left.match_state, right.match_state)
            if is_l2a:
                l2a_pairs += 1
                l2a_pair_count += 1
                l2a_students.add(left.student.custom_id)
                l2a_students.add(right.student.custom_id)
                l2a_code_distances.append(distance)
                per_scope_l2a_code_distances.append(distance)

            for threshold in thresholds:
                threshold_key = f"{threshold:.2f}"
                if distance <= threshold:
                    threshold_counts[threshold_key]["undirected_pair_count_l2b"] += 1
                    threshold_student_sets_l2b[threshold_key].add(left.student.custom_id)
                    threshold_student_sets_l2b[threshold_key].add(right.student.custom_id)
                    if is_l2a:
                        threshold_counts[threshold_key]["undirected_pair_count_l2a_and_l2b"] += 1
                        threshold_student_sets_l2a_l2b[threshold_key].add(left.student.custom_id)
                        threshold_student_sets_l2a_l2b[threshold_key].add(right.student.custom_id)

        all_pair_count += pair_count
        directed_l2a_pair_count += 2 * l2a_pairs
        sender_rows_with_l2a_peer += len(l2a_students)

        for threshold in thresholds:
            threshold_key = f"{threshold:.2f}"
            counts = threshold_counts[threshold_key]
            counts["directed_pair_count_l2b"] = 2 * counts["undirected_pair_count_l2b"]
            counts["sender_row_count_l2b"] = len(threshold_student_sets_l2b[threshold_key])
            counts["directed_pair_count_l2a_and_l2b"] = (
                2 * counts["undirected_pair_count_l2a_and_l2b"]
            )
            counts["sender_row_count_l2a_and_l2b"] = len(
                threshold_student_sets_l2a_l2b[threshold_key]
            )
            totals = threshold_totals[threshold_key]
            for key, value in counts.items():
                totals[key] += value

        scope_reports.append(
            {
                "scope_id": scope_id,
                "tertile": students[0].tertile if students else None,
                "n_buildable_rows": len(students),
                "all_undirected_pair_count": pair_count,
                "all_directed_pair_count": len(students) * (len(students) - 1)
                if len(students) >= 2
                else 0,
                "sender_row_count_with_any_peer": len(students) if len(students) >= 2 else 0,
                "l2a_undirected_pair_count": l2a_pairs,
                "l2a_directed_pair_count": 2 * l2a_pairs,
                "sender_row_count_l2a": len(l2a_students),
                "pass_vector_signature_counts": dict(sorted(pass_vector_signature_counts.items())),
                "all_pair_code_distance_stats": _stats(per_scope_all_code_distances),
                "l2a_pair_code_distance_stats": _stats(per_scope_l2a_code_distances),
                "thresholds": threshold_counts,
            }
        )

    scopes_with_l2a_pairs = sum(
        1 for report in scope_reports if report["l2a_undirected_pair_count"] > 0
    )

    return {
        "scope_count": total_scopes,
        "scope_count_with_at_least_two_buildable_rows": scopes_with_at_least_two_buildable_rows,
        "scope_count_with_l2a_pairs": scopes_with_l2a_pairs,
        "total_buildable_rows": total_buildable_rows,
        "sender_row_count_with_any_peer": sender_rows_with_any_peer,
        "sender_row_count_l2a": sender_rows_with_l2a_peer,
        "all_undirected_pair_count": all_pair_count,
        "all_directed_pair_count": directed_all_pair_count,
        "l2a_undirected_pair_count": l2a_pair_count,
        "l2a_directed_pair_count": directed_l2a_pair_count,
        "all_pair_code_distance_stats": _stats(all_code_distances),
        "l2a_pair_code_distance_stats": _stats(l2a_code_distances),
        "threshold_sweep": threshold_totals,
        "scope_reports": scope_reports,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    stage4 = report["stage4_attempt_depth"]
    buildable = report["buildable_universe"]
    matches = report["match_yield"]
    lines = [
        "# Match-L2 Fixed-n=2 Audit",
        "",
        "## Contract",
        f"- Schema version: `{report['schema_version']}`",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Scopes JSON: `{report['scopes_json']}`",
        f"- Preflight condition: `{buildable['condition_used_for_preflight']}`",
        f"- Fixed transition index (0-indexed): `{stage4['fixed_transition_index_0idx']}`",
        f"- Fixed visible attempt count: `{stage4['fixed_visible_attempt_count']}`",
        "- Match-L2A anchor: row-local `attempt_n` pass/fail vector",
        "- Match-L2B anchor: row-local `attempt_n` narrow-normalized code distance",
        "- L2B normalized code distance definition: `levenshtein_distance / (len(code_a) + len(code_b))` after narrow normalization",
        "",
        "## Counts",
        f"- Stage 4 retained students: `{stage4['total_retained_students']}`",
        f"- Buildable students after prompt preflight: `{buildable['total_buildable_students']}`",
        f"- Scopes with >=2 buildable rows: `{matches['scope_count_with_at_least_two_buildable_rows']}`",
        f"- Sender rows with any peer: `{matches['sender_row_count_with_any_peer']}`",
        f"- L2A sender rows: `{matches['sender_row_count_l2a']}`",
        f"- All directed pairs: `{matches['all_directed_pair_count']}`",
        f"- L2A directed pairs: `{matches['l2a_directed_pair_count']}`",
        "",
        "## Code Distance",
    ]
    l2a_stats = matches["l2a_pair_code_distance_stats"]
    if l2a_stats is None:
        lines.append("- No L2A pairs; no code-distance stats available.")
    else:
        lines.extend(
            [
                f"- L2A code-distance mean: `{l2a_stats['mean']:.6f}`",
                f"- L2A code-distance median: `{l2a_stats['median']:.6f}`",
                f"- L2A code-distance p90: `{l2a_stats['p90']:.6f}`",
            ]
        )
    lines.extend(["", "## Threshold Sweep"])
    for threshold_key, stats in matches["threshold_sweep"].items():
        lines.append(
            f"- `<= {threshold_key}`: "
            f"L2B sender rows `{stats['sender_row_count_l2b']}`, "
            f"L2A∩L2B sender rows `{stats['sender_row_count_l2a_and_l2b']}`, "
            f"L2B directed pairs `{stats['directed_pair_count_l2b']}`, "
            f"L2A∩L2B directed pairs `{stats['directed_pair_count_l2a_and_l2b']}`"
        )
    return "\n".join(lines) + "\n"


def build_report(
    *,
    scopes_json: Path,
    dataset_root: Path,
    condition: str,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    scope_records = _load_scope_records(scopes_json)
    stage1_scopes, stage1_report = stage1_canonical_exclusion(scope_records)
    scoped_assignments, stage2_report = stage2_assign_tertiles(stage1_scopes)
    scope_candidates, stage3_report = stage3_enumerate_scope_candidates(
        scoped_assignments, dataset_root
    )
    validated_by_scope, stage4_report = stage4_validate_attempt_depth(
        scoped_assignments, scope_candidates
    )
    buildable_by_scope, buildable_report = _build_matchable_students(
        scoped_assignments,
        validated_by_scope,
        dataset_root=dataset_root,
        condition=condition,
    )
    match_report = audit_l2_pairs(buildable_by_scope, thresholds=thresholds)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "scopes_json": str(scopes_json.resolve()),
        "condition": condition,
        "l2b_thresholds": list(thresholds),
        "stage1_canonical_exclusion": stage1_report,
        "stage2_tertiles": stage2_report,
        "stage3_log_enumeration": stage3_report,
        "stage4_attempt_depth": stage4_report,
        "buildable_universe": buildable_report,
        "match_yield": match_report,
    }


def main() -> int:
    args = parse_args()
    thresholds = _parse_thresholds(args.l2b_thresholds)
    validated_condition = validate_condition(args.condition)
    if not args.scopes_json.exists():
        raise SystemExit(f"Scopes JSON not found: {args.scopes_json}")
    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(
        scopes_json=args.scopes_json,
        dataset_root=args.dataset_root,
        condition=validated_condition,
        thresholds=thresholds,
    )

    report_path = args.out_dir / "report.json"
    markdown_path = args.out_dir / "report.md"
    _write_json(report_path, report)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"Wrote match audit report to {report_path}")
    print(f"Wrote match audit summary to {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
