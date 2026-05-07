from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from identity_perturbation.codebench_support.payload import failed_test_indices_0idx, pass_count


class V62FullTraceTargetError(ValueError):
    pass


def _contiguous_spans(lines: set[int]) -> list[dict[str, int]]:
    if not lines:
        return []
    sorted_lines = sorted(lines)
    spans: list[dict[str, int]] = []
    start = sorted_lines[0]
    end = sorted_lines[0]
    for line in sorted_lines[1:]:
        if line == end + 1:
            end = line
            continue
        spans.append(
            {
                "start_line_0idx": start,
                "end_line_0idx": end,
            }
        )
        start = end = line
    spans.append(
        {
            "start_line_0idx": start,
            "end_line_0idx": end,
        }
    )
    return spans


def _repair_spans_in_next_code(before_code: str, after_code: str) -> list[dict[str, object]]:
    before_lines = before_code.splitlines()
    after_lines = after_code.splitlines()
    if not after_lines:
        after_lines = [""]
    matcher = SequenceMatcher(None, before_lines, after_lines)
    spans: list[dict[str, object]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            edit_kind = "replace"
        elif tag == "insert":
            edit_kind = "insert"
        elif tag == "delete":
            edit_kind = "delete"
        else:
            raise V62FullTraceTargetError(f"Unsupported diff opcode tag: {tag!r}")

        if j1 < j2:
            start_line = j1
            end_line = j2 - 1
        else:
            anchor_line = j1 if j1 < len(after_lines) else len(after_lines) - 1
            if anchor_line < 0:
                raise V62FullTraceTargetError(
                    "Could not anchor a deletion-only repair span in next-code coordinates"
                )
            start_line = anchor_line
            end_line = anchor_line
        spans.append(
            {
                "start_line_0idx": start_line,
                "end_line_0idx": end_line,
                "edit_kind": edit_kind,
            }
        )
    return spans


def derive_repair_footprint(*, before_code: str, after_code: str) -> dict[str, object]:
    repair_spans = _repair_spans_in_next_code(before_code, after_code)
    changed_lines: set[int] = set()
    for span in repair_spans:
        changed_lines.update(
            range(
                int(span["start_line_0idx"]),
                int(span["end_line_0idx"]) + 1,
            )
        )
    return {
        "repair_spans_in_next_code": repair_spans,
        "changed_line_count_in_next_code": len(changed_lines),
        "changed_line_spans_in_next_code": _contiguous_spans(changed_lines),
    }


def _outcome_bucket(*, attempt_n: Any, attempt_n1: Any) -> str:
    current_passes = pass_count(attempt_n)
    next_passes = pass_count(attempt_n1)
    total_tests = len(attempt_n1.test_results)
    if next_passes == total_tests:
        return "solved"
    if next_passes > current_passes:
        return "partial_fix"
    if next_passes == current_passes:
        return "no_improvement"
    return "regressed"


def build_observed_repair_target(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index_0idx: int,
    attempt_n: Any,
    attempt_n1: Any,
) -> dict[str, object]:
    repair_footprint = derive_repair_footprint(
        before_code=attempt_n.code,
        after_code=attempt_n1.code,
    )

    return {
        "schema_version": "v6_2_observed_next_repair_target_v1",
        "source": {
            "class_id": class_id,
            "assessment_id": assessment_id,
            "exercise_id": exercise_id,
            "student_id": student_id,
            "transition_index_0idx": transition_index_0idx,
        },
        "attempt_n": {
            "attempt_index_0idx": attempt_n.attempt_index_0idx,
            "code": attempt_n.code,
            "grade": attempt_n.grade,
            "pass_count": pass_count(attempt_n),
            "failed_test_indices_0idx": list(failed_test_indices_0idx(attempt_n)),
        },
        "attempt_n1": {
            "attempt_index_0idx": attempt_n1.attempt_index_0idx,
            "code": attempt_n1.code,
            "grade": attempt_n1.grade,
            "pass_count": pass_count(attempt_n1),
            "failed_test_indices_0idx": list(failed_test_indices_0idx(attempt_n1)),
            "test_results": [
                {
                    "test_index_0idx": result.test_index_0idx,
                    "passed": result.passed,
                    "input": result.input,
                    "expected": result.expected,
                    "actual": result.actual,
                }
                for result in attempt_n1.test_results
            ],
        },
        "repair_target": {
            "outcome_bucket": _outcome_bucket(attempt_n=attempt_n, attempt_n1=attempt_n1),
            **repair_footprint,
        },
    }
