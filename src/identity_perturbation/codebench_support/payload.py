from __future__ import annotations

import difflib
from collections import Counter, defaultdict

from .executions import next_test_outcome, normalize_output
from .models import CohortWindow, ExecutionAttempt


class PayloadError(ValueError):
    pass


def pass_count(attempt: ExecutionAttempt) -> int:
    return sum(result.passed for result in attempt.test_results)


def failed_test_indices_0idx(attempt: ExecutionAttempt) -> tuple[int, ...]:
    return tuple(result.test_index_0idx for result in attempt.test_results if not result.passed)


def attempt_depth_bucket(attempt_index_0idx: int) -> str:
    if attempt_index_0idx <= 0:
        return "0"
    if attempt_index_0idx == 1:
        return "1"
    return "2_plus"


def current_pass_count_bucket(attempt: ExecutionAttempt) -> str:
    count = pass_count(attempt)
    if count == 0:
        return "0"
    if count == len(attempt.test_results):
        return "all"
    return "partial"


def eventual_success(window: CohortWindow) -> bool:
    return any(
        next_test_outcome(attempt) == "all_pass" for attempt in window.transition.all_attempts
    )


def mismatch_signature(expected: str, actual: str, test_input: str) -> str:
    expected_norm = normalize_output(expected)
    actual_norm = normalize_output(actual)
    input_norm = normalize_output(test_input)

    expected_lines = expected_norm.split("\n") if expected_norm else []
    actual_lines = actual_norm.split("\n") if actual_norm else []

    if actual_norm == "":
        return "empty_output"
    if actual_norm == input_norm:
        return "unchanged_input_echo"
    if len(actual_lines) < len(expected_lines):
        return "missing_output_lines"
    if len(actual_lines) > len(expected_lines):
        if len(set(actual_lines)) < len(actual_lines):
            return "repeated_output_value"
        return "extra_output_lines"
    if expected_lines and actual_lines and expected_lines[0] != actual_lines[0]:
        return "wrong_first_output_line"
    if expected_lines and actual_lines and expected_lines[-1] != actual_lines[-1]:
        return "wrong_final_output_line"
    return "other_output_mismatch"


def code_delta_summary(previous_code: str, current_code: str) -> str:
    previous_lines = previous_code.splitlines()
    current_lines = current_code.splitlines()
    matcher = difflib.SequenceMatcher(None, previous_lines, current_lines)
    changed_lines = 0
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed_lines += max(1, j2 - j1)

    tags: list[str] = []
    if ("while" in current_code or "for" in current_code) and not (
        "while" in previous_code or "for" in previous_code
    ):
        tags.append("introduced_loop")
    if ("if" in current_code or "elif" in current_code or "else:" in current_code) and not (
        "if" in previous_code or "elif" in previous_code or "else:" in previous_code
    ):
        tags.append("introduced_conditional")
    if "print(" in current_code and "print(" not in previous_code:
        tags.append("introduced_print")
    if "input(" in current_code and "input(" not in previous_code:
        tags.append("introduced_input")

    suffix = f" {' '.join(tags)}" if tags else ""
    return f"changed_{changed_lines}_lines{suffix}".strip()


def trajectory_note(window: CohortWindow) -> str:
    attempt_n = window.transition.attempt_n
    previous = window.transition.history[-1] if window.transition.history else None
    current_pass = pass_count(attempt_n)
    if previous is None:
        return f"first visible attempt with {current_pass} passing tests"

    previous_pass = pass_count(previous)
    gain = current_pass - previous_pass
    if gain > 0:
        return f"pass count improved from {previous_pass} to {current_pass}"
    if gain < 0:
        return f"pass count dropped from {previous_pass} to {current_pass}"
    return f"pass count stayed at {current_pass}"


def build_slice_header(windows: tuple[CohortWindow, ...]) -> dict:
    if not windows:
        raise PayloadError("Cannot build payload for empty window slice")
    first = windows[0]
    keys = {(w.class_id, w.assessment_id, w.exercise_id) for w in windows}
    if len(keys) != 1:
        raise PayloadError("All windows must belong to the same class_assessment_exercise slice")
    return {
        "cohort_unit": "class_assessment_exercise",
        "class_id": first.class_id,
        "assessment_id": first.assessment_id,
        "exercise_id": first.exercise_id,
        "n_students": len({w.student_id for w in windows}),
        "n_attempt_windows": len(windows),
    }


def build_aggregate_summary(windows: tuple[CohortWindow, ...]) -> dict:
    if not windows:
        raise PayloadError("Cannot build aggregate summary for empty window slice")

    depth_counts = Counter(
        attempt_depth_bucket(w.transition.attempt_n.attempt_index_0idx) for w in windows
    )
    failed_test_counter: Counter[int] = Counter()
    mismatch_counter: Counter[str] = Counter()
    total_failing_tests = 0
    current_pass_counts: list[int] = []
    previous_pass_counts: list[int] = []
    positive_gain = 0

    by_student: dict[str, list[CohortWindow]] = defaultdict(list)
    for window in windows:
        by_student[window.student_id].append(window)
        current_attempt = window.transition.attempt_n
        current_pass = pass_count(current_attempt)
        current_pass_counts.append(current_pass)
        for result in current_attempt.test_results:
            if not result.passed:
                total_failing_tests += 1
                failed_test_counter[result.test_index_0idx] += 1
                mismatch_counter[
                    mismatch_signature(result.expected, result.actual, result.input)
                ] += 1

        if window.transition.history:
            previous_attempt = window.transition.history[-1]
            previous_pass = pass_count(previous_attempt)
            previous_pass_counts.append(previous_pass)
            if current_pass > previous_pass:
                positive_gain += 1

    students = list(by_student)
    eventual_successes = sum(
        1 for student_id in students if any(eventual_success(w) for w in by_student[student_id])
    )
    failures_before_success = 0
    for student_id in students:
        attempts = by_student[student_id][0].transition.all_attempts
        saw_failure = False
        saw_success = False
        for attempt in attempts:
            outcome = next_test_outcome(attempt)
            if outcome != "all_pass":
                saw_failure = True
            if outcome == "all_pass":
                saw_success = True
                break
        if saw_failure and saw_success:
            failures_before_success += 1

    n_windows = len(windows)
    n_students = len(students)
    return {
        "attempt_depth_distribution": dict(depth_counts),
        "failure_before_success_rate": failures_before_success / n_students,
        "eventual_success_rate": eventual_successes / n_students,
        "common_failed_tests": [
            {"test_index_0idx": test_index_0idx, "rate": count / n_windows}
            for test_index_0idx, count in failed_test_counter.most_common(5)
        ],
        "common_expected_actual_mismatches": [
            {"signature": signature, "rate": count / total_failing_tests}
            for signature, count in mismatch_counter.most_common(5)
        ],
        "pass_count_progression": {
            "mean_current_pass_count": sum(current_pass_counts) / len(current_pass_counts),
            "mean_previous_pass_count": (
                sum(previous_pass_counts) / len(previous_pass_counts)
                if previous_pass_counts
                else 0.0
            ),
            "positive_gain_rate": positive_gain / len(previous_pass_counts)
            if previous_pass_counts
            else 0.0,
        },
    }


def representative_group_key(window: CohortWindow) -> tuple:
    return (
        attempt_depth_bucket(window.transition.attempt_n.attempt_index_0idx),
        current_pass_count_bucket(window.transition.attempt_n),
        eventual_success(window),
        failed_test_indices_0idx(window.transition.attempt_n),
    )


def _window_rank_tuple(window: CohortWindow) -> tuple:
    return (
        window.transition.attempt_n.attempt_index_0idx,
        pass_count(window.transition.attempt_n),
        window.student_id,
    )


def build_representative_cards(
    windows: tuple[CohortWindow, ...], max_cards: int = 10
) -> list[dict]:
    if not windows:
        raise PayloadError("Cannot build representative cards for empty window slice")

    grouped: dict[tuple, list[CohortWindow]] = defaultdict(list)
    for window in windows:
        grouped[representative_group_key(window)].append(window)

    group_items = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    selected: list[CohortWindow] = []

    for _group_key, group_windows in group_items:
        ordered = sorted(group_windows, key=_window_rank_tuple)
        selected.append(ordered[len(ordered) // 2])
        if len(selected) == min(max_cards, len(windows)):
            break

    if len(selected) < min(max_cards, len(windows)):
        remaining = [
            window for window in sorted(windows, key=_window_rank_tuple) if window not in selected
        ]
        selected.extend(remaining[: min(max_cards, len(windows)) - len(selected)])

    alias_by_student: dict[str, str] = {}
    for index, student_id in enumerate(sorted({window.student_id for window in selected}), start=1):
        alias_by_student[student_id] = f"S{index:02d}"

    cards: list[dict] = []
    for window in sorted(
        selected, key=lambda w: (alias_by_student[w.student_id], _window_rank_tuple(w))
    ):
        prior_summaries = []
        for previous_attempt in window.transition.history[-2:]:
            current_attempt = window.transition.all_attempts[
                previous_attempt.attempt_index_0idx + 1
            ]
            prior_summaries.append(
                {
                    "attempt_index_0idx": previous_attempt.attempt_index_0idx,
                    "grade": previous_attempt.grade,
                    "pass_count": pass_count(previous_attempt),
                    "failed_test_indices_0idx": list(failed_test_indices_0idx(previous_attempt)),
                    "code_delta_summary": code_delta_summary(
                        previous_attempt.code, current_attempt.code
                    ),
                }
            )

        cards.append(
            {
                "student_alias": alias_by_student[window.student_id],
                "attempt_index_0idx": window.transition.attempt_n.attempt_index_0idx,
                "current_code": window.transition.attempt_n.code,
                "current_test_results": [
                    {
                        "test_index_0idx": result.test_index_0idx,
                        "passed": result.passed,
                        "input": result.input,
                        "expected": result.expected,
                        "actual": result.actual,
                    }
                    for result in window.transition.attempt_n.test_results
                ],
                "prior_attempt_summaries": prior_summaries,
                "trajectory_note": trajectory_note(window),
            }
        )
    return cards


def build_payload(windows: tuple[CohortWindow, ...], max_cards: int = 10) -> dict:
    return {
        "slice_header": build_slice_header(windows),
        "aggregate_summary": build_aggregate_summary(windows),
        "representative_cards": build_representative_cards(windows, max_cards=max_cards),
    }
