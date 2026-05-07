from __future__ import annotations

from pathlib import Path

import pytest

from identity_perturbation.codebench_support.models import ExecutionAttempt
from identity_perturbation.codebench_support.models import TestResult as V5TestResult
from identity_perturbation.prediction_audit.audit_match_l2_pairs import audit_l2_pairs
from identity_perturbation.prediction_audit.pair_matching import (
    MatchState,
    PairMatchingError,
    build_match_state,
    l2a_error_match,
    l2b_code_match,
    normalized_code_distance,
    pass_fail_vector,
    pass_vector_signature,
)
from identity_perturbation.prediction_audit.select_branching_probe import ValidatedStudentCandidate


def _attempt(*, index: int, code: str, passed: tuple[bool, ...]) -> ExecutionAttempt:
    return ExecutionAttempt(
        timestamp=f"2024-01-01 10:00:0{index}",
        attempt_index_0idx=index,
        code=code,
        grade=100.0 if all(passed) else 0.0,
        test_results=tuple(
            V5TestResult(
                test_index_0idx=test_index,
                passed=value,
                input=str(test_index),
                expected="ok" if value else "bad",
                actual="ok" if value else "wrong",
            )
            for test_index, value in enumerate(passed)
        ),
    )


def _student(custom_suffix: str) -> ValidatedStudentCandidate:
    return ValidatedStudentCandidate(
        class_id="101",
        assessment_id="201",
        exercise_id="301",
        student_id=f"9{custom_suffix}",
        execution_path=Path("/tmp/exec.log"),
        codemirror_path=Path("/tmp/cm.log"),
        total_attempt_count=3,
        interior_n=1,
        attempt_n_code="x=1\nprint(x)\n",
        attempt_n1_code="x=2\nprint(x)\n",
    )


def test_build_match_state_uses_row_local_attempt() -> None:
    attempt0 = _attempt(index=0, code="print(0)\n", passed=(False, False))
    attempt1 = _attempt(index=1, code="print(1)\n", passed=(False, True))

    state0 = build_match_state(attempt0)
    state1 = build_match_state(attempt1)

    assert pass_fail_vector(attempt1) == (False, True)
    assert state0.pass_vector == (False, False)
    assert state1.pass_vector == (False, True)
    assert state1.attempt_index_0idx == 1


def test_build_match_state_rejects_empty_test_results() -> None:
    attempt = ExecutionAttempt(
        timestamp="2024-01-01 10:00:00",
        attempt_index_0idx=1,
        code="print(1)\n",
        grade=0.0,
        test_results=(),
    )

    with pytest.raises(PairMatchingError, match="empty test_results"):
        build_match_state(attempt)


def test_l2a_and_l2b_helpers() -> None:
    left = build_match_state(_attempt(index=1, code="x=1\nprint(x)\n", passed=(False, True)))
    right = build_match_state(_attempt(index=1, code="x = 1\nprint(x)\n", passed=(False, True)))
    other = build_match_state(_attempt(index=1, code="y=99\n", passed=(True, False)))

    assert l2a_error_match(left, right) is True
    assert l2a_error_match(left, other) is False
    assert normalized_code_distance("x=1\n", "x=1\n") == 0.0
    assert l2b_code_match(left, right, threshold=0.10) is True
    assert pass_vector_signature(left.pass_vector) == "FP"


def test_audit_l2_pairs_counts_sender_rows_and_thresholds() -> None:
    from identity_perturbation.prediction_audit.audit_match_l2_pairs import BuildableStudentRecord

    students = [
        BuildableStudentRecord(
            scope_id="101:201:301",
            tertile="T1",
            student=_student("001"),
            match_state=MatchState(
                attempt_index_0idx=1,
                pass_vector=(False, False),
                failed_test_indices_0idx=(0, 1),
                test_count=2,
                normalized_code="x=1\nprint(x)",
                raw_code="x=1\nprint(x)\n",
            ),
        ),
        BuildableStudentRecord(
            scope_id="101:201:301",
            tertile="T1",
            student=_student("002"),
            match_state=MatchState(
                attempt_index_0idx=1,
                pass_vector=(False, False),
                failed_test_indices_0idx=(0, 1),
                test_count=2,
                normalized_code="x=2\nprint(x)",
                raw_code="x=2\nprint(x)\n",
            ),
        ),
        BuildableStudentRecord(
            scope_id="101:201:301",
            tertile="T1",
            student=_student("003"),
            match_state=MatchState(
                attempt_index_0idx=1,
                pass_vector=(True, False),
                failed_test_indices_0idx=(1,),
                test_count=2,
                normalized_code="y=99",
                raw_code="y=99\n",
            ),
        ),
    ]

    report = audit_l2_pairs({"101:201:301": students}, thresholds=(0.20, 0.60))

    assert report["scope_count"] == 1
    assert report["scope_count_with_at_least_two_buildable_rows"] == 1
    assert report["sender_row_count_with_any_peer"] == 3
    assert report["l2a_undirected_pair_count"] == 1
    assert report["l2a_directed_pair_count"] == 2
    assert report["sender_row_count_l2a"] == 2
    assert report["threshold_sweep"]["0.60"]["undirected_pair_count_l2a_and_l2b"] == 1
    assert report["threshold_sweep"]["0.60"]["sender_row_count_l2a_and_l2b"] == 2
