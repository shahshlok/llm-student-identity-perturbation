from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from identity_perturbation.codebench_support.models import ExecutionAttempt
from identity_perturbation.codebench_support.payload import failed_test_indices_0idx
from identity_perturbation.prediction_audit.match_policy import narrow_normalize_code_for_match


class PairMatchingError(ValueError):
    pass


@dataclass(frozen=True)
class MatchState:
    attempt_index_0idx: int
    pass_vector: tuple[bool, ...]
    failed_test_indices_0idx: tuple[int, ...]
    test_count: int
    normalized_code: str
    raw_code: str


def pass_fail_vector(attempt: ExecutionAttempt) -> tuple[bool, ...]:
    return tuple(result.passed for result in attempt.test_results)


def build_match_state(attempt: ExecutionAttempt) -> MatchState:
    vector = pass_fail_vector(attempt)
    if not vector:
        raise PairMatchingError(
            f"Attempt {attempt.attempt_index_0idx} has empty test_results; cannot build match state"
        )
    return MatchState(
        attempt_index_0idx=attempt.attempt_index_0idx,
        pass_vector=vector,
        failed_test_indices_0idx=failed_test_indices_0idx(attempt),
        test_count=len(vector),
        normalized_code=narrow_normalize_code_for_match(attempt.code),
        raw_code=attempt.code,
    )


def l2a_error_match(left: MatchState, right: MatchState) -> bool:
    return left.pass_vector == right.pass_vector


def _levenshtein_distance(text_a: str, text_b: str) -> int:
    if text_a == text_b:
        return 0
    if len(text_a) < len(text_b):
        text_a, text_b = text_b, text_a
    previous = list(range(len(text_b) + 1))
    for i, char_a in enumerate(text_a, start=1):
        current = [i]
        for j, char_b in enumerate(text_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def normalized_code_distance(left_code: str, right_code: str) -> float:
    left = narrow_normalize_code_for_match(left_code)
    right = narrow_normalize_code_for_match(right_code)
    total_length = len(left) + len(right)
    if total_length == 0:
        return 0.0
    return _levenshtein_distance(left, right) / total_length


def normalized_match_state_code_distance(left: MatchState, right: MatchState) -> float:
    return normalized_code_distance(left.normalized_code, right.normalized_code)


def l2b_code_match(left: MatchState, right: MatchState, *, threshold: float) -> bool:
    if not 0.0 <= threshold <= 1.0:
        raise PairMatchingError(f"L2B threshold must be between 0 and 1, got {threshold}")
    return normalized_match_state_code_distance(left, right) <= threshold


def pass_vector_signature(vector: tuple[bool, ...]) -> str:
    return "".join("P" if value else "F" for value in vector)


def match_state_to_dict(state: MatchState) -> dict[str, Any]:
    return {
        "attempt_index_0idx": state.attempt_index_0idx,
        "pass_vector": list(state.pass_vector),
        "pass_vector_signature": pass_vector_signature(state.pass_vector),
        "failed_test_indices_0idx": list(state.failed_test_indices_0idx),
        "test_count": state.test_count,
        "normalized_code_char_count": len(state.normalized_code),
    }
