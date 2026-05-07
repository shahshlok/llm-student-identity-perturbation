from __future__ import annotations

from pathlib import Path

from identity_perturbation.codebench_support.executions import next_test_outcome
from identity_perturbation.codebench_support.models import ExecutionAttempt, ExecutionTransition, ReplayTrace
from identity_perturbation.codebench_support.payload import code_delta_summary, failed_test_indices_0idx, pass_count

from .payload import V6PayloadError, align_attempt_trace_cards
from .trace_card import AttemptTraceCard, trace_card_to_prompt_dict
from .trajectory import (
    build_attempt_semantic_tape_payload,
    build_semantic_event_tape,
    load_raw_submit_bounded_interval,
)


class V61PayloadError(ValueError):
    pass


def _test_result_to_prompt_dict(result: object) -> dict[str, object]:
    return {
        "test_index_0idx": result.test_index_0idx,
        "passed": result.passed,
        "input": result.input,
        "expected": result.expected,
        "actual": result.actual,
    }


def _attempt_summary(attempt: ExecutionAttempt) -> dict[str, object]:
    return {
        "attempt_index_0idx": attempt.attempt_index_0idx,
        "timestamp": attempt.timestamp,
        "grade": attempt.grade,
        "pass_count": pass_count(attempt),
        "failed_test_indices_0idx": list(failed_test_indices_0idx(attempt)),
        "code": attempt.code,
        "test_results": [_test_result_to_prompt_dict(result) for result in attempt.test_results],
    }


def _aligned_submit_index_for_attempt(
    *,
    attempt_index_0idx: int,
    aligned_cards: dict[int, AttemptTraceCard],
) -> int:
    trace_card = aligned_cards.get(attempt_index_0idx)
    if trace_card is None:
        raise V61PayloadError(f"Missing aligned trace card for attempt {attempt_index_0idx}")
    return trace_card.submit_index


def _build_attempt_n_semantic_tape(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index: int,
    transition: ExecutionTransition,
    aligned_cards: dict[int, AttemptTraceCard],
    codemirror_log_path: Path,
    idle_gap_seconds: float,
    include_keyhandled: bool,
    include_navigation: bool,
) -> dict[str, object]:
    attempt_n_index = transition.attempt_n.attempt_index_0idx
    submit_index = _aligned_submit_index_for_attempt(
        attempt_index_0idx=attempt_n_index,
        aligned_cards=aligned_cards,
    )
    raw_interval = load_raw_submit_bounded_interval(
        codemirror_log_path=codemirror_log_path,
        submit_index_0idx=submit_index,
    )
    semantic_tape = build_semantic_event_tape(
        interval_events=raw_interval,
        idle_gap_seconds=idle_gap_seconds,
        include_keyhandled=include_keyhandled,
        include_navigation=include_navigation,
        output_line_limit=None,
    )
    payload = build_attempt_semantic_tape_payload(
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        student_id=student_id,
        transition_index_0idx=transition_index,
        current_trace_card=trace_card_to_prompt_dict(aligned_cards[attempt_n_index]),
        raw_interval=raw_interval,
        semantic_tape=semantic_tape,
    )
    return payload.model_dump(by_alias=True)


def _prior_history_entries(
    *,
    transition: ExecutionTransition,
    aligned_cards: dict[int, AttemptTraceCard],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for attempt in transition.history:
        next_attempt = transition.all_attempts[attempt.attempt_index_0idx + 1]
        trace_card = aligned_cards.get(attempt.attempt_index_0idx)
        if trace_card is None:
            raise V61PayloadError(
                f"Missing aligned trace card for prior attempt {attempt.attempt_index_0idx}"
            )
        entries.append(
            {
                "attempt_index_0idx": attempt.attempt_index_0idx,
                "timestamp": attempt.timestamp,
                "grade": attempt.grade,
                "pass_count": pass_count(attempt),
                "failed_test_indices_0idx": list(failed_test_indices_0idx(attempt)),
                "next_test_outcome": next_test_outcome(next_attempt),
                "code_delta_summary": code_delta_summary(attempt.code, next_attempt.code),
                "trace_card": trace_card_to_prompt_dict(trace_card),
            }
        )
    return entries


def build_v61_payload(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index: int,
    transition: ExecutionTransition,
    trace: ReplayTrace,
    trace_cards: tuple[AttemptTraceCard, ...],
    codemirror_log_path: Path,
    idle_gap_seconds: float = 30.0,
    include_keyhandled: bool = False,
    include_navigation: bool = True,
) -> dict[str, object]:
    visible_attempts = (*transition.history, transition.attempt_n)
    try:
        aligned_cards = align_attempt_trace_cards(
            visible_attempts,
            trace=trace,
            trace_cards=trace_cards,
        )
    except V6PayloadError as exc:
        raise V61PayloadError(str(exc)) from exc

    payload = {
        "student_header": {
            "student_id": student_id,
            "class_id": class_id,
            "assessment_id": assessment_id,
            "exercise_id": exercise_id,
            "target_attempt_index_0idx": transition.attempt_n.attempt_index_0idx,
            "n_prior_attempts": len(transition.history),
            "n_total_tests": len(transition.attempt_n.test_results),
            "attempt_n_pass_count": pass_count(transition.attempt_n),
            "attempt_n_failed_test_indices_0idx": list(
                failed_test_indices_0idx(transition.attempt_n)
            ),
        },
        "attempt_n": {
            **_attempt_summary(transition.attempt_n),
            "semantic_tape": _build_attempt_n_semantic_tape(
                class_id=class_id,
                assessment_id=assessment_id,
                exercise_id=exercise_id,
                student_id=student_id,
                transition_index=transition_index,
                transition=transition,
                aligned_cards=aligned_cards,
                codemirror_log_path=codemirror_log_path,
                idle_gap_seconds=idle_gap_seconds,
                include_keyhandled=include_keyhandled,
                include_navigation=include_navigation,
            ),
        },
        "prior_history": _prior_history_entries(
            transition=transition,
            aligned_cards=aligned_cards,
        ),
    }
    return payload
