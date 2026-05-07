from __future__ import annotations

from identity_perturbation.codebench_support.codemirror import normalize_code_for_match
from identity_perturbation.codebench_support.executions import next_test_outcome
from identity_perturbation.codebench_support.models import ExecutionAttempt, ExecutionTransition, ReplayTrace, SubmitSnapshot
from identity_perturbation.codebench_support.payload import code_delta_summary, failed_test_indices_0idx, pass_count

from .models import AttemptTraceCard
from .trace_card import trace_card_to_prompt_dict


class V6PayloadError(ValueError):
    pass


def _candidate_snapshot_indices(
    code: str,
    snapshots: tuple[SubmitSnapshot, ...],
) -> tuple[int, ...]:
    normalized = normalize_code_for_match(code)
    return tuple(
        index
        for index, snapshot in enumerate(snapshots)
        if normalize_code_for_match(snapshot.code) == normalized
    )


def _attempt_alignment_candidates(
    attempts: tuple[ExecutionAttempt, ...],
    snapshots: tuple[SubmitSnapshot, ...],
) -> tuple[tuple[int, ...], ...]:
    candidates: list[tuple[int, ...]] = []
    for attempt in attempts:
        attempt_candidates = _candidate_snapshot_indices(attempt.code, snapshots)
        if not attempt_candidates:
            raise V6PayloadError(
                f"Execution attempt {attempt.attempt_index_0idx} code not found in any replay snapshot"
            )
        candidates.append(attempt_candidates)
    return tuple(candidates)


def _resolve_monotonic_alignment(
    attempts: tuple[ExecutionAttempt, ...],
    snapshots: tuple[SubmitSnapshot, ...],
) -> tuple[int, ...]:
    candidates_by_attempt = _attempt_alignment_candidates(attempts, snapshots)
    solutions: list[tuple[int, ...]] = []

    def backtrack(attempt_pos: int, chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if attempt_pos == len(attempts):
            solutions.append(tuple(chosen))
            return

        min_index = chosen[-1] + 1 if chosen else 0
        for snapshot_index in candidates_by_attempt[attempt_pos]:
            if snapshot_index < min_index:
                continue
            chosen.append(snapshot_index)
            backtrack(attempt_pos + 1, chosen)
            chosen.pop()

    backtrack(0, [])

    if not solutions:
        raise V6PayloadError("No monotonic execution-attempt to replay-snapshot alignment exists")
    if len(solutions) != 1:
        raise V6PayloadError("Ambiguous execution-attempt to replay-snapshot alignment")
    return solutions[0]


def align_attempt_trace_cards(
    attempts: tuple[ExecutionAttempt, ...],
    trace: ReplayTrace,
    trace_cards: tuple[AttemptTraceCard, ...],
) -> dict[int, AttemptTraceCard]:
    if len(trace.snapshots) != len(trace_cards):
        raise V6PayloadError(
            "Replay snapshots and trace cards must have identical lengths for strict alignment"
        )

    aligned_snapshot_indices = _resolve_monotonic_alignment(attempts, trace.snapshots)
    aligned_cards: dict[int, AttemptTraceCard] = {}
    for attempt, snapshot_index in zip(attempts, aligned_snapshot_indices, strict=False):
        card = trace_cards[snapshot_index]
        if card.submit_index != snapshot_index:
            raise V6PayloadError(
                f"Trace card submit_index mismatch at snapshot_index={snapshot_index}"
            )
        aligned_cards[attempt.attempt_index_0idx] = card
    return aligned_cards


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


def _prior_history_entries(
    transition: ExecutionTransition,
    aligned_cards: dict[int, AttemptTraceCard],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for attempt in transition.history:
        next_attempt = transition.all_attempts[attempt.attempt_index_0idx + 1]
        trace_card = aligned_cards.get(attempt.attempt_index_0idx)
        if trace_card is None:
            raise V6PayloadError(
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


def _trace_cards_pre_n(
    transition: ExecutionTransition,
    aligned_cards: dict[int, AttemptTraceCard],
) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    for attempt_index_0idx in range(transition.attempt_n.attempt_index_0idx):
        trace_card = aligned_cards.get(attempt_index_0idx)
        if trace_card is None:
            raise V6PayloadError(
                f"Missing aligned trace card for visible attempt {attempt_index_0idx}"
            )
        card_payload = trace_card_to_prompt_dict(trace_card)
        card_payload["attempt_index_0idx"] = attempt_index_0idx
        cards.append(card_payload)
    return cards


def build_payload(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition: ExecutionTransition,
    trace: ReplayTrace,
    trace_cards: tuple[AttemptTraceCard, ...],
) -> dict[str, object]:
    visible_attempts = (*transition.history, transition.attempt_n)
    aligned_cards = align_attempt_trace_cards(
        visible_attempts,
        trace=trace,
        trace_cards=trace_cards,
    )
    attempt_n_card = aligned_cards.get(transition.attempt_n.attempt_index_0idx)
    if attempt_n_card is None:
        raise V6PayloadError(
            f"Missing aligned trace card for attempt_n index {transition.attempt_n.attempt_index_0idx}"
        )

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
            "trace_card": trace_card_to_prompt_dict(attempt_n_card),
        },
        "prior_history": _prior_history_entries(
            transition=transition,
            aligned_cards=aligned_cards,
        ),
        "trace_cards_pre_n": _trace_cards_pre_n(
            transition=transition,
            aligned_cards=aligned_cards,
        ),
    }
    return payload
