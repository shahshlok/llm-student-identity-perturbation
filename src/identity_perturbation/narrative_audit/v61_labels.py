from __future__ import annotations

from pathlib import Path

from identity_perturbation.codebench_support.models import ExecutionTransition, ReplayTrace

from .payload import V6PayloadError, align_attempt_trace_cards
from .semantic_schema import ObservedNextEpisodeTape
from .trace_card import AttemptTraceCard
from .trajectory import build_semantic_event_tape, load_raw_submit_bounded_interval


class V61LabelsError(ValueError):
    pass


def build_observed_next_episode(
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
    idle_gap_seconds: float,
    include_keyhandled: bool,
    include_navigation: bool,
) -> dict[str, object]:
    visible_attempts = (*transition.history, transition.attempt_n, transition.attempt_n1)
    try:
        aligned_cards = align_attempt_trace_cards(
            visible_attempts,
            trace=trace,
            trace_cards=trace_cards,
        )
    except V6PayloadError as exc:
        raise V61LabelsError(str(exc)) from exc

    trace_card = aligned_cards.get(transition.attempt_n1.attempt_index_0idx)
    if trace_card is None:
        raise V61LabelsError(
            f"Missing aligned trace card for attempt_n1 {transition.attempt_n1.attempt_index_0idx}"
        )

    raw_interval = load_raw_submit_bounded_interval(
        codemirror_log_path=codemirror_log_path,
        submit_index_0idx=trace_card.submit_index,
    )
    semantic_events = build_semantic_event_tape(
        interval_events=raw_interval,
        idle_gap_seconds=idle_gap_seconds,
        include_keyhandled=include_keyhandled,
        include_navigation=include_navigation,
        output_line_limit=None,
    )

    payload = ObservedNextEpisodeTape.model_validate(
        {
            "schema_version": "v6_1_observed_next_episode_v1",
            "transition": {
                "class_id": class_id,
                "assessment_id": assessment_id,
                "exercise_id": exercise_id,
                "student_id": student_id,
                "transition_index_0idx": transition_index,
            },
            "attempt_n1_index_0idx": transition.attempt_n1.attempt_index_0idx,
            "semantic_tape_summary": {
                "raw_interval_event_count": len(raw_interval),
                "semantic_event_count": len(semantic_events),
                "change_event_count": sum(
                    1 for event in semantic_events if event["event_type"] == "change"
                ),
                "saida_testar_count": sum(
                    1 for event in semantic_events if event["event_type"] == "saida_testar"
                ),
                "submit_count": sum(
                    1 for event in semantic_events if event["event_type"] == "submit"
                ),
                "kill_program_count": sum(
                    1 for event in semantic_events if event["event_type"] == "kill_program"
                ),
                "idle_gap_count": sum(
                    1 for event in semantic_events if event["event_type"] == "idle_gap"
                ),
            },
            "semantic_event_tape": semantic_events,
        }
    )
    return payload.model_dump(by_alias=True)
