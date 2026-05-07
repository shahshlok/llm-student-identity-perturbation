from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Final

from identity_perturbation.codebench_support.codemirror import infer_initial_code, parse_codemirror_log

from .trace_card import V6TraceCardError, build_attempt_trace_cards, trace_card_to_prompt_dict
from .trace_shuffle import (
    V6TraceShuffleError,
    load_cross_class_random_trace_cards,
    load_within_class_random_trace_cards,
    load_within_exercise_shuffled_trace_cards,
)
from .trajectory import (
    build_attempt_semantic_tape_payload,
    build_semantic_event_tape,
    load_raw_submit_bounded_interval,
)

V61Condition = str
FULL_V61_CONDITION: Final[V61Condition] = "full_v61"
NO_TRACE_CONDITION: Final[V61Condition] = "no_trace"
TRACE_SHUFFLED_WITHIN_EXERCISE_CONDITION: Final[V61Condition] = "trace_shuffled_within_exercise"
TRACE_SHUFFLED_WITHIN_CLASS_RANDOM_CONDITION: Final[V61Condition] = (
    "trace_shuffled_within_class_random"
)
TRACE_SHUFFLED_CROSS_CLASS_RANDOM_CONDITION: Final[V61Condition] = (
    "trace_shuffled_cross_class_random"
)
RANDOMIZED_TRACE_SHUFFLE_CONDITIONS: Final[tuple[V61Condition, ...]] = (
    TRACE_SHUFFLED_WITHIN_CLASS_RANDOM_CONDITION,
    TRACE_SHUFFLED_CROSS_CLASS_RANDOM_CONDITION,
)
SUPPORTED_CONDITIONS: Final[tuple[V61Condition, ...]] = (
    FULL_V61_CONDITION,
    NO_TRACE_CONDITION,
    TRACE_SHUFFLED_WITHIN_EXERCISE_CONDITION,
    TRACE_SHUFFLED_WITHIN_CLASS_RANDOM_CONDITION,
    TRACE_SHUFFLED_CROSS_CLASS_RANDOM_CONDITION,
)


class V61ConditionError(ValueError):
    pass


def validate_condition(condition: str) -> V61Condition:
    if condition not in SUPPORTED_CONDITIONS:
        raise V61ConditionError(
            f"Unsupported v6.1 condition {condition!r}; expected one of {list(SUPPORTED_CONDITIONS)}"
        )
    return condition


def condition_output_dirname(condition: str, shuffle_seed: int | None) -> str | None:
    validated = validate_condition(condition)
    if shuffle_seed is not None and validated not in RANDOMIZED_TRACE_SHUFFLE_CONDITIONS:
        raise V61ConditionError(
            "shuffle_seed is only supported for randomized trace shuffle conditions"
        )
    if validated == FULL_V61_CONDITION:
        return None
    if validated in RANDOMIZED_TRACE_SHUFFLE_CONDITIONS:
        if shuffle_seed is None:
            raise V61ConditionError(
                f"Condition {validated!r} requires a non-null shuffle_seed for reproducibility"
            )
        return f"{validated}_seed{shuffle_seed}"
    return validated


def _visible_attempt_count(payload: dict[str, object]) -> int:
    student_header = payload.get("student_header")
    if not isinstance(student_header, dict):
        raise V61ConditionError("Payload student_header must be an object")
    return int(student_header["n_prior_attempts"]) + 1


def _withheld_semantic_tape(reason: str) -> dict[str, object]:
    return {
        "schema_version": "v6_1_attempt_semantic_tape_withheld_v1",
        "status": "withheld",
        "reason": reason,
    }


def _apply_no_trace(payload: dict[str, object]) -> dict[str, object]:
    conditioned = deepcopy(payload)
    conditioned["condition"] = NO_TRACE_CONDITION
    attempt_n = conditioned.get("attempt_n")
    if not isinstance(attempt_n, dict):
        raise V61ConditionError("Payload attempt_n must be an object")
    attempt_n["semantic_tape"] = _withheld_semantic_tape(
        "No CodeMirror log evidence is provided in this condition."
    )
    prior_history = conditioned.get("prior_history")
    if not isinstance(prior_history, list):
        raise V61ConditionError("Payload prior_history must be a list")
    for entry in prior_history:
        if not isinstance(entry, dict):
            raise V61ConditionError("Payload prior_history entries must be objects")
        entry.pop("trace_card", None)
    return conditioned


def _donor_attempt_n_semantic_tape(
    *,
    data_root: Path,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index_0idx: int,
    submit_index_0idx: int,
    idle_gap_seconds: float,
    include_keyhandled: bool,
    include_navigation: bool,
) -> dict[str, object]:
    cm_path = (
        data_root
        / class_id
        / "users"
        / student_id
        / "codemirror"
        / f"{assessment_id}_{exercise_id}.log"
    )
    if not cm_path.exists():
        raise V61ConditionError(f"Donor CodeMirror log not found: {cm_path}")
    try:
        cm_events = parse_codemirror_log(cm_path)
        initial_code = infer_initial_code(cm_events, "")
        donor_cards = build_attempt_trace_cards(cm_events, initial_code=initial_code)
        if submit_index_0idx < 0 or submit_index_0idx >= len(donor_cards):
            raise V61ConditionError(
                f"Donor submit index {submit_index_0idx} is out of range for {cm_path}"
            )
        raw_interval = load_raw_submit_bounded_interval(
            codemirror_log_path=cm_path,
            submit_index_0idx=submit_index_0idx,
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
            transition_index_0idx=transition_index_0idx,
            current_trace_card=trace_card_to_prompt_dict(donor_cards[submit_index_0idx]),
            raw_interval=raw_interval,
            semantic_tape=semantic_tape,
        )
    except (V6TraceCardError, V6TraceShuffleError) as exc:
        raise V61ConditionError(str(exc)) from exc
    return payload.model_dump(by_alias=True)


def apply_condition_to_payload(
    payload: dict[str, object],
    *,
    condition: V61Condition,
    data_root: Path,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index_0idx: int,
    idle_gap_seconds: float,
    include_keyhandled: bool,
    include_navigation: bool,
    shuffle_seed: int | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    validated = validate_condition(condition)
    if validated == FULL_V61_CONDITION:
        conditioned = deepcopy(payload)
        conditioned["condition"] = validated
        return conditioned, None

    if validated == NO_TRACE_CONDITION:
        return _apply_no_trace(payload), None

    if validated not in (
        TRACE_SHUFFLED_WITHIN_EXERCISE_CONDITION,
        TRACE_SHUFFLED_WITHIN_CLASS_RANDOM_CONDITION,
        TRACE_SHUFFLED_CROSS_CLASS_RANDOM_CONDITION,
    ):
        raise AssertionError(f"Unhandled v6.1 condition: {validated}")

    visible_attempt_count = _visible_attempt_count(payload)
    submit_index_0idx = visible_attempt_count - 1
    try:
        if validated == TRACE_SHUFFLED_WITHIN_EXERCISE_CONDITION:
            donor_cards, metadata = load_within_exercise_shuffled_trace_cards(
                data_root=data_root,
                class_id=class_id,
                assessment_id=assessment_id,
                exercise_id=exercise_id,
                target_student_id=student_id,
                visible_attempt_count=visible_attempt_count,
            )
        elif validated == TRACE_SHUFFLED_WITHIN_CLASS_RANDOM_CONDITION:
            if shuffle_seed is None:
                raise V61ConditionError("trace_shuffled_within_class_random requires shuffle_seed")
            donor_cards, metadata = load_within_class_random_trace_cards(
                data_root=data_root,
                class_id=class_id,
                assessment_id=assessment_id,
                exercise_id=exercise_id,
                target_student_id=student_id,
                visible_attempt_count=visible_attempt_count,
                rng_seed=shuffle_seed,
            )
        else:
            if shuffle_seed is None:
                raise V61ConditionError("trace_shuffled_cross_class_random requires shuffle_seed")
            donor_cards, metadata = load_cross_class_random_trace_cards(
                data_root=data_root,
                class_id=class_id,
                assessment_id=assessment_id,
                exercise_id=exercise_id,
                target_student_id=student_id,
                visible_attempt_count=visible_attempt_count,
                rng_seed=shuffle_seed,
            )
    except V6TraceShuffleError as exc:
        raise V61ConditionError(str(exc)) from exc

    conditioned = deepcopy(payload)
    conditioned["condition"] = validated
    conditioned["condition_metadata"] = deepcopy(metadata)

    attempt_n = conditioned.get("attempt_n")
    if not isinstance(attempt_n, dict):
        raise V61ConditionError("Payload attempt_n must be an object")
    donor_student_id = str(metadata["trace_donor_student_id"])
    donor_class_id = str(metadata["trace_donor_class_id"])
    donor_assessment_id = str(metadata["trace_donor_assessment_id"])
    attempt_n["semantic_tape"] = _donor_attempt_n_semantic_tape(
        data_root=data_root,
        class_id=donor_class_id,
        assessment_id=donor_assessment_id,
        exercise_id=exercise_id,
        student_id=donor_student_id,
        transition_index_0idx=transition_index_0idx,
        submit_index_0idx=submit_index_0idx,
        idle_gap_seconds=idle_gap_seconds,
        include_keyhandled=include_keyhandled,
        include_navigation=include_navigation,
    )

    prior_history = conditioned.get("prior_history")
    if not isinstance(prior_history, list):
        raise V61ConditionError("Payload prior_history must be a list")
    if len(prior_history) != visible_attempt_count - 1:
        raise V61ConditionError(
            f"Expected {visible_attempt_count - 1} prior history entries, got {len(prior_history)}"
        )
    for attempt_index, entry in enumerate(prior_history):
        if not isinstance(entry, dict):
            raise V61ConditionError("Payload prior_history entries must be objects")
        entry["trace_card"] = trace_card_to_prompt_dict(donor_cards[attempt_index])

    return conditioned, metadata
