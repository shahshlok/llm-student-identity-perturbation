from __future__ import annotations

from copy import deepcopy
from math import ceil
from typing import Final

V6Condition = str
FULL_V6_CONDITION: Final[V6Condition] = "full_v6"
NO_TRACE_CONDITION: Final[V6Condition] = "no_trace"
TRACE_SHUFFLED_WITHIN_EXERCISE_CONDITION: Final[V6Condition] = "trace_shuffled_within_exercise"
TRACE_DOSAGE_25_CONDITION: Final[V6Condition] = "trace_dosage_25"
TRACE_DOSAGE_50_CONDITION: Final[V6Condition] = "trace_dosage_50"
TRACE_DOSAGE_75_CONDITION: Final[V6Condition] = "trace_dosage_75"
TRACE_DOSAGE_FRACTIONS: Final[dict[V6Condition, float]] = {
    TRACE_DOSAGE_25_CONDITION: 0.25,
    TRACE_DOSAGE_50_CONDITION: 0.50,
    TRACE_DOSAGE_75_CONDITION: 0.75,
}
SUPPORTED_CONDITIONS: Final[tuple[V6Condition, ...]] = (
    FULL_V6_CONDITION,
    NO_TRACE_CONDITION,
    TRACE_SHUFFLED_WITHIN_EXERCISE_CONDITION,
    TRACE_DOSAGE_25_CONDITION,
    TRACE_DOSAGE_50_CONDITION,
    TRACE_DOSAGE_75_CONDITION,
)


class V6ConditionError(ValueError):
    pass


def validate_condition(condition: str) -> V6Condition:
    if condition not in SUPPORTED_CONDITIONS:
        raise V6ConditionError(
            f"Unsupported v6 condition {condition!r}; expected one of {list(SUPPORTED_CONDITIONS)}"
        )
    return condition


def _visible_attempt_count(conditioned: dict[str, object]) -> int:
    student_header = conditioned["student_header"]
    if not isinstance(student_header, dict):
        raise V6ConditionError("Payload student_header must be an object")
    return int(student_header["n_prior_attempts"]) + 1


def _apply_trace_visibility_window(
    conditioned: dict[str, object],
    *,
    visible_attempt_indices: set[int],
) -> dict[str, object]:
    visible_attempt_count = _visible_attempt_count(conditioned)
    expected_indices = set(range(visible_attempt_count))
    if not visible_attempt_indices.issubset(expected_indices):
        raise V6ConditionError(
            f"Visible attempt indices {sorted(visible_attempt_indices)} are outside "
            f"expected range 0..{visible_attempt_count - 1}"
        )

    attempt_n_index = visible_attempt_count - 1
    attempt_n = conditioned["attempt_n"]
    if not isinstance(attempt_n, dict):
        raise V6ConditionError("Payload attempt_n must be an object")
    if attempt_n_index in visible_attempt_indices:
        if "trace_card" not in attempt_n:
            raise V6ConditionError("Payload attempt_n is missing trace_card")
    else:
        attempt_n.pop("trace_card", None)

    prior_history = conditioned["prior_history"]
    if not isinstance(prior_history, list):
        raise V6ConditionError("Payload prior_history must be a list")
    if len(prior_history) != attempt_n_index:
        raise V6ConditionError(
            f"Expected {attempt_n_index} prior history entries, got {len(prior_history)}"
        )
    for attempt_index, entry in enumerate(prior_history):
        if not isinstance(entry, dict):
            raise V6ConditionError("Payload prior_history entries must be objects")
        if attempt_index in visible_attempt_indices:
            if "trace_card" not in entry:
                raise V6ConditionError(f"Prior history entry {attempt_index} is missing trace_card")
        else:
            entry.pop("trace_card", None)

    trace_cards_pre_n = conditioned["trace_cards_pre_n"]
    if not isinstance(trace_cards_pre_n, list):
        raise V6ConditionError("Payload trace_cards_pre_n must be a list")

    filtered_trace_cards_pre_n: list[dict[str, object]] = []
    for trace_card in trace_cards_pre_n:
        if not isinstance(trace_card, dict):
            raise V6ConditionError("Payload trace_cards_pre_n entries must be objects")
        attempt_index = int(trace_card["attempt_index_0idx"])
        if attempt_index in visible_attempt_indices:
            filtered_trace_cards_pre_n.append(trace_card)
    conditioned["trace_cards_pre_n"] = filtered_trace_cards_pre_n
    return conditioned


def _apply_trace_dosage_condition(
    conditioned: dict[str, object],
    *,
    condition: V6Condition,
) -> dict[str, object]:
    fraction = TRACE_DOSAGE_FRACTIONS[condition]
    visible_attempt_count = _visible_attempt_count(conditioned)
    keep_count = max(1, ceil(visible_attempt_count * fraction))
    start_index = visible_attempt_count - keep_count
    visible_attempt_indices = set(range(start_index, visible_attempt_count))
    conditioned = _apply_trace_visibility_window(
        conditioned,
        visible_attempt_indices=visible_attempt_indices,
    )
    metadata = conditioned.get("condition_metadata")
    if metadata is None:
        metadata = {}
        conditioned["condition_metadata"] = metadata
    if not isinstance(metadata, dict):
        raise V6ConditionError("condition_metadata must be an object when present")
    metadata["trace_dosage_fraction"] = fraction
    metadata["trace_dosage_keep_count"] = keep_count
    metadata["trace_dosage_visible_attempt_indices"] = sorted(visible_attempt_indices)
    return conditioned


def apply_condition_to_payload(
    payload: dict[str, object],
    *,
    condition: V6Condition,
    shuffled_trace_cards: list[dict[str, object]] | None = None,
    condition_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    validated = validate_condition(condition)
    conditioned = deepcopy(payload)
    conditioned["condition"] = validated
    if condition_metadata is not None:
        conditioned["condition_metadata"] = deepcopy(condition_metadata)

    if validated == FULL_V6_CONDITION:
        return conditioned

    if validated == NO_TRACE_CONDITION:
        return _apply_trace_visibility_window(conditioned, visible_attempt_indices=set())

    if validated == TRACE_SHUFFLED_WITHIN_EXERCISE_CONDITION:
        if shuffled_trace_cards is None:
            raise V6ConditionError("trace_shuffled_within_exercise requires shuffled_trace_cards")
        student_header = conditioned["student_header"]
        if not isinstance(student_header, dict):
            raise V6ConditionError("Payload student_header must be an object")
        visible_attempt_count = int(student_header["n_prior_attempts"]) + 1
        if len(shuffled_trace_cards) != visible_attempt_count:
            raise V6ConditionError(
                f"Expected {visible_attempt_count} shuffled trace cards, got {len(shuffled_trace_cards)}"
            )

        attempt_n = conditioned["attempt_n"]
        if not isinstance(attempt_n, dict):
            raise V6ConditionError("Payload attempt_n must be an object")
        attempt_n["trace_card"] = deepcopy(shuffled_trace_cards[-1])

        prior_history = conditioned["prior_history"]
        if not isinstance(prior_history, list):
            raise V6ConditionError("Payload prior_history must be a list")
        if len(prior_history) != visible_attempt_count - 1:
            raise V6ConditionError(
                f"Expected {visible_attempt_count - 1} prior history entries, got {len(prior_history)}"
            )
        for attempt_index, entry in enumerate(prior_history):
            if not isinstance(entry, dict):
                raise V6ConditionError("Payload prior_history entries must be objects")
            entry["trace_card"] = deepcopy(shuffled_trace_cards[attempt_index])

        shuffled_pre_n = []
        for attempt_index, trace_card in enumerate(shuffled_trace_cards[:-1]):
            card_payload = deepcopy(trace_card)
            card_payload["attempt_index_0idx"] = attempt_index
            shuffled_pre_n.append(card_payload)
        conditioned["trace_cards_pre_n"] = shuffled_pre_n
        return conditioned

    if validated in TRACE_DOSAGE_FRACTIONS:
        return _apply_trace_dosage_condition(conditioned, condition=validated)

    raise AssertionError(f"Unhandled v6 condition: {validated}")
