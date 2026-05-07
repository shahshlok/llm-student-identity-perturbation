from __future__ import annotations

from ..helpers import (
    event_types,
    expected_score,
    normalized_edit_similarity,
    summarize_mean,
    top_hypothesis,
)
from ..types import EvaluationExample

NAME = "event_type_edit_similarity"


def evaluate_row(example: EvaluationExample) -> dict[str, object]:
    observed = event_types(example.observed_events)
    predicted_top1 = event_types(top_hypothesis(example).events)
    top1 = normalized_edit_similarity(predicted_top1, observed)
    expected = expected_score(
        example,
        lambda events: normalized_edit_similarity(event_types(events), observed),
    )
    return {
        "top1_similarity": top1,
        "expected_similarity": expected,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "mean_top1_similarity": summarize_mean(rows, "top1_similarity"),
        "mean_expected_similarity": summarize_mean(rows, "expected_similarity"),
    }
