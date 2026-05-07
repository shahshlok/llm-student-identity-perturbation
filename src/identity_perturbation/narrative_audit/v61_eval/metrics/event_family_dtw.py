from __future__ import annotations

from ..helpers import dtw_similarity, event_families, expected_score, summarize_mean, top_hypothesis
from ..types import EvaluationExample

NAME = "event_family_dtw"


def evaluate_row(example: EvaluationExample) -> dict[str, object]:
    observed = event_families(example.observed_events)
    predicted_top1 = event_families(top_hypothesis(example).events)
    top1 = dtw_similarity(predicted_top1, observed)
    expected = expected_score(
        example, lambda events: dtw_similarity(event_families(events), observed)
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
