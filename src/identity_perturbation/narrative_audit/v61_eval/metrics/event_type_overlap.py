from __future__ import annotations

from ..helpers import event_types, expected_score, multiset_jaccard, summarize_mean, top_hypothesis
from ..types import EvaluationExample

NAME = "event_type_overlap"


def evaluate_row(example: EvaluationExample) -> dict[str, object]:
    observed = event_types(example.observed_events)
    predicted_top1 = event_types(top_hypothesis(example).events)
    top1 = multiset_jaccard(predicted_top1, observed)
    expected = expected_score(
        example, lambda events: multiset_jaccard(event_types(events), observed)
    )
    return {
        "top1_jaccard": top1,
        "expected_jaccard": expected,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "mean_top1_jaccard": summarize_mean(rows, "top1_jaccard"),
        "mean_expected_jaccard": summarize_mean(rows, "expected_jaccard"),
    }
