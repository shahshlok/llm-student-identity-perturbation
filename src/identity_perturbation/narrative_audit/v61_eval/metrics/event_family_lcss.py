from __future__ import annotations

from ..helpers import event_families, expected_score, lcss_ratio, summarize_mean, top_hypothesis
from ..types import EvaluationExample

NAME = "event_family_lcss"


def evaluate_row(example: EvaluationExample) -> dict[str, object]:
    observed = event_families(example.observed_events)
    predicted_top1 = event_families(top_hypothesis(example).events)
    top1 = lcss_ratio(predicted_top1, observed)
    expected = expected_score(example, lambda events: lcss_ratio(event_families(events), observed))
    return {
        "top1_lcss_ratio": top1,
        "expected_lcss_ratio": expected,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "mean_top1_lcss_ratio": summarize_mean(rows, "top1_lcss_ratio"),
        "mean_expected_lcss_ratio": summarize_mean(rows, "expected_lcss_ratio"),
    }
