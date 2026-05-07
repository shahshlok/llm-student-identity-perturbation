from __future__ import annotations

from ..helpers import (
    event_types,
    exact_prefix_match,
    expected_score,
    shared_prefix_length,
    summarize_mean,
    summarize_rate,
    top_hypothesis,
)
from ..types import EvaluationExample

NAME = "event_type_prefix"


def evaluate_row(example: EvaluationExample) -> dict[str, object]:
    observed = event_types(example.observed_events)
    predicted_top1 = event_types(top_hypothesis(example).events)
    shared = shared_prefix_length(predicted_top1, observed)
    shared_norm = shared / max(len(observed), 1)
    expected_norm = expected_score(
        example,
        lambda events: shared_prefix_length(event_types(events), observed) / max(len(observed), 1),
    )
    return {
        "observed_prefix": observed[: example.prefix_k],
        "predicted_prefix": predicted_top1[: example.prefix_k],
        "top1_exact_match": exact_prefix_match(predicted_top1, observed, example.prefix_k),
        "top1_shared_prefix_length": shared,
        "top1_shared_prefix_norm": shared_norm,
        "expected_shared_prefix_norm": expected_norm,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "top1_exact_match_rate": summarize_rate(rows, "top1_exact_match"),
        "mean_top1_shared_prefix_norm": summarize_mean(rows, "top1_shared_prefix_norm"),
        "mean_expected_shared_prefix_norm": summarize_mean(rows, "expected_shared_prefix_norm"),
    }
