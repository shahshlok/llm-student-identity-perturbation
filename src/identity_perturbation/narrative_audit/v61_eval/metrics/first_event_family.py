from __future__ import annotations

from ..helpers import (
    categorical_distribution,
    event_family,
    first_event_type,
    label_rank,
    multiclass_brier_score,
    reciprocal_rank,
    safe_log_loss,
    summarize_mean,
    summarize_rate,
    top_label,
    topk_hit,
)
from ..types import EvaluationExample

NAME = "first_event_family"


def _label(events):
    return event_family(first_event_type(events))


def evaluate_row(example: EvaluationExample) -> dict[str, object]:
    truth = _label(example.observed_events)
    distribution = categorical_distribution(example, _label)
    rank = label_rank(distribution, truth)
    truth_mass = distribution.get(truth, 0.0)
    predicted = top_label(distribution)
    return {
        "observed": truth,
        "predicted_top1": predicted,
        "top1_match": predicted == truth,
        "truth_probability_mass": truth_mass,
        "top2_hit": topk_hit(rank, 2),
        "top3_hit": topk_hit(rank, 3),
        "mrr": reciprocal_rank(rank),
        "brier_score": multiclass_brier_score(distribution, truth),
        "log_loss": safe_log_loss(truth_mass),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "top1_accuracy": summarize_rate(rows, "top1_match"),
        "mean_truth_probability_mass": summarize_mean(rows, "truth_probability_mass"),
        "top2_hit_rate": summarize_rate(rows, "top2_hit"),
        "top3_hit_rate": summarize_rate(rows, "top3_hit"),
        "mean_mrr": summarize_mean(rows, "mrr"),
        "mean_brier_score": summarize_mean(rows, "brier_score"),
        "mean_log_loss": summarize_mean(rows, "log_loss"),
    }
