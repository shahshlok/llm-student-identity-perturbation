from __future__ import annotations

from ..helpers import (
    binary_brier_score,
    contains_type,
    safe_log_loss,
    summarize_mean,
    summarize_rate,
    top_hypothesis,
    truth_mass,
)
from ..types import EvaluationExample

NAME = "idle_gap_presence"


def evaluate_row(example: EvaluationExample) -> dict[str, object]:
    observed = contains_type(example.observed_events, "idle_gap")
    predicted_true_probability = truth_mass(
        example, lambda events: contains_type(events, "idle_gap")
    )
    predicted_top1 = contains_type(top_hypothesis(example).events, "idle_gap")
    truth_probability_mass = (
        predicted_true_probability if observed else 1.0 - predicted_true_probability
    )
    return {
        "observed": observed,
        "predicted_top1": predicted_top1,
        "top1_match": predicted_top1 == observed,
        "truth_probability_mass": truth_probability_mass,
        "predicted_true_probability": predicted_true_probability,
        "brier_score": binary_brier_score(predicted_true_probability, observed),
        "log_loss": safe_log_loss(truth_probability_mass),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "top1_accuracy": summarize_rate(rows, "top1_match"),
        "mean_truth_probability_mass": summarize_mean(rows, "truth_probability_mass"),
        "mean_predicted_true_probability": summarize_mean(rows, "predicted_true_probability"),
        "mean_brier_score": summarize_mean(rows, "brier_score"),
        "mean_log_loss": summarize_mean(rows, "log_loss"),
    }
