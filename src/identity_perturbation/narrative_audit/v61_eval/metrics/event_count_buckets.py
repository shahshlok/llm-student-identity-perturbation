from __future__ import annotations

from ..helpers import (
    categorical_distribution,
    event_count_bucket_tuple,
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

NAME = "event_count_buckets"


def _parse_tuple(value: str) -> dict[str, str]:
    pieces = value.split("|")
    parsed: dict[str, str] = {}
    for piece in pieces:
        key, bucket = piece.split("=", 1)
        parsed[key] = bucket
    return parsed


def evaluate_row(example: EvaluationExample) -> dict[str, object]:
    truth = event_count_bucket_tuple(example.observed_events)
    distribution = categorical_distribution(example, event_count_bucket_tuple)
    rank = label_rank(distribution, truth)
    truth_mass = distribution.get(truth, 0.0)
    predicted = top_label(distribution)
    truth_dict = _parse_tuple(truth)
    predicted_dict = _parse_tuple(predicted)
    return {
        "observed": truth_dict,
        "predicted_top1": predicted_dict,
        "top1_match": predicted == truth,
        "top1_edit_bucket_match": predicted_dict["edit"] == truth_dict["edit"],
        "top1_run_bucket_match": predicted_dict["run"] == truth_dict["run"],
        "top1_pause_bucket_match": predicted_dict["pause"] == truth_dict["pause"],
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
        "edit_bucket_top1_accuracy": summarize_rate(rows, "top1_edit_bucket_match"),
        "run_bucket_top1_accuracy": summarize_rate(rows, "top1_run_bucket_match"),
        "pause_bucket_top1_accuracy": summarize_rate(rows, "top1_pause_bucket_match"),
        "mean_truth_probability_mass": summarize_mean(rows, "truth_probability_mass"),
        "top2_hit_rate": summarize_rate(rows, "top2_hit"),
        "top3_hit_rate": summarize_rate(rows, "top3_hit"),
        "mean_mrr": summarize_mean(rows, "mrr"),
        "mean_brier_score": summarize_mean(rows, "brier_score"),
        "mean_log_loss": summarize_mean(rows, "log_loss"),
    }
