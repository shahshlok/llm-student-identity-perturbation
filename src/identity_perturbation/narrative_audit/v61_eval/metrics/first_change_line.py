from __future__ import annotations

from ..helpers import (
    first_change_line,
    hypothesis_rank,
    line_within_tolerance,
    reciprocal_rank,
    summarize_mean,
    summarize_rate,
    top_hypothesis,
    topk_hit,
    truth_mass,
)
from ..types import EvaluationExample

NAME = "first_change_line"


def evaluate_row(example: EvaluationExample) -> dict[str, object]:
    observed = first_change_line(example.observed_events)
    predicted_top1 = first_change_line(top_hypothesis(example).events)
    exact_mass = truth_mass(example, lambda events: first_change_line(events) == observed)
    within_1_mass = truth_mass(
        example,
        lambda events: bool(line_within_tolerance(first_change_line(events), observed, 1)),
    )
    within_2_mass = truth_mass(
        example,
        lambda events: bool(line_within_tolerance(first_change_line(events), observed, 2)),
    )
    exact_rank = hypothesis_rank(example, lambda events: first_change_line(events) == observed)
    within_2_rank = hypothesis_rank(
        example,
        lambda events: bool(line_within_tolerance(first_change_line(events), observed, 2)),
    )
    return {
        "observed": observed,
        "predicted_top1": predicted_top1,
        "top1_exact_match": predicted_top1 == observed,
        "truth_probability_mass_exact": exact_mass,
        "top1_within_1_match": line_within_tolerance(predicted_top1, observed, 1),
        "truth_probability_mass_within_1": within_1_mass,
        "top1_within_2_match": line_within_tolerance(predicted_top1, observed, 2),
        "truth_probability_mass_within_2": within_2_mass,
        "top2_hit_exact": topk_hit(exact_rank, 2),
        "top3_hit_exact": topk_hit(exact_rank, 3),
        "mrr_exact": reciprocal_rank(exact_rank),
        "top2_hit_within_2": topk_hit(within_2_rank, 2),
        "top3_hit_within_2": topk_hit(within_2_rank, 3),
        "mrr_within_2": reciprocal_rank(within_2_rank),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "top1_accuracy_exact": summarize_rate(rows, "top1_exact_match"),
        "mean_truth_probability_mass_exact": summarize_mean(rows, "truth_probability_mass_exact"),
        "top1_accuracy_within_1": summarize_rate(rows, "top1_within_1_match"),
        "mean_truth_probability_mass_within_1": summarize_mean(
            rows, "truth_probability_mass_within_1"
        ),
        "top1_accuracy_within_2": summarize_rate(rows, "top1_within_2_match"),
        "mean_truth_probability_mass_within_2": summarize_mean(
            rows, "truth_probability_mass_within_2"
        ),
        "top2_hit_rate_exact": summarize_rate(rows, "top2_hit_exact"),
        "top3_hit_rate_exact": summarize_rate(rows, "top3_hit_exact"),
        "mean_mrr_exact": summarize_mean(rows, "mrr_exact"),
        "top2_hit_rate_within_2": summarize_rate(rows, "top2_hit_within_2"),
        "top3_hit_rate_within_2": summarize_rate(rows, "top3_hit_within_2"),
        "mean_mrr_within_2": summarize_mean(rows, "mrr_within_2"),
    }
