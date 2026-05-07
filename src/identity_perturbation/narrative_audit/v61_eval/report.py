from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.openai_batch import _save_json

PRIMARY_METRICS = (
    "first_active_event_type",
    "first_change_line",
    "run_presence",
    "idle_gap_presence",
    "episode_motif",
    "event_type_overlap",
    "event_type_edit_similarity",
    "event_family_lcss",
    "event_family_dtw",
)

SUPPLEMENTARY_METRICS = (
    "first_event_type",
    "first_event_family",
    "change_presence",
    "event_type_prefix",
    "event_family_edit_similarity",
    "event_type_lcss",
    "event_count_buckets",
)


def write_evaluation_artifacts(
    *,
    evaluation: dict[str, Any],
    evaluation_json_path: Path,
    evaluation_md_path: Path,
    run_name: str,
    model: str,
    reasoning_effort: str,
) -> None:
    _save_json(evaluation_json_path, evaluation)
    scores = evaluation["scores"]
    metrics = scores["metrics"]
    lines = [
        "# V6.1 Evaluation",
        "",
        f"- Run name: `{run_name}`",
        f"- Model: `{model}`",
        f"- Reasoning effort: `{reasoning_effort}`",
        f"- Evaluated predictions: `{scores['n_predictions']}`",
        f"- Prefix length: `{scores['prefix_k']}`",
        "",
        "## Primary Metrics",
        "",
        "| Metric | Key Results |",
        "| --- | --- |",
    ]
    lines.extend(_primary_metric_lines(metrics))
    lines.extend(
        [
            "",
            "## Supplementary Metrics",
            "",
            "| Metric | Key Results |",
            "| --- | --- |",
        ]
    )
    lines.extend(_supplementary_metric_lines(metrics))

    first_event = metrics["first_event_type"]
    diagnostics = first_event["diagnostics"]
    lines.extend(
        [
            "",
            "## First Event Diagnostics",
            "",
            f"- Observed counts: `{json.dumps(diagnostics['observed_counts'], sort_keys=True)}`",
            f"- Predicted top-1 counts: `{json.dumps(diagnostics['predicted_top1_counts'], sort_keys=True)}`",
            "",
            "| Observed first event | Count | Top-1 | Truth mass | MRR |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for event_type, payload in diagnostics["by_observed"].items():
        lines.append(
            f"| {event_type} | {payload['count']} | {payload['top1_accuracy']:.3f} | "
            f"{payload['mean_truth_probability_mass']:.3f} | {payload['mean_mrr']:.3f} |"
        )

    lines.extend(
        [
            "",
            "| Observed | Predicted | Count |",
            "| --- | --- | --- |",
        ]
    )
    for item in diagnostics["top_confusions"]:
        lines.append(f"| {item['observed']} | {item['predicted']} | {item['count']} |")

    evaluation_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _primary_metric_lines(metrics: dict[str, Any]) -> list[str]:
    first_active = metrics["first_active_event_type"]
    first_change = metrics["first_change_line"]
    run_presence = metrics["run_presence"]
    idle_gap = metrics["idle_gap_presence"]
    motif = metrics["episode_motif"]
    overlap = metrics["event_type_overlap"]
    edit_sim = metrics["event_type_edit_similarity"]
    family_lcss = metrics["event_family_lcss"]
    family_dtw = metrics["event_family_dtw"]
    return [
        f"| First active event type | top-1 `{first_active['top1_accuracy']:.3f}`, truth mass `{first_active['mean_truth_probability_mass']:.3f}`, MRR `{first_active['mean_mrr']:.3f}` |",
        f"| First change line | exact `{first_change['top1_accuracy_exact']:.3f}`, +/-1 `{first_change['top1_accuracy_within_1']:.3f}`, +/-2 `{first_change['top1_accuracy_within_2']:.3f}` |",
        f"| Run presence | top-1 `{run_presence['top1_accuracy']:.3f}`, truth mass `{run_presence['mean_truth_probability_mass']:.3f}`, Brier `{run_presence['mean_brier_score']:.3f}` |",
        f"| Idle-gap presence | top-1 `{idle_gap['top1_accuracy']:.3f}`, truth mass `{idle_gap['mean_truth_probability_mass']:.3f}`, Brier `{idle_gap['mean_brier_score']:.3f}` |",
        f"| Episode motif | top-1 `{motif['top1_accuracy']:.3f}`, truth mass `{motif['mean_truth_probability_mass']:.3f}`, MRR `{motif['mean_mrr']:.3f}` |",
        f"| Event-type overlap | top-1 `{overlap['mean_top1_jaccard']:.3f}`, expected `{overlap['mean_expected_jaccard']:.3f}` |",
        f"| Event-type edit similarity | top-1 `{edit_sim['mean_top1_similarity']:.3f}`, expected `{edit_sim['mean_expected_similarity']:.3f}` |",
        f"| Event-family LCSS | top-1 `{family_lcss['mean_top1_lcss_ratio']:.3f}`, expected `{family_lcss['mean_expected_lcss_ratio']:.3f}` |",
        f"| Event-family DTW | top-1 `{family_dtw['mean_top1_similarity']:.3f}`, expected `{family_dtw['mean_expected_similarity']:.3f}` |",
    ]


def _supplementary_metric_lines(metrics: dict[str, Any]) -> list[str]:
    first_event = metrics["first_event_type"]
    first_event_family = metrics["first_event_family"]
    change_presence = metrics["change_presence"]
    prefix = metrics["event_type_prefix"]
    family_edit = metrics["event_family_edit_similarity"]
    event_lcss = metrics["event_type_lcss"]
    count_buckets = metrics["event_count_buckets"]
    return [
        f"| First event type | top-1 `{first_event['top1_accuracy']:.3f}`, truth mass `{first_event['mean_truth_probability_mass']:.3f}`, MRR `{first_event['mean_mrr']:.3f}` |",
        f"| First event family | top-1 `{first_event_family['top1_accuracy']:.3f}`, truth mass `{first_event_family['mean_truth_probability_mass']:.3f}`, MRR `{first_event_family['mean_mrr']:.3f}` |",
        f"| Change presence | top-1 `{change_presence['top1_accuracy']:.3f}`, truth mass `{change_presence['mean_truth_probability_mass']:.3f}`, Brier `{change_presence['mean_brier_score']:.3f}` |",
        f"| Event-type prefix | exact `{prefix['top1_exact_match_rate']:.3f}`, top-1 shared `{prefix['mean_top1_shared_prefix_norm']:.3f}`, expected shared `{prefix['mean_expected_shared_prefix_norm']:.3f}` |",
        f"| Event-family edit similarity | top-1 `{family_edit['mean_top1_similarity']:.3f}`, expected `{family_edit['mean_expected_similarity']:.3f}` |",
        f"| Event-type LCSS | top-1 `{event_lcss['mean_top1_lcss_ratio']:.3f}`, expected `{event_lcss['mean_expected_lcss_ratio']:.3f}` |",
        f"| Event count buckets | full top-1 `{count_buckets['top1_accuracy']:.3f}`, edit `{count_buckets['edit_bucket_top1_accuracy']:.3f}`, run `{count_buckets['run_bucket_top1_accuracy']:.3f}`, pause `{count_buckets['pause_bucket_top1_accuracy']:.3f}` |",
    ]
