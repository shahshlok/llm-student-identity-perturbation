from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.openai_batch import _load_json, _save_json, _utc_now

from ..v61_metric.registry import PREDICTIVE_METRICS, TRAJECTORY_METRICS

EVALUATION_SCHEMA_VERSION = "v6_1_batch_evaluation_v2"
COMPARISON_SCHEMA_VERSION = "v6_1_condition_comparison_v1"


class V61ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class ConditionEvaluation:
    label: str
    path: Path
    payload: dict[str, Any]


def load_condition_evaluation(*, label: str, path: Path) -> ConditionEvaluation:
    if not path.exists():
        raise V61ComparisonError(f"Evaluation JSON not found for {label}: {path}")
    payload = _load_json(path)
    if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise V61ComparisonError(
            f"Unexpected evaluation schema for {label} in {path}: {payload.get('schema_version')}"
        )
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise V61ComparisonError(f"Evaluation missing scores object for {label}: {path}")
    metrics = scores.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise V61ComparisonError(f"Evaluation missing metrics object for {label}: {path}")
    return ConditionEvaluation(label=label, path=path, payload=payload)


def build_condition_comparison(
    *,
    conditions: list[ConditionEvaluation],
) -> dict[str, Any]:
    if len(conditions) < 2:
        raise V61ComparisonError("Need at least two conditions for comparison")

    predictive_metric_names = [metric.NAME for metric in PREDICTIVE_METRICS]
    trajectory_metric_names = [metric.NAME for metric in TRAJECTORY_METRICS]

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "conditions": [
            {
                "label": condition.label,
                "path": str(condition.path.resolve()),
                "run_name": condition.payload["run_name"],
                "model": condition.payload["model"],
                "reasoning_effort": condition.payload["reasoning_effort"],
                "n_predictions": condition.payload["scores"]["n_predictions"],
            }
            for condition in conditions
        ],
        "metric_groups": {
            "predictive": predictive_metric_names,
            "trajectory": trajectory_metric_names,
        },
        "metrics": {
            metric_name: _compare_metric(metric_name=metric_name, conditions=conditions)
            for metric_name in predictive_metric_names + trajectory_metric_names
        },
    }


def write_condition_comparison(
    *,
    comparison: dict[str, Any],
    comparison_json_path: Path,
    comparison_md_path: Path,
) -> None:
    _save_json(comparison_json_path, comparison)
    lines = [
        "# V6.1 Condition Comparison",
        "",
        "## Conditions",
        "",
        "| Label | Run name | Predictions | Evaluation |",
        "| --- | --- | --- | --- |",
    ]
    for condition in comparison["conditions"]:
        lines.append(
            f"| {condition['label']} | `{condition['run_name']}` | {condition['n_predictions']} | "
            f"`{condition['path']}` |"
        )

    for group_name, metric_names in comparison["metric_groups"].items():
        lines.extend(["", f"## {group_name.title()} Metrics", ""])
        for metric_name in metric_names:
            metric_payload = comparison["metrics"][metric_name]
            lines.extend(
                [
                    f"### `{metric_name}`",
                    "",
                    "| Summary key | "
                    + " | ".join(condition["label"] for condition in comparison["conditions"])
                    + " |",
                    "| --- | " + " | ".join("---" for _ in comparison["conditions"]) + " |",
                ]
            )
            for key in metric_payload["summary_keys"]:
                values = [
                    _format_scalar(metric_payload["by_condition"][condition["label"]][key])
                    for condition in comparison["conditions"]
                ]
                lines.append("| " + " | ".join([key, *values]) + " |")

    comparison_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compare_metric(
    *,
    metric_name: str,
    conditions: list[ConditionEvaluation],
) -> dict[str, Any]:
    per_condition: dict[str, dict[str, Any]] = {}
    summary_keys: list[str] = []
    for condition in conditions:
        metric_summary = condition.payload["scores"]["metrics"].get(metric_name)
        if not isinstance(metric_summary, dict):
            raise V61ComparisonError(
                f"Metric {metric_name} missing from condition {condition.label}: {condition.path}"
            )
        scalar_summary = {key: value for key, value in metric_summary.items() if _is_scalar(value)}
        if not scalar_summary:
            raise V61ComparisonError(
                f"Metric {metric_name} has no scalar summary keys in condition {condition.label}"
            )
        if not summary_keys:
            summary_keys = list(scalar_summary.keys())
        elif list(scalar_summary.keys()) != summary_keys:
            raise V61ComparisonError(
                f"Metric {metric_name} summary keys differ across conditions: "
                f"{summary_keys} vs {list(scalar_summary.keys())}"
            )
        per_condition[condition.label] = scalar_summary

    predictive_names = {metric.NAME for metric in PREDICTIVE_METRICS}
    metric_kind = "predictive" if metric_name in predictive_names else "trajectory"
    return {
        "metric_kind": metric_kind,
        "summary_keys": summary_keys,
        "by_condition": per_condition,
    }


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _format_scalar(value: Any) -> str:
    if value is None:
        return "`n/a`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, str):
        return f"`{value}`"
    raise V61ComparisonError(f"Unsupported scalar value for markdown formatting: {value!r}")
