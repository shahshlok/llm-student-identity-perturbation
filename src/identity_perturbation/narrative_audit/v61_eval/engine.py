from __future__ import annotations

from typing import Any

from ..v61_metric.registry import ALL_METRICS
from .loader import build_examples, load_hydrated_predictions

EVALUATION_SCHEMA_VERSION = "v6_1_batch_evaluation_v2"


def evaluate_hydrated_predictions(*, hydrated_path, prefix_k: int) -> dict[str, Any]:
    hydrated_payload = load_hydrated_predictions(hydrated_path)
    examples = build_examples(hydrated_payload=hydrated_payload, prefix_k=prefix_k)

    rows: list[dict[str, Any]] = []
    for example in examples:
        metric_rows = {metric.NAME: metric.evaluate_row(example) for metric in ALL_METRICS}
        rows.append(
            {
                "custom_id": example.custom_id,
                "class_id": example.class_id,
                "assessment_id": example.assessment_id,
                "exercise_id": example.exercise_id,
                "student_id": example.student_id,
                "transition_index_0idx": example.transition_index_0idx,
                "metrics": metric_rows,
            }
        )

    scores = {
        "n_predictions": len(rows),
        "prefix_k": prefix_k,
        "metric_order": [metric.NAME for metric in ALL_METRICS],
        "metrics": {
            metric.NAME: metric.summarize([row["metrics"][metric.NAME] for row in rows])
            for metric in ALL_METRICS
        },
        "rows": rows,
    }
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "scores": scores,
    }
