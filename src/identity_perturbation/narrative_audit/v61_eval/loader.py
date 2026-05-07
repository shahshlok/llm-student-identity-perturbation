from __future__ import annotations

from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.openai_batch import _load_json

from .types import EvaluationExample, Hypothesis

HYDRATED_SCHEMA_VERSION = "v6_1_batch_hydrated_predictions_v1"
OBSERVED_EPISODE_SCHEMA_VERSION = "v6_1_observed_next_episode_v1"


class V61EvaluationLoadError(ValueError):
    pass


def load_hydrated_predictions(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise V61EvaluationLoadError(f"Hydrated predictions not found: {path}")
    payload = _load_json(path)
    if payload.get("schema_version") != HYDRATED_SCHEMA_VERSION:
        raise V61EvaluationLoadError(
            f"Unexpected hydrated prediction schema in {path}: {payload.get('schema_version')}"
        )
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise V61EvaluationLoadError("No hydrated predictions available for v6.1 evaluation")
    return payload


def load_observed_episode(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise V61EvaluationLoadError(f"Observed next episode not found: {path}")
    payload = _load_json(path)
    if payload.get("schema_version") != OBSERVED_EPISODE_SCHEMA_VERSION:
        raise V61EvaluationLoadError(
            f"Unexpected observed next episode schema in {path}: {payload.get('schema_version')}"
        )
    events = payload.get("semantic_event_tape")
    if not isinstance(events, list) or not events:
        raise V61EvaluationLoadError(
            f"Observed next episode missing non-empty semantic_event_tape: {path}"
        )
    return payload


def build_examples(
    *,
    hydrated_payload: dict[str, Any],
    prefix_k: int,
) -> list[EvaluationExample]:
    if prefix_k < 1:
        raise V61EvaluationLoadError(f"prefix_k must be positive, got {prefix_k}")
    examples: list[EvaluationExample] = []
    for prediction in hydrated_payload["predictions"]:
        observed = load_observed_episode(Path(prediction["observed_next_episode_path"]))
        response = prediction["response"]
        hypotheses_payload = response.get("next_episode_hypotheses")
        if not isinstance(hypotheses_payload, list) or not hypotheses_payload:
            raise V61EvaluationLoadError(
                f"Prediction {prediction['custom_id']} missing non-empty next_episode_hypotheses"
            )
        hypotheses = []
        for row in hypotheses_payload:
            hypotheses.append(
                Hypothesis(
                    label=str(row["label"]),
                    probability=float(row["estimated_probability"]),
                    student_state_summary=str(row["student_state_summary"]),
                    events=tuple(row["predicted_event_tape"]),
                )
            )
        examples.append(
            EvaluationExample(
                custom_id=str(prediction["custom_id"]),
                class_id=str(prediction["class_id"]),
                assessment_id=str(prediction["assessment_id"]),
                exercise_id=str(prediction["exercise_id"]),
                student_id=str(prediction["student_id"]),
                transition_index_0idx=int(prediction["transition_index_0idx"]),
                prefix_k=prefix_k,
                observed_events=tuple(observed["semantic_event_tape"]),
                hypotheses=tuple(hypotheses),
            )
        )
    return examples
