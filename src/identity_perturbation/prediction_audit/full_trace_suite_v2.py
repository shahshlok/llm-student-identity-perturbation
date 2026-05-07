"""Multi-row analyses that turn v6.2 scored predictions into the claims
we actually want to make about latent student modelling.

The per-row scorer tells us *how well* a prediction fit its observed
target.  That alone cannot answer the research question the v6.1 study
already showed has teeth: are LLMs modelling **this student**, or just
the exercise mode?  To answer that we need three things the v1
pipeline does not produce:

1. Per-exercise majority baselines
   For each exercise, derive a trace-blind "copy the modal observed
   repair" prediction and score it.  Every metric then has a natural
   zero against which to report lift.

2. Identity discrimination (A-vs-B on the same exercise)
   Given two rows on the same exercise from different students, we
   re-score student A's top-1 prediction against student B's observed
   target.  If the model is picking up student-level signal, score
   against own-truth should beat score against other-truth.  We report
   mean(self) - mean(other) and the AUC (fraction of directed pairs
   where self > other), plus a paired Wilcoxon-lite p-value computed
   from sign tests so the suite has no scipy dependency.

3. Condition-level deltas and pooling
   When later runs carry full / no_trace / trace_shuffled labels we can
   slice the same aggregate by condition and report paired deltas at
   matched (student, exercise, attempt_n) keys.

All three live here.  The v1 scorer is left untouched.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from identity_perturbation.prediction_audit.full_trace_scorer_v2 import (
    DEFAULT_EDIT_STEP_WINDOW_RADIUS,
    SCHEMA_VERSION,
    _full_code_metrics,
    _trajectory_metrics,
    score_full_trace_prediction_v2,
)
from identity_perturbation.prediction_audit.pair_matching import normalized_code_distance


class FullTraceSuiteV2Error(ValueError):
    pass


# ---------------------------------------------------------------------------
# Row identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowKey:
    """Parsed identity for a scored row.

    v6.2 ``custom_id`` strings look like ``"589:5897:6353:9735:1"`` —
    ``class_id : student_id : exercise_id : task_id : attempt_n``.  The
    key lets the suite slice rows by exercise (for baselines) and by
    student (for A-vs-B discrimination) without re-reading each bundle
    manifest.
    """

    class_id: str
    student_id: str
    exercise_id: str
    task_id: str
    attempt_n: str

    @property
    def exercise_scope(self) -> str:
        return f"{self.class_id}:{self.exercise_id}:{self.task_id}"

    @classmethod
    def from_custom_id(cls, custom_id: str) -> RowKey:
        parts = custom_id.split(":")
        if len(parts) != 5:
            raise FullTraceSuiteV2Error(
                f"custom_id must have 5 colon-separated parts, got {custom_id!r}"
            )
        class_id, student_id, exercise_id, task_id, attempt_n = parts
        if not all([class_id, student_id, exercise_id, task_id, attempt_n]):
            raise FullTraceSuiteV2Error(f"custom_id has empty parts: {custom_id!r}")
        return cls(
            class_id=class_id,
            student_id=student_id,
            exercise_id=exercise_id,
            task_id=task_id,
            attempt_n=attempt_n,
        )


@dataclass
class ScoredRow:
    """One already-scored row carried through the suite.

    ``scored`` must be the object returned by
    ``score_full_trace_prediction_v2`` (schema
    ``v6_2_full_trace_scored_prediction_v2``).  Any other scorer
    version is rejected so downstream aggregation stays consistent.
    """

    custom_id: str
    key: RowKey
    condition: str
    scored: dict[str, Any]
    response_payload: dict[str, Any]
    observed_repair_target: dict[str, Any]
    observed_coarse_path: dict[str, Any]
    attempt_n_pass_fail_vector: tuple[bool, ...] = field(default_factory=tuple)
    attempt_n_pass_vector_signature: str = ""
    attempt_n_normalized_code: str | None = None

    def __post_init__(self) -> None:
        schema_version = self.scored.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise FullTraceSuiteV2Error(
                f"ScoredRow requires schema {SCHEMA_VERSION!r}, got {schema_version!r}"
            )
        if self.attempt_n_pass_fail_vector:
            derived_signature = _pass_vector_signature(self.attempt_n_pass_fail_vector)
            if self.attempt_n_pass_vector_signature:
                if self.attempt_n_pass_vector_signature != derived_signature:
                    raise FullTraceSuiteV2Error(
                        "ScoredRow attempt_n_pass_vector_signature does not match attempt_n_pass_fail_vector"
                    )
            else:
                self.attempt_n_pass_vector_signature = derived_signature

    @property
    def l2a_group_key(self) -> tuple[str, str]:
        if not self.attempt_n_pass_vector_signature:
            raise FullTraceSuiteV2Error(
                f"ScoredRow {self.custom_id} is missing attempt_n_pass_vector_signature"
            )
        return (self.key.exercise_scope, self.attempt_n_pass_vector_signature)

    @property
    def normalized_code_for_l2b(self) -> str:
        if self.attempt_n_normalized_code is None:
            raise FullTraceSuiteV2Error(
                f"ScoredRow {self.custom_id} is missing attempt_n_normalized_code"
            )
        return self.attempt_n_normalized_code


# ---------------------------------------------------------------------------
# Aggregation across rows
# ---------------------------------------------------------------------------


VIEW_NAMES: tuple[str, ...] = (
    "oracle_at_3",
    "expected",
    "rank_weighted",
    "top_1",
    "best_hypothesis_rank",
)

FAMILY_NAMES: tuple[str, ...] = ("repair", "trajectory", "full_code")


def _collect_view_values(
    rows: Iterable[ScoredRow],
    *,
    family: str,
    metric: str,
    view: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        family_view = row.scored["views"][family][metric]
        values.append(float(family_view[view]))
    return values


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def aggregate_rows(rows: list[ScoredRow]) -> dict[str, Any]:
    """Compute mean/median per (family, metric, view) across rows.

    The shape mirrors ``views`` inside a single scored prediction so
    downstream tooling can drop it straight into a condition report.
    """

    if not rows:
        raise FullTraceSuiteV2Error("Cannot aggregate zero rows")
    aggregate: dict[str, Any] = {"n_rows": len(rows), "families": {}}
    reference_views = rows[0].scored["views"]
    for family in FAMILY_NAMES:
        aggregate["families"][family] = {}
        for metric in reference_views.get(family, {}):
            aggregate["families"][family][metric] = {}
            for view in VIEW_NAMES:
                values = _collect_view_values(rows, family=family, metric=metric, view=view)
                aggregate["families"][family][metric][view] = {
                    "mean": _mean(values),
                    "median": _median(values),
                    "n": len(values),
                }
    return aggregate


# ---------------------------------------------------------------------------
# Per-exercise majority baseline
# ---------------------------------------------------------------------------


def _majority_copy_prediction(
    *,
    exercise_scope: str,
    rows: list[ScoredRow],
) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(predicted_next_code, predicted_next_trajectory)`` for the
    trace-blind majority baseline on this exercise scope.

    Strategy (intentionally boring so the baseline is faithful to
    "you didn't read the trace"):

    * predicted_next_code = the observed attempt_n+1 code most
      frequently seen across students on this exercise; ties broken by
      first appearance order
    * predicted_next_trajectory = the most common bounded coarse
      action-type shape across the same group; edit line ranges inside
      are filled from the first matching row after projecting observed
      paths down to the model's max-8-step contract

    The baseline is scored using the same v2 scorer to keep all
    numerical comparisons apples-to-apples.
    """

    if not rows:
        raise FullTraceSuiteV2Error(f"No rows in scope {exercise_scope!r}")
    code_counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for order, row in enumerate(rows):
        code = str(row.observed_repair_target["attempt_n1"]["code"])
        code_counts[code] = code_counts.get(code, 0) + 1
        first_seen.setdefault(code, order)
    majority_code = max(
        code_counts.keys(),
        key=lambda code: (code_counts[code], -first_seen[code]),
    )
    shape_counts: dict[tuple[str, ...], int] = {}
    shape_first_seen: dict[tuple[str, ...], int] = {}
    shape_samples: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for order, row in enumerate(rows):
        steps = row.observed_coarse_path["attempt_n1"]["coarse_path_steps"]
        projected_steps = _project_observed_steps_to_predicted_trajectory(steps)
        shape = tuple(step["action_type"] for step in projected_steps)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        shape_first_seen.setdefault(shape, order)
        shape_samples.setdefault(shape, projected_steps)
    majority_shape = max(
        shape_counts.keys(),
        key=lambda shape: (shape_counts[shape], -shape_first_seen[shape]),
    )
    majority_trajectory = list(shape_samples[majority_shape])
    return majority_code, majority_trajectory


def _observed_step_to_predicted_step(step: dict[str, Any]) -> dict[str, Any]:
    """Translate an observed-path step into the predicted-step schema.

    Observed paths store ``None`` for non-edit line indices; the
    predicted-path pydantic model requires ``-1`` sentinels on
    ``local_run`` and ``submit`` steps.  This conversion is the only
    place the two coordinate conventions meet, so it lives as a tiny
    dedicated helper instead of polluting the callers.
    """

    action_type = step["action_type"]
    if action_type == "edit":
        start = int(step["target_start_line_0idx"])
        end = int(step["target_end_line_0idx"])
        return {
            "action_type": "edit",
            "target_start_line_0idx": start,
            "target_end_line_0idx": end,
        }
    return {
        "action_type": action_type,
        "target_start_line_0idx": -1,
        "target_end_line_0idx": -1,
    }


def _merge_edit_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        raise FullTraceSuiteV2Error("Cannot merge an empty edit-step group")
    if any(step.get("action_type") != "edit" for step in steps):
        raise FullTraceSuiteV2Error("Edit-step merge received a non-edit step")
    starts = [int(step["target_start_line_0idx"]) for step in steps]
    ends = [int(step["target_end_line_0idx"]) for step in steps]
    return {
        "action_type": "edit",
        "target_start_line_0idx": min(starts),
        "target_end_line_0idx": max(ends),
    }


def _collapse_predicted_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    for step in steps:
        action_type = step["action_type"]
        if action_type == "edit":
            if collapsed and collapsed[-1]["action_type"] == "edit":
                collapsed[-1] = _merge_edit_steps([collapsed[-1], step])
            else:
                collapsed.append(dict(step))
            continue
        if collapsed and collapsed[-1]["action_type"] == action_type:
            continue
        collapsed.append(dict(step))
    return collapsed


def _project_observed_steps_to_predicted_trajectory(
    observed_steps: list[dict[str, Any]],
    *,
    max_steps: int = 8,
) -> list[dict[str, Any]]:
    """Project an observed coarse path down to the model's bounded contract.

    The model can only emit 1..8 coarse steps, so baseline synthesis
    must summarize raw observed paths into the same space before it can
    be validated or scored. We first collapse adjacent identical action
    types, then, if needed, bin the pre-submit sequence into at most
    ``max_steps - 1`` ordered chunks and merge each chunk into one
    coarse step.
    """

    if max_steps < 2:
        raise FullTraceSuiteV2Error(
            "Projected trajectory requires room for at least one pre-submit step and submit"
        )
    if not observed_steps:
        raise FullTraceSuiteV2Error("Observed coarse path steps cannot be empty")
    if observed_steps[-1].get("action_type") != "submit":
        raise FullTraceSuiteV2Error("Observed coarse path must end in submit")

    predicted_steps = [_observed_step_to_predicted_step(step) for step in observed_steps]
    collapsed = _collapse_predicted_steps(predicted_steps)
    if collapsed[-1]["action_type"] != "submit":
        raise FullTraceSuiteV2Error("Projected trajectory must end in submit")
    pre_submit = collapsed[:-1]
    if not pre_submit:
        raise FullTraceSuiteV2Error(
            "Projected trajectory must include at least one pre-submit step"
        )
    if not any(step["action_type"] == "edit" for step in pre_submit):
        raise FullTraceSuiteV2Error(
            "Projected trajectory must include at least one pre-submit edit"
        )
    if len(collapsed) <= max_steps:
        return collapsed

    max_pre_submit_steps = max_steps - 1
    projected_pre_submit: list[dict[str, Any]] = []
    for index in range(max_pre_submit_steps):
        start = math.floor(index * len(pre_submit) / max_pre_submit_steps)
        end = math.floor((index + 1) * len(pre_submit) / max_pre_submit_steps)
        chunk = pre_submit[start:end]
        if not chunk:
            continue
        edit_steps = [step for step in chunk if step["action_type"] == "edit"]
        if edit_steps:
            projected_pre_submit.append(_merge_edit_steps(edit_steps))
        else:
            projected_pre_submit.append(
                {
                    "action_type": "local_run",
                    "target_start_line_0idx": -1,
                    "target_end_line_0idx": -1,
                }
            )
    projected = _collapse_predicted_steps(projected_pre_submit) + [collapsed[-1]]
    if len(projected) > max_steps:
        raise FullTraceSuiteV2Error(
            f"Projected baseline trajectory still exceeds {max_steps} steps after bounding"
        )
    if projected[-1]["action_type"] != "submit":
        raise FullTraceSuiteV2Error("Projected baseline trajectory must end in submit")
    return projected


def _synthesize_baseline_response(
    *,
    predicted_code: str,
    predicted_trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wrap the majority baseline as a valid 3-hypothesis response.

    The scorer requires exactly three hypotheses summing to 1.  We
    emit the same baseline three times with equal probability.  The
    Oracle@3 / Top-1 / rank_weighted views will all equal the raw
    score in this case (they must, because there is only one
    candidate), which is the correct behaviour for a no-choice
    baseline.
    """

    hypothesis = {
        "label": "majority_copy_baseline",
        "estimated_probability": 1.0 / 3.0,
        "predicted_next_code": predicted_code,
        "predicted_next_trajectory": predicted_trajectory,
    }
    return {
        "schema_version": "v6_2_full_trace_prediction_v3",
        "hypotheses": [
            {**hypothesis, "label": f"majority_copy_baseline_{i + 1}"} for i in range(3)
        ],
    }


def score_majority_baselines(rows: list[ScoredRow]) -> dict[str, dict[str, Any]]:
    """Score the trace-blind majority baseline per exercise scope.

    Returns one baseline entry per exercise scope. The baseline
    prediction itself is scope-level, but the scoring must still be
    done per row because different students in the same scope can have
    different observed ``n+1`` targets. Reusing a single
    representative-row score across the entire scope would flatten real
    lift differences into artifacts.

    Rows with a unique exercise scope get a degenerate baseline that
    equals their own observation — so the lift on that row is exactly
    zero and you know the number cannot be used for personalization
    claims. That is the honest reporting of the "nothing to baseline
    against" case.
    """

    scopes: dict[str, list[ScoredRow]] = {}
    for row in rows:
        scopes.setdefault(row.key.exercise_scope, []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for scope, scope_rows in scopes.items():
        code, trajectory = _majority_copy_prediction(
            exercise_scope=scope,
            rows=scope_rows,
        )
        baseline_response = _synthesize_baseline_response(
            predicted_code=code,
            predicted_trajectory=trajectory,
        )
        representative = scope_rows[0]
        per_row_scored: dict[str, dict[str, Any]] = {}
        for row in scope_rows:
            per_row_scored[row.custom_id] = score_full_trace_prediction_v2(
                response_payload=baseline_response,
                observed_repair_target=row.observed_repair_target,
                observed_coarse_path=row.observed_coarse_path,
            )
        out[scope] = {
            "scope": scope,
            "n_rows_in_scope": len(scope_rows),
            "unique_scope": len(scope_rows) == 1,
            "baseline_response": baseline_response,
            "scored_against_representative_row": representative.custom_id,
            "scored": per_row_scored[representative.custom_id],
            "per_row_scored": per_row_scored,
        }
    return out


def model_lift_over_baseline(
    rows: list[ScoredRow],
    baselines: dict[str, dict[str, Any]],
    *,
    family: str,
    metric: str,
    view: str,
) -> dict[str, Any]:
    """Paired per-row delta (model - baseline) for one metric/view.

    Every row is paired with the baseline computed on its exercise
    scope.  We report the mean, median, fraction of rows where the
    model beats the baseline, and a sign-test p-value.  Rows whose
    scope contains only that one student get their delta reported but
    flagged as ``unique_scope`` so downstream readers can filter them
    out before making personalization claims.
    """

    deltas: list[float] = []
    wins = 0
    unique_scope_count = 0
    per_row: list[dict[str, Any]] = []
    for row in rows:
        baseline_entry = baselines.get(row.key.exercise_scope)
        if baseline_entry is None:
            raise FullTraceSuiteV2Error(
                f"No baseline for scope {row.key.exercise_scope}; "
                "ensure rows and baselines share the same corpus"
            )
        model_score = float(row.scored["views"][family][metric][view])
        per_row_scored = baseline_entry.get("per_row_scored")
        if isinstance(per_row_scored, dict) and row.custom_id in per_row_scored:
            baseline_payload = per_row_scored[row.custom_id]
        else:
            baseline_payload = baseline_entry["scored"]
        baseline_score = float(baseline_payload["views"][family][metric][view])
        delta = model_score - baseline_score
        deltas.append(delta)
        if delta > 0:
            wins += 1
        if baseline_entry["unique_scope"]:
            unique_scope_count += 1
        per_row.append(
            {
                "custom_id": row.custom_id,
                "exercise_scope": row.key.exercise_scope,
                "unique_scope": baseline_entry["unique_scope"],
                "model_score": model_score,
                "baseline_score": baseline_score,
                "delta": delta,
            }
        )
    return {
        "family": family,
        "metric": metric,
        "view": view,
        "n_rows": len(deltas),
        "n_unique_scope_rows": unique_scope_count,
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
        "win_rate": wins / len(deltas) if deltas else float("nan"),
        "sign_test_p_two_sided": _sign_test_p_value(deltas),
        "per_row": per_row,
    }


# ---------------------------------------------------------------------------
# Identity discrimination (A-vs-B on same exercise)
# ---------------------------------------------------------------------------


IDENTITY_DISCRIMINATION_VIEWS: tuple[str, ...] = ("top_1", "expected", "rank_weighted")


def _top1_hypothesis_from_row(row: ScoredRow) -> dict[str, Any]:
    response = row.response_payload
    hypotheses = response["hypotheses"]
    probabilities = [float(h["estimated_probability"]) for h in hypotheses]
    top1_index = max(range(len(hypotheses)), key=lambda i: probabilities[i])
    return dict(hypotheses[top1_index])


def _observed_truth_as_response(row: ScoredRow) -> dict[str, Any]:
    """Wrap a row's observed next attempt as a synthetic top-1 response.

    This lets post-hoc peer analyses reuse the exact same scorer contract
    as model predictions. The resulting score answers: if student A had
    "predicted" their own true next move, how well would that move fit a
    matched peer B?
    """

    hypothesis = {
        "predicted_next_code": str(row.observed_repair_target["attempt_n1"]["code"]),
        "predicted_next_trajectory": _project_observed_steps_to_predicted_trajectory(
            row.observed_coarse_path["attempt_n1"]["coarse_path_steps"]
        ),
    }
    return {
        "schema_version": "v6_2_full_trace_prediction_v3",
        "hypotheses": [
            {
                "label": "observed_truth_h1",
                "estimated_probability": 0.999998,
                **hypothesis,
            },
            {
                "label": "observed_truth_h2",
                "estimated_probability": 0.000001,
                **hypothesis,
            },
            {
                "label": "observed_truth_h3",
                "estimated_probability": 0.000001,
                **hypothesis,
            },
        ],
    }


def _pass_vector_signature(vector: tuple[bool, ...]) -> str:
    return "".join("P" if value else "F" for value in vector)


def _rows_grouped_by_l2a(rows: list[ScoredRow]) -> dict[tuple[str, str], list[ScoredRow]]:
    groups: dict[tuple[str, str], list[ScoredRow]] = {}
    for row in rows:
        groups.setdefault(row.l2a_group_key, []).append(row)
    return groups


def _validated_l2b_threshold(l2b_threshold: float | None) -> float | None:
    if l2b_threshold is None:
        return None
    threshold = float(l2b_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise FullTraceSuiteV2Error(f"L2B threshold must be between 0 and 1, got {l2b_threshold!r}")
    return threshold


def _pair_code_distance(
    row_a: ScoredRow,
    row_b: ScoredRow,
    *,
    l2b_threshold: float | None,
) -> float | None:
    threshold = _validated_l2b_threshold(l2b_threshold)
    if threshold is None:
        return None
    return normalized_code_distance(
        row_a.normalized_code_for_l2b,
        row_b.normalized_code_for_l2b,
    )


def _analysis_identity(
    *,
    l2b_threshold: float | None,
) -> dict[str, Any]:
    threshold = _validated_l2b_threshold(l2b_threshold)
    if threshold is None:
        return {
            "match_mode": "l2a",
            "l2b_threshold": None,
        }
    return {
        "match_mode": "l2a_and_l2b",
        "l2b_threshold": threshold,
    }


def compute_identity_discrimination(
    rows: list[ScoredRow],
    *,
    family: str,
    metric: str,
    view: str = "top_1",
    l2b_threshold: float | None = None,
) -> dict[str, Any]:
    """A-vs-B identity discrimination on the chosen metric/view.

    For every directed pair ``(A, B)`` on the same exercise scope with
    ``student_A != student_B`` we score A's original response payload
    against B's truth under the requested aggregation view. Scoring A
    against A (self-score) already exists in the row. Personalization
    evidence looks like:

    * ``mean_self - mean_other > 0``
    * AUC = ``P(self > other) > 0.5``

    Supported views are deliberately limited to ``top_1``,
    ``expected``, and ``rank_weighted``. ``oracle_at_3`` is excluded
    because it gives the model extra chances to be right, and
    ``best_hypothesis_rank`` is ordinal rather than a same-direction
    quality score.

    Rows are always compared within the same frozen L2A group
    ``(exercise_scope, attempt_n_pass_vector_signature)``. When
    ``l2b_threshold`` is provided, pairs are additionally filtered to
    those whose row-local `attempt_n` narrow-normalized code distance is
    within that threshold.

    The pairwise A->B deltas are descriptive, not independent samples.
    The reported sign-test p-value is therefore computed over one mean
    discrimination delta per frozen match group, not over all directed
    pairs.
    """

    if view not in IDENTITY_DISCRIMINATION_VIEWS:
        raise FullTraceSuiteV2Error(
            f"Identity discrimination view must be one of {IDENTITY_DISCRIMINATION_VIEWS}, got {view!r}"
        )
    threshold = _validated_l2b_threshold(l2b_threshold)

    grouped_rows = _rows_grouped_by_l2a(rows)
    group_to_row_deltas: dict[tuple[str, str], list[float]] = {}

    self_scores: list[float] = []
    other_scores: list[float] = []
    paired: list[dict[str, Any]] = []
    per_row_sender: list[dict[str, Any]] = []
    per_scope: list[dict[str, Any]] = []
    n_pairs = 0

    for (scope, signature), scope_rows in grouped_rows.items():
        if len(scope_rows) < 2:
            continue
        distinct_students = len({row.key.student_id for row in scope_rows})
        if distinct_students < 2:
            continue
        for row_a in scope_rows:
            row_other_scores: list[float] = []
            for row_b in scope_rows:
                if row_a is row_b:
                    continue
                if row_a.key.student_id == row_b.key.student_id:
                    continue
                pair_code_distance = _pair_code_distance(
                    row_a,
                    row_b,
                    l2b_threshold=threshold,
                )
                if pair_code_distance is not None and pair_code_distance > threshold:
                    continue
                scored_against_b = score_full_trace_prediction_v2(
                    response_payload=row_a.response_payload,
                    observed_repair_target=row_b.observed_repair_target,
                    observed_coarse_path=row_b.observed_coarse_path,
                )
                self_score = float(row_a.scored["views"][family][metric][view])
                other_score = float(scored_against_b["views"][family][metric][view])
                self_scores.append(self_score)
                other_scores.append(other_score)
                row_other_scores.append(other_score)
                paired.append(
                    {
                        "scope": scope,
                        "match_group_signature": signature,
                        "row_a": row_a.custom_id,
                        "row_b": row_b.custom_id,
                        "self_score": self_score,
                        "other_score": other_score,
                        "discrim_delta": self_score - other_score,
                        "pair_code_distance": pair_code_distance,
                    }
                )
                n_pairs += 1
            if row_other_scores:
                mean_other_for_row = _mean(row_other_scores)
                row_delta = float(row_a.scored["views"][family][metric][view]) - mean_other_for_row
                group_to_row_deltas.setdefault((scope, signature), []).append(row_delta)
                per_row_sender.append(
                    {
                        "scope": scope,
                        "match_group_signature": signature,
                        "row_a": row_a.custom_id,
                        "self_score": float(row_a.scored["views"][family][metric][view]),
                        "mean_other_score": mean_other_for_row,
                        "discrim_delta": row_delta,
                        "n_other_rows": len(row_other_scores),
                    }
                )
    per_match_group: list[dict[str, Any]] = []
    scope_to_group_means: dict[str, list[float]] = {}
    for scope, signature in sorted(group_to_row_deltas):
        group_row_deltas = group_to_row_deltas[(scope, signature)]
        group_mean = _mean(group_row_deltas)
        per_match_group.append(
            {
                "scope": scope,
                "match_group_signature": signature,
                "mean_discrim_delta": group_mean,
                "median_discrim_delta": _median(group_row_deltas),
                "n_sender_rows": len(group_row_deltas),
            }
        )
        scope_to_group_means.setdefault(scope, []).append(group_mean)
    for scope in sorted(scope_to_group_means):
        group_means = scope_to_group_means[scope]
        per_scope.append(
            {
                "scope": scope,
                "mean_discrim_delta": _mean(group_means),
                "median_discrim_delta": _median(group_means),
                "n_match_groups": len(group_means),
            }
        )

    auc_numerator = 0
    ties = 0
    for pair in paired:
        if pair["self_score"] > pair["other_score"]:
            auc_numerator += 1
        elif pair["self_score"] == pair["other_score"]:
            ties += 1
    auc = (auc_numerator + 0.5 * ties) / n_pairs if n_pairs > 0 else float("nan")
    deltas = [pair["discrim_delta"] for pair in paired]
    match_group_mean_deltas = [float(entry["mean_discrim_delta"]) for entry in per_match_group]
    return {
        "family": family,
        "metric": metric,
        "view": view,
        **_analysis_identity(l2b_threshold=threshold),
        "n_pairs": n_pairs,
        "n_distinct_scopes_with_pairs": len(scope_to_group_means),
        "n_match_groups_with_pairs": len(per_match_group),
        "n_group_clusters": len(per_match_group),
        "n_scope_clusters": len(per_scope),
        "mean_self": _mean(self_scores),
        "mean_other": _mean(other_scores),
        "mean_discrim_delta": _mean(deltas),
        "median_discrim_delta": _median(deltas),
        "discrim_auc_self_gt_other": auc,
        "sign_test_p_two_sided": _sign_test_p_value(match_group_mean_deltas),
        "per_row_sender": per_row_sender,
        "per_match_group": per_match_group,
        "per_scope": per_scope,
        "pairs": paired,
    }


def compute_twin_prediction_similarity(
    rows: list[ScoredRow],
    *,
    family: str,
    metric: str,
    l2b_threshold: float | None = None,
) -> dict[str, Any]:
    """Peer-prediction similarity within the frozen L2A groups.

    For every directed pair ``(A, B)`` in the same frozen match group,
    compare A's top-1 prediction directly to B's top-1 prediction. This
    is the genericity diagnostic: if predictions collapse to a per-task
    prototype, pairwise similarity stays high even when self-vs-other
    still looks superficially good.

    Twin similarity is intentionally limited to prediction-to-prediction
    families that make sense without ground truth:

    * ``full_code`` via direct code similarity metrics
    * ``trajectory`` via direct coarse-path similarity metrics
    """

    threshold = _validated_l2b_threshold(l2b_threshold)
    grouped_rows = _rows_grouped_by_l2a(rows)
    group_to_similarities: dict[tuple[str, str], list[float]] = {}
    paired: list[dict[str, Any]] = []
    per_row_sender: list[dict[str, Any]] = []
    per_scope: list[dict[str, Any]] = []
    similarities: list[float] = []

    for (scope, signature), scope_rows in grouped_rows.items():
        if len(scope_rows) < 2:
            continue
        distinct_students = len({row.key.student_id for row in scope_rows})
        if distinct_students < 2:
            continue
        for row_a in scope_rows:
            sender_similarities: list[float] = []
            top1_a = _top1_hypothesis_from_row(row_a)
            for row_b in scope_rows:
                if row_a is row_b:
                    continue
                if row_a.key.student_id == row_b.key.student_id:
                    continue
                pair_code_distance = _pair_code_distance(
                    row_a,
                    row_b,
                    l2b_threshold=threshold,
                )
                if pair_code_distance is not None and pair_code_distance > threshold:
                    continue
                top1_b = _top1_hypothesis_from_row(row_b)
                if family == "full_code":
                    metric_values = _full_code_metrics(
                        before_code=str(row_a.observed_repair_target["attempt_n"]["code"]),
                        predicted_code=str(top1_a["predicted_next_code"]),
                        observed_code=str(top1_b["predicted_next_code"]),
                    )
                elif family == "trajectory":
                    metric_values = _trajectory_metrics(
                        predicted_steps=list(top1_a["predicted_next_trajectory"]),
                        observed_steps=list(top1_b["predicted_next_trajectory"]),
                        edit_window_radius=DEFAULT_EDIT_STEP_WINDOW_RADIUS,
                    )
                else:
                    raise FullTraceSuiteV2Error(
                        f"Twin prediction similarity does not support family {family!r}"
                    )
                if metric not in metric_values:
                    raise FullTraceSuiteV2Error(
                        f"Twin prediction similarity metric {metric!r} is not available in family {family!r}"
                    )
                similarity = float(metric_values[metric])
                similarities.append(similarity)
                sender_similarities.append(similarity)
                group_to_similarities.setdefault((scope, signature), []).append(similarity)
                paired.append(
                    {
                        "scope": scope,
                        "match_group_signature": signature,
                        "row_a": row_a.custom_id,
                        "row_b": row_b.custom_id,
                        "similarity": similarity,
                        "pair_code_distance": pair_code_distance,
                    }
                )
            if sender_similarities:
                per_row_sender.append(
                    {
                        "scope": scope,
                        "match_group_signature": signature,
                        "row_a": row_a.custom_id,
                        "mean_twin_similarity": _mean(sender_similarities),
                        "median_twin_similarity": _median(sender_similarities),
                        "n_other_rows": len(sender_similarities),
                    }
                )
    per_match_group: list[dict[str, Any]] = []
    scope_to_group_means: dict[str, list[float]] = {}
    for scope, signature in sorted(group_to_similarities):
        group_similarities = group_to_similarities[(scope, signature)]
        group_mean = _mean(group_similarities)
        per_match_group.append(
            {
                "scope": scope,
                "match_group_signature": signature,
                "mean_twin_similarity": group_mean,
                "median_twin_similarity": _median(group_similarities),
                "n_pairs": len(group_similarities),
            }
        )
        scope_to_group_means.setdefault(scope, []).append(group_mean)
    for scope in sorted(scope_to_group_means):
        group_means = scope_to_group_means[scope]
        per_scope.append(
            {
                "scope": scope,
                "mean_twin_similarity": _mean(group_means),
                "median_twin_similarity": _median(group_means),
                "n_match_groups": len(group_means),
            }
        )

    return {
        "family": family,
        "metric": metric,
        "view": "top_1_prediction_vs_top_1_prediction",
        **_analysis_identity(l2b_threshold=threshold),
        "n_pairs": len(paired),
        "n_distinct_scopes_with_pairs": len(scope_to_group_means),
        "n_match_groups_with_pairs": len(per_match_group),
        "mean_twin_similarity": _mean(similarities),
        "median_twin_similarity": _median(similarities),
        "per_row_sender": per_row_sender,
        "per_match_group": per_match_group,
        "per_scope": per_scope,
        "pairs": paired,
    }


def compute_reality_divergence(
    rows: list[ScoredRow],
    *,
    family: str,
    metric: str,
    l2b_threshold: float | None = None,
) -> dict[str, Any]:
    """Observed-next-attempt similarity within the frozen L2A groups.

    For every directed matched pair ``(A, B)``, wrap A's real next
    attempt as a synthetic top-1 response and score it against B's real
    next attempt through the standard scorer. This keeps the comparison
    in the same scorer space as Test 2A: a bounded prediction payload
    against B's raw observed target.

    Higher ``reality_similarity`` means matched peers truly make similar
    next moves on that metric. We intentionally do not derive a generic
    ``1 - similarity`` divergence term here because some scorer metrics
    (for example bounded gain) are not globally constrained to
    ``[0, 1]``.
    """

    threshold = _validated_l2b_threshold(l2b_threshold)
    grouped_rows = _rows_grouped_by_l2a(rows)
    group_to_similarities: dict[tuple[str, str], list[float]] = {}
    paired: list[dict[str, Any]] = []
    per_row_sender: list[dict[str, Any]] = []
    per_scope: list[dict[str, Any]] = []
    similarities: list[float] = []

    for (scope, signature), scope_rows in grouped_rows.items():
        if len(scope_rows) < 2:
            continue
        distinct_students = len({row.key.student_id for row in scope_rows})
        if distinct_students < 2:
            continue
        for row_a in scope_rows:
            sender_similarities: list[float] = []
            response_a = _observed_truth_as_response(row_a)
            for row_b in scope_rows:
                if row_a is row_b:
                    continue
                if row_a.key.student_id == row_b.key.student_id:
                    continue
                pair_code_distance = _pair_code_distance(
                    row_a,
                    row_b,
                    l2b_threshold=threshold,
                )
                if pair_code_distance is not None and pair_code_distance > threshold:
                    continue
                scored_against_b = score_full_trace_prediction_v2(
                    response_payload=response_a,
                    observed_repair_target=row_b.observed_repair_target,
                    observed_coarse_path=row_b.observed_coarse_path,
                )
                similarity = float(scored_against_b["views"][family][metric]["top_1"])
                similarities.append(similarity)
                sender_similarities.append(similarity)
                group_to_similarities.setdefault((scope, signature), []).append(similarity)
                paired.append(
                    {
                        "scope": scope,
                        "match_group_signature": signature,
                        "row_a": row_a.custom_id,
                        "row_b": row_b.custom_id,
                        "reality_similarity": similarity,
                        "pair_code_distance": pair_code_distance,
                    }
                )
            if sender_similarities:
                per_row_sender.append(
                    {
                        "scope": scope,
                        "match_group_signature": signature,
                        "row_a": row_a.custom_id,
                        "mean_reality_similarity": _mean(sender_similarities),
                        "median_reality_similarity": _median(sender_similarities),
                        "n_other_rows": len(sender_similarities),
                    }
                )
    per_match_group: list[dict[str, Any]] = []
    scope_to_group_means: dict[str, list[float]] = {}
    for scope, signature in sorted(group_to_similarities):
        group_similarities = group_to_similarities[(scope, signature)]
        group_mean = _mean(group_similarities)
        per_match_group.append(
            {
                "scope": scope,
                "match_group_signature": signature,
                "mean_reality_similarity": group_mean,
                "median_reality_similarity": _median(group_similarities),
                "n_pairs": len(group_similarities),
            }
        )
        scope_to_group_means.setdefault(scope, []).append(group_mean)
    for scope in sorted(scope_to_group_means):
        group_means = scope_to_group_means[scope]
        scope_mean = _mean(group_means)
        per_scope.append(
            {
                "scope": scope,
                "mean_reality_similarity": scope_mean,
                "median_reality_similarity": _median(group_means),
                "n_match_groups": len(group_means),
            }
        )

    return {
        "family": family,
        "metric": metric,
        "view": "observed_next_attempt_vs_observed_next_attempt",
        **_analysis_identity(l2b_threshold=threshold),
        "n_pairs": len(paired),
        "n_distinct_scopes_with_pairs": len(scope_to_group_means),
        "n_match_groups_with_pairs": len(per_match_group),
        "mean_reality_similarity": _mean(similarities),
        "median_reality_similarity": _median(similarities),
        "per_row_sender": per_row_sender,
        "per_match_group": per_match_group,
        "per_scope": per_scope,
        "pairs": paired,
    }


# ---------------------------------------------------------------------------
# Condition deltas (full vs no_trace vs trace_shuffled)
# ---------------------------------------------------------------------------


def paired_condition_delta(
    rows_by_condition: dict[str, list[ScoredRow]],
    *,
    family: str,
    metric: str,
    view: str,
    left: str,
    right: str,
) -> dict[str, Any]:
    """Paired delta ``left - right`` at matched row keys.

    Rows are paired on ``(class, student, exercise, task, attempt_n)``
    so a left/right comparison is always on the same transition.
    Unmatched rows are reported separately so the reader can tell
    whether a condition contrast is being inflated by the matched
    subset.
    """

    if left not in rows_by_condition or right not in rows_by_condition:
        raise FullTraceSuiteV2Error(
            f"Conditions must be present in rows_by_condition; got keys {sorted(rows_by_condition)}"
        )
    left_index = {row.custom_id: row for row in rows_by_condition[left]}
    right_index = {row.custom_id: row for row in rows_by_condition[right]}
    shared_ids = sorted(set(left_index) & set(right_index))
    unmatched_left = sorted(set(left_index) - set(right_index))
    unmatched_right = sorted(set(right_index) - set(left_index))
    deltas: list[float] = []
    per_row: list[dict[str, Any]] = []
    wins = 0
    for custom_id in shared_ids:
        left_score = float(left_index[custom_id].scored["views"][family][metric][view])
        right_score = float(right_index[custom_id].scored["views"][family][metric][view])
        delta = left_score - right_score
        deltas.append(delta)
        if delta > 0:
            wins += 1
        per_row.append(
            {
                "custom_id": custom_id,
                "left": left_score,
                "right": right_score,
                "delta": delta,
            }
        )
    return {
        "left": left,
        "right": right,
        "family": family,
        "metric": metric,
        "view": view,
        "n_pairs": len(shared_ids),
        "n_unmatched_left": len(unmatched_left),
        "n_unmatched_right": len(unmatched_right),
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
        "win_rate": wins / len(deltas) if deltas else float("nan"),
        "sign_test_p_two_sided": _sign_test_p_value(deltas),
        "per_row": per_row,
    }


# ---------------------------------------------------------------------------
# Sign-test p-value (no scipy)
# ---------------------------------------------------------------------------


def _log_binomial_coefficient(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _binomial_tail_two_sided(
    *,
    positives: int,
    n_nonzero: int,
) -> float:
    if n_nonzero == 0:
        return 1.0
    if positives < 0 or positives > n_nonzero:
        raise FullTraceSuiteV2Error("positives must be in [0, n_nonzero]")
    log_half = math.log(0.5)
    log_probs: list[float] = []
    for k in range(n_nonzero + 1):
        log_probs.append(_log_binomial_coefficient(n_nonzero, k) + n_nonzero * log_half)
    observed_log_prob = log_probs[positives]
    mass = 0.0
    for log_prob in log_probs:
        if log_prob <= observed_log_prob + 1e-12:
            mass += math.exp(log_prob)
    return min(1.0, mass)


def _sign_test_p_value(deltas: list[float]) -> float:
    """Two-sided exact sign test, zero-excluded.

    Null hypothesis: ``P(delta > 0) = P(delta < 0) = 0.5`` among
    non-zero deltas.  Returns ``NaN`` when there are zero non-zero
    deltas because the test has no resolution there.
    """

    positives = sum(1 for d in deltas if d > 0)
    negatives = sum(1 for d in deltas if d < 0)
    n_nonzero = positives + negatives
    if n_nonzero == 0:
        return float("nan")
    return _binomial_tail_two_sided(positives=positives, n_nonzero=n_nonzero)


# ---------------------------------------------------------------------------
# Scope-cluster bootstrap on saved pairwise analyses
# ---------------------------------------------------------------------------


def _require_matching_analysis_metadata(
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    *,
    fields: tuple[str, ...],
) -> None:
    for required_field in fields:
        if required_field not in left_result:
            raise FullTraceSuiteV2Error(
                f"Analysis metadata is missing required field {required_field!r} on the left result"
            )
        if required_field not in right_result:
            raise FullTraceSuiteV2Error(
                f"Analysis metadata is missing required field {required_field!r} on the right result"
            )
        left_value = left_result.get(required_field)
        right_value = right_result.get(required_field)
        if left_value != right_value:
            raise FullTraceSuiteV2Error(
                f"Analysis metadata mismatch on {required_field!r}: {left_value!r} != {right_value!r}"
            )


def _cluster_scope_summaries_from_pairs(
    result: dict[str, Any],
    *,
    value_key: str,
) -> tuple[dict[str, dict[str, float]], dict[str, set[tuple[str, str]]]]:
    pairs = result.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise FullTraceSuiteV2Error(
            f"Cluster bootstrap requires a non-empty pairs list for value {value_key!r}"
        )
    scope_summaries: dict[str, dict[str, float]] = {}
    scope_pair_ids: dict[str, set[tuple[str, str]]] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise FullTraceSuiteV2Error("Cluster bootstrap pair entries must be dicts")
        scope = pair.get("scope")
        if not isinstance(scope, str) or not scope:
            raise FullTraceSuiteV2Error(
                "Cluster bootstrap pair entries must carry a non-empty scope"
            )
        row_a = pair.get("row_a")
        row_b = pair.get("row_b")
        if not isinstance(row_a, str) or not row_a:
            raise FullTraceSuiteV2Error(
                "Cluster bootstrap pair entries must carry a non-empty row_a"
            )
        if not isinstance(row_b, str) or not row_b:
            raise FullTraceSuiteV2Error(
                "Cluster bootstrap pair entries must carry a non-empty row_b"
            )
        if value_key not in pair:
            raise FullTraceSuiteV2Error(
                f"Cluster bootstrap pair entry for scope {scope!r} is missing value {value_key!r}"
            )
        value = float(pair[value_key])
        if math.isnan(value):
            raise FullTraceSuiteV2Error(
                f"Cluster bootstrap pair entry for scope {scope!r} has NaN value for {value_key!r}"
            )
        summary = scope_summaries.setdefault(scope, {"sum": 0.0, "n_pairs": 0.0})
        pair_ids = scope_pair_ids.setdefault(scope, set())
        pair_id = (row_a, row_b)
        if pair_id in pair_ids:
            raise FullTraceSuiteV2Error(
                f"Cluster bootstrap pair list contains a duplicate directed pair in scope {scope!r}: "
                f"{row_a!r} -> {row_b!r}"
            )
        pair_ids.add(pair_id)
        summary["sum"] += value
        summary["n_pairs"] += 1.0
    if not scope_summaries:
        raise FullTraceSuiteV2Error(f"Cluster bootstrap found zero scopes for value {value_key!r}")
    return scope_summaries, scope_pair_ids


def _validate_identical_scope_clusters(
    cluster_maps: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    if not cluster_maps:
        raise FullTraceSuiteV2Error("Cluster bootstrap requires at least one cluster map")
    labels = list(cluster_maps)
    reference_label = labels[0]
    reference_scopes = set(cluster_maps[reference_label])
    if not reference_scopes:
        raise FullTraceSuiteV2Error(
            f"Cluster bootstrap cluster map {reference_label!r} has zero scopes"
        )
    for label in labels[1:]:
        scopes = set(cluster_maps[label])
        if scopes != reference_scopes:
            missing = sorted(reference_scopes - scopes)
            extra = sorted(scopes - reference_scopes)
            details: list[str] = []
            if missing:
                details.append(
                    f"missing scopes {missing[:5]!r}"
                    + ("" if len(missing) <= 5 else f" (+{len(missing) - 5} more)")
                )
            if extra:
                details.append(
                    f"unexpected scopes {extra[:5]!r}"
                    + ("" if len(extra) <= 5 else f" (+{len(extra) - 5} more)")
                )
            raise FullTraceSuiteV2Error(
                f"Cluster bootstrap requires identical scope sets; {label!r} differs from {reference_label!r}: "
                + "; ".join(details)
            )
    return sorted(reference_scopes)


def _validate_identical_scope_pair_sets(
    pair_maps: dict[str, dict[str, set[tuple[str, str]]]],
) -> list[str]:
    if not pair_maps:
        raise FullTraceSuiteV2Error("Cluster bootstrap requires at least one pair map")
    labels = list(pair_maps)
    reference_label = labels[0]
    reference_scopes = set(pair_maps[reference_label])
    if not reference_scopes:
        raise FullTraceSuiteV2Error(
            f"Cluster bootstrap pair map {reference_label!r} has zero scopes"
        )
    for label in labels[1:]:
        scopes = set(pair_maps[label])
        if scopes != reference_scopes:
            missing = sorted(reference_scopes - scopes)
            extra = sorted(scopes - reference_scopes)
            details: list[str] = []
            if missing:
                details.append(
                    f"missing scopes {missing[:5]!r}"
                    + ("" if len(missing) <= 5 else f" (+{len(missing) - 5} more)")
                )
            if extra:
                details.append(
                    f"unexpected scopes {extra[:5]!r}"
                    + ("" if len(extra) <= 5 else f" (+{len(extra) - 5} more)")
                )
            raise FullTraceSuiteV2Error(
                f"Cluster bootstrap requires identical scope sets; {label!r} differs from {reference_label!r}: "
                + "; ".join(details)
            )
    for scope in sorted(reference_scopes):
        reference_pairs = pair_maps[reference_label][scope]
        if not reference_pairs:
            raise FullTraceSuiteV2Error(
                f"Cluster bootstrap scope {scope!r} has zero directed pairs in {reference_label!r}"
            )
        for label in labels[1:]:
            pairs = pair_maps[label][scope]
            if pairs != reference_pairs:
                missing = sorted(reference_pairs - pairs)
                extra = sorted(pairs - reference_pairs)
                details: list[str] = []
                if missing:
                    details.append(
                        f"missing directed pairs {missing[:3]!r}"
                        + ("" if len(missing) <= 3 else f" (+{len(missing) - 3} more)")
                    )
                if extra:
                    details.append(
                        f"unexpected directed pairs {extra[:3]!r}"
                        + ("" if len(extra) <= 3 else f" (+{len(extra) - 3} more)")
                    )
                raise FullTraceSuiteV2Error(
                    "Cluster bootstrap requires identical directed peer pairs within each scope; "
                    f"{label!r} differs from {reference_label!r} in scope {scope!r}: "
                    + "; ".join(details)
                )
    return sorted(reference_scopes)


def _cluster_weighted_mean(
    scope_summaries: dict[str, dict[str, float]],
    sampled_scopes: list[str],
) -> float:
    total_sum = 0.0
    total_pairs = 0.0
    for scope in sampled_scopes:
        summary = scope_summaries[scope]
        total_sum += float(summary["sum"])
        total_pairs += float(summary["n_pairs"])
    if total_pairs <= 0.0:
        raise FullTraceSuiteV2Error("Cluster bootstrap sampled zero pair mass")
    return total_sum / total_pairs


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise FullTraceSuiteV2Error("Cannot take a percentile of zero values")
    if not 0.0 <= q <= 1.0:
        raise FullTraceSuiteV2Error(f"Percentile q must be in [0, 1], got {q!r}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    if lower_index == upper_index:
        return lower_value
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _scope_flip_randomization_test(
    weighted_scope_contributions: list[float],
    *,
    randomization_samples: int,
    seed: int,
) -> dict[str, Any]:
    if randomization_samples <= 0:
        raise FullTraceSuiteV2Error(
            f"randomization_samples must be positive, got {randomization_samples}"
        )
    if not weighted_scope_contributions:
        raise FullTraceSuiteV2Error(
            "Scope randomization test requires at least one scope contribution"
        )
    if any(math.isnan(value) for value in weighted_scope_contributions):
        raise FullTraceSuiteV2Error("Scope randomization test received NaN scope contributions")
    observed_abs = abs(sum(weighted_scope_contributions))
    tolerance = 1e-12
    n_scopes = len(weighted_scope_contributions)
    if n_scopes <= 20:
        total_randomizations = 1 << n_scopes
        extreme = 0
        for mask in range(total_randomizations):
            stat = 0.0
            for index, contribution in enumerate(weighted_scope_contributions):
                stat += contribution if (mask >> index) & 1 else -contribution
            if abs(stat) + tolerance >= observed_abs:
                extreme += 1
        return {
            "p_two_sided_against_zero": extreme / total_randomizations,
            "method": "exact_scope_sign_flip",
            "n_randomizations": total_randomizations,
        }

    rng = random.Random(seed)
    extreme = 0
    for _ in range(randomization_samples):
        stat = 0.0
        for contribution in weighted_scope_contributions:
            stat += contribution if rng.getrandbits(1) else -contribution
        if abs(stat) + tolerance >= observed_abs:
            extreme += 1
    return {
        "p_two_sided_against_zero": (extreme + 1.0) / (randomization_samples + 1.0),
        "method": "monte_carlo_scope_sign_flip",
        "n_randomizations": randomization_samples,
    }


def cluster_bootstrap_identity_condition_contrast(
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    *,
    left_label: str,
    right_label: str,
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Scope-cluster bootstrap for the Test 2A condition contrast.

    The statistic is the condition-level identity discrimination delta:
    ``mean_discrim_delta(left) - mean_discrim_delta(right)``.
    Scopes are resampled with replacement and each sampled scope carries
    all of its directed pair deltas.
    """

    if bootstrap_samples <= 0:
        raise FullTraceSuiteV2Error(f"bootstrap_samples must be positive, got {bootstrap_samples}")
    _require_matching_analysis_metadata(
        left_result,
        right_result,
        fields=("family", "metric", "view", "match_mode", "l2b_threshold"),
    )
    left_clusters, left_pairs = _cluster_scope_summaries_from_pairs(
        left_result,
        value_key="discrim_delta",
    )
    right_clusters, right_pairs = _cluster_scope_summaries_from_pairs(
        right_result,
        value_key="discrim_delta",
    )
    scope_list = _validate_identical_scope_pair_sets(
        {left_label: left_pairs, right_label: right_pairs}
    )
    if len(scope_list) < 2:
        raise FullTraceSuiteV2Error(
            f"Cluster bootstrap requires at least 2 scope clusters, got {len(scope_list)}"
        )
    left_mean = _cluster_weighted_mean(left_clusters, scope_list)
    right_mean = _cluster_weighted_mean(right_clusters, scope_list)
    observed_delta = left_mean - right_mean

    rng = random.Random(seed)
    bootstrap_deltas: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_scopes = [
            scope_list[rng.randrange(len(scope_list))] for _ in range(len(scope_list))
        ]
        sample_left_mean = _cluster_weighted_mean(left_clusters, sampled_scopes)
        sample_right_mean = _cluster_weighted_mean(right_clusters, sampled_scopes)
        bootstrap_deltas.append(sample_left_mean - sample_right_mean)
    bootstrap_deltas.sort()

    total_pairs = sum(float(left_clusters[scope]["n_pairs"]) for scope in scope_list)
    if total_pairs <= 0.0:
        raise FullTraceSuiteV2Error("Cluster bootstrap found zero total directed pairs")
    scope_summaries = []
    weighted_scope_contributions: list[float] = []
    for scope in scope_list:
        left_summary = left_clusters[scope]
        right_summary = right_clusters[scope]
        left_scope_mean = float(left_summary["sum"]) / float(left_summary["n_pairs"])
        right_scope_mean = float(right_summary["sum"]) / float(right_summary["n_pairs"])
        weighted_scope_contribution = (
            float(left_summary["sum"]) - float(right_summary["sum"])
        ) / total_pairs
        weighted_scope_contributions.append(weighted_scope_contribution)
        scope_summaries.append(
            {
                "scope": scope,
                "left_mean_discrim_delta": left_scope_mean,
                "right_mean_discrim_delta": right_scope_mean,
                "scope_delta_of_deltas": left_scope_mean - right_scope_mean,
                "pair_weighted_scope_contribution_to_delta": weighted_scope_contribution,
                "left_n_pairs": int(left_summary["n_pairs"]),
                "right_n_pairs": int(right_summary["n_pairs"]),
            }
        )
    if not math.isclose(
        sum(weighted_scope_contributions), observed_delta, rel_tol=0.0, abs_tol=1e-12
    ):
        raise FullTraceSuiteV2Error(
            "Internal error: pair-weighted scope contributions do not sum to the observed delta"
        )
    randomization = _scope_flip_randomization_test(
        weighted_scope_contributions,
        randomization_samples=bootstrap_samples,
        seed=seed,
    )

    return {
        "analysis": "test2a_identity_discrimination",
        "cluster_unit": "scope",
        "left_condition": left_label,
        "right_condition": right_label,
        "family": left_result["family"],
        "metric": left_result["metric"],
        "view": left_result["view"],
        "match_mode": left_result["match_mode"],
        "l2b_threshold": left_result["l2b_threshold"],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "n_scope_clusters": len(scope_list),
        "left_mean_discrim_delta": left_mean,
        "right_mean_discrim_delta": right_mean,
        "delta_of_deltas": observed_delta,
        "ci95_lower": _percentile(bootstrap_deltas, 0.025),
        "ci95_upper": _percentile(bootstrap_deltas, 0.975),
        "scope_flip_randomization_method": randomization["method"],
        "scope_flip_randomization_count": randomization["n_randomizations"],
        "scope_flip_randomization_p_two_sided_against_zero": randomization[
            "p_two_sided_against_zero"
        ],
        "bootstrap_prob_delta_gt_zero": sum(1 for value in bootstrap_deltas if value > 0.0)
        / len(bootstrap_deltas),
        "scope_summaries": scope_summaries,
    }


def cluster_bootstrap_overcollapse_condition_contrast(
    left_twin: dict[str, Any],
    right_twin: dict[str, Any],
    left_reality: dict[str, Any],
    right_reality: dict[str, Any],
    *,
    left_label: str,
    right_label: str,
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Scope-cluster bootstrap for the peer-reality-anchored Test 2B contrast.

    The statistic is the condition-level over-collapse delta:
    ``(mean_twin_similarity - mean_reality_similarity)_left - (same)_right``.
    """

    if bootstrap_samples <= 0:
        raise FullTraceSuiteV2Error(f"bootstrap_samples must be positive, got {bootstrap_samples}")
    _require_matching_analysis_metadata(
        left_twin,
        right_twin,
        fields=("family", "metric", "match_mode", "l2b_threshold"),
    )
    _require_matching_analysis_metadata(
        left_reality,
        right_reality,
        fields=("family", "metric", "match_mode", "l2b_threshold"),
    )
    _require_matching_analysis_metadata(
        left_twin,
        left_reality,
        fields=("family", "metric", "match_mode", "l2b_threshold"),
    )
    _require_matching_analysis_metadata(
        right_twin,
        right_reality,
        fields=("family", "metric", "match_mode", "l2b_threshold"),
    )
    left_twin_clusters, left_twin_pairs = _cluster_scope_summaries_from_pairs(
        left_twin,
        value_key="similarity",
    )
    right_twin_clusters, right_twin_pairs = _cluster_scope_summaries_from_pairs(
        right_twin,
        value_key="similarity",
    )
    left_reality_clusters, left_reality_pairs = _cluster_scope_summaries_from_pairs(
        left_reality,
        value_key="reality_similarity",
    )
    right_reality_clusters, right_reality_pairs = _cluster_scope_summaries_from_pairs(
        right_reality,
        value_key="reality_similarity",
    )
    scope_list = _validate_identical_scope_pair_sets(
        {
            f"{left_label}_twin": left_twin_pairs,
            f"{right_label}_twin": right_twin_pairs,
            f"{left_label}_reality": left_reality_pairs,
            f"{right_label}_reality": right_reality_pairs,
        }
    )
    if len(scope_list) < 2:
        raise FullTraceSuiteV2Error(
            f"Cluster bootstrap requires at least 2 scope clusters, got {len(scope_list)}"
        )

    left_twin_mean = _cluster_weighted_mean(left_twin_clusters, scope_list)
    right_twin_mean = _cluster_weighted_mean(right_twin_clusters, scope_list)
    left_reality_mean = _cluster_weighted_mean(left_reality_clusters, scope_list)
    right_reality_mean = _cluster_weighted_mean(right_reality_clusters, scope_list)
    left_overcollapse = left_twin_mean - left_reality_mean
    right_overcollapse = right_twin_mean - right_reality_mean
    observed_delta = left_overcollapse - right_overcollapse

    rng = random.Random(seed)
    bootstrap_deltas: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_scopes = [
            scope_list[rng.randrange(len(scope_list))] for _ in range(len(scope_list))
        ]
        sample_left_twin = _cluster_weighted_mean(left_twin_clusters, sampled_scopes)
        sample_right_twin = _cluster_weighted_mean(right_twin_clusters, sampled_scopes)
        sample_left_reality = _cluster_weighted_mean(left_reality_clusters, sampled_scopes)
        sample_right_reality = _cluster_weighted_mean(right_reality_clusters, sampled_scopes)
        bootstrap_deltas.append(
            (sample_left_twin - sample_left_reality) - (sample_right_twin - sample_right_reality)
        )
    bootstrap_deltas.sort()

    total_twin_pairs = sum(float(left_twin_clusters[scope]["n_pairs"]) for scope in scope_list)
    total_reality_pairs = sum(
        float(left_reality_clusters[scope]["n_pairs"]) for scope in scope_list
    )
    if total_twin_pairs <= 0.0 or total_reality_pairs <= 0.0:
        raise FullTraceSuiteV2Error(
            "Cluster bootstrap found zero total pair mass in over-collapse inputs"
        )
    scope_summaries = []
    weighted_scope_contributions: list[float] = []
    for scope in scope_list:
        left_twin_scope = left_twin_clusters[scope]
        right_twin_scope = right_twin_clusters[scope]
        left_reality_scope = left_reality_clusters[scope]
        right_reality_scope = right_reality_clusters[scope]
        left_twin_scope_mean = float(left_twin_scope["sum"]) / float(left_twin_scope["n_pairs"])
        right_twin_scope_mean = float(right_twin_scope["sum"]) / float(right_twin_scope["n_pairs"])
        left_reality_scope_mean = float(left_reality_scope["sum"]) / float(
            left_reality_scope["n_pairs"]
        )
        right_reality_scope_mean = float(right_reality_scope["sum"]) / float(
            right_reality_scope["n_pairs"]
        )
        weighted_scope_contribution = (
            float(left_twin_scope["sum"]) - float(right_twin_scope["sum"])
        ) / total_twin_pairs - (
            float(left_reality_scope["sum"]) - float(right_reality_scope["sum"])
        ) / total_reality_pairs
        weighted_scope_contributions.append(weighted_scope_contribution)
        scope_summaries.append(
            {
                "scope": scope,
                "left_overcollapse": left_twin_scope_mean - left_reality_scope_mean,
                "right_overcollapse": right_twin_scope_mean - right_reality_scope_mean,
                "scope_delta_of_overcollapse": (
                    left_twin_scope_mean
                    - left_reality_scope_mean
                    - right_twin_scope_mean
                    + right_reality_scope_mean
                ),
                "left_twin_mean": left_twin_scope_mean,
                "right_twin_mean": right_twin_scope_mean,
                "left_reality_mean": left_reality_scope_mean,
                "right_reality_mean": right_reality_scope_mean,
                "pair_weighted_scope_contribution_to_delta": weighted_scope_contribution,
                "left_n_pairs": int(left_twin_scope["n_pairs"]),
                "right_n_pairs": int(right_twin_scope["n_pairs"]),
            }
        )
    if not math.isclose(
        sum(weighted_scope_contributions), observed_delta, rel_tol=0.0, abs_tol=1e-12
    ):
        raise FullTraceSuiteV2Error(
            "Internal error: pair-weighted over-collapse contributions do not sum to the observed delta"
        )
    randomization = _scope_flip_randomization_test(
        weighted_scope_contributions,
        randomization_samples=bootstrap_samples,
        seed=seed,
    )

    return {
        "analysis": "test2b_peer_reality_anchored_overcollapse",
        "cluster_unit": "scope",
        "left_condition": left_label,
        "right_condition": right_label,
        "family": left_twin["family"],
        "metric": left_twin["metric"],
        "view": "top_1_prediction_vs_peer_reality",
        "match_mode": left_twin["match_mode"],
        "l2b_threshold": left_twin["l2b_threshold"],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "n_scope_clusters": len(scope_list),
        "left_mean_twin_similarity": left_twin_mean,
        "right_mean_twin_similarity": right_twin_mean,
        "left_mean_reality_similarity": left_reality_mean,
        "right_mean_reality_similarity": right_reality_mean,
        "left_overcollapse": left_overcollapse,
        "right_overcollapse": right_overcollapse,
        "delta_of_overcollapse": observed_delta,
        "ci95_lower": _percentile(bootstrap_deltas, 0.025),
        "ci95_upper": _percentile(bootstrap_deltas, 0.975),
        "scope_flip_randomization_method": randomization["method"],
        "scope_flip_randomization_count": randomization["n_randomizations"],
        "scope_flip_randomization_p_two_sided_against_zero": randomization[
            "p_two_sided_against_zero"
        ],
        "bootstrap_prob_delta_gt_zero": sum(1 for value in bootstrap_deltas if value > 0.0)
        / len(bootstrap_deltas),
        "scope_summaries": scope_summaries,
    }


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_scored_row(
    *,
    scored_prediction_json_path: Path,
    custom_id: str,
    condition: str,
    observed_repair_target: dict[str, Any],
    observed_coarse_path: dict[str, Any],
    response_payload: dict[str, Any],
) -> ScoredRow:
    """Lift one scored JSON artifact into a ``ScoredRow``.

    The caller supplies the already-loaded observed target files and
    response payload so this module stays independent of the existing
    bundle-loading utilities; tests and the CLI wrap this with the
    appropriate file IO.
    """

    scored_payload = json.loads(Path(scored_prediction_json_path).read_text(encoding="utf-8"))
    scored = scored_payload.get("scored_prediction", scored_payload)
    return ScoredRow(
        custom_id=custom_id,
        key=RowKey.from_custom_id(custom_id),
        condition=condition,
        scored=scored,
        response_payload=response_payload,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )


__all__ = [
    "FullTraceSuiteV2Error",
    "RowKey",
    "ScoredRow",
    "VIEW_NAMES",
    "FAMILY_NAMES",
    "aggregate_rows",
    "score_majority_baselines",
    "model_lift_over_baseline",
    "compute_identity_discrimination",
    "compute_reality_divergence",
    "compute_twin_prediction_similarity",
    "cluster_bootstrap_identity_condition_contrast",
    "cluster_bootstrap_overcollapse_condition_contrast",
    "paired_condition_delta",
    "load_scored_row",
]
