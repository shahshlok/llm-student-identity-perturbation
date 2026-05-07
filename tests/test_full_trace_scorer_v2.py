"""Tests for the revised v6.2 full-trace scorer and suite.

Each test targets one of the measurement-drift fixes called out in the
v1 audit:

* normalization consistency between repair footprint and gain
* hunk-aligned content similarity rejecting blob-style token bleed
* bounded gain keeping Expected numerically sane
* graded local-run count, region-overlap trajectory view
* rank-weighted aggregation matching handcrafted expectations
* demoted full-code structural similarity being reported as diagnostic
* baseline lift reporting zero on unique-scope rows
* identity discrimination returning sensible AUC on a hand-built case
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from identity_perturbation.prediction_audit.full_trace_scorer_v2 import (
    DEFAULT_EDIT_STEP_WINDOW_RADIUS,
    DEFAULT_FOOTPRINT_WINDOW_RADIUS,
    RANK_WEIGHTS,
    SCHEMA_VERSION,
    FullTraceScorerV2Error,
    _best_hunk_alignment,
    score_full_trace_prediction_v2,
)
from identity_perturbation.prediction_audit.full_trace_suite_v2 import (
    FullTraceSuiteV2Error,
    RowKey,
    ScoredRow,
    _project_observed_steps_to_predicted_trajectory,
    aggregate_rows,
    cluster_bootstrap_identity_condition_contrast,
    cluster_bootstrap_overcollapse_condition_contrast,
    compute_identity_discrimination,
    compute_reality_divergence,
    compute_twin_prediction_similarity,
    model_lift_over_baseline,
    paired_condition_delta,
    score_majority_baselines,
)
from identity_perturbation.prediction_audit.match_policy import narrow_normalize_code_for_match
from identity_perturbation.prediction_audit.report_full_trace_run_v2 import _l2b_threshold_key, _parse_l2b_thresholds
from identity_perturbation.prediction_audit.score_full_trace_bundle import (
    FullTraceBundleScoringError,
    _load_bundle_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


ATTEMPT_N_CODE = (
    "def solve(nums):\n"
    "    total = 0\n"
    "    for n in nums:\n"
    "        total = total + n\n"
    "    return total\n"
)

OBSERVED_NEXT_CODE = (
    "def solve(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total\n"
)

OBSERVED_STEPS = [
    {"action_type": "edit", "target_start_line_0idx": 3, "target_end_line_0idx": 3},
    {"action_type": "local_run", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
    {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
]


def _repair_target(before: str, after: str) -> dict[str, Any]:
    return {
        "schema_version": "v6_2_observed_next_repair_target_v1",
        "attempt_n": {"code": before},
        "attempt_n1": {"code": after},
    }


def _coarse_path(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "v6_2_observed_coarse_path_v1",
        "attempt_n1": {"coarse_path_steps": steps},
    }


def _response(
    *,
    hypotheses: list[tuple[str, float, str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    return {
        "schema_version": "v6_2_full_trace_prediction_v3",
        "hypotheses": [
            {
                "label": label,
                "estimated_probability": prob,
                "predicted_next_code": code,
                "predicted_next_trajectory": trajectory,
            }
            for label, prob, code, trajectory in hypotheses
        ],
    }


def _trivial_trajectory(edit_start: int = 3, edit_end: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "action_type": "edit",
            "target_start_line_0idx": edit_start,
            "target_end_line_0idx": edit_end,
        },
        {"action_type": "local_run", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
    ]


def _long_alternating_observed_steps(repeats: int = 36) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index in range(repeats):
        steps.append(
            {
                "action_type": "local_run",
                "target_start_line_0idx": -1,
                "target_end_line_0idx": -1,
            }
        )
        steps.append(
            {
                "action_type": "edit",
                "target_start_line_0idx": index % 4,
                "target_end_line_0idx": index % 4,
            }
        )
    steps.append(
        {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1}
    )
    return steps


def _three_copies(
    *,
    code: str,
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    return _response(hypotheses=[(f"h{i + 1}", 1.0 / 3.0, code, trajectory) for i in range(3)])


# ---------------------------------------------------------------------------
# Normalization consistency between repair footprint and gain
# ---------------------------------------------------------------------------


def test_footprint_is_whitespace_invariant_like_gain() -> None:
    """v1 bug: diff ran on raw strings, so a trailing space changed hunks.

    In v2 both halves normalize first.  A prediction that differs from
    the observation only in trailing whitespace should produce a
    perfect strict footprint F1 (empty diff both sides) and a perfect
    bounded gain.
    """

    noisy_observed = OBSERVED_NEXT_CODE.replace("total += n", "total += n  ")
    response = _three_copies(
        code=OBSERVED_NEXT_CODE,
        trajectory=_trivial_trajectory(),
    )
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=_repair_target(ATTEMPT_N_CODE, noisy_observed),
        observed_coarse_path=_coarse_path(OBSERVED_STEPS),
    )
    repair = scored["views"]["repair"]
    assert repair["strict_footprint_f1"]["top_1"] == pytest.approx(1.0)
    assert repair["windowed_footprint_f1"]["top_1"] == pytest.approx(1.0)
    assert repair["code_gain_over_copy_bounded"]["top_1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Hunk-aligned content similarity
# ---------------------------------------------------------------------------


def test_right_token_wrong_hunk_is_penalized_vs_correct_order() -> None:
    """Two-hunk observation: a prediction that swaps added text between
    hunks must score lower than the same two tokens placed in the right
    hunks.  v1's blob-F1 concatenated all added/removed text across
    hunks, so the swap scored identically to the correct order -- this
    test is the v2-vs-v1 divergence in miniature.
    """

    before = "alpha\nb\nc\nd\nbeta\n"
    observed = "A\nb\nc\nd\nB\n"
    correct_response = _three_copies(
        code=observed,
        trajectory=_trivial_trajectory(edit_start=0, edit_end=0),
    )
    swapped_response = _three_copies(
        code="B\nb\nc\nd\nA\n",
        trajectory=_trivial_trajectory(edit_start=0, edit_end=0),
    )
    correct_scored = score_full_trace_prediction_v2(
        response_payload=correct_response,
        observed_repair_target=_repair_target(before, observed),
        observed_coarse_path=_coarse_path(OBSERVED_STEPS),
    )
    swapped_scored = score_full_trace_prediction_v2(
        response_payload=swapped_response,
        observed_repair_target=_repair_target(before, observed),
        observed_coarse_path=_coarse_path(OBSERVED_STEPS),
    )
    correct = correct_scored["views"]["repair"]["aligned_content_f1"]["top_1"]
    swapped = swapped_scored["views"]["repair"]["aligned_content_f1"]["top_1"]
    assert swapped < correct, (
        "Aligned content F1 must penalize right-tokens-in-wrong-hunks; "
        f"correct={correct}, swapped={swapped}"
    )


def test_structural_and_identifier_f1_split_isolates_naming() -> None:
    """Structural F1 stays high when only identifier choice differs."""

    before = "def f():\n    result = 0\n"
    observed = "def f():\n    result = 1\n"
    predicted = "def f():\n    answer = 1\n"
    response = _three_copies(
        code=predicted,
        trajectory=_trivial_trajectory(edit_start=1, edit_end=1),
    )
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=_repair_target(before, observed),
        observed_coarse_path=_coarse_path(OBSERVED_STEPS),
    )
    structural = scored["views"]["repair"]["aligned_content_structural_f1"]["top_1"]
    identifier = scored["views"]["repair"]["aligned_content_identifier_f1"]["top_1"]
    assert structural > identifier, (
        "Naming-only difference should preserve structural agreement "
        f"while lowering identifier agreement; got structural={structural}, "
        f"identifier={identifier}"
    )


def test_exact_hunk_alignment_avoids_greedy_counterexample() -> None:
    """Regression test for the old greedy hunk matcher.

    These span sets form a small counterexample where greedy max-IoU
    matching leaves total overlap on the table. The exact matcher
    should recover the higher global optimum.
    """

    predicted_spans = [
        {-1},
        {-1, 0},
        {-1, 0, 1},
    ]
    observed_spans = [
        {-1},
        {-1, 0, 1, 2},
        {0, 1},
    ]
    matched, mean_iou = _best_hunk_alignment(predicted_spans, observed_spans)
    total_iou = sum(iou for _, _, iou in matched)
    assert len(matched) == 3
    assert total_iou == pytest.approx(13.0 / 6.0)
    assert mean_iou == pytest.approx(13.0 / 18.0)


# ---------------------------------------------------------------------------
# Bounded gain
# ---------------------------------------------------------------------------


def test_code_gain_bounded_clips_in_minus_one_one() -> None:
    before = "x = 1\n"
    observed = "x = 2\n"
    very_wrong = "y = " + ("0\n" * 200)
    response = _three_copies(
        code=very_wrong,
        trajectory=_trivial_trajectory(edit_start=0, edit_end=0),
    )
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=_repair_target(before, observed),
        observed_coarse_path=_coarse_path(OBSERVED_STEPS),
    )
    raw_gain = scored["views"]["repair"]["code_gain_over_copy_raw"]["top_1"]
    bounded_gain = scored["views"]["repair"]["code_gain_over_copy_bounded"]["top_1"]
    assert raw_gain < -10.0, "Raw gain should still be allowed to go very negative for diagnostics"
    assert -1.0 <= bounded_gain <= 0.0
    assert bounded_gain > -1.0 - 1e-9


def test_code_gain_bounded_reaches_one_on_exact_match() -> None:
    response = _three_copies(
        code=OBSERVED_NEXT_CODE,
        trajectory=_trivial_trajectory(),
    )
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=_repair_target(ATTEMPT_N_CODE, OBSERVED_NEXT_CODE),
        observed_coarse_path=_coarse_path(OBSERVED_STEPS),
    )
    assert scored["views"]["repair"]["code_gain_over_copy_bounded"]["top_1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Graded trajectory metrics
# ---------------------------------------------------------------------------


def test_local_run_count_agreement_is_graded_not_binary() -> None:
    one_run_traj = _trivial_trajectory()
    two_run_traj = [
        {"action_type": "edit", "target_start_line_0idx": 3, "target_end_line_0idx": 3},
        {"action_type": "local_run", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        {"action_type": "local_run", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
    ]
    response = _three_copies(code=OBSERVED_NEXT_CODE, trajectory=two_run_traj)
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=_repair_target(ATTEMPT_N_CODE, OBSERVED_NEXT_CODE),
        observed_coarse_path=_coarse_path(
            [
                one_run_traj[0],
                one_run_traj[1],
                {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
            ]
        ),
    )
    trajectory = scored["views"]["trajectory"]
    agreement = trajectory["local_run_count_agreement"]["top_1"]
    assert agreement == pytest.approx(0.5), (
        f"With pred=2 and obs=1, graded agreement should be 1 - 1/2 = 0.5; got {agreement}"
    )
    assert trajectory["local_run_presence_match"]["top_1"] == pytest.approx(1.0)


def test_region_overlap_unordered_views_catch_scrambled_order() -> None:
    """Both sides edit the same two regions but in opposite order.

    Alignment DP gives partial credit; region overlap should be 1.
    """

    before = "\n".join(f"L{i}" for i in range(10)) + "\n"
    observed = before.replace("L3", "X3").replace("L7", "X7")
    predicted = observed
    pred_steps = [
        {"action_type": "edit", "target_start_line_0idx": 7, "target_end_line_0idx": 7},
        {"action_type": "edit", "target_start_line_0idx": 3, "target_end_line_0idx": 3},
        {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
    ]
    obs_steps = [
        {"action_type": "edit", "target_start_line_0idx": 3, "target_end_line_0idx": 3},
        {"action_type": "edit", "target_start_line_0idx": 7, "target_end_line_0idx": 7},
        {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
    ]
    response = _three_copies(code=predicted, trajectory=pred_steps)
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=_repair_target(before, observed),
        observed_coarse_path=_coarse_path(obs_steps),
    )
    trajectory = scored["views"]["trajectory"]
    assert trajectory["edit_region_overlap_unordered"]["top_1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rank-weighted aggregation
# ---------------------------------------------------------------------------


def test_rank_weighted_uses_fixed_weights_on_probability_ranked_hypotheses() -> None:
    """Hand-compute rank_weighted for ``exact_next_code_match``.

    Hypothesis 2 is the correct code (score 1), h1 and h3 are wrong.
    Probabilities sort (h1=0.5, h2=0.3, h3=0.2) → rank order
    [h1, h2, h3].  Weights (0.5, 0.3, 0.2) applied to
    [score(h1), score(h2), score(h3)] = [0, 1, 0] gives 0.3.
    """

    wrong_code = OBSERVED_NEXT_CODE.replace("total += n", "total = total * n")
    response = _response(
        hypotheses=[
            ("h1_wrong", 0.5, wrong_code, _trivial_trajectory()),
            ("h2_right", 0.3, OBSERVED_NEXT_CODE, _trivial_trajectory()),
            ("h3_wrong", 0.2, wrong_code, _trivial_trajectory()),
        ]
    )
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=_repair_target(ATTEMPT_N_CODE, OBSERVED_NEXT_CODE),
        observed_coarse_path=_coarse_path(OBSERVED_STEPS),
    )
    exact_view = scored["views"]["full_code"]["exact_next_code_match"]
    assert exact_view["oracle_at_3"] == pytest.approx(1.0)
    assert exact_view["top_1"] == pytest.approx(0.0)
    assert exact_view["rank_weighted"] == pytest.approx(RANK_WEIGHTS[1])
    assert exact_view["best_hypothesis_rank"] == 2


# ---------------------------------------------------------------------------
# Full-code family demotion
# ---------------------------------------------------------------------------


def test_structural_lift_is_near_zero_when_prediction_equals_baseline() -> None:
    """A model that predicts ``attempt_n`` verbatim buys zero lift."""

    response = _three_copies(code=ATTEMPT_N_CODE, trajectory=_trivial_trajectory())
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=_repair_target(ATTEMPT_N_CODE, OBSERVED_NEXT_CODE),
        observed_coarse_path=_coarse_path(OBSERVED_STEPS),
    )
    assert scored["views"]["full_code"]["structural_lift_over_copy"]["top_1"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert scored["views"]["full_code"]["full_code_structural_similarity_diagnostic"]["top_1"] > 0.5


# ---------------------------------------------------------------------------
# Suite: baselines and discrimination
# ---------------------------------------------------------------------------


def _make_scored_row(
    *,
    custom_id: str,
    condition: str,
    attempt_n_code: str,
    observed_next_code: str,
    predicted_next_code: str,
    observed_steps: list[dict[str, Any]] | None = None,
    predicted_steps: list[dict[str, Any]] | None = None,
    attempt_n_pass_fail_vector: tuple[bool, ...] = (True,),
    attempt_n_normalized_code: str | None = None,
    response_payload: dict[str, Any] | None = None,
) -> ScoredRow:
    observed_steps = observed_steps if observed_steps is not None else OBSERVED_STEPS
    predicted_steps = predicted_steps if predicted_steps is not None else _trivial_trajectory()
    response = (
        response_payload
        if response_payload is not None
        else _three_copies(code=predicted_next_code, trajectory=predicted_steps)
    )
    observed_repair_target = _repair_target(attempt_n_code, observed_next_code)
    observed_coarse_path = _coarse_path(observed_steps)
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )
    return ScoredRow(
        custom_id=custom_id,
        key=RowKey.from_custom_id(custom_id),
        condition=condition,
        scored=scored,
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
        attempt_n_pass_fail_vector=attempt_n_pass_fail_vector,
        attempt_n_pass_vector_signature="".join(
            "P" if value else "F" for value in attempt_n_pass_fail_vector
        ),
        attempt_n_normalized_code=(
            narrow_normalize_code_for_match(attempt_n_code)
            if attempt_n_normalized_code is None
            else attempt_n_normalized_code
        ),
    )


def test_baselines_and_lift_on_multiple_students_same_exercise() -> None:
    """Two students on the same exercise with a shared true next-code.

    The majority baseline equals that next-code, so a model that
    also returns it should show zero lift.  A model that returns a
    different-but-correct variant should show positive lift on
    content similarity, demonstrating that lift reporting works.
    """

    row_perfect_a = _make_scored_row(
        custom_id="589:5897:6353:9735:1",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
    )
    row_perfect_b = _make_scored_row(
        custom_id="589:5897:6353:9735:2",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
    )
    baselines = score_majority_baselines([row_perfect_a, row_perfect_b])
    scope = row_perfect_a.key.exercise_scope
    assert scope in baselines
    assert baselines[scope]["unique_scope"] is False

    lift = model_lift_over_baseline(
        [row_perfect_a, row_perfect_b],
        baselines,
        family="full_code",
        metric="exact_next_code_match",
        view="top_1",
    )
    assert lift["mean_delta"] == pytest.approx(0.0)
    assert lift["win_rate"] == pytest.approx(0.0)


def test_baseline_lift_rescores_per_row_within_scope() -> None:
    """Different truths in one scope must not share one baseline score."""

    before = "x = 0\n"
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 1\n",
        predicted_next_code="x = 1\n",
        observed_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
        predicted_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 2\n",
        predicted_next_code="x = 2\n",
        observed_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
        predicted_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
    )
    baselines = score_majority_baselines([row_a, row_b])
    scope = row_a.key.exercise_scope
    per_row_scored = baselines[scope]["per_row_scored"]
    assert per_row_scored[row_a.custom_id]["views"]["full_code"]["exact_next_code_match"][
        "top_1"
    ] == pytest.approx(1.0)
    assert per_row_scored[row_b.custom_id]["views"]["full_code"]["exact_next_code_match"][
        "top_1"
    ] == pytest.approx(0.0)

    lift = model_lift_over_baseline(
        [row_a, row_b],
        baselines,
        family="full_code",
        metric="exact_next_code_match",
        view="top_1",
    )
    assert lift["mean_delta"] == pytest.approx(0.5)
    assert lift["win_rate"] == pytest.approx(0.5)


def test_unique_scope_row_is_flagged_and_delta_is_zero() -> None:
    row = _make_scored_row(
        custom_id="590:5843:1187:9880:1",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
    )
    baselines = score_majority_baselines([row])
    scope = row.key.exercise_scope
    assert baselines[scope]["unique_scope"] is True
    lift = model_lift_over_baseline(
        [row],
        baselines,
        family="full_code",
        metric="exact_next_code_match",
        view="top_1",
    )
    assert lift["n_unique_scope_rows"] == 1
    assert lift["mean_delta"] == pytest.approx(0.0)


def test_majority_baseline_projects_long_observed_trajectories_into_schema_budget() -> None:
    row_long = _make_scored_row(
        custom_id="590:7607:5629:1801:1",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
        observed_steps=_long_alternating_observed_steps(),
        predicted_steps=_trivial_trajectory(),
    )
    row_short = _make_scored_row(
        custom_id="590:9819:5629:1801:1",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
        observed_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
        predicted_steps=_trivial_trajectory(edit_start=0, edit_end=0),
    )

    baselines = score_majority_baselines([row_long, row_short])
    scope = row_long.key.exercise_scope
    baseline_response = baselines[scope]["baseline_response"]
    for hypothesis in baseline_response["hypotheses"]:
        assert len(hypothesis["predicted_next_trajectory"]) <= 8
        assert hypothesis["predicted_next_trajectory"][-1]["action_type"] == "submit"


def test_identity_discrimination_sees_self_better_than_other() -> None:
    """Two students on the same exercise with genuinely different next-code.

    Each model prediction exactly matches its own student's truth,
    so scoring A's prediction against B's truth must drop.  Mean
    self-score > mean other-score and AUC must equal 1.0.
    """

    before = "x = 0\n"
    truth_a = "x = 1\n"
    truth_b = "x = 2\n"
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code=truth_a,
        predicted_next_code=truth_a,
        observed_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
        predicted_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code=truth_b,
        predicted_next_code=truth_b,
        observed_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
        predicted_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
    )
    discrim = compute_identity_discrimination(
        [row_a, row_b],
        family="full_code",
        metric="exact_next_code_match",
    )
    assert discrim["n_pairs"] == 2
    assert discrim["n_scope_clusters"] == 1
    assert discrim["mean_self"] == pytest.approx(1.0)
    assert discrim["mean_other"] == pytest.approx(0.0)
    assert discrim["discrim_auc_self_gt_other"] == pytest.approx(1.0)
    assert discrim["sign_test_p_two_sided"] == pytest.approx(1.0)


def test_identity_discrimination_near_chance_when_model_is_generic() -> None:
    """If both students have the same true next-code and the model returns
    that generic answer, self-score equals other-score for every pair.
    AUC should be exactly 0.5 (all ties).
    """

    before = "x = 0\n"
    shared_truth = "x = 1\n"
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code=shared_truth,
        predicted_next_code=shared_truth,
        observed_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
        predicted_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code=shared_truth,
        predicted_next_code=shared_truth,
        observed_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
        predicted_steps=[
            {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
            {"action_type": "submit", "target_start_line_0idx": -1, "target_end_line_0idx": -1},
        ],
    )
    discrim = compute_identity_discrimination(
        [row_a, row_b],
        family="full_code",
        metric="exact_next_code_match",
    )
    assert discrim["n_pairs"] == 2
    assert discrim["discrim_auc_self_gt_other"] == pytest.approx(0.5)


def test_identity_discrimination_only_pairs_within_same_l2a_group() -> None:
    before = "x = 0\n"
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 1\n",
        predicted_next_code="x = 1\n",
        attempt_n_pass_fail_vector=(False, True),
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 2\n",
        predicted_next_code="x = 2\n",
        attempt_n_pass_fail_vector=(False, True),
    )
    row_c = _make_scored_row(
        custom_id="589:3333:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 3\n",
        predicted_next_code="x = 3\n",
        attempt_n_pass_fail_vector=(True, False),
    )

    discrim = compute_identity_discrimination(
        [row_a, row_b, row_c],
        family="full_code",
        metric="exact_next_code_match",
    )

    assert discrim["n_pairs"] == 2
    assert discrim["n_match_groups_with_pairs"] == 1
    assert discrim["n_group_clusters"] == 1
    assert {pair["row_b"] for pair in discrim["pairs"]} == {
        "589:1111:6353:9735:1",
        "589:2222:6353:9735:1",
    }


def test_twin_prediction_similarity_uses_same_l2a_group_only() -> None:
    before = "x = 0\n"
    shared_prediction = "x = 10\n"
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 1\n",
        predicted_next_code=shared_prediction,
        attempt_n_pass_fail_vector=(False, True),
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 2\n",
        predicted_next_code=shared_prediction,
        attempt_n_pass_fail_vector=(False, True),
    )
    row_c = _make_scored_row(
        custom_id="589:3333:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 3\n",
        predicted_next_code="x = 99\n",
        attempt_n_pass_fail_vector=(True, False),
    )

    twin = compute_twin_prediction_similarity(
        [row_a, row_b, row_c],
        family="full_code",
        metric="exact_next_code_match",
    )

    assert twin["n_pairs"] == 2
    assert twin["n_match_groups_with_pairs"] == 1
    assert twin["mean_twin_similarity"] == pytest.approx(1.0)
    assert {pair["row_b"] for pair in twin["pairs"]} == {
        "589:1111:6353:9735:1",
        "589:2222:6353:9735:1",
    }


def test_identity_discrimination_rejects_missing_l2a_metadata() -> None:
    response = _three_copies(code=OBSERVED_NEXT_CODE, trajectory=_trivial_trajectory())
    observed_repair_target = _repair_target(ATTEMPT_N_CODE, OBSERVED_NEXT_CODE)
    observed_coarse_path = _coarse_path(OBSERVED_STEPS)
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )
    row = ScoredRow(
        custom_id="589:1111:6353:9735:1",
        key=RowKey.from_custom_id("589:1111:6353:9735:1"),
        condition="full",
        scored=scored,
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )

    with pytest.raises(FullTraceSuiteV2Error, match="missing attempt_n_pass_vector_signature"):
        compute_identity_discrimination(
            [row],
            family="full_code",
            metric="exact_next_code_match",
        )


def test_twin_prediction_similarity_rejects_missing_l2a_metadata() -> None:
    response = _three_copies(code=OBSERVED_NEXT_CODE, trajectory=_trivial_trajectory())
    observed_repair_target = _repair_target(ATTEMPT_N_CODE, OBSERVED_NEXT_CODE)
    observed_coarse_path = _coarse_path(OBSERVED_STEPS)
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )
    row = ScoredRow(
        custom_id="589:1111:6353:9735:1",
        key=RowKey.from_custom_id("589:1111:6353:9735:1"),
        condition="full",
        scored=scored,
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )

    with pytest.raises(FullTraceSuiteV2Error, match="missing attempt_n_pass_vector_signature"):
        compute_twin_prediction_similarity(
            [row],
            family="full_code",
            metric="exact_next_code_match",
        )


def test_identity_discrimination_l2b_filters_pairs_by_code_distance() -> None:
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code="x=1\n",
        observed_next_code="x=2\n",
        predicted_next_code="x=2\n",
        attempt_n_pass_fail_vector=(False, True),
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code="x = 1\n",
        observed_next_code="x=2\n",
        predicted_next_code="x=2\n",
        attempt_n_pass_fail_vector=(False, True),
    )
    row_c = _make_scored_row(
        custom_id="589:3333:6353:9735:1",
        condition="full",
        attempt_n_code="totally_different = 999\n",
        observed_next_code="x=2\n",
        predicted_next_code="x=2\n",
        attempt_n_pass_fail_vector=(False, True),
    )

    discrim = compute_identity_discrimination(
        [row_a, row_b, row_c],
        family="full_code",
        metric="exact_next_code_match",
        l2b_threshold=0.30,
    )

    assert discrim["match_mode"] == "l2a_and_l2b"
    assert discrim["l2b_threshold"] == pytest.approx(0.30)
    assert discrim["n_pairs"] == 2
    assert {(pair["row_a"], pair["row_b"]) for pair in discrim["pairs"]} == {
        ("589:1111:6353:9735:1", "589:2222:6353:9735:1"),
        ("589:2222:6353:9735:1", "589:1111:6353:9735:1"),
    }


def test_identity_discrimination_l2b_rejects_missing_code_anchor() -> None:
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code="x=1\n",
        observed_next_code="x=2\n",
        predicted_next_code="x=2\n",
        attempt_n_pass_fail_vector=(False, True),
    )
    row_a.attempt_n_normalized_code = None
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code="x=1\n",
        observed_next_code="x=2\n",
        predicted_next_code="x=2\n",
        attempt_n_pass_fail_vector=(False, True),
    )

    with pytest.raises(FullTraceSuiteV2Error, match="missing attempt_n_normalized_code"):
        compute_identity_discrimination(
            [row_a, row_b],
            family="full_code",
            metric="exact_next_code_match",
            l2b_threshold=0.10,
        )


def test_reality_divergence_uses_same_l2a_group_only() -> None:
    before = "x = 0\n"
    shared_truth = "x = 1\n"
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code=shared_truth,
        predicted_next_code="unused_a\n",
        attempt_n_pass_fail_vector=(False, True),
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code=shared_truth,
        predicted_next_code="unused_b\n",
        attempt_n_pass_fail_vector=(False, True),
    )
    row_c = _make_scored_row(
        custom_id="589:3333:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 99\n",
        predicted_next_code="unused_c\n",
        attempt_n_pass_fail_vector=(True, False),
    )

    reality = compute_reality_divergence(
        [row_a, row_b, row_c],
        family="full_code",
        metric="exact_next_code_match",
    )

    assert reality["n_pairs"] == 2
    assert reality["n_match_groups_with_pairs"] == 1
    assert reality["mean_reality_similarity"] == pytest.approx(1.0)
    assert {pair["row_b"] for pair in reality["pairs"]} == {
        "589:1111:6353:9735:1",
        "589:2222:6353:9735:1",
    }


def test_reality_divergence_rejects_missing_l2a_metadata() -> None:
    response = _three_copies(code=OBSERVED_NEXT_CODE, trajectory=_trivial_trajectory())
    observed_repair_target = _repair_target(ATTEMPT_N_CODE, OBSERVED_NEXT_CODE)
    observed_coarse_path = _coarse_path(OBSERVED_STEPS)
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )
    row = ScoredRow(
        custom_id="589:1111:6353:9735:1",
        key=RowKey.from_custom_id("589:1111:6353:9735:1"),
        condition="full",
        scored=scored,
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )

    with pytest.raises(FullTraceSuiteV2Error, match="missing attempt_n_pass_vector_signature"):
        compute_reality_divergence(
            [row],
            family="full_code",
            metric="exact_next_code_match",
        )


def test_reality_divergence_normalizes_null_line_spans_in_observed_paths() -> None:
    before = "x = 0\n"
    observed_steps = [
        {"action_type": "edit", "target_start_line_0idx": 0, "target_end_line_0idx": 0},
        {"action_type": "local_run", "target_start_line_0idx": None, "target_end_line_0idx": None},
        {"action_type": "submit", "target_start_line_0idx": None, "target_end_line_0idx": None},
    ]
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 1\n",
        predicted_next_code="unused_a\n",
        observed_steps=observed_steps,
        attempt_n_pass_fail_vector=(False, True),
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 1\n",
        predicted_next_code="unused_b\n",
        observed_steps=observed_steps,
        attempt_n_pass_fail_vector=(False, True),
    )

    reality = compute_reality_divergence(
        [row_a, row_b],
        family="trajectory",
        metric="trajectory_alignment_score",
    )

    assert reality["n_pairs"] == 2
    assert reality["mean_reality_similarity"] == pytest.approx(1.0)


def test_reality_divergence_projects_long_observed_paths_to_model_contract() -> None:
    before = "x = 0\n"
    long_steps = _long_alternating_observed_steps()
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 1\n",
        predicted_next_code="unused_a\n",
        observed_steps=long_steps,
        attempt_n_pass_fail_vector=(False, True),
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 1\n",
        predicted_next_code="unused_b\n",
        observed_steps=long_steps,
        attempt_n_pass_fail_vector=(False, True),
    )

    reality = compute_reality_divergence(
        [row_a, row_b],
        family="trajectory",
        metric="trajectory_alignment_score",
    )
    expected_similarity = score_full_trace_prediction_v2(
        response_payload={
            "schema_version": "v6_2_full_trace_prediction_v3",
            "hypotheses": [
                {
                    "label": "observed_truth_h1",
                    "estimated_probability": 0.999998,
                    "predicted_next_code": "x = 1\n",
                    "predicted_next_trajectory": _project_observed_steps_to_predicted_trajectory(
                        long_steps
                    ),
                },
                {
                    "label": "observed_truth_h2",
                    "estimated_probability": 0.000001,
                    "predicted_next_code": "x = 1\n",
                    "predicted_next_trajectory": _project_observed_steps_to_predicted_trajectory(
                        long_steps
                    ),
                },
                {
                    "label": "observed_truth_h3",
                    "estimated_probability": 0.000001,
                    "predicted_next_code": "x = 1\n",
                    "predicted_next_trajectory": _project_observed_steps_to_predicted_trajectory(
                        long_steps
                    ),
                },
            ],
        },
        observed_repair_target=row_b.observed_repair_target,
        observed_coarse_path=row_b.observed_coarse_path,
    )["views"]["trajectory"]["trajectory_alignment_score"]["top_1"]

    assert reality["n_pairs"] == 2
    assert reality["mean_reality_similarity"] == pytest.approx(expected_similarity)


def test_identity_discrimination_counts_match_groups_separately_within_one_scope() -> None:
    before = "x = 0\n"
    rows = [
        _make_scored_row(
            custom_id="589:1111:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="x = 1\n",
            predicted_next_code="x = 1\n",
            attempt_n_pass_fail_vector=(False, True),
        ),
        _make_scored_row(
            custom_id="589:2222:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="x = 2\n",
            predicted_next_code="x = 2\n",
            attempt_n_pass_fail_vector=(False, True),
        ),
        _make_scored_row(
            custom_id="589:3333:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="y = 1\n",
            predicted_next_code="y = 1\n",
            attempt_n_pass_fail_vector=(True, False),
        ),
        _make_scored_row(
            custom_id="589:4444:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="y = 2\n",
            predicted_next_code="y = 2\n",
            attempt_n_pass_fail_vector=(True, False),
        ),
    ]

    discrim = compute_identity_discrimination(
        rows,
        family="full_code",
        metric="exact_next_code_match",
    )

    assert discrim["n_pairs"] == 4
    assert discrim["n_match_groups_with_pairs"] == 2
    assert discrim["n_group_clusters"] == 2
    assert discrim["n_scope_clusters"] == 1
    assert len(discrim["per_match_group"]) == 2
    assert {entry["match_group_signature"] for entry in discrim["per_match_group"]} == {"FP", "PF"}


def test_identity_discrimination_supports_expected_and_rank_weighted_views() -> None:
    before = "x = 0\n"
    wrong = "x = 99\n"
    response_a = _response(
        hypotheses=[
            ("a_top1_wrong", 0.7, wrong, _trivial_trajectory()),
            ("a_rank2_right", 0.2, "x = 1\n", _trivial_trajectory()),
            ("a_rank3_wrong", 0.1, wrong, _trivial_trajectory()),
        ]
    )
    response_b = _response(
        hypotheses=[
            ("b_top1_wrong", 0.7, wrong, _trivial_trajectory()),
            ("b_rank2_right", 0.2, "x = 2\n", _trivial_trajectory()),
            ("b_rank3_wrong", 0.1, wrong, _trivial_trajectory()),
        ]
    )
    row_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 1\n",
        predicted_next_code=wrong,
        attempt_n_pass_fail_vector=(False, True),
        response_payload=response_a,
    )
    row_b = _make_scored_row(
        custom_id="589:2222:6353:9735:1",
        condition="full",
        attempt_n_code=before,
        observed_next_code="x = 2\n",
        predicted_next_code=wrong,
        attempt_n_pass_fail_vector=(False, True),
        response_payload=response_b,
    )

    top1 = compute_identity_discrimination(
        [row_a, row_b],
        family="full_code",
        metric="exact_next_code_match",
        view="top_1",
    )
    expected = compute_identity_discrimination(
        [row_a, row_b],
        family="full_code",
        metric="exact_next_code_match",
        view="expected",
    )
    rank_weighted = compute_identity_discrimination(
        [row_a, row_b],
        family="full_code",
        metric="exact_next_code_match",
        view="rank_weighted",
    )

    assert top1["view"] == "top_1"
    assert top1["mean_self"] == pytest.approx(0.0)
    assert top1["mean_other"] == pytest.approx(0.0)
    assert expected["view"] == "expected"
    assert expected["mean_self"] == pytest.approx(0.2)
    assert expected["mean_other"] == pytest.approx(0.0)
    assert rank_weighted["view"] == "rank_weighted"
    assert rank_weighted["mean_self"] == pytest.approx(RANK_WEIGHTS[1])
    assert rank_weighted["mean_other"] == pytest.approx(0.0)


def test_identity_discrimination_rejects_unsupported_view() -> None:
    row = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
    )

    with pytest.raises(FullTraceSuiteV2Error, match="Identity discrimination view must be one of"):
        compute_identity_discrimination(
            [row],
            family="full_code",
            metric="exact_next_code_match",
            view="oracle_at_3",
        )


def test_twin_prediction_similarity_counts_match_groups_separately_within_one_scope() -> None:
    before = "x = 0\n"
    rows = [
        _make_scored_row(
            custom_id="589:1111:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="x = 1\n",
            predicted_next_code="shared_a\n",
            attempt_n_pass_fail_vector=(False, True),
        ),
        _make_scored_row(
            custom_id="589:2222:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="x = 2\n",
            predicted_next_code="shared_a\n",
            attempt_n_pass_fail_vector=(False, True),
        ),
        _make_scored_row(
            custom_id="589:3333:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="y = 1\n",
            predicted_next_code="shared_b\n",
            attempt_n_pass_fail_vector=(True, False),
        ),
        _make_scored_row(
            custom_id="589:4444:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="y = 2\n",
            predicted_next_code="shared_b\n",
            attempt_n_pass_fail_vector=(True, False),
        ),
    ]

    twin = compute_twin_prediction_similarity(
        rows,
        family="full_code",
        metric="exact_next_code_match",
    )

    assert twin["n_pairs"] == 4
    assert twin["n_match_groups_with_pairs"] == 2
    assert len(twin["per_match_group"]) == 2
    assert {entry["match_group_signature"] for entry in twin["per_match_group"]} == {"FP", "PF"}


def test_reality_divergence_counts_match_groups_separately_within_one_scope() -> None:
    before = "x = 0\n"
    rows = [
        _make_scored_row(
            custom_id="589:1111:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="x = 1\n",
            predicted_next_code="unused_1\n",
            attempt_n_pass_fail_vector=(False, True),
        ),
        _make_scored_row(
            custom_id="589:2222:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="x = 2\n",
            predicted_next_code="unused_2\n",
            attempt_n_pass_fail_vector=(False, True),
        ),
        _make_scored_row(
            custom_id="589:3333:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="y = 1\n",
            predicted_next_code="unused_3\n",
            attempt_n_pass_fail_vector=(True, False),
        ),
        _make_scored_row(
            custom_id="589:4444:6353:9735:1",
            condition="full",
            attempt_n_code=before,
            observed_next_code="y = 2\n",
            predicted_next_code="unused_4\n",
            attempt_n_pass_fail_vector=(True, False),
        ),
    ]

    reality = compute_reality_divergence(
        rows,
        family="full_code",
        metric="exact_next_code_match",
    )

    assert reality["n_pairs"] == 4
    assert reality["n_match_groups_with_pairs"] == 2
    assert len(reality["per_match_group"]) == 2
    assert {entry["match_group_signature"] for entry in reality["per_match_group"]} == {"FP", "PF"}


def _fake_pair_result(
    *,
    value_key: str,
    scope_to_values: dict[str, list[float]] | None = None,
    scope_to_pair_rows: dict[str, list[tuple[str, str, float]]] | None = None,
    family: str = "trajectory",
    metric: str = "trajectory_alignment_score",
    view: str = "top_1",
    include_analysis_metadata: bool = True,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    if scope_to_values is None and scope_to_pair_rows is None:
        raise ValueError("fake pair result requires scope_to_values or scope_to_pair_rows")
    if scope_to_values is not None and scope_to_pair_rows is not None:
        raise ValueError(
            "fake pair result accepts either scope_to_values or scope_to_pair_rows, not both"
        )
    if scope_to_values is not None:
        for scope, values in scope_to_values.items():
            for index, value in enumerate(values):
                pairs.append(
                    {
                        "scope": scope,
                        "row_a": f"{scope}:a:{index}",
                        "row_b": f"{scope}:b:{index}",
                        value_key: value,
                    }
                )
    else:
        assert scope_to_pair_rows is not None
        for scope, rows in scope_to_pair_rows.items():
            for row_a, row_b, value in rows:
                pairs.append(
                    {
                        "scope": scope,
                        "row_a": row_a,
                        "row_b": row_b,
                        value_key: value,
                    }
                )
    result = {
        "family": family,
        "metric": metric,
        "view": view,
        "pairs": pairs,
    }
    if include_analysis_metadata:
        result["match_mode"] = "l2a"
        result["l2b_threshold"] = None
    return result


TEST2B_SHARED_METRICS: tuple[tuple[str, str], ...] = (
    ("trajectory", "trajectory_alignment_score"),
    ("trajectory", "edit_region_overlap_unordered"),
    ("trajectory", "local_run_count_agreement"),
    ("full_code", "exact_next_code_match"),
)


def _fake_real_report_twin_entries(
    scope_to_values: dict[str, list[float]],
) -> list[dict[str, Any]]:
    entries = [
        _fake_pair_result(
            value_key="similarity",
            scope_to_values=scope_to_values,
            family=family,
            metric=metric,
            include_analysis_metadata=False,
        )
        for family, metric in TEST2B_SHARED_METRICS
    ]
    entries.append(
        _fake_pair_result(
            value_key="similarity",
            scope_to_values=scope_to_values,
            family="full_code",
            metric="full_code_structural_similarity_diagnostic",
            include_analysis_metadata=False,
        )
    )
    return entries


def _fake_real_report_reality_entries(
    scope_to_values: dict[str, list[float]],
) -> list[dict[str, Any]]:
    return [
        _fake_pair_result(
            value_key="reality_similarity",
            scope_to_values=scope_to_values,
            family=family,
            metric=metric,
            include_analysis_metadata=False,
        )
        for family, metric in TEST2B_SHARED_METRICS
    ]


def _clone_entries_with_l2b_metadata(
    entries: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    cloned_entries: list[dict[str, Any]] = []
    for entry in entries:
        cloned = copy.deepcopy(entry)
        cloned["match_mode"] = "l2a_and_l2b"
        cloned["l2b_threshold"] = threshold
        cloned_entries.append(cloned)
    return cloned_entries


def _with_fake_l2b_sections(report: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(report)
    thresholds = (0.2, 0.25, 0.3)
    default_scope_to_values = {
        "589:6353:9735": [0.40, 0.40],
        "590:6354:9782": [0.50, 0.50],
    }
    identity_source = cloned.get("identity_discrimination")
    if isinstance(identity_source, list) and identity_source:
        identity_entries = identity_source
    else:
        identity_entries = [
            _fake_pair_result(
                value_key="discrim_delta",
                scope_to_values=default_scope_to_values,
                include_analysis_metadata=False,
            )
        ]
    twin_source = cloned.get("twin_prediction_similarity")
    if isinstance(twin_source, list) and twin_source:
        twin_entries = twin_source
    else:
        twin_entries = _fake_real_report_twin_entries(default_scope_to_values)
    reality_source = cloned.get("reality_peer_similarity")
    if isinstance(reality_source, list) and reality_source:
        reality_entries = reality_source
    else:
        reality_entries = _fake_real_report_reality_entries(default_scope_to_values)

    cloned["l2b_thresholds"] = list(thresholds)
    cloned["identity_discrimination_l2b"] = {
        _l2b_threshold_key(threshold): _clone_entries_with_l2b_metadata(
            identity_entries,
            threshold=threshold,
        )
        for threshold in thresholds
    }
    cloned["twin_prediction_similarity_l2b"] = {
        _l2b_threshold_key(threshold): _clone_entries_with_l2b_metadata(
            twin_entries,
            threshold=threshold,
        )
        for threshold in thresholds
    }
    cloned["reality_peer_similarity_l2b"] = {
        _l2b_threshold_key(threshold): _clone_entries_with_l2b_metadata(
            reality_entries,
            threshold=threshold,
        )
        for threshold in thresholds
    }
    return cloned


def test_cluster_bootstrap_identity_condition_contrast_weights_scopes_by_pair_count() -> None:
    left = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_pair_rows={
            "589:6353:9735": [
                ("a1", "b1", 1.0),
                ("a2", "b2", 1.0),
                ("a3", "b3", 1.0),
            ],
            "590:6354:9782": [("c1", "d1", 0.0)],
        },
    )
    right = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_pair_rows={
            "589:6353:9735": [
                ("a1", "b1", 0.0),
                ("a2", "b2", 0.0),
                ("a3", "b3", 0.0),
            ],
            "590:6354:9782": [("c1", "d1", 1.0)],
        },
    )

    contrast = cluster_bootstrap_identity_condition_contrast(
        left,
        right,
        left_label="full",
        right_label="no_trace",
        bootstrap_samples=400,
        seed=7,
    )

    assert contrast["n_scope_clusters"] == 2
    assert contrast["delta_of_deltas"] == pytest.approx(0.5)
    assert contrast["ci95_lower"] < 0.0
    assert contrast["ci95_upper"] > 0.0
    assert contrast["scope_flip_randomization_p_two_sided_against_zero"] == pytest.approx(1.0)
    assert contrast["scope_summaries"][0][
        "pair_weighted_scope_contribution_to_delta"
    ] == pytest.approx(0.75)
    assert contrast["scope_summaries"][1][
        "pair_weighted_scope_contribution_to_delta"
    ] == pytest.approx(-0.25)


def test_cluster_bootstrap_identity_condition_contrast_rejects_pair_set_mismatch_within_scope() -> (
    None
):
    left = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_pair_rows={"589:6353:9735": [("a1", "b1", 1.0)]},
    )
    right = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_pair_rows={"589:6353:9735": [("other_a", "other_b", 0.0)]},
    )

    with pytest.raises(FullTraceSuiteV2Error, match="identical directed peer pairs"):
        cluster_bootstrap_identity_condition_contrast(
            left,
            right,
            left_label="full",
            right_label="no_trace",
            bootstrap_samples=100,
            seed=3,
        )


def test_cluster_bootstrap_overcollapse_condition_contrast_rejects_pair_set_mismatch() -> None:
    left_twin = _fake_pair_result(
        value_key="similarity",
        scope_to_pair_rows={"589:6353:9735": [("a1", "b1", 0.6)]},
    )
    right_twin = _fake_pair_result(
        value_key="similarity",
        scope_to_pair_rows={"589:6353:9735": [("a1", "b1", 0.5)]},
    )
    left_reality = _fake_pair_result(
        value_key="reality_similarity",
        scope_to_pair_rows={"589:6353:9735": [("x1", "y1", 0.2)]},
    )
    right_reality = _fake_pair_result(
        value_key="reality_similarity",
        scope_to_pair_rows={"589:6353:9735": [("x1", "y1", 0.2)]},
    )

    with pytest.raises(FullTraceSuiteV2Error, match="identical directed peer pairs"):
        cluster_bootstrap_overcollapse_condition_contrast(
            left_twin,
            right_twin,
            left_reality,
            right_reality,
            left_label="full",
            right_label="no_trace",
            bootstrap_samples=100,
            seed=11,
        )


def test_cluster_bootstrap_overcollapse_condition_contrast_weights_pair_mass() -> None:
    left_twin = _fake_pair_result(
        value_key="similarity",
        scope_to_pair_rows={
            "589:6353:9735": [
                ("a1", "b1", 0.8),
                ("a2", "b2", 0.8),
                ("a3", "b3", 0.8),
            ],
            "590:6354:9782": [("c1", "d1", 0.2)],
        },
    )
    right_twin = _fake_pair_result(
        value_key="similarity",
        scope_to_pair_rows={
            "589:6353:9735": [
                ("a1", "b1", 0.5),
                ("a2", "b2", 0.5),
                ("a3", "b3", 0.5),
            ],
            "590:6354:9782": [("c1", "d1", 0.4)],
        },
    )
    left_reality = _fake_pair_result(
        value_key="reality_similarity",
        scope_to_pair_rows={
            "589:6353:9735": [
                ("a1", "b1", 0.1),
                ("a2", "b2", 0.1),
                ("a3", "b3", 0.1),
            ],
            "590:6354:9782": [("c1", "d1", 0.0)],
        },
    )
    right_reality = _fake_pair_result(
        value_key="reality_similarity",
        scope_to_pair_rows={
            "589:6353:9735": [
                ("a1", "b1", 0.1),
                ("a2", "b2", 0.1),
                ("a3", "b3", 0.1),
            ],
            "590:6354:9782": [("c1", "d1", 0.1)],
        },
    )

    contrast = cluster_bootstrap_overcollapse_condition_contrast(
        left_twin,
        right_twin,
        left_reality,
        right_reality,
        left_label="full",
        right_label="no_trace",
        bootstrap_samples=400,
        seed=11,
    )

    assert contrast["n_scope_clusters"] == 2
    assert contrast["delta_of_overcollapse"] == pytest.approx(0.2)
    assert contrast["ci95_lower"] < 0.0
    assert contrast["ci95_upper"] > 0.0
    assert contrast["scope_flip_randomization_p_two_sided_against_zero"] == pytest.approx(1.0)
    assert contrast["scope_summaries"][0][
        "pair_weighted_scope_contribution_to_delta"
    ] == pytest.approx(0.225)
    assert contrast["scope_summaries"][1][
        "pair_weighted_scope_contribution_to_delta"
    ] == pytest.approx(-0.025)


def test_cluster_bootstrap_identity_condition_contrast_resamples_scopes() -> None:
    left = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_values={
            "589:6353:9735": [1.0, 1.0],
            "590:6354:9782": [1.0, 1.0],
        },
    )
    right = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_values={
            "589:6353:9735": [0.0, 0.0],
            "590:6354:9782": [0.0, 0.0],
        },
    )

    contrast = cluster_bootstrap_identity_condition_contrast(
        left,
        right,
        left_label="full",
        right_label="no_trace",
        bootstrap_samples=200,
        seed=7,
    )

    assert contrast["n_scope_clusters"] == 2
    assert contrast["delta_of_deltas"] == pytest.approx(1.0)
    assert contrast["ci95_lower"] == pytest.approx(1.0)
    assert contrast["ci95_upper"] == pytest.approx(1.0)
    assert contrast["scope_flip_randomization_p_two_sided_against_zero"] == pytest.approx(0.5)
    assert contrast["bootstrap_prob_delta_gt_zero"] == pytest.approx(1.0)
    assert len(contrast["scope_summaries"]) == 2


def test_cluster_bootstrap_overcollapse_condition_contrast_resamples_scopes() -> None:
    left_twin = _fake_pair_result(
        value_key="similarity",
        scope_to_values={
            "589:6353:9735": [0.60, 0.60],
            "590:6354:9782": [0.60, 0.60],
        },
    )
    right_twin = _fake_pair_result(
        value_key="similarity",
        scope_to_values={
            "589:6353:9735": [0.50, 0.50],
            "590:6354:9782": [0.50, 0.50],
        },
    )
    left_reality = _fake_pair_result(
        value_key="reality_similarity",
        scope_to_values={
            "589:6353:9735": [0.20, 0.20],
            "590:6354:9782": [0.20, 0.20],
        },
    )
    right_reality = _fake_pair_result(
        value_key="reality_similarity",
        scope_to_values={
            "589:6353:9735": [0.20, 0.20],
            "590:6354:9782": [0.20, 0.20],
        },
    )

    contrast = cluster_bootstrap_overcollapse_condition_contrast(
        left_twin,
        right_twin,
        left_reality,
        right_reality,
        left_label="full",
        right_label="no_trace",
        bootstrap_samples=200,
        seed=11,
    )

    assert contrast["n_scope_clusters"] == 2
    assert contrast["left_overcollapse"] == pytest.approx(0.40)
    assert contrast["right_overcollapse"] == pytest.approx(0.30)
    assert contrast["delta_of_overcollapse"] == pytest.approx(0.10)
    assert contrast["ci95_lower"] == pytest.approx(0.10)
    assert contrast["ci95_upper"] == pytest.approx(0.10)
    assert contrast["scope_flip_randomization_p_two_sided_against_zero"] == pytest.approx(0.5)
    assert contrast["bootstrap_prob_delta_gt_zero"] == pytest.approx(1.0)


def test_cluster_bootstrap_identity_condition_contrast_rejects_scope_mismatch() -> None:
    left = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_values={"589:6353:9735": [1.0]},
    )
    right = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_values={"590:6354:9782": [0.0]},
    )

    with pytest.raises(FullTraceSuiteV2Error, match="identical scope sets"):
        cluster_bootstrap_identity_condition_contrast(
            left,
            right,
            left_label="full",
            right_label="no_trace",
            bootstrap_samples=100,
            seed=3,
        )


def test_cluster_bootstrap_identity_condition_contrast_rejects_single_scope_cluster() -> None:
    left = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_values={"589:6353:9735": [1.0, 1.0]},
    )
    right = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_values={"589:6353:9735": [0.0, 0.0]},
    )

    with pytest.raises(FullTraceSuiteV2Error, match="at least 2 scope clusters"):
        cluster_bootstrap_identity_condition_contrast(
            left,
            right,
            left_label="full",
            right_label="no_trace",
            bootstrap_samples=100,
            seed=3,
        )


def test_cluster_bootstrap_overcollapse_condition_contrast_rejects_single_scope_cluster() -> None:
    left_twin = _fake_pair_result(
        value_key="similarity",
        scope_to_values={"589:6353:9735": [0.60, 0.60]},
    )
    right_twin = _fake_pair_result(
        value_key="similarity",
        scope_to_values={"589:6353:9735": [0.50, 0.50]},
    )
    left_reality = _fake_pair_result(
        value_key="reality_similarity",
        scope_to_values={"589:6353:9735": [0.20, 0.20]},
    )
    right_reality = _fake_pair_result(
        value_key="reality_similarity",
        scope_to_values={"589:6353:9735": [0.20, 0.20]},
    )

    with pytest.raises(FullTraceSuiteV2Error, match="at least 2 scope clusters"):
        cluster_bootstrap_overcollapse_condition_contrast(
            left_twin,
            right_twin,
            left_reality,
            right_reality,
            left_label="full",
            right_label="no_trace",
            bootstrap_samples=100,
            seed=11,
        )


# ---------------------------------------------------------------------------
# Aggregation and condition deltas
# ---------------------------------------------------------------------------


def test_aggregate_rows_returns_expected_shape() -> None:
    row = _make_scored_row(
        custom_id="589:5897:6353:9735:1",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
    )
    aggregate = aggregate_rows([row])
    assert aggregate["n_rows"] == 1
    repair = aggregate["families"]["repair"]
    assert "strict_footprint_f1" in repair
    assert "windowed_footprint_f1" in repair
    assert "aligned_content_f1" in repair
    assert "code_gain_over_copy_bounded" in repair
    trajectory = aggregate["families"]["trajectory"]
    assert "edit_region_overlap_unordered" in trajectory
    assert "local_run_count_agreement" in trajectory
    full_code = aggregate["families"]["full_code"]
    assert "structural_lift_over_copy" in full_code


def test_aggregate_rows_preserves_best_hypothesis_rank() -> None:
    wrong_code = OBSERVED_NEXT_CODE.replace("total += n", "total = total * n")
    response = _response(
        hypotheses=[
            ("h1_wrong", 0.5, wrong_code, _trivial_trajectory()),
            ("h2_right", 0.3, OBSERVED_NEXT_CODE, _trivial_trajectory()),
            ("h3_wrong", 0.2, wrong_code, _trivial_trajectory()),
        ]
    )
    observed_repair_target = _repair_target(ATTEMPT_N_CODE, OBSERVED_NEXT_CODE)
    observed_coarse_path = _coarse_path(OBSERVED_STEPS)
    scored = score_full_trace_prediction_v2(
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )
    row = ScoredRow(
        custom_id="589:5897:6353:9735:1",
        key=RowKey.from_custom_id("589:5897:6353:9735:1"),
        condition="full",
        scored=scored,
        response_payload=response,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )
    aggregate = aggregate_rows([row])
    best_rank = aggregate["families"]["full_code"]["exact_next_code_match"]["best_hypothesis_rank"]
    assert best_rank["mean"] == pytest.approx(2.0)
    assert best_rank["median"] == pytest.approx(2.0)


def test_paired_condition_delta_handles_matched_and_unmatched_rows() -> None:
    row_full = _make_scored_row(
        custom_id="589:5897:6353:9735:1",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
    )
    row_no_trace = _make_scored_row(
        custom_id="589:5897:6353:9735:1",
        condition="no_trace",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=ATTEMPT_N_CODE,
    )
    row_full_only = _make_scored_row(
        custom_id="589:5897:6353:9735:2",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
    )
    delta = paired_condition_delta(
        {"full": [row_full, row_full_only], "no_trace": [row_no_trace]},
        family="full_code",
        metric="exact_next_code_match",
        view="top_1",
        left="full",
        right="no_trace",
    )
    assert delta["n_pairs"] == 1
    assert delta["n_unmatched_left"] == 1
    assert delta["n_unmatched_right"] == 0
    assert delta["mean_delta"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_row_key_rejects_malformed_custom_id() -> None:
    with pytest.raises(FullTraceSuiteV2Error):
        RowKey.from_custom_id("only:three:parts")


def test_schema_version_exposed() -> None:
    assert SCHEMA_VERSION == "v6_2_full_trace_scored_prediction_v2"
    assert DEFAULT_FOOTPRINT_WINDOW_RADIUS >= 0
    assert DEFAULT_EDIT_STEP_WINDOW_RADIUS >= 0


def test_scorer_rejects_observed_code_equal_to_attempt_n() -> None:
    response = _three_copies(code=OBSERVED_NEXT_CODE, trajectory=_trivial_trajectory())
    with pytest.raises(FullTraceScorerV2Error):
        score_full_trace_prediction_v2(
            response_payload=response,
            observed_repair_target=_repair_target(ATTEMPT_N_CODE, ATTEMPT_N_CODE),
            observed_coarse_path=_coarse_path(OBSERVED_STEPS),
        )


def test_sign_test_p_value_is_one_on_split_outcomes() -> None:
    """With equal positives and negatives the two-sided sign-test p must
    be 1.  Verifies the no-scipy implementation behaves sanely."""

    row_tie_a = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="full",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
    )
    row_tie_b = _make_scored_row(
        custom_id="589:1111:6353:9735:1",
        condition="no_trace",
        attempt_n_code=ATTEMPT_N_CODE,
        observed_next_code=OBSERVED_NEXT_CODE,
        predicted_next_code=OBSERVED_NEXT_CODE,
    )
    delta = paired_condition_delta(
        {"full": [row_tie_a], "no_trace": [row_tie_b]},
        family="full_code",
        metric="exact_next_code_match",
        view="top_1",
        left="full",
        right="no_trace",
    )
    assert math.isnan(delta["sign_test_p_two_sided"]) or delta[
        "sign_test_p_two_sided"
    ] == pytest.approx(1.0)


def test_l2b_threshold_keys_do_not_collapse_distinct_values() -> None:
    thresholds = _parse_l2b_thresholds("0.201, 0.204")

    assert thresholds == pytest.approx((0.201, 0.204))
    assert [_l2b_threshold_key(value) for value in thresholds] == ["0.201", "0.204"]


def _write_report_bundle(
    *,
    base_dir: Path,
    custom_id: str,
    condition: str,
    observed_next_code: str,
    response_payload: dict[str, Any],
    attempt_n_pass_fail_vector: tuple[bool, ...] = (False, True),
    observed_steps: list[dict[str, Any]] | None = None,
    include_attempt_n_normalized_code: bool = True,
) -> tuple[Path, dict[str, Any]]:
    bundle_dir = base_dir / custom_id.replace(":", "_")
    bundle_dir.mkdir()
    observed_repair_target = _repair_target(ATTEMPT_N_CODE, observed_next_code)
    observed_coarse_path = _coarse_path(
        observed_steps if observed_steps is not None else OBSERVED_STEPS
    )
    (bundle_dir / "observed_next_repair_target.json").write_text(
        json.dumps(observed_repair_target, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "observed_next_coarse_path.json").write_text(
        json.dumps(observed_coarse_path, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "v6_2_full_trace_prototype_bundle_v6",
        "custom_id": custom_id,
        "condition": condition,
        "observed_next_repair_target_path": "observed_next_repair_target.json",
        "observed_next_coarse_path_path": "observed_next_coarse_path.json",
        "attempt_n_pass_fail_vector": list(attempt_n_pass_fail_vector),
        "attempt_n_pass_vector_signature": "".join(
            "P" if value else "F" for value in attempt_n_pass_fail_vector
        ),
    }
    if include_attempt_n_normalized_code:
        manifest["attempt_n_normalized_code"] = narrow_normalize_code_for_match(ATTEMPT_N_CODE)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    batch_item = {
        "id": f"batch_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "id": f"resp_{custom_id}",
                "status": "completed",
                "incomplete_details": None,
                "output": [
                    {
                        "id": f"msg_{custom_id}",
                        "type": "message",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(response_payload, ensure_ascii=False),
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    }
                ],
            },
        },
        "error": None,
    }
    return bundle_dir, batch_item


def test_cli_scores_synthetic_batch_output_jsonl(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    custom_id = "589:5897:6353:9735:1"
    observed_repair_target = _repair_target(ATTEMPT_N_CODE, OBSERVED_NEXT_CODE)
    observed_coarse_path = _coarse_path(OBSERVED_STEPS)
    (bundle_dir / "observed_next_repair_target.json").write_text(
        json.dumps(observed_repair_target, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "observed_next_coarse_path.json").write_text(
        json.dumps(observed_coarse_path, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "v6_2_full_trace_prototype_bundle_v6",
        "custom_id": custom_id,
        "condition": "full",
        "observed_next_repair_target_path": "observed_next_repair_target.json",
        "observed_next_coarse_path_path": "observed_next_coarse_path.json",
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    response_payload = _three_copies(code=OBSERVED_NEXT_CODE, trajectory=_trivial_trajectory())
    batch_body = {
        "id": "resp_cli_test",
        "status": "completed",
        "incomplete_details": None,
        "output": [
            {
                "id": "msg_cli_test",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(response_payload, ensure_ascii=False),
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ],
    }
    batch_item = {
        "id": "batch_req_cli_test",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": batch_body,
        },
        "error": None,
    }
    batch_output_jsonl = tmp_path / "output.jsonl"
    batch_output_jsonl.write_text(
        json.dumps(batch_item, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "scored_v2.json"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.score_full_trace_bundle_v2",
            "--bundle-dir",
            str(bundle_dir),
            "--batch-output-jsonl",
            str(batch_output_jsonl),
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert out_path.exists()
    scored_bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert scored_bundle["schema_version"] == "v6_2_full_trace_scored_bundle_v2"
    assert scored_bundle["scorer_schema_version"] == SCHEMA_VERSION
    assert scored_bundle["custom_id"] == custom_id


def test_run_report_cli_emits_reality_peer_similarity(tmp_path: Path) -> None:
    bundles_root = tmp_path / "bundles"
    bundles_root.mkdir()
    response_a = _three_copies(code="x = 1\n", trajectory=_trivial_trajectory())
    response_b = _three_copies(code="x = 2\n", trajectory=_trivial_trajectory())
    bundle_a, batch_item_a = _write_report_bundle(
        base_dir=bundles_root,
        custom_id="589:1111:6353:9735:1",
        condition="full",
        observed_next_code="x = 1\n",
        response_payload=response_a,
    )
    bundle_b, batch_item_b = _write_report_bundle(
        base_dir=bundles_root,
        custom_id="589:2222:6353:9735:1",
        condition="full",
        observed_next_code="x = 2\n",
        response_payload=response_b,
    )
    run_manifest = {
        "schema_version": "v6_2_full_trace_run_manifest_v1",
        "bundle_map": {
            "589:1111:6353:9735:1": {"bundle_dir": str(bundle_a)},
            "589:2222:6353:9735:1": {"bundle_dir": str(bundle_b)},
        },
    }
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "output.jsonl"
    output_jsonl.write_text(
        "\n".join(
            [
                json.dumps(batch_item_a, ensure_ascii=False),
                json.dumps(batch_item_b, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "report.json"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.report_full_trace_run_v2",
            "--run-manifest",
            str(run_manifest_path),
            "--output-jsonl",
            str(output_jsonl),
            "--condition",
            "full",
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[report] loading manifest:" in result.stderr
    assert "[report] identity discrimination:" in result.stderr
    assert "[report] writing report atomically:" in result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert "reality_peer_similarity" in report
    assert "reality_divergence" not in report
    assert report["l2b_thresholds"] == [
        0.02,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
    ]
    assert "identity_discrimination_l2b" in report
    assert "0.2" in report["identity_discrimination_l2b"]
    exact = next(
        entry
        for entry in report["reality_peer_similarity"]
        if entry["family"] == "full_code" and entry["metric"] == "exact_next_code_match"
    )
    assert exact["n_pairs"] == 2
    assert exact["mean_reality_similarity"] == pytest.approx(0.0)
    l2b_exact = next(
        entry
        for entry in report["identity_discrimination_l2b"]["0.2"]
        if entry["family"] == "full_code"
        and entry["metric"] == "exact_next_code_match"
        and entry["view"] == "top_1"
    )
    assert l2b_exact["n_pairs"] == 2
    assert l2b_exact["match_mode"] == "l2a_and_l2b"


def test_compare_condition_reports_cli_emits_cluster_bootstrap_artifact(tmp_path: Path) -> None:
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={
                        "589:6353:9735": [0.30, 0.30],
                        "590:6354:9782": [0.20, 0.20],
                    },
                    include_analysis_metadata=False,
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {
                    "589:6353:9735": [0.40, 0.40],
                    "590:6354:9782": [0.50, 0.50],
                }
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {
                    "589:6353:9735": [0.20, 0.20],
                    "590:6354:9782": [0.20, 0.20],
                }
            ),
        }
    )
    right_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "no_trace",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={
                        "589:6353:9735": [0.10, 0.10],
                        "590:6354:9782": [0.05, 0.05],
                    },
                    include_analysis_metadata=False,
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {
                    "589:6353:9735": [0.35, 0.35],
                    "590:6354:9782": [0.45, 0.45],
                }
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {
                    "589:6353:9735": [0.20, 0.20],
                    "590:6354:9782": [0.20, 0.20],
                }
            ),
        }
    )
    left_path = tmp_path / "full_report.json"
    right_path = tmp_path / "no_trace_report.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    out_path = tmp_path / "cluster_bootstrap.json"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(out_path),
            "--bootstrap-samples",
            "200",
            "--seed",
            "5",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Test 2A scope-cluster bootstrap" in result.stdout
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "v6_2_condition_cluster_bootstrap_report_v1"
    assert report["left_condition"] == "full"
    assert report["right_condition"] == "no_trace"
    assert report["l2b_thresholds"] == [0.2, 0.25, 0.3]
    assert len(report["identity_discrimination_cluster_bootstrap"]) == 1
    assert sorted(report["identity_discrimination_l2b_cluster_bootstrap"]) == ["0.2", "0.25", "0.3"]
    assert sorted(report["peer_reality_anchored_overcollapse_l2b_cluster_bootstrap"]) == [
        "0.2",
        "0.25",
        "0.3",
    ]
    assert len(report["peer_reality_anchored_overcollapse_cluster_bootstrap"]) == len(
        TEST2B_SHARED_METRICS
    )
    assert report["identity_discrimination_cluster_bootstrap"][0][
        "scope_flip_randomization_p_two_sided_against_zero"
    ] == pytest.approx(0.5)


def test_compare_condition_reports_cli_fails_on_missing_condition(tmp_path: Path) -> None:
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "identity_discrimination": [],
            "twin_prediction_similarity": [],
            "reality_peer_similarity": [],
        }
    )
    right_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "no_trace",
            "identity_discrimination": [],
            "twin_prediction_similarity": [],
            "reality_peer_similarity": [],
        }
    )
    left_path = tmp_path / "left_missing_condition.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "left report must carry a non-empty string condition" in (result.stderr + result.stdout)


def test_compare_condition_reports_cli_fails_when_identity_view_filter_removes_all_entries(
    tmp_path: Path,
) -> None:
    report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={"589:6353:9735": [0.30, 0.30]},
                    include_analysis_metadata=False,
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.40, 0.40], "590:6354:9782": [0.50, 0.50]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    right_path.write_text(
        json.dumps({**report, "condition": "no_trace"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
            "--identity-views",
            "expected",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing requested identity views" in (result.stderr + result.stdout)


def test_compare_condition_reports_cli_fails_when_any_requested_identity_view_is_missing(
    tmp_path: Path,
) -> None:
    report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={"589:6353:9735": [0.30, 0.30]},
                    include_analysis_metadata=False,
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.40, 0.40], "590:6354:9782": [0.50, 0.50]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    right_path.write_text(
        json.dumps({**report, "condition": "no_trace"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
            "--identity-views",
            "top_1,expected",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing requested identity views" in (result.stderr + result.stdout)


def test_compare_condition_reports_cli_honors_identity_view_filter_and_ignores_unrequested_view_drift(
    tmp_path: Path,
) -> None:
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={
                        "589:6353:9735": [0.30, 0.30],
                        "590:6354:9782": [0.20, 0.20],
                    },
                    view="top_1",
                    include_analysis_metadata=False,
                ),
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={
                        "589:6353:9735": [0.10, 0.10],
                        "590:6354:9782": [0.15, 0.15],
                    },
                    view="expected",
                    include_analysis_metadata=False,
                ),
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {
                    "589:6353:9735": [0.40, 0.40],
                    "590:6354:9782": [0.50, 0.50],
                }
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {
                    "589:6353:9735": [0.20, 0.20],
                    "590:6354:9782": [0.20, 0.20],
                }
            ),
        }
    )
    right_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "no_trace",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={
                        "589:6353:9735": [0.05, 0.05],
                        "590:6354:9782": [0.10, 0.10],
                    },
                    view="top_1",
                    include_analysis_metadata=False,
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {
                    "589:6353:9735": [0.35, 0.35],
                    "590:6354:9782": [0.45, 0.45],
                }
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {
                    "589:6353:9735": [0.20, 0.20],
                    "590:6354:9782": [0.20, 0.20],
                }
            ),
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    out_path = tmp_path / "out.json"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(out_path),
            "--identity-views",
            "top_1",
            "--bootstrap-samples",
            "200",
            "--seed",
            "5",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Test 2A scope-cluster bootstrap" in result.stdout
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["identity_views"] == ["top_1"]
    assert len(report["identity_discrimination_cluster_bootstrap"]) == 1
    assert report["identity_discrimination_cluster_bootstrap"][0]["view"] == "top_1"


def test_compare_condition_reports_cli_fails_on_pair_set_drift_within_shared_scope(
    tmp_path: Path,
) -> None:
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_pair_rows={"589:6353:9735": [("a1", "b1", 0.30)]},
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.40, 0.40], "590:6354:9782": [0.50, 0.50]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    right_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "no_trace",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_pair_rows={"589:6353:9735": [("other_a", "other_b", 0.10)]},
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.35, 0.35], "590:6354:9782": [0.45, 0.45]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "identical directed peer pairs within each scope" in (result.stderr + result.stdout)


def test_compare_condition_reports_cli_fails_on_reversed_pair_direction(tmp_path: Path) -> None:
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_pair_rows={"589:6353:9735": [("a1", "b1", 0.30)]},
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.40, 0.40], "590:6354:9782": [0.50, 0.50]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    right_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "no_trace",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_pair_rows={"589:6353:9735": [("b1", "a1", 0.10)]},
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.35, 0.35], "590:6354:9782": [0.45, 0.45]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "identical directed peer pairs within each scope" in (result.stderr + result.stdout)


def test_compare_condition_reports_cli_accepts_real_test2b_metric_inventory_with_twin_only_diagnostic(
    tmp_path: Path,
) -> None:
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={
                        "589:6353:9735": [0.30, 0.30],
                        "590:6354:9782": [0.20, 0.20],
                    },
                    include_analysis_metadata=False,
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.40, 0.40], "590:6354:9782": [0.50, 0.50]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    right_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "no_trace",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={
                        "589:6353:9735": [0.10, 0.10],
                        "590:6354:9782": [0.05, 0.05],
                    },
                    include_analysis_metadata=False,
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.35, 0.35], "590:6354:9782": [0.45, 0.45]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    out_path = tmp_path / "out.json"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(out_path),
            "--bootstrap-samples",
            "200",
            "--seed",
            "5",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Test 2B peer-reality-anchored over-collapse bootstrap" in result.stdout
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(report["peer_reality_anchored_overcollapse_cluster_bootstrap"]) == len(
        TEST2B_SHARED_METRICS
    )
    assert sorted(report["identity_discrimination_l2b_cluster_bootstrap"]) == ["0.2", "0.25", "0.3"]


def test_compare_condition_reports_cli_fails_on_l2b_threshold_section_mismatch(
    tmp_path: Path,
) -> None:
    report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={"589:6353:9735": [0.30, 0.30]},
                    include_analysis_metadata=False,
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.40, 0.40], "590:6354:9782": [0.50, 0.50]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    left_report = copy.deepcopy(report)
    right_report = copy.deepcopy(report)
    right_report["condition"] = "no_trace"
    del right_report["identity_discrimination_l2b"]["0.25"]

    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "threshold mismatch" in (result.stderr + result.stdout)


def test_compare_condition_reports_cli_fails_on_missing_twin_view_metadata(
    tmp_path: Path,
) -> None:
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={"589:6353:9735": [0.30, 0.30]},
                    include_analysis_metadata=False,
                )
            ],
            "twin_prediction_similarity": _fake_real_report_twin_entries(
                {"589:6353:9735": [0.40, 0.40], "590:6354:9782": [0.50, 0.50]}
            ),
            "reality_peer_similarity": _fake_real_report_reality_entries(
                {"589:6353:9735": [0.20, 0.20], "590:6354:9782": [0.20, 0.20]}
            ),
        }
    )
    right_report = copy.deepcopy(left_report)
    right_report["condition"] = "no_trace"
    left_report["twin_prediction_similarity"][0]["view"] = None
    left_report["twin_prediction_similarity_l2b"]["0.2"][0]["view"] = None

    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "left twin_prediction_similarity" in (result.stderr + result.stdout)
    assert "field 'view' must be a non-empty string" in (result.stderr + result.stdout)


def test_compare_condition_reports_cli_fails_cleanly_on_missing_analysis_metadata_field(
    tmp_path: Path,
) -> None:
    malformed_identity = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_values={"589:6353:9735": [0.30, 0.30]},
    )
    del malformed_identity["family"]
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [malformed_identity],
            "twin_prediction_similarity": [
                _fake_pair_result(
                    value_key="similarity",
                    scope_to_values={"589:6353:9735": [0.40, 0.40]},
                )
            ],
            "reality_peer_similarity": [
                _fake_pair_result(
                    value_key="reality_similarity",
                    scope_to_values={"589:6353:9735": [0.20, 0.20]},
                )
            ],
        }
    )
    right_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "no_trace",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={"589:6353:9735": [0.10, 0.10]},
                )
            ],
            "twin_prediction_similarity": [
                _fake_pair_result(
                    value_key="similarity",
                    scope_to_values={"589:6353:9735": [0.35, 0.35]},
                )
            ],
            "reality_peer_similarity": [
                _fake_pair_result(
                    value_key="reality_similarity",
                    scope_to_values={"589:6353:9735": [0.20, 0.20]},
                )
            ],
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "left identity_discrimination entry is missing required field 'family'" in (
        result.stderr + result.stdout
    )


def test_compare_condition_reports_cli_fails_cleanly_on_null_analysis_metadata_value(
    tmp_path: Path,
) -> None:
    malformed_identity = _fake_pair_result(
        value_key="discrim_delta",
        scope_to_values={"589:6353:9735": [0.30, 0.30]},
    )
    malformed_identity["family"] = None
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [malformed_identity],
            "twin_prediction_similarity": [
                _fake_pair_result(
                    value_key="similarity",
                    scope_to_values={"589:6353:9735": [0.40, 0.40]},
                )
            ],
            "reality_peer_similarity": [
                _fake_pair_result(
                    value_key="reality_similarity",
                    scope_to_values={"589:6353:9735": [0.20, 0.20]},
                )
            ],
        }
    )
    right_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "no_trace",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={"589:6353:9735": [0.10, 0.10]},
                )
            ],
            "twin_prediction_similarity": [
                _fake_pair_result(
                    value_key="similarity",
                    scope_to_values={"589:6353:9735": [0.35, 0.35]},
                )
            ],
            "reality_peer_similarity": [
                _fake_pair_result(
                    value_key="reality_similarity",
                    scope_to_values={"589:6353:9735": [0.20, 0.20]},
                )
            ],
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "left identity_discrimination entry field 'family' must be a non-empty string" in (
        result.stderr + result.stdout
    )


def test_compare_condition_reports_cli_fails_on_null_report_section(tmp_path: Path) -> None:
    left_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": None,
            "twin_prediction_similarity": [],
            "reality_peer_similarity": [],
        }
    )
    right_report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "no_trace",
            "identity_discrimination": [],
            "twin_prediction_similarity": [],
            "reality_peer_similarity": [],
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(
        json.dumps(left_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    right_path.write_text(
        json.dumps(right_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "left report section 'identity_discrimination' must be a list" in (
        result.stderr + result.stdout
    )


def test_compare_condition_reports_cli_fails_when_test2b_metrics_are_empty(tmp_path: Path) -> None:
    report = _with_fake_l2b_sections(
        {
            "schema_version": "v6_2_full_trace_run_report_v2",
            "condition": "full",
            "identity_discrimination": [
                _fake_pair_result(
                    value_key="discrim_delta",
                    scope_to_values={
                        "589:6353:9735": [0.30, 0.30],
                        "590:6354:9782": [0.20, 0.20],
                    },
                )
            ],
            "twin_prediction_similarity": [],
            "reality_peer_similarity": [],
        }
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    right_path.write_text(
        json.dumps({**report, "condition": "no_trace"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.compare_condition_reports_v2",
            "--left-report",
            str(left_path),
            "--right-report",
            str(right_path),
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No peer-reality-anchored over-collapse metrics are available to compare" in (
        result.stderr + result.stdout
    )


def test_run_report_cli_fails_on_missing_l2b_anchor_in_manifest(tmp_path: Path) -> None:
    bundles_root = tmp_path / "bundles"
    bundles_root.mkdir()
    response_a = _three_copies(code="x = 1\n", trajectory=_trivial_trajectory())
    response_b = _three_copies(code="x = 2\n", trajectory=_trivial_trajectory())
    bundle_a, batch_item_a = _write_report_bundle(
        base_dir=bundles_root,
        custom_id="589:1111:6353:9735:1",
        condition="full",
        observed_next_code="x = 1\n",
        response_payload=response_a,
        include_attempt_n_normalized_code=False,
    )
    bundle_b, batch_item_b = _write_report_bundle(
        base_dir=bundles_root,
        custom_id="589:2222:6353:9735:1",
        condition="full",
        observed_next_code="x = 2\n",
        response_payload=response_b,
        include_attempt_n_normalized_code=False,
    )
    run_manifest = {
        "schema_version": "v6_2_full_trace_run_manifest_v1",
        "bundle_map": {
            "589:1111:6353:9735:1": {"bundle_dir": str(bundle_a)},
            "589:2222:6353:9735:1": {"bundle_dir": str(bundle_b)},
        },
    }
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "output.jsonl"
    output_jsonl.write_text(
        "\n".join(
            [
                json.dumps(batch_item_a, ensure_ascii=False),
                json.dumps(batch_item_b, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "report.json"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.report_full_trace_run_v2",
            "--run-manifest",
            str(run_manifest_path),
            "--output-jsonl",
            str(output_jsonl),
            "--condition",
            "full",
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing non-empty attempt_n_normalized_code in manifest" in (
        result.stderr + result.stdout
    )


def test_run_report_cli_fails_on_missing_output_row(tmp_path: Path) -> None:
    bundles_root = tmp_path / "bundles"
    bundles_root.mkdir()
    response = _three_copies(code="x = 1\n", trajectory=_trivial_trajectory())
    bundle_a, batch_item_a = _write_report_bundle(
        base_dir=bundles_root,
        custom_id="589:1111:6353:9735:1",
        condition="full",
        observed_next_code="x = 1\n",
        response_payload=response,
    )
    bundle_b, _batch_item_b = _write_report_bundle(
        base_dir=bundles_root,
        custom_id="589:2222:6353:9735:1",
        condition="full",
        observed_next_code="x = 2\n",
        response_payload=response,
    )
    run_manifest = {
        "schema_version": "v6_2_full_trace_run_manifest_v1",
        "bundle_map": {
            "589:1111:6353:9735:1": {"bundle_dir": str(bundle_a)},
            "589:2222:6353:9735:1": {"bundle_dir": str(bundle_b)},
        },
    }
    run_manifest_path = tmp_path / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_jsonl = tmp_path / "output.jsonl"
    output_jsonl.write_text(
        json.dumps(batch_item_a, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "report.json"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "identity_perturbation.prediction_audit.report_full_trace_run_v2",
            "--run-manifest",
            str(run_manifest_path),
            "--output-jsonl",
            str(output_jsonl),
            "--condition",
            "full",
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not match frozen manifest" in (result.stderr + result.stdout)


def test_bundle_manifest_v6_requires_condition(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    manifest = {
        "schema_version": "v6_2_full_trace_prototype_bundle_v6",
        "custom_id": "589:5897:6353:9735:1",
        "observed_next_repair_target_path": "observed_next_repair_target.json",
        "observed_next_coarse_path_path": "observed_next_coarse_path.json",
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FullTraceBundleScoringError, match="requires a non-empty condition"):
        _load_bundle_manifest(bundle_dir)
