# v6.2 Full-Trace Scoring Contract v2

## Status

Working contract for the hardened `*_v2` scorer/suite in `src/identity_perturbation/prediction_audit/`.

This document defines:
- the per-row metric families
- the aggregation views over 3 hypotheses
- the multi-row baseline and discrimination analyses
- the interpretation boundary for the new fields

Reference implementation:
- `src/identity_perturbation/prediction_audit/full_trace_scorer_v2.py`
- `src/identity_perturbation/prediction_audit/full_trace_suite_v2.py`
- `src/identity_perturbation/prediction_audit/score_full_trace_bundle_v2.py`
- `src/identity_perturbation/prediction_audit/report_full_trace_run_v2.py`

Current verification on `v6.1-metricsv2`:
- `uv run pytest tests/test_full_trace_scorer_v2.py` passes (`23` tests)
- recovered pilot report artifact:
  - `data/v62/batch_runs/full_trace_same_task_family_2plus_pilot10_v2/scores_v2/report.json`
  - `9/10` pilot rows scored because one raw batch output line is missing

## Per-Row Scoring

The model still returns exactly 3 hypotheses, each with:
- `estimated_probability`
- `predicted_next_code`
- `predicted_next_trajectory`

The scorer still returns:
- per-hypothesis scores
- aggregated views over the 3 hypotheses

Schema version:
- `v6_2_full_trace_scored_prediction_v2`

## Repair Family

Repair scoring is still in the `attempt_n` coordinate frame, but v2 changes the mechanics.

### Repair Changes

- diff on narrow-normalized code, not raw code
- keep strict footprint scoring for exact locality
- add windowed footprint scoring for near-miss locality
- align repair hunks with exact max-sum-IoU matching
- score repair content per aligned hunk, not blob-concatenated text
- split aligned content into:
  - `aligned_content_structural_f1`
  - `aligned_content_identifier_f1`
- keep raw code gain for diagnostics
- use bounded code gain for primary aggregation

### Repair Metrics

- `strict_footprint_precision`
- `strict_footprint_recall`
- `strict_footprint_f1`
- `windowed_footprint_precision`
- `windowed_footprint_recall`
- `windowed_footprint_f1`
- `windowed_footprint_mean_iou`
- `aligned_content_f1`
- `aligned_content_structural_f1`
- `aligned_content_identifier_f1`
- `code_gain_over_copy_raw`
- `code_gain_over_copy_bounded`

### Interpretation Boundary

- `strict_footprint_*` is the exact locality view
- `windowed_footprint_*` is the tolerant locality view
- `aligned_content_identifier_f1` is the most sensitive content metric for student-specific naming choices
- `code_gain_over_copy_bounded` is the main gain metric for aggregation
- `code_gain_over_copy_raw` is diagnostic only

## Trajectory Family

Trajectory scoring still excludes the terminal `submit` from the credit-bearing path.

### Trajectory Metrics

- `trajectory_alignment_score`
- `edit_span_overlap`
- `edit_region_overlap_unordered`
- `local_run_presence_match`
- `local_run_count_agreement`

### Interpretation Boundary

- `trajectory_alignment_score` is the ordered path view
- `edit_region_overlap_unordered` is the order-insensitive “same region” view
- `local_run_count_agreement` replaces the old binary count match with graded agreement

## Full-Code Family

Whole-file metrics remain visible, but v2 makes the “lift over doing nothing” explicit.

### Full-Code Metrics

- `exact_next_code_match`
- `structural_lift_over_copy`
- `full_code_structural_similarity_diagnostic`
- `worse_than_copy_rate`

### Interpretation Boundary

- `exact_next_code_match` is still the strict final-file view
- `structural_lift_over_copy` is the main non-exact whole-file metric
- `full_code_structural_similarity_diagnostic` is diagnostic only

## Aggregation Views Over 3 Hypotheses

v2 keeps the original views and adds one rank-based view.

### Views

- `oracle_at_3`
- `expected`
- `rank_weighted`
- `top_1`
- `best_hypothesis_rank`

### Interpretation Boundary

- `oracle_at_3` is candidate-support coverage
- `expected` uses the model's reported probabilities directly
- `rank_weighted` uses fixed weights `(0.5, 0.3, 0.2)` on probability rank order
- `top_1` is the deployed-answer view
- `best_hypothesis_rank` is the selection-quality view

## Multi-Row Baseline Analysis

v2 adds a trace-blind majority baseline per exercise scope.

### Baseline Rule

For each scope `(class_id, exercise_id, task_id)`:
- majority code = most common observed `attempt_n1.code`
- majority trajectory = most common observed coarse action-type shape

Important implementation rule:
- the baseline prediction is scope-level
- the baseline score must be recomputed per row against that row's observed target

This prevents one representative-row score from being reused across all rows in the scope.

## Identity Discrimination

v2 keeps the A-vs-B same-scope discrimination analysis, but the inferential boundary is stricter.

### Descriptive Outputs

- `mean_self`
- `mean_other`
- `mean_discrim_delta`
- `median_discrim_delta`
- `discrim_auc_self_gt_other`

### Statistical Rule

Directed A->B pairs are not independent observations.

So:
- pairwise deltas remain descriptive
- the reported `sign_test_p_two_sided` is computed over one mean discrimination delta per scope

Supporting outputs:
- `n_scope_clusters`
- `per_row_sender`
- `per_scope`
- `pairs`

## CLI

Bundle scorer entrypoint:
- `python -m identity_perturbation.prediction_audit.score_full_trace_bundle_v2`

Accepted response sources:
- `--prediction-json`
- `--batch-output-item-json`
- `--batch-output-jsonl`

Bundle output schema:
- `v6_2_full_trace_scored_bundle_v2`

## What v2 Does Not Claim

- it does not make whole-file overlap the main metric
- it does not treat pairwise discrimination deltas as independent samples
- it does not claim model probabilities are calibrated
- it does not collapse repair, trajectory, and full-code into one composite score
