# Artifact Description

## Purpose

This repository is the public artifact for the identity-perturbation study on novice programming data.

It is designed for third-party review and replication of the following:

- what was measured,
- how matched conditions were built,
- how scores were produced,
- what those scores do and do not support.

The artifact is not a production deployment package.

## What is included and why

### 1) Narrative audit

Purpose:

- examine whether narrative predictions appear individualized,
- detect template and grounding differences across trace conditions,
- check whether narratives track the focal student more than context.

Primary locations:

- `data/narrative_audit/full_trace/hydrated_predictions.json`
- `data/narrative_audit/no_trace/hydrated_predictions.json`
- `data/narrative_audit/trace_shuffled/hydrated_predictions.json`
- `data/narrative_audit/comparisons/narrative_pattern_analysis.json`
- `data/narrative_audit/comparisons/narrative_deep_analysis.json`
- `data/narrative_audit/comparisons/cognitive_accuracy_tests.json`

Key code:

- `src/identity_perturbation/narrative_audit/narrative_analysis.py`
- `src/identity_perturbation/narrative_audit/narrative_deep_analysis.py`

### 2) Prediction audit

Purpose:

- test whether prediction quality is concentrated on focal students or mostly recoverable from shared state,
- evaluate discrimination between focal and matched-peer predictions,
- compare full-trace versus no-trace evidence conditions.

Primary locations:

- `data/prediction_audit/selection_preparation/prep_report.json`
- `data/prediction_audit/final_full_trace/manifest.json`
- `data/prediction_audit/final_no_trace/manifest.json`
- `data/prediction_audit/final_full_trace/scores_v2/report.json`
- `data/prediction_audit/final_no_trace/scores_v2/report.json`
- `data/prediction_audit/final_condition_comparison/report.json`
- `data/prediction_audit/final_condition_comparison/README.md` (if present in your snapshot)

Key code:

- `src/identity_perturbation/prediction_audit/report_full_trace_run_v2.py`
- `src/identity_perturbation/prediction_audit/compare_condition_reports_v2.py`
- `src/identity_perturbation/prediction_audit/full_trace_scorer_v2.py`
- `src/identity_perturbation/prediction_audit/full_trace_suite_v2.py`
- `src/identity_perturbation/prediction_audit/score_full_trace_bundle_v2.py`

## Final quantitative scope

For this public artifact, the final analyzed prediction scope is:

- 91 matched assessment-state scopes,
- 209 prediction rows per condition,
- 310 directed focal-peer comparisons,
- two conditions: `full_trace` and `no_trace`.

The selection stage includes 4,290 candidate scope records (`selection_preparation/prep_report.json`) prior to filtering and matching. The 4,290 count is a pre-filter universe, not the final scoring denominator.

## Condition definitions

- `full_trace`: includes code, test outcome context, and trace context leading to target attempts.
- `no_trace`: includes code and test outcome context but excludes trace context.
- `trace_shuffled`: controls narrative and contextual robustness; retained for narrative-layer diagnostics.

These conditions are paired with identical scope matching in order to isolate whether added signal changes prediction identity behavior.

## Inference strategy

The primary comparison is a matched-pair directional test:

- each row produces focal→peer comparisons within the same assessment-state scope,
- identity discrimination is calculated per-scope and aggregated by scope cluster to avoid false independence assumptions,
- condition-level contrast uses bootstrap over scope clusters with seed `42` and `10000` resamples.

`L2A` scope matching is the primary analytic denominator. `L2B` is used as a sensitivity analysis.

## What this artifact can support

- Interpretation of the identity-specific versus state-specific behavior claimed in the study.
- Recalculation of final prediction score tables from archived outputs.
- Independent review of whether claim boundaries are respected in interpretation.

## What this artifact does not support

- Claims of true belief reconstruction in production settings.
- Guaranteed student-level behavioral inferences from model outputs.
- Replacement for raw internal data access beyond what is shared in this artifact.
