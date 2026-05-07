# Artifact Description

## Purpose

This artifact supports the empirical claim that personalized-sounding LLM student-model outputs should not be treated as evidence of student-specific prediction. It provides the code, prompts, model outputs, scoring results, and comparison reports used for the narrative and prediction audits.

## Main Empirical Components

### Narrative Audit

Location: `data/narrative_audit/`

The narrative audit contains 117 hydrated predictions per primary condition and the derived lexical/template analyses used to evaluate whether summaries appear individualized rather than templated.

Primary files:

- `data/narrative_audit/full_trace/hydrated_predictions.json`
- `data/narrative_audit/no_trace/hydrated_predictions.json`
- `data/narrative_audit/comparisons/narrative_pattern_analysis.json`
- `data/narrative_audit/comparisons/narrative_deep_analysis.json`

Primary code:

- `src/identity_perturbation/narrative_audit/narrative_analysis.py`
- `src/identity_perturbation/narrative_audit/narrative_deep_analysis.py`

### Prediction Audit

Location: `data/prediction_audit/`

The prediction audit is the main identity-perturbation analysis. It compares the top-ranked predicted next submission for a focal student against both the focal student's real next submission and matched peers' real next submissions.

Primary files:

- `data/prediction_audit/final_full_trace/manifest.json`
- `data/prediction_audit/final_full_trace/scores_v2/report.json`
- `data/prediction_audit/final_no_trace/manifest.json`
- `data/prediction_audit/final_no_trace/scores_v2/report.json`
- `data/prediction_audit/final_condition_comparison/report.json`

Primary code:

- `src/identity_perturbation/prediction_audit/report_full_trace_run_v2.py`
- `src/identity_perturbation/prediction_audit/compare_condition_reports_v2.py`
- `src/identity_perturbation/prediction_audit/full_trace_suite_v2.py`
- `src/identity_perturbation/prediction_audit/full_trace_scorer_v2.py`

## Final Prediction-Audit Denominator

The final denominator is:

- 91 matched assessment-state scopes
- 209 prediction rows per condition
- 310 directed focal-peer comparisons

The larger `selection_preparation` directory records the multi-semester candidate universe before matching and filtering. Its `prep_report.json` records `combined_scope_record_count = 4290`; this is not the final scored scope count.

## Conditions

`full_trace` includes submitted-code and execution evidence through the focal attempt, plus process trace evidence before the observed next submission.

`no_trace` includes submitted-code and execution evidence through the focal attempt, but excludes IDE/process trace evidence. It is not a cold-start condition.

## Inference

The primary comparison report uses matched assessment-state scope as the bootstrap cluster. The final condition comparison uses 10,000 bootstrap resamples with seed 42. The L2B code-distance sweep is sensitivity analysis; L2A matched assessment state is the primary denominator.
