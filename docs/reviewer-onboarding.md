# Reviewer Onboarding Guide

This document is for someone reading the identity-perturbation artifact before touching source code.

## 1) Study intent and what question it answers

The study asks whether LLM student-model responses are targeted to a focal student versus driven by shared context.

- If a response changes when we replace the focal student with a matched peer in the same state, that supports student-sensitive modeling.
- If responses stay unchanged, that suggests strong state dependence and limited student-specific tracking.

The public evidence is split into:

- **Narrative layer:** whether language and explanatory content indicates personalization.
- **Prediction layer:** whether predicted next submissions are better than peer-matched baselines under tightly controlled matching.

## 2) The two evidence channels

### Narrative audit

Files:

- `data/narrative_audit/full_trace/hydrated_predictions.json`
- `data/narrative_audit/no_trace/hydrated_predictions.json`
- `data/narrative_audit/trace_shuffled/hydrated_predictions.json`
- `data/narrative_audit/comparisons/narrative_pattern_analysis.json`

Primary interpretation artifacts:

- `data/narrative_audit/comparisons/narrative_deep_analysis.json`
- `data/narrative_audit/comparisons/cognitive_accuracy_tests.json`
- `data/narrative_audit/comparisons/narrative_pattern_analysis.json`

### Prediction audit

Primary files:

- `data/prediction_audit/final_full_trace/manifest.json`
- `data/prediction_audit/final_no_trace/manifest.json`
- `data/prediction_audit/final_full_trace/scores_v2/report.json`
- `data/prediction_audit/final_no_trace/scores_v2/report.json`
- `data/prediction_audit/final_condition_comparison/report.json`

Reference entrypoints:

- `src/identity_perturbation/prediction_audit/report_full_trace_run_v2.py`
- `src/identity_perturbation/prediction_audit/compare_condition_reports_v2.py`
- `src/identity_perturbation/prediction_audit/full_trace_scorer_v2.py`

## 3) Publicly checkable final numbers

Start by reading:

```bash
cat data/prediction_audit/final_full_trace/scores_v2/report.json
cat data/prediction_audit/final_no_trace/scores_v2/report.json
cat data/prediction_audit/final_condition_comparison/report.json
```

Focus first on:

- the same-scope discrimination block,
- the cluster-level support counts,
- the baseline method note,
- and the score-family definitions in `docs/scoring_contract.md`.

## 4) Recommended interpretation sequence

1. Read `docs/artifact.md` and `docs/data_provenance.md` to understand sample construction.
2. Read `docs/scoring_contract.md` for exact metric definitions and what is not claimed.
3. Read the comparison outputs and note whether full-trace/no-trace divergences align with directionality claims.
4. Run the scoring commands in `docs/reproducibility.md` to regenerate from the same archived outputs.
5. Only then read `src/` code for implementation confirmation.

## 5) Reproducibility with archived outputs

The artifact includes archived batch outputs, so interpretation checks do not require live provider calls.

```bash
uv run python -m identity_perturbation.prediction_audit.report_full_trace_run_v2 \
  --run-manifest data/prediction_audit/final_full_trace/manifest.json \
  --output-jsonl data/prediction_audit/openai_batch_outputs/batch_69e8290b6d78819098933a8a0e8e5a11_output.jsonl \
  --condition full \
  --out data/prediction_audit/final_full_trace/scores_v2/report.json
```

Repeat the same for no-trace and then:

```bash
uv run python -m identity_perturbation.prediction_audit.compare_condition_reports_v2 \
  --left-report data/prediction_audit/final_full_trace/scores_v2/report.json \
  --right-report data/prediction_audit/final_no_trace/scores_v2/report.json \
  --out data/prediction_audit/final_condition_comparison/report.json
```

## 6) What this artifact does not do

- It does not include a full raw CodeBench snapshot; only the de-identified artifacts needed for replication are included.
- It does not claim student identity recovery in a deployment or high-stakes setting.
- It does not include the paper PDF itself.

## 7) Quick cross-reference map

- `docs/artifact.md` — what is included and how it maps to claims
- `docs/data_provenance.md` — provenance and privacy boundary
- `docs/scoring_contract.md` — metric contract for `*_v2` scoring
- `docs/reproducibility.md` — exact rerun commands
