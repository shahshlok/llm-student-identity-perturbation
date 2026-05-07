# Reproducibility Guide

This guide provides a deterministic verification path for the identity-perturbation artifact.

## Environment

- Python 3.10+
- `uv`

```bash
uv sync
```

## Reproducibility levels

### Level 1: artifact-level validation (no API keys)

Inspect archived outputs only:

```bash
cat data/prediction_audit/final_full_trace/scores_v2/report.json
cat data/prediction_audit/final_no_trace/scores_v2/report.json
cat data/prediction_audit/final_condition_comparison/report.json
cat data/narrative_audit/comparisons/narrative_pattern_analysis.json
cat data/narrative_audit/comparisons/narrative_deep_analysis.json
```

This confirms what was produced in the archived run.

### Level 2: deterministic re-scoring from archived batch outputs

This is the primary rerun target for reviewer verification and should match the included artifacts.

```bash
uv run python -m identity_perturbation.prediction_audit.report_full_trace_run_v2 \
  --run-manifest data/prediction_audit/final_full_trace/manifest.json \
  --output-jsonl data/prediction_audit/openai_batch_outputs/batch_69e8290b6d78819098933a8a0e8e5a11_output.jsonl \
  --condition full \
  --out data/prediction_audit/final_full_trace/scores_v2/report.json

uv run python -m identity_perturbation.prediction_audit.report_full_trace_run_v2 \
  --run-manifest data/prediction_audit/final_no_trace/manifest.json \
  --output-jsonl data/prediction_audit/openai_batch_outputs/batch_69e828fa3ff08190a26e5caf70d4a2be_output.jsonl \
  --condition no_trace \
  --out data/prediction_audit/final_no_trace/scores_v2/report.json

uv run python -m identity_perturbation.prediction_audit.compare_condition_reports_v2 \
  --left-report data/prediction_audit/final_full_trace/scores_v2/report.json \
  --right-report data/prediction_audit/final_no_trace/scores_v2/report.json \
  --out data/prediction_audit/final_condition_comparison/report.json \
  --bootstrap-samples 10000 \
  --seed 42
```

### Level 3: narrative layer checks

```bash
uv run python -m identity_perturbation.narrative_audit.narrative_analysis
```

This regenerates narrative comparison artifacts under `data/narrative_audit/comparisons/`.

## Optional: smoke test of the Python package

```bash
uv run pytest -q
```

The test suite is intended as a local integrity check, not as the publication equivalence check.

## Optional: full regeneration (API-bound)

Recreating raw model outputs requires API access to compatible model families and is not the primary artifact validation path.

For generation commands, use `uv run python miscons`-style equivalents in the source scripts inside `src/identity_perturbation`, with provider keys configured in environment.

## Important reproducibility notes

- Reproducing with archived responses should be deterministic.
- Re-submitting the raw requests through current APIs can differ due to model drift, sampling controls, and service-side changes.
