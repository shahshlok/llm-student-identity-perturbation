# Reproducibility Guide

## Environment

The artifact uses Python 3.10 or newer and is managed with `uv`.

```bash
uv sync
```

## Test Suite

```bash
uv run pytest -q
```

Expected result for the packaged artifact:

```text
102 passed
```

## Re-score the Final Prediction Runs

Full-trace condition:

```bash
uv run python -m identity_perturbation.prediction_audit.report_full_trace_run_v2 \
  --run-manifest data/prediction_audit/final_full_trace/manifest.json \
  --output-jsonl data/prediction_audit/openai_batch_outputs/batch_69e8290b6d78819098933a8a0e8e5a11_output.jsonl \
  --condition full \
  --out data/prediction_audit/final_full_trace/scores_v2/report.json
```

No-trace condition:

```bash
uv run python -m identity_perturbation.prediction_audit.report_full_trace_run_v2 \
  --run-manifest data/prediction_audit/final_no_trace/manifest.json \
  --output-jsonl data/prediction_audit/openai_batch_outputs/batch_69e828fa3ff08190a26e5caf70d4a2be_output.jsonl \
  --condition no_trace \
  --out data/prediction_audit/final_no_trace/scores_v2/report.json
```

These commands are deterministic given the archived batch outputs and scoring code.

## Re-run the Condition Comparison

```bash
uv run python -m identity_perturbation.prediction_audit.compare_condition_reports_v2 \
  --left-report data/prediction_audit/final_full_trace/scores_v2/report.json \
  --right-report data/prediction_audit/final_no_trace/scores_v2/report.json \
  --out data/prediction_audit/final_condition_comparison/report.json \
  --bootstrap-samples 10000 \
  --seed 42
```

## Re-run the Narrative Analysis

```bash
uv run python -m identity_perturbation.narrative_audit.narrative_analysis
```

This rewrites `data/narrative_audit/comparisons/narrative_pattern_analysis.json`.

## External API Calls

The repository includes the model request JSONL files and completed OpenAI batch output JSONL files for the final runs. Reproducing the scoring and comparison artifacts does not require live API access. Re-submitting the original LLM batches would require access to a compatible model and may not reproduce byte-identical outputs because model-serving systems can change over time.
