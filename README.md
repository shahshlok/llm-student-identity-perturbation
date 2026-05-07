# LLM Student Identity Perturbation

This repository is the public research artifact for:

**Do LLM Student Models Track the Student or the Situation? A Matched-Peer Study in Novice Programming**

The artifact supports an identity-perturbation audit of LLM student models on novice Python programming data. The central question is whether an LLM prediction is targeted to the focal student, or whether it mainly reflects the shared programming situation: the same exercise and the same current test-outcome state.

## Artifact Summary

The study has two linked empirical parts:

1. **Narrative audit**: 117 LLM-generated narrative cases are analyzed for variation, grounding, and repeated template language.
2. **Prediction audit**: a matched-peer identity-perturbation analysis compares LLM predictions for students in the same assessed programming state.

The final prediction-audit run uses:

- 4 CodeBench semesters: `2022-2`, `2023-1`, `2023-2`, and `2024-1`
- 91 matched assessment-state scopes
- 209 prediction rows per condition
- 310 directed focal-peer comparisons
- 2 prediction conditions: `full_trace` and `no_trace`
- model: `gpt-5.4`
- reasoning effort: `xhigh`
- scope-cluster bootstrap inference with 10,000 resamples and seed 42

The larger selection-preparation artifact records 4,290 candidate scope records before matching, preflight, and L2A retention.

## Repository Layout

```text
.
├── data/
│   ├── prediction_audit/
│   │   ├── selection_preparation/        # multi-semester candidate-scope preparation
│   │   ├── final_full_trace/             # final full-trace requests, bundles, and scores
│   │   ├── final_no_trace/               # final no-trace requests, bundles, and scores
│   │   ├── final_condition_comparison/   # cluster-bootstrap comparison report
│   │   └── openai_batch_outputs/         # raw OpenAI batch output JSONL files
│   └── narrative_audit/
│       ├── full_trace/
│       ├── no_trace/
│       ├── trace_shuffled/
│       └── comparisons/
├── src/identity_perturbation/
│   ├── codebench_support/                # parsers and shared CodeBench helpers
│   ├── prediction_audit/                 # matched-peer scoring and comparison code
│   └── narrative_audit/                  # narrative audit code
├── docs/
│   ├── artifact.md
│   ├── reproducibility.md
│   ├── data_provenance.md
│   └── scoring_contract.md
└── tests/
```

## Quick Start

Install dependencies with `uv`:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest -q
```

Re-score the final prediction runs:

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
```

Re-run the condition comparison:

```bash
uv run python -m identity_perturbation.prediction_audit.compare_condition_reports_v2 \
  --left-report data/prediction_audit/final_full_trace/scores_v2/report.json \
  --right-report data/prediction_audit/final_no_trace/scores_v2/report.json \
  --out data/prediction_audit/final_condition_comparison/report.json
```

## Data Use Note

The artifact contains de-identified CodeBench-derived programming records, model requests, model outputs, and scoring artifacts. Identifiers are dataset keys rather than student names or demographic attributes. Local source-machine paths have been replaced with artifact-relative paths or placeholder roots such as `<CODEBENCH_DATA_ROOT>`.

See [`docs/data_provenance.md`](docs/data_provenance.md) for the data boundary and [`docs/reproducibility.md`](docs/reproducibility.md) for verification commands.
