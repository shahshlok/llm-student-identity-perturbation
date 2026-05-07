# LLM Student Identity Perturbation

## Purpose

This repository is the public artifact for:

**Do LLM Student Models Track the Student or the Situation? A Matched-Peer Study in Novice Programming**

The artifact documents an identity-perturbation benchmark on novice Python submissions. The central hypothesis is that a model's next-submission prediction should be targeted to the focal student, not just to the shared assessment state (same task, same current outcomes, same visible history shape).

## What is in this repository

- Narrative audit: comparison of generated narrative responses across conditions.
- Prediction audit: matched-peer identity perturbation comparisons on the final prompting conditions.
- Full scoring contracts, metrics, and comparison logic used in the paper.
- Minimal commands and scripts needed to verify key outputs.

## Empirical design summary

The public artifact records a two-part study:

1. **Narrative audit**
   - 117 hydrated narrative predictions across `full_trace`, `no_trace`, and `trace_shuffled` settings.
   - Tests template reuse, personalization signals, and narrative variation patterns.
2. **Prediction audit**
   - Main test compares a model's top candidate against focal students and matched peers.
   - Final scored comparison is done on:
     - 4 CodeBench semesters: `2022-2`, `2023-1`, `2023-2`, `2024-1`
     - 91 matched assessment-state scopes
     - 209 predictions per condition
     - 310 directed focal-peer comparisons
   - Two conditions: `full_trace` and `no_trace`.

Model and scorer configuration used for the packaged final runs:

- model: `gpt-5.4`
- reasoning effort: `xhigh`
- condition comparison bootstrap: 10,000 samples, random seed 42
- scoring contract: `v6_2_full_trace_scored_prediction_v2` (v2 metrics family)

The selection-preparation stage records 4,290 candidate rows before matching and strict scope filtering.

## Repository layout

```text
.
├── data/
│   ├── prediction_audit/
│   │   ├── selection_preparation/        # candidate scope preparation
│   │   ├── final_full_trace/             # final full-trace prompt bundles and scores
│   │   ├── final_no_trace/               # final no-trace prompt bundles and scores
│   │   ├── final_condition_comparison/   # bootstrap comparison output
│   │   └── openai_batch_outputs/         # archived batch response JSONL
│   └── narrative_audit/
│       ├── full_trace/
│       ├── no_trace/
│       ├── trace_shuffled/
│       └── comparisons/
├── src/identity_perturbation/
│   ├── codebench_support/                # dataset parsing and benchmark utilities
│   ├── prediction_audit/                 # matching, scoring, and condition comparison
│   └── narrative_audit/                  # narrative processing and comparison tooling
├── docs/
│   ├── artifact.md
│   ├── reproducibility.md
│   ├── data_provenance.md
│   └── scoring_contract.md
└── tests/
```

## Reproducibility playbook (from artifacts)

Install dependencies:

```bash
uv sync
```

Re-score the final full-trace and no-trace runs:

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

Recompute the condition comparison:

```bash
uv run python -m identity_perturbation.prediction_audit.compare_condition_reports_v2 \
  --left-report data/prediction_audit/final_full_trace/scores_v2/report.json \
  --right-report data/prediction_audit/final_no_trace/scores_v2/report.json \
  --out data/prediction_audit/final_condition_comparison/report.json
```

Run narrative summary checks:

```bash
uv run python -m identity_perturbation.narrative_audit.narrative_analysis
```

Run the test suite if you want a code-quality smoke check:

```bash
uv run pytest -q
```

## How to interpret the outputs

- The primary inference target is whether full-trace condition improves identity discrimination over no-trace while controlling for shared states.
- The interpretation boundary is reported in `docs/scoring_contract.md` and `docs/artifact.md`.
- The paper-level focus is discrimination quality under same-state matching, not absolute generation quality.
- All identifiers are dataset keys; no student names or protected attributes are included in this public artifact.

## Reviewer path

Start here before opening implementation:

1. `docs/artifact.md` (artifact summary and study components)
2. `docs/data_provenance.md` (sample construction and boundaries)
3. `docs/scoring_contract.md` (metric definitions and limitations)
4. `docs/reproducibility.md` (exact rerun commands)

Then move to `src/identity_perturbation/...` only if you need implementation inspection.

See also [docs/reviewer-onboarding.md](docs/reviewer-onboarding.md) for a structured first-pass checklist.
