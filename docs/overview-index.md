# Identity Perturbation Documentation Index

## What this package is for

This repository is a reviewer-facing artifact for the identity-perturbation study. It is arranged so the full logic can be validated before reading implementation.

## Recommended reading order (non-technical)

1. `README.md`
   - Research goal, layout, and top-level interpretation guidance.
2. `docs/overview-index.md` (this file)
   - Canonical navigation and glossary.
3. `docs/reviewer-onboarding.md`
   - Checklist-style onboarding for a first review pass.
4. `docs/artifact.md`
   - What was included and why it supports each claim.
5. `docs/data_provenance.md`
   - Data boundaries and de-identification handling.
6. `docs/scoring_contract.md`
   - Metric definitions and interpretation boundary.
7. `docs/reproducibility.md`
   - Exact rerun steps from archived responses.

## Recommended reading order (method-first)

1. `docs/artifact.md`
2. `docs/scoring_contract.md`
3. `docs/data_provenance.md`
4. `docs/reproducibility.md`
5. Code inspection (`src/identity_perturbation/...`) only after claim mapping is complete.

## Claim-to-artifact validation map

Use `docs/claim-to-artifact-map.md` (added as part of this pass) to trace every key claim to concrete files.

## Directory walkthrough

- `data/narrative_audit/`
  - Narrative predictions (`full_trace`, `no_trace`, `trace_shuffled`) and comparison artifacts.
- `data/prediction_audit/`
  - Selection prep, final condition manifests, scored reports, archived model outputs, and condition comparison.
- `docs/`
  - Artifact description, scoring contract, reproducibility, provenance, and reviewer onboarding.
- `src/identity_perturbation/`
  - Scoring, prediction matching, and narrative analysis implementation.

## Minimal reproduction flow

- Level A: artifact check only
  - Inspect final reports in `data/prediction_audit/.../scores_v2/report.json` and `final_condition_comparison/report.json`.
- Level B: deterministic rerun from archived outputs
  - Use commands in `docs/reproducibility.md`.
- Level C: narrative audit refresh
  - Run `identity_perturbation.narrative_audit.narrative_analysis`.

## Glossary

- **focal student**: student whose next-step target is being compared against matched peers.
- **peer perturbation**: replacement of focal student with matched peer inside the same assessment state.
- **L2A scope**: primary matching scope used for final inference.
- **cluster bootstrap**: bootstrap resampling at the scope level used for condition comparison inference.
- **full_trace**: includes trace context; **no_trace** excludes trace context.

## Interpretation guardrails

- Identity-support claims are bounded to the matched-pair experimental design.
- The artifact supports inference about personalization vs context-driven prediction, not omniscient student identity reconstruction.
