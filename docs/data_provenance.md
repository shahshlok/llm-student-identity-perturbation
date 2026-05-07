# Data Provenance

## Source Dataset

The study uses de-identified CodeBench records from introductory Python programming courses. The prediction audit spans four CodeBench semesters:

- `2022-2`
- `2023-1`
- `2023-2`
- `2024-1`

The records include source submissions, execution outcomes, test-level feedback, and IDE/process traces. The public artifact does not include student names, demographic attributes, or institution-facing identifiers. Student identifiers are dataset keys.

## Public Artifact Boundary

This repository includes the derived artifacts needed to inspect and reproduce scoring:

- selected prompt bundles for the final prediction audit
- OpenAI batch request JSONL files
- OpenAI batch output JSONL files
- scored prediction reports
- cluster-bootstrap comparison report
- narrative-audit hydrated predictions and comparison reports
- code used to parse, score, aggregate, and compare these artifacts

The repository does not include the original full local CodeBench semester roots. The final prompt bundles and reports contain de-identified code snippets and trace-derived records for the retained study sample.

## Selection Flow

The final prediction audit starts from the multi-semester preparation artifacts in `data/prediction_audit/selection_preparation/`. The preparation report records 4,290 candidate scope records before downstream filtering.

The final scored denominator is:

- 91 matched assessment-state scopes
- 209 prediction rows per condition
- 310 directed focal-peer comparisons

The reduction from candidate scope records to final scored scopes comes from requiring the fixed third-visible-attempt setup, buildable prompt bundles, and at least two students sharing the same row-local test pass/fail vector within the same class/task/exercise scope.

## Path Redaction

Local machine paths from the original worktree were replaced with artifact-relative paths where possible. Historical source paths that refer to non-packaged raw roots are represented with placeholders such as `<SOURCE_WORKTREE>` and `<CODEBENCH_DATA_ROOT>`. The original preparation directory included local symlinks under `merged_root/`; those symlinks are deliberately omitted from this public artifact.

## Privacy and Ethics

The artifact reports and bundles are intended for research review and replication of the study computations. They should not be used to identify individual learners. Public reporting should use aggregate behavior and anonymized dataset keys.
