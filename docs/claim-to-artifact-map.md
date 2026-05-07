# Claim-to-Artifact Map

Use this table to quickly check every major claim against concrete reproducible files.

| Claim | Evidence artifacts | How to verify |
| --- | --- | --- |
| The model output is primarily context-driven rather than strictly focal-student-driven, and identity sensitivity is limited by same-state matching | `data/prediction_audit/final_full_trace/scores_v2/report.json`, `data/prediction_audit/final_no_trace/scores_v2/report.json`, `data/prediction_audit/final_condition_comparison/report.json` | Compare focal/peer discrimination blocks and condition-delta fields. |
| Full-trace vs no-trace supports the same core matching framework while isolating trace contribution | `data/prediction_audit/final_full_trace/`, `data/prediction_audit/final_no_trace/`, `docs/data_provenance.md` | Confirm identical scope-level preprocessing constraints and inspect contrast in comparison report. |
| Final sample is based on 91 matched scopes, 209 rows per condition, 310 directed comparisons | `docs/artifact.md`, `data/prediction_audit/final_condition_comparison/report.json`, `data/prediction_audit/selection_preparation/prep_report.json` | Read denominator statement in artifact + compare pre-selection and final comparison counts in prep report. |
| Narrative layer indicates limited personalization in wording | `data/narrative_audit/comparisons/narrative_pattern_analysis.json`, `data/narrative_audit/comparisons/narrative_deep_analysis.json`, `data/narrative_audit/comparisons/cognitive_accuracy_tests.json` | Review repetition/variation and accuracy diagnostics in these comparison files. |
| Score contract `v2` metrics are the basis for interpretation | `docs/scoring_contract.md`, `src/identity_perturbation/prediction_audit/full_trace_scorer_v2.py`, scored `scores_v2/report.json` files | Verify listed metrics and definitions in the contract doc, then confirm presence in scored outputs. |
| Cluster-level bootstrap inference is used for condition comparison | `data/prediction_audit/final_condition_comparison/report.json`, `docs/scoring_contract.md` | Confirm bootstrap sample size and seed and inspect output fields for scope-level aggregation. |
| Artifact is reproducible from archived outputs without live API calls | `docs/reproducibility.md`, `data/prediction_audit/openai_batch_outputs/*.jsonl` | Run Level B commands in reproducibility and compare regenerated reports to checked-in ones. |
| Public artifact does not claim direct deployment-grade identity inference | `README.md`, `docs/artifact.md`, `docs/data_provenance.md` | Read claim-boundary sections before making external claims. |
