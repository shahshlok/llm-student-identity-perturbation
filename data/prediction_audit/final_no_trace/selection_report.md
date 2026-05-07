# V6.2 Branching Probe Selection Report

Pre-commit diversity distribution (stage-1 retained scopes):
- Retained scopes before selection: `4041`
- Mean pairwise distance distribution: `min=0.000000`, `p33=0.000000`, `median=0.000000`, `p67=0.000000`, `max=0.000000`
- Fixed cut-points for sensitivity analysis: `0.15`, `0.30`, `0.45`

| Diversity bucket | Scope count |
| --- | ---: |
| <= 0.15 | 4041 |
| (0.15, 0.30] | 0 |
| (0.30, 0.45] | 0 |
| > 0.45 | 0 |

## Final Output

- Final scope counts: `T1=91`, `T2=0`, `T3=0`
- Final scope total: `91`
- Final bundle total: `209`
- Documented rebalance: `{"T1": 91, "T2": 0, "T3": 0}`
- Manifest: `data/prediction_audit/final_no_trace/manifest.json`

## Stage 1 Canonical Exclusion

- Input scopes: `4290`
- Dropped canonical scopes: `249`
- Retained scopes: `4041`

## Stage 2 Diversity Tertiles

- P33 boundary: `0.000000`
- P67 boundary: `0.000000`
- Retained-tertile counts: `T1=4041`, `T2=0`, `T3=0`

## Stage 3 Candidate Enumeration

- Students with both logs: `73635`
- Students dropped for missing logs: `59699`

## Stage 4 Attempt-Depth Validation

- Fixed transition index (0-indexed): `2`
- Fixed visible attempt count: `3`
- Minimum total attempts required: `4`
- Candidate students checked: `73635`
- Dropped parse failures: `18247`
- Dropped insufficient attempts: `50943`
- Retained students: `4445`
- Parse failure rate over stage-3 candidates: `24.78%`

## Stage 5 L2A Matching

- Scopes with matched rows: `91`
- Rows retained in frozen matched cohort: `209`
- Retained L2A groups: `91`
- Scopes dropped for <2 retained students: `2974`
- Scopes dropped by bundle preflight: `924`
- Scopes dropped for no non-singleton L2A group: `52`

## Stage 6 Stratified Sampling

- Requested quotas: `T1=31`, `T2=30`, `T3=30`
- Final sampled counts before emission: `T1=91`, `T2=0`, `T3=0`
- Final sampled row counts before emission: `T1=209`, `T2=0`, `T3=0`
- Reserve scopes after sampling: `0`
- Reserve rows after sampling: `0`

## Stage 7 Bundle Emission

- Successful scopes emitted: `91`
- Successful bundles emitted: `209`
- Emission failures: `0`

## Stage 8 Self-Validation

- Status: `passed`
- Validated final scope counts: `T1=91`, `T2=0`, `T3=0`
- Unique custom_ids: `209`
