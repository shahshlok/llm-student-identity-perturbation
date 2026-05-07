# Cognitive Accuracy Tests

- Created at: `2026-04-13T09:33:00+00:00`

## Conditions

| Condition | Rows | Run Dir |
| --- | ---: | --- |
| full | 117 | `<SOURCE_WORKTREE>/data/v61_batch_runs/gpt54_medium_v61_clean126_buildable117_v6` |
| no_trace | 117 | `<SOURCE_WORKTREE>/data/v61_batch_runs/gpt54_medium_v61_clean126_buildable117_no_trace_v1` |
| trace_shuffled | 116 | `<SOURCE_WORKTREE>/data/v61_batch_runs/gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1` |

## Test 1: Line Mention Matching

Whether explicit line mentions are close to the observed first changed line.

| Condition | Line mentions | Mention rate | Hit rate within +-2 |
| --- | ---: | ---: | ---: |
| full | 22 | 0.188 | 0.864 |
| no_trace | 1 | 0.009 | 1.000 |
| trace_shuffled | 6 | 0.052 | 0.833 |

- `full` vs `no_trace` paired n: `1`
- Paired means: `1.000` vs `1.000`
- Wilcoxon p: `1.000`
- Takeaway: `full` and `no_trace` are tied on the paired subset.

- `full` vs `trace_shuffled` paired n: `3`
- Paired means: `1.000` vs `1.000`
- Wilcoxon p: `1.000`
- Takeaway: `full` and `trace_shuffled` are tied on the paired subset.

## Test 2: Length as Cognitive Proxy

Whether longer top summaries correlate with better top-1 prediction accuracy.

| Condition | Word count vs Jaccard rho | p | Word count vs edit similarity rho | p |
| --- | ---: | ---: | ---: | ---: |
| full | 0.090 | 0.336 | 0.093 | 0.316 |
| no_trace | 0.008 | 0.928 | 0.022 | 0.815 |
| trace_shuffled | 0.123 | 0.190 | 0.099 | 0.288 |

## Test 3: Action-Type Alignment

Whether action types implied by the summary appear in the observed next episode.

| Condition | Action extraction n | Extraction rate | Mean alignment |
| --- | ---: | ---: | ---: |
| full | 73 | 0.624 | 0.863 |
| no_trace | 54 | 0.462 | 0.861 |
| trace_shuffled | 72 | 0.621 | 0.926 |

- `full` vs `no_trace` paired n: `32`
- Paired means: `0.917` vs `0.891`
- Wilcoxon p: `0.570`
- Takeaway: `full` is higher on the paired subset (delta `0.026`, p `0.570`).

- `full` vs `trace_shuffled` paired n: `50`
- Paired means: `0.883` vs `0.933`
- Wilcoxon p: `0.260`
- Takeaway: `trace_shuffled` is higher on the paired subset (delta `-0.050`, p `0.260`).

## Test 4: Specificity-Accuracy Correlation

Whether more specific summaries correlate with higher composite trajectory accuracy.

- Composite accuracy: `(event_type_overlap.top1_jaccard + event_type_edit_similarity.top1_similarity) / 2`

| Condition | Word count rho | p | Code refs rho | p | Cognitive verbs rho | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.092 | 0.324 | 0.167 | 0.071 | 0.218 | 0.018 |
| no_trace | 0.015 | 0.876 | 0.129 | 0.165 | 0.029 | 0.754 |
| trace_shuffled | 0.113 | 0.226 | 0.033 | 0.727 | 0.120 | 0.199 |
