# V6.1 Paired Statistical Comparison

## Conditions

| Label | Run name | N |
| --- | --- | --- |
| full | `gpt54_medium_v61_clean126_buildable117_v6` | 117 |
| no_trace | `gpt54_medium_v61_clean126_buildable117_no_trace_v1` | 117 |
| trace_shuffled | `gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1` | 116 |

Bootstrap samples: 2000, seed: 42

## Results

| Metric | Pair | N paired | Mean A | Mean B | Delta | 95% CI | Wilcoxon p | Majority | Rand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `first_active_event_type` (top1_accuracy) | full - no_trace | 117 | 0.6325 | 0.6325 | +0.0000 | [-0.0342, +0.0342] | 1.0000 | 0.658 | 0.529 |
|  | full - trace_shuffled | 116 | 0.6293 | 0.6121 | +0.0172 | [-0.0259, +0.0603] | 0.4142 |  |  |
| `run_presence` (top1_accuracy) | full - no_trace | 117 | 0.7350 | 0.5214 | +0.2137 | [+0.0855, +0.3333] | 0.0011 ** | 0.521 | 0.501 |
|  | full - trace_shuffled | 116 | 0.7328 | 0.5431 | +0.1897 | [+0.0690, +0.3017] | 0.0019 ** |  |  |
| `first_change_line` (top1_accuracy_within_2) | full - no_trace | 117 | 0.7009 | 0.6752 | +0.0256 | [-0.0171, +0.0684] | 0.2568 | 0.137 | 0.065 |
|  | full - trace_shuffled | 116 | 0.6983 | 0.6552 | +0.0431 | [-0.0086, +0.0948] | 0.0956 |  |  |
| `episode_motif` (top1_accuracy) | full - no_trace | 117 | 0.3248 | 0.1453 | +0.1795 | [+0.0940, +0.2650] | 0.0002 *** | 0.462 | 0.237 |
|  | full - trace_shuffled | 116 | 0.3190 | 0.1983 | +0.1207 | [+0.0172, +0.2155] | 0.0196 * |  |  |
| `event_type_overlap` (mean_top1_jaccard) | full - no_trace | 117 | 0.3220 | 0.2761 | +0.0459 | [+0.0240, +0.0686] | 0.0000 *** | n/a | n/a |
|  | full - trace_shuffled | 116 | 0.3240 | 0.3047 | +0.0193 | [-0.0046, +0.0432] | 0.0730 |  |  |
| `event_family_dtw` (mean_top1_similarity) | full - no_trace | 117 | 0.8408 | 0.8113 | +0.0295 | [+0.0060, +0.0528] | 0.0065 ** | n/a | n/a |
|  | full - trace_shuffled | 116 | 0.8394 | 0.8284 | +0.0110 | [-0.0199, +0.0381] | 0.1416 |  |  |

**Legend**: Delta = Mean_A - Mean_B (positive means A > B). \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001 (Wilcoxon signed-rank, two-sided). Majority = majority-class baseline accuracy. Rand = weighted-random (sum p_i^2) baseline accuracy.
