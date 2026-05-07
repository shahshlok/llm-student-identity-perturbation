# V6.1 Condition Comparison

## Conditions

| Label | Run name | Predictions | Evaluation |
| --- | --- | --- | --- |
| full | `gpt54_medium_v61_clean126_buildable117_v6` | 117 | `<SOURCE_WORKTREE>/data/v61_batch_runs/gpt54_medium_v61_clean126_buildable117_v6/evaluation.json` |
| no_trace | `gpt54_medium_v61_clean126_buildable117_no_trace_v1` | 117 | `<SOURCE_WORKTREE>/data/v61_batch_runs/gpt54_medium_v61_clean126_buildable117_no_trace_v1/evaluation.json` |
| trace_shuffled | `gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1` | 116 | `<SOURCE_WORKTREE>/data/v61_batch_runs/gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1/evaluation.json` |

## Predictive Metrics

### `first_event_type`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_accuracy | 0.504 | 0.615 | 0.405 |
| mean_truth_probability_mass | 0.500 | 0.599 | 0.431 |
| top2_hit_rate | 0.641 | 0.632 | 0.603 |
| top3_hit_rate | 0.658 | 0.650 | 0.612 |
| mean_mrr | 0.578 | 0.630 | 0.511 |
| mean_brier_score | 0.855 | 0.723 | 0.924 |
| mean_log_loss | 9.742 | 9.764 | 11.029 |
### `first_event_family`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_accuracy | 0.504 | 0.615 | 0.405 |
| mean_truth_probability_mass | 0.500 | 0.599 | 0.431 |
| top2_hit_rate | 0.641 | 0.632 | 0.603 |
| top3_hit_rate | 0.658 | 0.650 | 0.612 |
| mean_mrr | 0.578 | 0.630 | 0.511 |
| mean_brier_score | 0.855 | 0.723 | 0.924 |
| mean_log_loss | 9.742 | 9.764 | 11.029 |
### `first_active_event_type`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_accuracy | 0.632 | 0.632 | 0.612 |
| mean_truth_probability_mass | 0.624 | 0.628 | 0.599 |
| top2_hit_rate | 0.744 | 0.650 | 0.655 |
| top3_hit_rate | 0.761 | 0.667 | 0.672 |
| mean_mrr | 0.694 | 0.647 | 0.639 |
| mean_brier_score | 0.632 | 0.698 | 0.699 |
| mean_log_loss | 6.871 | 9.286 | 9.169 |
### `change_presence`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_accuracy | 0.957 | 0.957 | 0.966 |
| mean_truth_probability_mass | 0.953 | 0.957 | 0.956 |
| mean_predicted_true_probability | 0.953 | 0.957 | 0.956 |
| mean_brier_score | 0.028 | 0.030 | 0.028 |
| mean_log_loss | 0.299 | 0.090 | 0.299 |
### `run_presence`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_accuracy | 0.735 | 0.521 | 0.543 |
| mean_truth_probability_mass | 0.706 | 0.507 | 0.527 |
| mean_predicted_true_probability | 0.427 | 0.770 | 0.392 |
| mean_brier_score | 0.254 | 0.438 | 0.406 |
| mean_log_loss | 5.820 | 9.726 | 8.200 |
### `idle_gap_presence`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_accuracy | 0.607 | 0.667 | 0.552 |
| mean_truth_probability_mass | 0.604 | 0.666 | 0.560 |
| mean_predicted_true_probability | 0.198 | 0.055 | 0.302 |
| mean_brier_score | 0.358 | 0.303 | 0.346 |
| mean_log_loss | 7.584 | 6.798 | 5.667 |
### `first_change_line`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_accuracy_exact | 0.444 | 0.419 | 0.422 |
| mean_truth_probability_mass_exact | 0.356 | 0.384 | 0.375 |
| top1_accuracy_within_1 | 0.590 | 0.590 | 0.569 |
| mean_truth_probability_mass_within_1 | 0.529 | 0.569 | 0.535 |
| top1_accuracy_within_2 | 0.701 | 0.675 | 0.655 |
| mean_truth_probability_mass_within_2 | 0.635 | 0.650 | 0.632 |
| top2_hit_rate_exact | 0.462 | 0.504 | 0.483 |
| top3_hit_rate_exact | 0.547 | 0.607 | 0.578 |
| mean_mrr_exact | 0.488 | 0.498 | 0.486 |
| top2_hit_rate_within_2 | 0.726 | 0.752 | 0.724 |
| top3_hit_rate_within_2 | 0.829 | 0.829 | 0.784 |
| mean_mrr_within_2 | 0.752 | 0.739 | 0.712 |
### `event_count_buckets`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_accuracy | 0.085 | 0.034 | 0.060 |
| edit_bucket_top1_accuracy | 0.222 | 0.137 | 0.241 |
| run_bucket_top1_accuracy | 0.573 | 0.333 | 0.448 |
| pause_bucket_top1_accuracy | 0.564 | 0.667 | 0.483 |
| mean_truth_probability_mass | 0.083 | 0.038 | 0.051 |
| top2_hit_rate | 0.171 | 0.068 | 0.095 |
| top3_hit_rate | 0.188 | 0.068 | 0.103 |
| mean_mrr | 0.132 | 0.051 | 0.080 |
| mean_brier_score | 1.445 | 1.616 | 1.466 |
| mean_log_loss | 22.418 | 25.793 | 24.864 |

## Trajectory Metrics

### `event_type_prefix`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_exact_match_rate | 0.051 | 0.043 | 0.043 |
| mean_top1_shared_prefix_norm | 0.138 | 0.136 | 0.092 |
| mean_expected_shared_prefix_norm | 0.131 | 0.132 | 0.098 |
### `event_type_overlap`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| mean_top1_jaccard | 0.322 | 0.276 | 0.305 |
| mean_expected_jaccard | 0.313 | 0.262 | 0.294 |
### `event_type_edit_similarity`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| mean_top1_similarity | 0.307 | 0.268 | 0.303 |
| mean_expected_similarity | 0.302 | 0.254 | 0.294 |
### `event_family_edit_similarity`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| mean_top1_similarity | 0.307 | 0.268 | 0.303 |
| mean_expected_similarity | 0.302 | 0.254 | 0.294 |
### `event_type_lcss`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| mean_top1_lcss_ratio | 0.313 | 0.271 | 0.308 |
| mean_expected_lcss_ratio | 0.309 | 0.258 | 0.299 |
### `event_family_lcss`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| mean_top1_lcss_ratio | 0.313 | 0.271 | 0.308 |
| mean_expected_lcss_ratio | 0.309 | 0.258 | 0.299 |
### `event_family_dtw`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| mean_top1_similarity | 0.841 | 0.811 | 0.828 |
| mean_expected_similarity | 0.834 | 0.802 | 0.814 |
### `episode_motif`

| Summary key | full | no_trace | trace_shuffled |
| --- | --- | --- | --- |
| top1_accuracy | 0.325 | 0.145 | 0.198 |
| mean_truth_probability_mass | 0.302 | 0.136 | 0.178 |
| top2_hit_rate | 0.402 | 0.197 | 0.259 |
| top3_hit_rate | 0.402 | 0.205 | 0.284 |
| mean_mrr | 0.365 | 0.174 | 0.237 |
| mean_brier_score | 1.125 | 1.396 | 1.290 |
| mean_log_loss | 16.485 | 22.085 | 19.946 |
