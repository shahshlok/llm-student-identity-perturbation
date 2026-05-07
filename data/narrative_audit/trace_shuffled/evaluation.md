# V6.1 Evaluation

- Run name: `gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1`
- Model: `gpt-5.4`
- Reasoning effort: `medium`
- Evaluated predictions: `116`
- Prefix length: `3`

## Primary Metrics

| Metric | Key Results |
| --- | --- |
| First active event type | top-1 `0.612`, truth mass `0.599`, MRR `0.639` |
| First change line | exact `0.422`, +/-1 `0.569`, +/-2 `0.655` |
| Run presence | top-1 `0.543`, truth mass `0.527`, Brier `0.406` |
| Idle-gap presence | top-1 `0.552`, truth mass `0.560`, Brier `0.346` |
| Episode motif | top-1 `0.198`, truth mass `0.178`, MRR `0.237` |
| Event-type overlap | top-1 `0.305`, expected `0.294` |
| Event-type edit similarity | top-1 `0.303`, expected `0.294` |
| Event-family LCSS | top-1 `0.308`, expected `0.299` |
| Event-family DTW | top-1 `0.828`, expected `0.814` |

## Supplementary Metrics

| Metric | Key Results |
| --- | --- |
| First event type | top-1 `0.405`, truth mass `0.431`, MRR `0.511` |
| First event family | top-1 `0.405`, truth mass `0.431`, MRR `0.511` |
| Change presence | top-1 `0.966`, truth mass `0.956`, Brier `0.028` |
| Event-type prefix | exact `0.043`, top-1 shared `0.092`, expected shared `0.098` |
| Event-family edit similarity | top-1 `0.303`, expected `0.294` |
| Event-type LCSS | top-1 `0.308`, expected `0.299` |
| Event count buckets | full top-1 `0.060`, edit `0.241`, run `0.448`, pause `0.483` |

## First Event Diagnostics

- Observed counts: `{"change": 76, "saida_testar": 36, "tab_click": 4}`
- Predicted top-1 counts: `{"change": 73, "idle_gap": 41, "tab_click": 2}`

| Observed first event | Count | Top-1 | Truth mass | MRR |
| --- | --- | --- | --- | --- |
| change | 76 | 0.618 | 0.651 | 0.768 |
| saida_testar | 36 | 0.000 | 0.009 | 0.014 |
| tab_click | 4 | 0.000 | 0.062 | 0.125 |

| Observed | Predicted | Count |
| --- | --- | --- |
| change | change | 47 |
| change | idle_gap | 27 |
| saida_testar | change | 24 |
| saida_testar | idle_gap | 12 |
| tab_click | idle_gap | 2 |
| change | tab_click | 2 |
| tab_click | change | 2 |
