# V6.1 Evaluation

- Run name: `gpt54_medium_v61_clean126_buildable117_no_trace_v1`
- Model: `gpt-5.4`
- Reasoning effort: `medium`
- Evaluated predictions: `117`
- Prefix length: `3`

## Primary Metrics

| Metric | Key Results |
| --- | --- |
| First active event type | top-1 `0.632`, truth mass `0.628`, MRR `0.647` |
| First change line | exact `0.419`, +/-1 `0.590`, +/-2 `0.675` |
| Run presence | top-1 `0.521`, truth mass `0.507`, Brier `0.438` |
| Idle-gap presence | top-1 `0.667`, truth mass `0.666`, Brier `0.303` |
| Episode motif | top-1 `0.145`, truth mass `0.136`, MRR `0.174` |
| Event-type overlap | top-1 `0.276`, expected `0.262` |
| Event-type edit similarity | top-1 `0.268`, expected `0.254` |
| Event-family LCSS | top-1 `0.271`, expected `0.258` |
| Event-family DTW | top-1 `0.811`, expected `0.802` |

## Supplementary Metrics

| Metric | Key Results |
| --- | --- |
| First event type | top-1 `0.615`, truth mass `0.599`, MRR `0.630` |
| First event family | top-1 `0.615`, truth mass `0.599`, MRR `0.630` |
| Change presence | top-1 `0.957`, truth mass `0.957`, Brier `0.030` |
| Event-type prefix | exact `0.043`, top-1 shared `0.136`, expected shared `0.132` |
| Event-family edit similarity | top-1 `0.268`, expected `0.254` |
| Event-type LCSS | top-1 `0.271`, expected `0.258` |
| Event count buckets | full top-1 `0.034`, edit `0.137`, run `0.333`, pause `0.667` |

## First Event Diagnostics

- Observed counts: `{"change": 77, "saida_testar": 36, "tab_click": 4}`
- Predicted top-1 counts: `{"change": 110, "idle_gap": 4, "submit": 3}`

| Observed first event | Count | Top-1 | Truth mass | MRR |
| --- | --- | --- | --- | --- |
| change | 77 | 0.935 | 0.910 | 0.957 |
| saida_testar | 36 | 0.000 | 0.000 | 0.000 |
| tab_click | 4 | 0.000 | 0.000 | 0.000 |

| Observed | Predicted | Count |
| --- | --- | --- |
| change | change | 72 |
| saida_testar | change | 36 |
| change | idle_gap | 3 |
| tab_click | change | 2 |
| change | submit | 2 |
| tab_click | submit | 1 |
| tab_click | idle_gap | 1 |
