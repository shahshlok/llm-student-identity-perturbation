# V6.1 Evaluation

- Run name: `gpt54_medium_v61_clean126_buildable117_v6`
- Model: `gpt-5.4`
- Reasoning effort: `medium`
- Evaluated predictions: `117`
- Prefix length: `3`

## Primary Metrics

| Metric | Key Results |
| --- | --- |
| First active event type | top-1 `0.632`, truth mass `0.624`, MRR `0.694` |
| First change line | exact `0.444`, +/-1 `0.590`, +/-2 `0.701` |
| Run presence | top-1 `0.735`, truth mass `0.706`, Brier `0.254` |
| Idle-gap presence | top-1 `0.607`, truth mass `0.604`, Brier `0.358` |
| Episode motif | top-1 `0.325`, truth mass `0.302`, MRR `0.365` |
| Event-type overlap | top-1 `0.322`, expected `0.313` |
| Event-type edit similarity | top-1 `0.307`, expected `0.302` |
| Event-family LCSS | top-1 `0.313`, expected `0.309` |
| Event-family DTW | top-1 `0.841`, expected `0.834` |

## Supplementary Metrics

| Metric | Key Results |
| --- | --- |
| First event type | top-1 `0.504`, truth mass `0.500`, MRR `0.578` |
| First event family | top-1 `0.504`, truth mass `0.500`, MRR `0.578` |
| Change presence | top-1 `0.957`, truth mass `0.953`, Brier `0.028` |
| Event-type prefix | exact `0.051`, top-1 shared `0.138`, expected shared `0.131` |
| Event-family edit similarity | top-1 `0.307`, expected `0.302` |
| Event-type LCSS | top-1 `0.313`, expected `0.309` |
| Event count buckets | full top-1 `0.085`, edit `0.222`, run `0.573`, pause `0.564` |

## First Event Diagnostics

- Observed counts: `{"change": 77, "saida_testar": 36, "tab_click": 4}`
- Predicted top-1 counts: `{"change": 93, "idle_gap": 19, "tab_click": 5}`

| Observed first event | Count | Top-1 | Truth mass | MRR |
| --- | --- | --- | --- | --- |
| change | 77 | 0.753 | 0.730 | 0.807 |
| saida_testar | 36 | 0.000 | 0.043 | 0.111 |
| tab_click | 4 | 0.250 | 0.190 | 0.375 |

| Observed | Predicted | Count |
| --- | --- | --- |
| change | change | 58 |
| saida_testar | change | 32 |
| change | idle_gap | 16 |
| change | tab_click | 3 |
| saida_testar | idle_gap | 3 |
| tab_click | change | 3 |
| tab_click | tab_click | 1 |
| saida_testar | tab_click | 1 |
