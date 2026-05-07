# Student State Summary Comparison

- Matched transitions across all three conditions: `117`
- Example selection rule: `5` lowest overall similarity transitions and `5` highest overall similarity transitions.
- Pairwise similarity metrics below use corpus-level TF-IDF cosine similarity and normalized Levenshtein ratio.

## Result Files

- `full`: `<SOURCE_WORKTREE>/data/v61_batch_runs/gpt54_medium_v61_clean126_buildable117_v6/output.jsonl`
- `no_trace`: `<SOURCE_WORKTREE>/data/v61_batch_runs/gpt54_medium_v61_clean126_buildable117_no_trace_v1/output.jsonl`
- `trace_shuffled`: `<SOURCE_WORKTREE>/data/v61_batch_runs/gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1/output.jsonl`

## Example Transitions

### 1. `589:5897:3189:9741:2` (low similarity)

- Overall mean similarity: `0.292`
- `full` vs `no_trace`: TF-IDF `0.032`, Levenshtein `0.474`
- `full` vs `trace_shuffled`: TF-IDF `0.055`, Levenshtein `0.552`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.069`, Levenshtein `0.567`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | Likely next the student targets lines 10-14, because the actual outputs are much too large for vetor[0] and vetor[1], matching the current pattern of looping through all medias inside each qualifying horas case. |
| `no_trace` | Student likely notices the inner loop multiplies counts and rewrites the logic per student. |
| `trace_shuffled` | The failed outputs strongly point to overcounting, so the student may now target the nested loop and the reversed attendance counter together. |

### 2. `589:5846:1605:9746:2` (low similarity)

- Overall mean similarity: `0.292`
- `full` vs `no_trace`: TF-IDF `0.110`, Levenshtein `0.443`
- `full` vs `trace_shuffled`: TF-IDF `0.078`, Levenshtein `0.480`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.029`, Levenshtein `0.613`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | All 3 tests passed, the platform reported success, and the student ended after a short pause rather than continuing to edit. Given the heavy effort in this attempt and the successful outcome, they likely consider the task done and leave the exercise. |
| `no_trace` | The student likely sees the exercise as complete and may submit without further code changes. |
| `trace_shuffled` | All tests pass, so the student likely feels finished and does not immediately make another repair. |

### 3. `589:5897:3189:9741:4` (low similarity)

- Overall mean similarity: `0.299`
- `full` vs `no_trace`: TF-IDF `0.027`, Levenshtein `0.448`
- `full` vs `trace_shuffled`: TF-IDF `0.047`, Levenshtein `0.628`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.141`, Levenshtein `0.504`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | The student seems focused on count updates and where cont1 and cont2 should be reset. They may still believe the main bug is local counter handling rather than the full double-loop design. |
| `no_trace` | Student likely notices the nested loop mixes every media with every hour. |
| `trace_shuffled` | The student likely notices the code is comparing every grade against each qualifying hours entry and resetting counters inside the loop, so a structural fix is plausible next. |

### 4. `591:5853:6788:9787:0` (low similarity)

- Overall mean similarity: `0.305`
- `full` vs `no_trace`: TF-IDF `0.023`, Levenshtein `0.609`
- `full` vs `trace_shuffled`: TF-IDF `0.057`, Levenshtein `0.594`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.022`, Levenshtein `0.528`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | Likely notices the current program shape is wrong and reuses the while/index scaffold while changing what is computed and printed. |
| `no_trace` | Student likely notices the expected output looks like one value plus one count, not a rounded total. |
| `trace_shuffled` | The failed outputs make the current logic look clearly off-task, and the prior trace suggests the student was already thinking about max and a counter but struggled to finish it correctly. |

### 5. `590:5843:1605:9815:0` (low similarity)

- Overall mean similarity: `0.316`
- `full` vs `no_trace`: TF-IDF `0.103`, Levenshtein `0.537`
- `full` vs `trace_shuffled`: TF-IDF `0.023`, Levenshtein `0.609`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.035`, Levenshtein `0.586`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | Likely notices that one failure comes from only processing four items and the other from printing an int when no division happens, so they may patch both visible lines at once. |
| `no_trace` | The student likely notices the six-value case is being truncated because only four positions are processed. |
| `trace_shuffled` | The longer failing test likely makes the hardcoded 4 stand out. The student may correct iteration length first and still miss the separate .0 formatting issue. |

### 6. `593:5886:6353:9843:0` (high similarity)

- Overall mean similarity: `0.684`
- `full` vs `no_trace`: TF-IDF `0.559`, Levenshtein `0.782`
- `full` vs `trace_shuffled`: TF-IDF `0.496`, Levenshtein `0.776`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.608`, Levenshtein `0.884`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | The student likely reads the diff, notices that only the farewell line is repeated, and makes a focused indentation fix on the last print line before submitting again. |
| `no_trace` | The student likely notices the repeated final message and fixes the indentation on the last print before submitting again. |
| `trace_shuffled` | The student likely notices the repeated final message and makes a minimal indentation fix to the last line before resubmitting. |

### 7. `591:5902:6353:9781:6` (high similarity)

- Overall mean similarity: `0.623`
- `full` vs `no_trace`: TF-IDF `0.399`, Levenshtein `0.733`
- `full` vs `trace_shuffled`: TF-IDF `0.364`, Levenshtein `0.703`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.689`, Levenshtein `0.852`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | The student likely notices that the countdown itself is correct and focuses on the last line's exact text. |
| `no_trace` | The student likely notices the exact output mismatch and makes a minimal fix on the last print line. |
| `trace_shuffled` | Student likely notices the output text mismatch and makes a minimal correction on the last line. |

### 8. `593:5886:1223:8870:0` (high similarity)

- Overall mean similarity: `0.573`
- `full` vs `no_trace`: TF-IDF `0.459`, Levenshtein `0.698`
- `full` vs `trace_shuffled`: TF-IDF `0.411`, Levenshtein `0.797`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.346`, Levenshtein `0.723`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | The student likely notices the output is missing only the last longest line and focuses on the second for-range upper bound. |
| `no_trace` | Student likely spots that only the last ascending line is missing and edits the second loop bound. |
| `trace_shuffled` | The student likely reads the failure as a missing final ascending line and focuses on the second range call. |

### 9. `591:5902:6353:9799:0` (high similarity)

- Overall mean similarity: `0.571`
- `full` vs `no_trace`: TF-IDF `0.462`, Levenshtein `0.764`
- `full` vs `trace_shuffled`: TF-IDF `0.240`, Levenshtein `0.767`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.403`, Levenshtein `0.791`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | Student likely notices the numeric countdown is right and focuses on the exact required closing phrase. |
| `no_trace` | The student likely notices the countdown works and focuses on the closing text mismatch. |
| `trace_shuffled` | The student likely notices the loop output is correct and focuses on the mismatched final message text. |

### 10. `590:5896:6353:7604:0` (high similarity)

- Overall mean similarity: `0.536`
- `full` vs `no_trace`: TF-IDF `0.388`, Levenshtein `0.722`
- `full` vs `trace_shuffled`: TF-IDF `0.335`, Levenshtein `0.732`
- `no_trace` vs `trace_shuffled`: TF-IDF `0.314`, Levenshtein `0.726`

| Condition | Top-1 `student_state_summary` |
| --- | --- |
| `full` | The student likely notices the program logic is now working and focuses on the last output line, making a minimal text correction and resubmitting. |
| `no_trace` | The student likely notices the countdown logic is correct and focuses on matching the final output text exactly. |
| `trace_shuffled` | The student likely notices the output mismatch is in the final phrase, not the loop, and makes a targeted text correction. |

