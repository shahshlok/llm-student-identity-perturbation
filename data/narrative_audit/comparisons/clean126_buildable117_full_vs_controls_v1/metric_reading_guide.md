# V6.1 Metric Reading Guide

This file explains what each `v6.1` metric is trying to measure, why it exists,
and what the current three-condition results look like on the `clean126_buildable117`
study slice.

Conditions:

| Label | Meaning |
| --- | --- |
| `full` | real student logs + code + outcomes |
| `no_trace` | code + outcomes, but no logs |
| `trace_shuffled` | code + outcomes + logs from another student on the same exercise |

The goal is not just to find high numbers. The goal is to see whether the model
seems to carry a useful hidden student-state that is behaviorally predictive,
and whether the real logs help reveal that state.

## 1. Predictive Metrics

These ask whether the model can predict meaningful next student behavior.

| Metric | What it means | Why it is here | Headline score | `full` | `no_trace` | `trace_shuffled` | Current read |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `first_event_type` | literal first logged action | catches very local next-step guessing | top-1 | `0.504` | `0.615` | `0.405` | brittle; rewards edit-first prior |
| `first_event_family` | first action, but collapsed to broad family | softer version of the first-event question | top-1 | `0.504` | `0.615` | `0.405` | currently duplicates `first_event_type` on this slice |
| `first_active_event_type` | first meaningful action, ignoring leading pauses | closer to “what does the student actually do next?” | top-1 | `0.632` | `0.632` | `0.612` | much better behaved than literal first event |
| `change_presence` | whether the next episode contains any edit at all | basic edit vs no-edit screen | top-1 | `0.957` | `0.957` | `0.966` | too easy; not very discriminative |
| `run_presence` | whether the student runs locally next | captures debugging / checking behavior | top-1 | `0.735` | `0.521` | `0.543` | one of the strongest wins for `full` |
| `idle_gap_presence` | whether the next episode includes a meaningful pause | may capture hesitation or rhythm | top-1 | `0.607` | `0.667` | `0.552` | mixed and hard to interpret |
| `first_change_line` | where the first edit lands | tests whether the model points at the right repair spot | within `+/-2` top-1 | `0.701` | `0.675` | `0.655` | useful; moderate win for `full` |
| `event_count_buckets` | rough amount of edit / run / pause behavior | coarse repair-style forecast | full bucket top-1 | `0.085` | `0.034` | `0.060` | noisy, but the sub-buckets are sometimes helpful |

### Notes on Predictive Metrics

`first_event_type` looks strong for `no_trace`, but that is mostly because `no_trace`
leans hard into predicting `change` first. It is the clearest example of a metric
that can be gamed by a shallow prior.

`run_presence` is much more interesting. It is hard for a shallow edit-first prior
to fake, and `full` beats both controls clearly.

For `first_change_line`, the exact-line score is harsh. The `+/-2` tolerance is the
better headline because it gives credit when the model points near the right repair
region instead of demanding the exact line number.

## 2. Trajectory Metrics

These ask whether the model captures the shape of the repair episode, not just
one label.

| Metric | What it means | Why it is here | Headline score | `full` | `no_trace` | `trace_shuffled` | Current read |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `event_type_prefix` | whether the very start of the predicted sequence matches | strict early-sequence check | shared-prefix norm | `0.138` | `0.136` | `0.092` | very brittle |
| `event_type_overlap` | whether the predicted and real event types overlap as a multiset | broad episode-shape agreement | top-1 Jaccard | `0.322` | `0.276` | `0.305` | solid win for `full` |
| `event_type_edit_similarity` | edit-distance style similarity on event types | rewards getting sequence content roughly right | top-1 similarity | `0.307` | `0.268` | `0.303` | `full` beats `no_trace`, close to shuffled |
| `event_family_edit_similarity` | same as above, but on broad families | softer sequence similarity | top-1 similarity | `0.307` | `0.268` | `0.303` | currently mirrors event-type version closely |
| `event_type_lcss` | longest common subsequence on event types | rewards partial ordered agreement | top-1 ratio | `0.313` | `0.271` | `0.308` | `full` beats `no_trace`, close to shuffled |
| `event_family_lcss` | same as above on broad families | softer ordered agreement | top-1 ratio | `0.313` | `0.271` | `0.308` | one of the cleaner broad-shape metrics |
| `event_family_dtw` | sequence similarity that allows flexible alignment | measures whether the overall rhythm is similar | top-1 similarity | `0.841` | `0.811` | `0.828` | small but meaningful win for `full` |
| `episode_motif` | compressed repair pattern like `edit->run->submit` | very readable high-level repair shape | top-1 | `0.325` | `0.145` | `0.198` | strongest trajectory metric right now |

### Notes on Trajectory Metrics

This is where `full` looks best overall.

The sequence metrics tell a clearer story than the brittle first-event metrics:
the real logs seem to help the model capture the broad repair shape.

`episode_motif` is especially promising because it is both interpretable and
statistically strong.

## 3. Causal Metrics

These do not live inside one row. They come from comparing conditions.

The causal question is:

- does `full` beat `no_trace`?
- does `full` beat `trace_shuffled`?

If `full > no_trace`, the logs matter at all.
If `full > trace_shuffled`, the true student’s logs may matter more than generic
same-exercise logs.

The current paired statistical comparison is here:

- [stat_comparison.md](../stat_comparison_v1/stat_comparison.md)

### Strongest Current Causal Results

| Metric | `full - no_trace` | Wilcoxon p | `full - trace_shuffled` | Wilcoxon p | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `run_presence` top-1 | `+0.214` | `0.0011` | `+0.190` | `0.0019` | strong evidence that real logs help |
| `episode_motif` top-1 | `+0.180` | `0.0002` | `+0.121` | `0.0196` | strong evidence on high-level repair shape |
| `event_type_overlap` | `+0.046` | `<0.001` | `+0.019` | `0.0730` | clear vs `no_trace`, weaker vs shuffled |
| `event_type_edit_similarity` | `+0.039` | `0.0003` | `+0.006` | `0.5178` | clear vs `no_trace`, not vs shuffled |
| `event_family_lcss` | `+0.042` | `0.0001` | `+0.008` | `0.4631` | clear vs `no_trace`, not vs shuffled |
| `event_family_dtw` | `+0.030` | `0.0065` | `+0.011` | `0.1416` | clear vs `no_trace`, not vs shuffled |
| `first_event_type` top-1 | `-0.111` | `0.0046` | `+0.095` | `0.0630` | misleading metric; favors edit-first prior |

## What the 3 Metric Types Say Right Now

| Metric type | Current result |
| --- | --- |
| `Predictive` | mixed but real signal |
| `Trajectory` | strongest part of the study right now |
| `Causal` | clear evidence that logs help, weaker evidence that the true student’s logs help more than generic shuffled logs |

## Current Best Read

The strongest current claim is:

`GPT-5.4 appears to use richer logs to model the next repair episode in a behaviorally useful way.`

The weaker part is:

`We do not yet have decisive evidence that this is a robust hidden state for this exact student, rather than a strong generic student-in-context state.`

## Good Candidates for the Headline Set

If we had to pick the most useful current metrics, these look strongest:

| Role | Metric |
| --- | --- |
| Predictive | `first_active_event_type` |
| Predictive | `run_presence` |
| Predictive | `first_change_line` within `+/-2` |
| Trajectory | `episode_motif` |
| Trajectory | `event_type_overlap` |
| Trajectory | `event_family_dtw` |
| Causal | `full - no_trace` and `full - trace_shuffled` on the metrics above |

## Likely Diagnostic-Only Metrics

These are still useful, but probably should not headline the claim:

| Metric | Why |
| --- | --- |
| `first_event_type` | too easy to game with an edit-first prior |
| `first_event_family` | currently duplicates `first_event_type` on this slice |
| `change_presence` | near-ceiling and not very discriminative |
| `event_type_prefix` | too brittle |
