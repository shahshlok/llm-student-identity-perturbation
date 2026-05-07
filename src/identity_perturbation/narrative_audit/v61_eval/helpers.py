from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Hashable, Iterable
from typing import Any

from .types import EvaluationExample, Hypothesis

EVENT_FAMILY_BY_TYPE = {
    "change": "edit",
    "saida_testar": "run",
    "submit": "submit",
    "kill_program": "runtime",
    "keyHandled": "navigate",
    "tab_click": "navigate",
    "idle_gap": "pause",
}


def event_family(event_type: str) -> str:
    if event_type not in EVENT_FAMILY_BY_TYPE:
        raise ValueError(f"Unknown event_type for family mapping: {event_type}")
    return EVENT_FAMILY_BY_TYPE[event_type]


def event_types(events: Iterable[dict[str, Any]]) -> list[str]:
    return [str(event["event_type"]) for event in events]


def event_families(events: Iterable[dict[str, Any]]) -> list[str]:
    return [event_family(str(event["event_type"])) for event in events]


def first_event_type(events: Iterable[dict[str, Any]]) -> str:
    events = tuple(events)
    if not events:
        raise ValueError("Event list is empty")
    return str(events[0]["event_type"])


def first_non_idle_event_type(events: Iterable[dict[str, Any]]) -> str | None:
    for event in events:
        event_type = str(event["event_type"])
        if event_type != "idle_gap":
            return event_type
    return None


def contains_type(events: Iterable[dict[str, Any]], event_type: str) -> bool:
    return any(str(event["event_type"]) == event_type for event in events)


def first_change_line(events: Iterable[dict[str, Any]]) -> int | None:
    for event in events:
        if str(event["event_type"]) != "change":
            continue
        if "primary_line_0idx" in event:
            line = int(event["primary_line_0idx"])
            return None if line < 0 else line
        if "from_line" in event:
            return int(event["from_line"])
        from_position = event.get("from")
        if isinstance(from_position, dict) and "line" in from_position:
            return int(from_position["line"])
        raise ValueError("change event missing primary_line_0idx / from_line / from.line")
    return None


def top_hypothesis(example: EvaluationExample) -> Hypothesis:
    return max(
        example.hypotheses,
        key=lambda hypothesis: (hypothesis.probability, hypothesis.label),
    )


def truth_mass(
    example: EvaluationExample,
    predicate: Callable[[tuple[dict[str, Any], ...]], bool],
) -> float:
    return sum(
        hypothesis.probability for hypothesis in example.hypotheses if predicate(hypothesis.events)
    )


def expected_score(
    example: EvaluationExample,
    score_fn: Callable[[tuple[dict[str, Any], ...]], float],
) -> float:
    return sum(
        hypothesis.probability * score_fn(hypothesis.events) for hypothesis in example.hypotheses
    )


def hypothesis_rank(
    example: EvaluationExample,
    predicate: Callable[[tuple[dict[str, Any], ...]], bool],
) -> int | None:
    ranked = sorted(
        example.hypotheses,
        key=lambda hypothesis: (-hypothesis.probability, hypothesis.label),
    )
    for index, hypothesis in enumerate(ranked, start=1):
        if predicate(hypothesis.events):
            return index
    return None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / rank


def topk_hit(rank: int | None, k: int) -> bool:
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    return rank is not None and rank <= k


def categorical_distribution(
    example: EvaluationExample,
    label_fn: Callable[[tuple[dict[str, Any], ...]], Hashable],
) -> dict[str, float]:
    distribution: dict[str, float] = {}
    for hypothesis in example.hypotheses:
        label = str(label_fn(hypothesis.events))
        distribution[label] = distribution.get(label, 0.0) + hypothesis.probability
    return distribution


def top_label(distribution: dict[str, float]) -> str:
    if not distribution:
        raise ValueError("Distribution is empty")
    return max(distribution.items(), key=lambda item: (item[1], item[0]))[0]


def truth_probability_from_distribution(distribution: dict[str, float], truth_label: str) -> float:
    return float(distribution.get(truth_label, 0.0))


def label_rank(distribution: dict[str, float], truth_label: str) -> int | None:
    ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    for index, (label, _) in enumerate(ranked, start=1):
        if label == truth_label:
            return index
    return None


def multiclass_brier_score(distribution: dict[str, float], truth_label: str) -> float:
    labels = set(distribution)
    labels.add(truth_label)
    total = 0.0
    for label in labels:
        predicted = distribution.get(label, 0.0)
        observed = 1.0 if label == truth_label else 0.0
        total += (predicted - observed) ** 2
    return total


def safe_log_loss(probability_of_truth: float, *, epsilon: float = 1e-12) -> float:
    clipped = min(max(probability_of_truth, epsilon), 1.0)
    return -math.log(clipped)


def binary_brier_score(predicted_true_probability: float, observed_true: bool) -> float:
    target = 1.0 if observed_true else 0.0
    return (predicted_true_probability - target) ** 2


def line_within_tolerance(
    predicted_line: int | None,
    observed_line: int | None,
    tolerance: int,
) -> bool | None:
    if observed_line is None:
        return None
    if predicted_line is None:
        return False
    return abs(predicted_line - observed_line) <= tolerance


def shared_prefix_length(left: list[str], right: list[str]) -> int:
    count = 0
    for lval, rval in zip(left, right, strict=False):
        if lval != rval:
            break
        count += 1
    return count


def exact_prefix_match(left: list[str], right: list[str], k: int) -> bool:
    if k < 1:
        raise ValueError(f"prefix k must be positive, got {k}")
    return left[:k] == right[:k]


def multiset_jaccard(left: list[str], right: list[str]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    keys = set(left_counts) | set(right_counts)
    if not keys:
        return 1.0
    intersection = sum(min(left_counts[key], right_counts[key]) for key in keys)
    union = sum(max(left_counts[key], right_counts[key]) for key in keys)
    if union == 0:
        return 1.0
    return intersection / union


def normalized_edit_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    distance = _levenshtein_distance(left, right)
    scale = max(len(left), len(right), 1)
    return 1.0 - (distance / scale)


def lcss_ratio(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    common = _lcss_length(left, right)
    scale = max(len(left), len(right), 1)
    return common / scale


def dtw_similarity(left: list[str], right: list[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    cost = _dtw_cost(left, right)
    scale = max(len(left), len(right), 1)
    similarity = 1.0 - (cost / scale)
    return max(0.0, similarity)


def compress_consecutive(values: list[str]) -> list[str]:
    if not values:
        return []
    compressed = [values[0]]
    for value in values[1:]:
        if value != compressed[-1]:
            compressed.append(value)
    return compressed


def episode_motif_from_events(events: Iterable[dict[str, Any]]) -> str:
    families = compress_consecutive(event_families(events))
    return "->".join(families) if families else "empty"


def count_bucket(count: int) -> str:
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def event_count_bucket_tuple(events: Iterable[dict[str, Any]]) -> str:
    types = event_types(events)
    edit_bucket = count_bucket(sum(1 for item in types if item == "change"))
    run_bucket = count_bucket(sum(1 for item in types if item == "saida_testar"))
    pause_bucket = count_bucket(sum(1 for item in types if item == "idle_gap"))
    return f"edit={edit_bucket}|run={run_bucket}|pause={pause_bucket}"


def summarize_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row[field] is not None]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def summarize_rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row[field] is not None]
    if not values:
        return None
    return sum(1.0 for value in values if bool(value)) / len(values)


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, lval in enumerate(left, start=1):
        cur = [i]
        for j, rval in enumerate(right, start=1):
            substitution = 0 if lval == rval else 1
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + substitution,
                )
            )
        prev = cur
    return prev[-1]


def _lcss_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, lval in enumerate(left, start=1):
        for j, rval in enumerate(right, start=1):
            if lval == rval:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _dtw_cost(left: list[str], right: list[str]) -> float:
    inf = float("inf")
    dp = [[inf] * (len(right) + 1) for _ in range(len(left) + 1)]
    dp[0][0] = 0.0
    for i, lval in enumerate(left, start=1):
        for j, rval in enumerate(right, start=1):
            cost = 0.0 if lval == rval else 1.0
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[-1][-1]
