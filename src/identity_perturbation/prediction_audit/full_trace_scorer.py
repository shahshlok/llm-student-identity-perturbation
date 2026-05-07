from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from identity_perturbation.prediction_audit.full_trace_target_schema import (
    FullTraceHypothesis,
    FullTracePredictionResponse,
)
from identity_perturbation.prediction_audit.match_policy import narrow_normalize_code_for_match


class FullTraceScorerError(ValueError):
    pass


PatchKind = Literal["replace", "delete", "insert"]


@dataclass(frozen=True)
class PatchHunk:
    kind: PatchKind
    source_start_line_0idx: int | None
    source_end_line_0idx: int | None
    insertion_anchor_0idx: int | None
    removed_text: str
    added_text: str


def _validate_observed_repair_target(observed_repair_target: dict[str, Any]) -> None:
    if observed_repair_target.get("schema_version") != "v6_2_observed_next_repair_target_v1":
        raise FullTraceScorerError("Observed repair target schema_version is invalid")
    if not isinstance(observed_repair_target.get("attempt_n"), dict):
        raise FullTraceScorerError("Observed repair target missing attempt_n")
    if not isinstance(observed_repair_target.get("attempt_n1"), dict):
        raise FullTraceScorerError("Observed repair target missing attempt_n1")


def _validate_observed_coarse_path(observed_coarse_path: dict[str, Any]) -> None:
    if observed_coarse_path.get("schema_version") != "v6_2_observed_coarse_path_v1":
        raise FullTraceScorerError("Observed coarse path schema_version is invalid")
    attempt_n1 = observed_coarse_path.get("attempt_n1")
    if not isinstance(attempt_n1, dict):
        raise FullTraceScorerError("Observed coarse path missing attempt_n1")
    steps = attempt_n1.get("coarse_path_steps")
    if not isinstance(steps, list) or not steps:
        raise FullTraceScorerError("Observed coarse path missing coarse_path_steps")
    if steps[-1].get("action_type") != "submit":
        raise FullTraceScorerError("Observed coarse path must end in submit")
    if not any(isinstance(step, dict) and step.get("action_type") == "edit" for step in steps[:-1]):
        raise FullTraceScorerError("Observed coarse path must include at least one pre-submit edit")


def _build_patch_hunks(*, before_code: str, after_code: str) -> list[PatchHunk]:
    before_lines = before_code.split("\n")
    after_lines = after_code.split("\n")
    matcher = SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    hunks: list[PatchHunk] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            hunks.append(
                PatchHunk(
                    kind="replace",
                    source_start_line_0idx=i1,
                    source_end_line_0idx=i2 - 1,
                    insertion_anchor_0idx=None,
                    removed_text="\n".join(before_lines[i1:i2]),
                    added_text="\n".join(after_lines[j1:j2]),
                )
            )
            continue
        if tag == "delete":
            hunks.append(
                PatchHunk(
                    kind="delete",
                    source_start_line_0idx=i1,
                    source_end_line_0idx=i2 - 1,
                    insertion_anchor_0idx=None,
                    removed_text="\n".join(before_lines[i1:i2]),
                    added_text="",
                )
            )
            continue
        if tag == "insert":
            hunks.append(
                PatchHunk(
                    kind="insert",
                    source_start_line_0idx=None,
                    source_end_line_0idx=None,
                    insertion_anchor_0idx=i1,
                    removed_text="",
                    added_text="\n".join(after_lines[j1:j2]),
                )
            )
            continue
        raise FullTraceScorerError(f"Unsupported diff opcode tag: {tag!r}")
    return hunks


def _footprint_units(hunks: list[PatchHunk]) -> set[str]:
    units: set[str] = set()
    for hunk in hunks:
        if hunk.kind == "insert":
            if hunk.insertion_anchor_0idx is None:
                raise FullTraceScorerError("Insert hunk missing insertion anchor")
            units.add(f"A:{hunk.insertion_anchor_0idx}")
            continue
        if hunk.source_start_line_0idx is None or hunk.source_end_line_0idx is None:
            raise FullTraceScorerError("Non-insert hunk missing source line range")
        for line in range(hunk.source_start_line_0idx, hunk.source_end_line_0idx + 1):
            units.add(f"L:{line}")
    return units


def _tokenize_delta_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|:=|->|[-+*/%<>=(){}\[\],.:;]|\S", text)


def _token_f1(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize_delta_text(text_a)
    tokens_b = _tokenize_delta_text(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    counts_a: dict[str, int] = {}
    counts_b: dict[str, int] = {}
    for token in tokens_a:
        counts_a[token] = counts_a.get(token, 0) + 1
    for token in tokens_b:
        counts_b[token] = counts_b.get(token, 0) + 1
    overlap = 0
    for token, count_a in counts_a.items():
        overlap += min(count_a, counts_b.get(token, 0))
    precision = overlap / len(tokens_a)
    recall = overlap / len(tokens_b)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _repair_content_similarity(
    predicted_hunks: list[PatchHunk], observed_hunks: list[PatchHunk]
) -> float:
    predicted_removed = "\n".join(
        hunk.removed_text for hunk in predicted_hunks if hunk.removed_text
    )
    observed_removed = "\n".join(hunk.removed_text for hunk in observed_hunks if hunk.removed_text)
    predicted_added = "\n".join(hunk.added_text for hunk in predicted_hunks if hunk.added_text)
    observed_added = "\n".join(hunk.added_text for hunk in observed_hunks if hunk.added_text)

    component_scores: list[float] = []
    if predicted_removed or observed_removed:
        component_scores.append(_token_f1(predicted_removed, observed_removed))
    if predicted_added or observed_added:
        component_scores.append(_token_f1(predicted_added, observed_added))
    if not component_scores:
        return 1.0
    return sum(component_scores) / len(component_scores)


def _levenshtein_distance(text_a: str, text_b: str) -> int:
    if text_a == text_b:
        return 0
    if len(text_a) < len(text_b):
        text_a, text_b = text_b, text_a
    previous = list(range(len(text_b) + 1))
    for i, char_a in enumerate(text_a, start=1):
        current = [i]
        for j, char_b in enumerate(text_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _code_gain_over_copy(*, before_code: str, predicted_code: str, observed_code: str) -> float:
    before_norm = narrow_normalize_code_for_match(before_code)
    predicted_norm = narrow_normalize_code_for_match(predicted_code)
    observed_norm = narrow_normalize_code_for_match(observed_code)
    baseline_distance = _levenshtein_distance(before_norm, observed_norm)
    if baseline_distance == 0:
        raise FullTraceScorerError(
            "Observed code is identical to attempt_n code; expected a real edit row"
        )
    predicted_distance = _levenshtein_distance(predicted_norm, observed_norm)
    return 1.0 - (predicted_distance / baseline_distance)


def _full_code_structural_similarity(*, predicted_code: str, observed_code: str) -> float:
    predicted_tokens = _tokenize_delta_text(narrow_normalize_code_for_match(predicted_code))
    observed_tokens = _tokenize_delta_text(narrow_normalize_code_for_match(observed_code))
    if not predicted_tokens and not observed_tokens:
        return 1.0
    if not predicted_tokens or not observed_tokens:
        return 0.0
    return SequenceMatcher(a=predicted_tokens, b=observed_tokens, autojunk=False).ratio()


def _line_span_iou(step_a: dict[str, Any], step_b: dict[str, Any]) -> float:
    if step_a["action_type"] != "edit" or step_b["action_type"] != "edit":
        raise FullTraceScorerError("Line span IoU requires two edit steps")
    a_start = int(step_a["target_start_line_0idx"])
    a_end = int(step_a["target_end_line_0idx"])
    b_start = int(step_b["target_start_line_0idx"])
    b_end = int(step_b["target_end_line_0idx"])
    a_set = set(range(a_start, a_end + 1))
    b_set = set(range(b_start, b_end + 1))
    union = a_set | b_set
    if not union:
        raise FullTraceScorerError("Edit step union cannot be empty")
    return len(a_set & b_set) / len(union)


def _pre_submit_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not steps or steps[-1]["action_type"] != "submit":
        raise FullTraceScorerError("Trajectory must end in submit")
    return steps[:-1]


def _aligned_step_pairs(
    predicted_steps: list[dict[str, Any]],
    observed_steps: list[dict[str, Any]],
) -> tuple[float, list[tuple[int, int, float]]]:
    pred = _pre_submit_steps(predicted_steps)
    obs = _pre_submit_steps(observed_steps)
    if not pred and not obs:
        return (1.0, [])
    rows = len(pred) + 1
    cols = len(obs) + 1
    dp = [[0.0 for _ in range(cols)] for _ in range(rows)]
    take_diag = [[False for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            pair_score = 0.0
            pred_type = pred[i - 1]["action_type"]
            obs_type = obs[j - 1]["action_type"]
            if pred_type == obs_type == "local_run":
                pair_score = 1.0
            elif pred_type == obs_type == "edit":
                pair_score = _line_span_iou(pred[i - 1], obs[j - 1])
            diag = dp[i - 1][j - 1] + pair_score
            up = dp[i - 1][j]
            left = dp[i][j - 1]
            best = max(diag, up, left)
            dp[i][j] = best
            take_diag[i][j] = best == diag and pair_score > 0.0
    i = len(pred)
    j = len(obs)
    aligned_pairs: list[tuple[int, int, float]] = []
    while i > 0 and j > 0:
        if take_diag[i][j]:
            pred_type = pred[i - 1]["action_type"]
            obs_type = obs[j - 1]["action_type"]
            if pred_type == obs_type == "local_run":
                pair_score = 1.0
            elif pred_type == obs_type == "edit":
                pair_score = _line_span_iou(pred[i - 1], obs[j - 1])
            else:
                raise FullTraceScorerError("Unexpected aligned pair during backtrack")
            aligned_pairs.append((i - 1, j - 1, pair_score))
            i -= 1
            j -= 1
            continue
        if dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    aligned_pairs.reverse()
    normalizer = max(len(pred), len(obs), 1)
    return (dp[-1][-1] / normalizer, aligned_pairs)


def _trajectory_metrics(
    *,
    predicted_steps: list[dict[str, Any]],
    observed_steps: list[dict[str, Any]],
) -> dict[str, float]:
    alignment_score, aligned_pairs = _aligned_step_pairs(predicted_steps, observed_steps)
    predicted_pre = _pre_submit_steps(predicted_steps)
    observed_pre = _pre_submit_steps(observed_steps)
    edit_overlaps = [
        score
        for pred_index, obs_index, score in aligned_pairs
        if predicted_pre[pred_index]["action_type"]
        == observed_pre[obs_index]["action_type"]
        == "edit"
    ]
    predicted_run_count = sum(1 for step in predicted_pre if step["action_type"] == "local_run")
    observed_run_count = sum(1 for step in observed_pre if step["action_type"] == "local_run")
    return {
        "trajectory_alignment_score": alignment_score,
        "edit_span_overlap": (sum(edit_overlaps) / len(edit_overlaps) if edit_overlaps else 0.0),
        "local_run_presence_match": float((predicted_run_count > 0) == (observed_run_count > 0)),
        "local_run_count_match": float(predicted_run_count == observed_run_count),
    }


def _repair_metrics(
    *,
    before_code: str,
    predicted_code: str,
    observed_code: str,
) -> dict[str, float]:
    predicted_hunks = _build_patch_hunks(before_code=before_code, after_code=predicted_code)
    observed_hunks = _build_patch_hunks(before_code=before_code, after_code=observed_code)
    predicted_units = _footprint_units(predicted_hunks)
    observed_units = _footprint_units(observed_hunks)
    overlap = len(predicted_units & observed_units)
    precision = overlap / len(predicted_units) if predicted_units else 0.0
    recall = overlap / len(observed_units) if observed_units else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "repair_footprint_precision": precision,
        "repair_footprint_recall": recall,
        "repair_footprint_f1": f1,
        "repair_content_similarity": _repair_content_similarity(predicted_hunks, observed_hunks),
        "code_gain_over_copy": _code_gain_over_copy(
            before_code=before_code,
            predicted_code=predicted_code,
            observed_code=observed_code,
        ),
    }


def _full_code_metrics(
    *,
    before_code: str,
    predicted_code: str,
    observed_code: str,
) -> dict[str, float]:
    predicted_norm = narrow_normalize_code_for_match(predicted_code)
    observed_norm = narrow_normalize_code_for_match(observed_code)
    code_gain = _code_gain_over_copy(
        before_code=before_code,
        predicted_code=predicted_code,
        observed_code=observed_code,
    )
    return {
        "exact_next_code_match": float(predicted_norm == observed_norm),
        "full_code_structural_similarity": _full_code_structural_similarity(
            predicted_code=predicted_code,
            observed_code=observed_code,
        ),
        "worse_than_copy_rate": float(code_gain < 0.0),
    }


def _aggregate_metric(
    *,
    hypothesis_scores: list[float],
    probabilities: list[float],
) -> dict[str, float | int]:
    if len(hypothesis_scores) != 3 or len(probabilities) != 3:
        raise FullTraceScorerError("Expected exactly 3 hypothesis scores and 3 probabilities")
    best_score = max(hypothesis_scores)
    top1_index = max(range(3), key=lambda index: probabilities[index])
    ranked_indices = sorted(range(3), key=lambda index: probabilities[index], reverse=True)
    best_indices = [index for index, score in enumerate(hypothesis_scores) if score == best_score]
    best_rank = min(ranked_indices.index(index) + 1 for index in best_indices)
    expected = sum(probabilities[index] * hypothesis_scores[index] for index in range(3))
    return {
        "oracle_at_3": best_score,
        "expected": expected,
        "best_hypothesis_rank": best_rank,
        "top_1": hypothesis_scores[top1_index],
    }


def _hypothesis_to_dict(hypothesis: FullTraceHypothesis) -> dict[str, Any]:
    return hypothesis.model_dump(mode="json")


def score_full_trace_prediction(
    *,
    response_payload: dict[str, Any],
    observed_repair_target: dict[str, Any],
    observed_coarse_path: dict[str, Any],
) -> dict[str, Any]:
    _validate_observed_repair_target(observed_repair_target)
    _validate_observed_coarse_path(observed_coarse_path)
    response = FullTracePredictionResponse.model_validate(response_payload)

    attempt_n_code = str(observed_repair_target["attempt_n"]["code"])
    observed_code = str(observed_repair_target["attempt_n1"]["code"])
    observed_steps = observed_coarse_path["attempt_n1"]["coarse_path_steps"]  # type: ignore[index]
    if not isinstance(observed_steps, list):
        raise FullTraceScorerError("Observed coarse path steps must be a list")

    per_hypothesis: list[dict[str, Any]] = []
    repair_metric_names = [
        "repair_footprint_precision",
        "repair_footprint_recall",
        "repair_footprint_f1",
        "repair_content_similarity",
        "code_gain_over_copy",
    ]
    trajectory_metric_names = [
        "trajectory_alignment_score",
        "edit_span_overlap",
        "local_run_presence_match",
        "local_run_count_match",
    ]
    full_code_metric_names = [
        "exact_next_code_match",
        "full_code_structural_similarity",
        "worse_than_copy_rate",
    ]

    for hypothesis in response.hypotheses:
        hypothesis_dict = _hypothesis_to_dict(hypothesis)
        predicted_steps = [
            step.model_dump(mode="json") for step in hypothesis.predicted_next_trajectory
        ]
        repair_metrics = _repair_metrics(
            before_code=attempt_n_code,
            predicted_code=hypothesis.predicted_next_code,
            observed_code=observed_code,
        )
        trajectory_metrics = _trajectory_metrics(
            predicted_steps=predicted_steps,
            observed_steps=observed_steps,
        )
        full_code_metrics = _full_code_metrics(
            before_code=attempt_n_code,
            predicted_code=hypothesis.predicted_next_code,
            observed_code=observed_code,
        )
        per_hypothesis.append(
            {
                "label": hypothesis.label,
                "estimated_probability": hypothesis.estimated_probability,
                "predicted_next_code": hypothesis.predicted_next_code,
                "predicted_next_trajectory": predicted_steps,
                "repair": repair_metrics,
                "trajectory": trajectory_metrics,
                "full_code": full_code_metrics,
                "raw_hypothesis": hypothesis_dict,
            }
        )

    probabilities = [float(item["estimated_probability"]) for item in per_hypothesis]

    def aggregate_family(metric_names: list[str], family_key: str) -> dict[str, Any]:
        family: dict[str, Any] = {}
        for metric_name in metric_names:
            family[metric_name] = _aggregate_metric(
                hypothesis_scores=[float(item[family_key][metric_name]) for item in per_hypothesis],
                probabilities=probabilities,
            )
        return family

    return {
        "schema_version": "v6_2_full_trace_scored_prediction_v1",
        "observed": {
            "attempt_n_code": attempt_n_code,
            "attempt_n1_code": observed_code,
            "coarse_path_steps": observed_steps,
        },
        "per_hypothesis": per_hypothesis,
        "views": {
            "repair": aggregate_family(repair_metric_names, "repair"),
            "trajectory": aggregate_family(trajectory_metric_names, "trajectory"),
            "full_code": aggregate_family(full_code_metric_names, "full_code"),
        },
    }
