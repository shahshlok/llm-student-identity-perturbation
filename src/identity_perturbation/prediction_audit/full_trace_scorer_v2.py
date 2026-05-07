"""v6.2 full-trace scorer, revised suite (opus).

This module is the v2 companion to ``full_trace_scorer.py``.  It keeps the
same public contract (3 hypotheses; repair / trajectory / full-code
families; Oracle@3 / Top-1 / Expected / best-rank views) but fixes the
measurement drifts identified in the v1 audit:

Repair family
    * normalize code before line-diffing so footprint scoring is
      consistent with full-code and gain metrics
    * add a soft (windowed) footprint F1 alongside the strict one; the
      strict anchor model punishes "insert one line off" as total miss
    * compute content similarity per aligned hunk, not over a concatenated
      blob, so positional agreement matters
    * split content similarity into structural vs identifier tokens so
      boilerplate overlap does not drown out student-specific word choice
    * expose a bounded code-gain metric in ``[-1, 1]`` so the Expected
      view is not dominated by one wildly-off hypothesis

Trajectory family
    * replace binary ``local_run_count_match`` with a graded version
    * keep presence match (still useful for quick diagnostics) but also
      add a region-overlap metric so we are not crediting "edit line 2-4"
      vs "edit line 2-4" only when alignment DP hooks them up

Full-code family
    * demote ``full_code_structural_similarity`` (ceiling-pinned) to a
      diagnostic-only field and add ``structural_lift_over_copy`` which
      subtracts the attempt_n-vs-observed baseline so boilerplate does
      not dominate

Aggregation views
    * keep Oracle@3 / Expected / Top-1 / best_rank
    * add a rank-weighted view with weights (0.5, 0.3, 0.2) on the
      probability-ranked hypotheses; this is stable even if the model's
      self-reported probabilities are miscalibrated

None of the v1 files are modified.  The v2 scorer returns a superset
schema ``v6_2_full_trace_scored_prediction_v2`` so downstream aggregators
can tell the versions apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import cache
from typing import Any, Literal

from identity_perturbation.prediction_audit.full_trace_target_schema import (
    FullTraceHypothesis,
    FullTracePredictionResponse,
)
from identity_perturbation.prediction_audit.match_policy import narrow_normalize_code_for_match


class FullTraceScorerV2Error(ValueError):
    pass


PatchKind = Literal["replace", "delete", "insert"]


@dataclass(frozen=True)
class PatchHunk:
    """A single line-level change between two normalized code blobs.

    ``replace`` and ``delete`` carry a half-closed source-line range in
    the ``before_code`` coordinate frame.  ``insert`` carries an
    anchor: the line index in ``before_code`` *after* which new lines
    appear.  Both frames live in the *normalized* before code, not the
    raw code, because the v1 scorer compared raw frames and that meant
    whitespace noise shifted the whole unit set around.
    """

    kind: PatchKind
    source_start_line_0idx: int | None
    source_end_line_0idx: int | None
    insertion_anchor_0idx: int | None
    removed_text: str
    added_text: str


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


_STRUCTURAL_TOKEN_RE = re.compile(r"==|!=|<=|>=|:=|->|[-+*/%<>=(){}\[\],.:;]|\S")
_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")
_NUMBER_TOKEN_RE = re.compile(r"\d+")
_PY_KEYWORDS = frozenset(
    {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
        "match",
        "case",
    }
)


def _tokenize_all(text: str) -> list[str]:
    """Tokenize to a flat list matching the v1 scorer, kept for parity checks."""

    return re.findall(
        r"[A-Za-z_]\w*|\d+|==|!=|<=|>=|:=|->|[-+*/%<>=(){}\[\],.:;]|\S",
        text,
    )


def _split_structural_and_identifier(text: str) -> tuple[list[str], list[str]]:
    """Return (structural, identifier) token lists.

    Keywords and numbers are treated as structural; user-chosen names
    (variables, function names, strings' alphabetic tokens) are treated
    as identifiers.  The split lets us report two separate F1 scores so
    the personalization analysis can see when a model matched the
    *shape* of an edit but not the student's naming choices.
    """

    structural: list[str] = []
    identifier: list[str] = []
    for token in _tokenize_all(text):
        if _IDENTIFIER_TOKEN_RE.fullmatch(token):
            if token in _PY_KEYWORDS:
                structural.append(token)
            else:
                identifier.append(token)
            continue
        if _NUMBER_TOKEN_RE.fullmatch(token):
            structural.append(token)
            continue
        structural.append(token)
    return structural, identifier


def _token_f1(tokens_a: list[str], tokens_b: list[str]) -> float:
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
    if overlap == 0:
        return 0.0
    precision = overlap / len(tokens_a)
    recall = overlap / len(tokens_b)
    return 2.0 * precision * recall / (precision + recall)


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


# ---------------------------------------------------------------------------
# Repair family
# ---------------------------------------------------------------------------


def _build_patch_hunks(*, before_code: str, after_code: str) -> list[PatchHunk]:
    """Diff in *normalized* line space so whitespace noise is absorbed.

    Compare against v1, which diffed the raw strings and therefore could
    flag a pure-whitespace line as changed.  That made footprint scoring
    inconsistent with ``code_gain_over_copy`` (which already normalized)
    and let formatting differences masquerade as semantic edits.
    """

    before_lines = narrow_normalize_code_for_match(before_code).split("\n")
    after_lines = narrow_normalize_code_for_match(after_code).split("\n")
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
        raise FullTraceScorerV2Error(f"Unsupported diff opcode tag: {tag!r}")
    return hunks


def _strict_footprint_units(hunks: list[PatchHunk]) -> set[str]:
    units: set[str] = set()
    for hunk in hunks:
        if hunk.kind == "insert":
            if hunk.insertion_anchor_0idx is None:
                raise FullTraceScorerV2Error("Insert hunk missing insertion anchor")
            units.add(f"A:{hunk.insertion_anchor_0idx}")
            continue
        if hunk.source_start_line_0idx is None or hunk.source_end_line_0idx is None:
            raise FullTraceScorerV2Error("Non-insert hunk missing source line range")
        for line in range(hunk.source_start_line_0idx, hunk.source_end_line_0idx + 1):
            units.add(f"L:{line}")
    return units


def _windowed_footprint_lines(hunks: list[PatchHunk], *, radius: int) -> list[set[int]]:
    """Return one expanded line set per hunk.

    Each hunk contributes ``set(range(start - radius, end + radius + 1))``,
    where an insert at anchor ``k`` is treated as a zero-width span at
    line ``k``.  Radius ``1`` means "one line either side".  The
    per-hunk list (not a flat union) lets callers compute IoU over best
    hunk-to-hunk alignment instead of over coarsely unioned sets.
    """

    if radius < 0:
        raise FullTraceScorerV2Error("Window radius must be non-negative")
    spans: list[set[int]] = []
    for hunk in hunks:
        if hunk.kind == "insert":
            if hunk.insertion_anchor_0idx is None:
                raise FullTraceScorerV2Error("Insert hunk missing insertion anchor")
            center = hunk.insertion_anchor_0idx
            spans.append(set(range(center - radius, center + radius + 1)))
            continue
        if hunk.source_start_line_0idx is None or hunk.source_end_line_0idx is None:
            raise FullTraceScorerV2Error("Non-insert hunk missing source line range")
        start = hunk.source_start_line_0idx - radius
        end = hunk.source_end_line_0idx + radius
        spans.append(set(range(start, end + 1)))
    return spans


def _best_hunk_alignment(
    predicted_spans: list[set[int]],
    observed_spans: list[set[int]],
) -> tuple[list[tuple[int, int, float]], float]:
    """Exact max-sum-IoU bipartite alignment.

    Returns the list of ``(predicted_index, observed_index, iou)``
    pairs that were matched plus the mean IoU across matched pairs.
    Unmatched hunks on either side contribute zero-IoU penalties via
    the F1-style aggregation performed by the caller.

    The v2 draft used a greedy matcher here. That was still an
    approximation: there are small counterexamples where greedy misses
    the best global assignment. Hunk counts in this task are tiny, so a
    bitmask DP gives an exact optimum with simpler correctness
    guarantees than a heuristic.
    """
    if not predicted_spans or not observed_spans:
        return [], 0.0

    transpose = len(predicted_spans) < len(observed_spans)
    if transpose:
        row_spans = observed_spans
        col_spans = predicted_spans
    else:
        row_spans = predicted_spans
        col_spans = observed_spans

    weights: list[list[float]] = []
    for row_span in row_spans:
        row_weights: list[float] = []
        for col_span in col_spans:
            union = row_span | col_span
            if not union:
                row_weights.append(0.0)
                continue
            row_weights.append(len(row_span & col_span) / len(union))
        weights.append(row_weights)

    def better(
        candidate_score: float,
        candidate_count: int,
        best_score: float,
        best_count: int,
    ) -> bool:
        if candidate_score > best_score + 1e-12:
            return True
        if abs(candidate_score - best_score) <= 1e-12 and candidate_count > best_count:
            return True
        return False

    @cache
    def best_from(row_index: int, used_mask: int) -> tuple[float, int]:
        if row_index == len(row_spans):
            return (0.0, 0)
        best_score, best_count = best_from(row_index + 1, used_mask)
        for col_index, weight in enumerate(weights[row_index]):
            if weight <= 0.0 or (used_mask & (1 << col_index)):
                continue
            rem_score, rem_count = best_from(row_index + 1, used_mask | (1 << col_index))
            candidate_score = weight + rem_score
            candidate_count = 1 + rem_count
            if better(candidate_score, candidate_count, best_score, best_count):
                best_score = candidate_score
                best_count = candidate_count
        return (best_score, best_count)

    def reconstruct(row_index: int, used_mask: int) -> list[tuple[int, int, float]]:
        if row_index == len(row_spans):
            return []
        best_score, best_count = best_from(row_index, used_mask)
        for col_index, weight in enumerate(weights[row_index]):
            if weight <= 0.0 or (used_mask & (1 << col_index)):
                continue
            rem_score, rem_count = best_from(row_index + 1, used_mask | (1 << col_index))
            candidate_score = weight + rem_score
            candidate_count = 1 + rem_count
            if abs(candidate_score - best_score) <= 1e-12 and candidate_count == best_count:
                rest = reconstruct(row_index + 1, used_mask | (1 << col_index))
                if transpose:
                    return [(col_index, row_index, weight), *rest]
                return [(row_index, col_index, weight), *rest]
        return reconstruct(row_index + 1, used_mask)

    matched = reconstruct(0, 0)
    mean_iou = sum(iou for _, _, iou in matched) / len(matched) if matched else 0.0
    return matched, mean_iou


def _strict_footprint_metrics(
    predicted_hunks: list[PatchHunk],
    observed_hunks: list[PatchHunk],
) -> dict[str, float]:
    predicted_units = _strict_footprint_units(predicted_hunks)
    observed_units = _strict_footprint_units(observed_hunks)
    overlap = len(predicted_units & observed_units)
    precision = overlap / len(predicted_units) if predicted_units else 0.0
    recall = overlap / len(observed_units) if observed_units else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "strict_footprint_precision": precision,
        "strict_footprint_recall": recall,
        "strict_footprint_f1": f1,
    }


def _windowed_footprint_metrics(
    predicted_hunks: list[PatchHunk],
    observed_hunks: list[PatchHunk],
    *,
    radius: int,
) -> dict[str, float]:
    """F1-style aggregation over best hunk-to-hunk IoU alignment.

    ``precision`` uses the number of predicted hunks as denominator
    (so an extra predicted hunk that has no real counterpart costs
    accuracy).  ``recall`` uses the number of observed hunks.  This is
    the graded generalization of the strict unit-set F1, but with
    near-miss tolerance and positional agreement intact.
    """

    predicted_spans = _windowed_footprint_lines(predicted_hunks, radius=radius)
    observed_spans = _windowed_footprint_lines(observed_hunks, radius=radius)
    if not predicted_spans and not observed_spans:
        return {
            "windowed_footprint_precision": 1.0,
            "windowed_footprint_recall": 1.0,
            "windowed_footprint_f1": 1.0,
            "windowed_footprint_mean_iou": 1.0,
        }
    if not predicted_spans or not observed_spans:
        return {
            "windowed_footprint_precision": 0.0,
            "windowed_footprint_recall": 0.0,
            "windowed_footprint_f1": 0.0,
            "windowed_footprint_mean_iou": 0.0,
        }
    matched, mean_iou = _best_hunk_alignment(predicted_spans, observed_spans)
    precision = sum(iou for _, _, iou in matched) / len(predicted_spans)
    recall = sum(iou for _, _, iou in matched) / len(observed_spans)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "windowed_footprint_precision": precision,
        "windowed_footprint_recall": recall,
        "windowed_footprint_f1": f1,
        "windowed_footprint_mean_iou": mean_iou,
    }


def _aligned_repair_content_similarity(
    predicted_hunks: list[PatchHunk],
    observed_hunks: list[PatchHunk],
    alignment: list[tuple[int, int, float]],
) -> dict[str, float]:
    """Per-aligned-hunk content similarity, split structural vs identifier.

    v1 concatenated all removed and added text blob-wise, then ran a
    single token F1.  That throws away which edit corresponds to which,
    so a model that wrote the right token in the wrong hunk still got
    credit.  Here we score each aligned hunk pair separately and
    average.  Unaligned hunks contribute zero, which is the content-
    side analogue of a missed footprint.
    """

    if not predicted_hunks and not observed_hunks:
        return {
            "aligned_content_f1": 1.0,
            "aligned_content_structural_f1": 1.0,
            "aligned_content_identifier_f1": 1.0,
        }
    if not alignment:
        return {
            "aligned_content_f1": 0.0,
            "aligned_content_structural_f1": 0.0,
            "aligned_content_identifier_f1": 0.0,
        }
    total_hunks_for_penalty = max(len(predicted_hunks), len(observed_hunks))
    combined_scores: list[float] = []
    structural_scores: list[float] = []
    identifier_scores: list[float] = []
    for pred_idx, obs_idx, _iou in alignment:
        pred_hunk = predicted_hunks[pred_idx]
        obs_hunk = observed_hunks[obs_idx]
        pred_text = pred_hunk.removed_text + "\n" + pred_hunk.added_text
        obs_text = obs_hunk.removed_text + "\n" + obs_hunk.added_text
        combined_scores.append(_token_f1(_tokenize_all(pred_text), _tokenize_all(obs_text)))
        pred_struct, pred_ident = _split_structural_and_identifier(pred_text)
        obs_struct, obs_ident = _split_structural_and_identifier(obs_text)
        structural_scores.append(_token_f1(pred_struct, obs_struct))
        identifier_scores.append(_token_f1(pred_ident, obs_ident))
    while len(combined_scores) < total_hunks_for_penalty:
        combined_scores.append(0.0)
        structural_scores.append(0.0)
        identifier_scores.append(0.0)
    return {
        "aligned_content_f1": sum(combined_scores) / len(combined_scores),
        "aligned_content_structural_f1": sum(structural_scores) / len(structural_scores),
        "aligned_content_identifier_f1": sum(identifier_scores) / len(identifier_scores),
    }


def _code_gain_over_copy_raw(
    *,
    before_code: str,
    predicted_code: str,
    observed_code: str,
) -> float:
    """v1-compatible raw gain; kept for diagnostics only.

    ``1 - d_pred / d_base`` is unbounded below and that is exactly what
    wrecked the pilot's ``Expected(code_gain_over_copy)``.  We keep the
    raw number for post-hoc inspection, but aggregation uses the
    bounded version below.
    """

    before_norm = narrow_normalize_code_for_match(before_code)
    predicted_norm = narrow_normalize_code_for_match(predicted_code)
    observed_norm = narrow_normalize_code_for_match(observed_code)
    baseline_distance = _levenshtein_distance(before_norm, observed_norm)
    if baseline_distance == 0:
        raise FullTraceScorerV2Error(
            "Observed code equals attempt_n code; row should be inadmissible"
        )
    predicted_distance = _levenshtein_distance(predicted_norm, observed_norm)
    return 1.0 - (predicted_distance / baseline_distance)


def _code_gain_bounded(
    *,
    before_code: str,
    predicted_code: str,
    observed_code: str,
) -> float:
    """Bounded ``[-1, 1]`` code-gain.

    ``(d_base - d_pred) / max(d_base, d_pred)`` has the same sign as the
    raw formula but is cleanly bounded, so ``Expected`` is numerically
    stable when one hypothesis is wildly off.  At ``d_pred = 0`` it
    returns ``1`` (perfect); at ``d_pred = d_base`` it returns ``0`` (no
    lift); as ``d_pred → ∞`` it approaches ``-1``.
    """

    before_norm = narrow_normalize_code_for_match(before_code)
    predicted_norm = narrow_normalize_code_for_match(predicted_code)
    observed_norm = narrow_normalize_code_for_match(observed_code)
    baseline_distance = _levenshtein_distance(before_norm, observed_norm)
    if baseline_distance == 0:
        raise FullTraceScorerV2Error(
            "Observed code equals attempt_n code; row should be inadmissible"
        )
    predicted_distance = _levenshtein_distance(predicted_norm, observed_norm)
    denominator = max(baseline_distance, predicted_distance)
    return (baseline_distance - predicted_distance) / denominator


def _repair_metrics(
    *,
    before_code: str,
    predicted_code: str,
    observed_code: str,
    window_radius: int,
) -> dict[str, float]:
    predicted_hunks = _build_patch_hunks(before_code=before_code, after_code=predicted_code)
    observed_hunks = _build_patch_hunks(before_code=before_code, after_code=observed_code)
    strict = _strict_footprint_metrics(predicted_hunks, observed_hunks)
    windowed = _windowed_footprint_metrics(
        predicted_hunks,
        observed_hunks,
        radius=window_radius,
    )
    predicted_spans = _windowed_footprint_lines(predicted_hunks, radius=window_radius)
    observed_spans = _windowed_footprint_lines(observed_hunks, radius=window_radius)
    alignment, _ = _best_hunk_alignment(predicted_spans, observed_spans)
    content = _aligned_repair_content_similarity(predicted_hunks, observed_hunks, alignment)
    raw_gain = _code_gain_over_copy_raw(
        before_code=before_code,
        predicted_code=predicted_code,
        observed_code=observed_code,
    )
    bounded_gain = _code_gain_bounded(
        before_code=before_code,
        predicted_code=predicted_code,
        observed_code=observed_code,
    )
    return {
        **strict,
        **windowed,
        **content,
        "code_gain_over_copy_raw": raw_gain,
        "code_gain_over_copy_bounded": bounded_gain,
    }


# ---------------------------------------------------------------------------
# Trajectory family
# ---------------------------------------------------------------------------


def _pre_submit_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not steps or steps[-1]["action_type"] != "submit":
        raise FullTraceScorerV2Error("Trajectory must end in submit")
    return steps[:-1]


def _edit_step_line_set(step: dict[str, Any], *, radius: int) -> set[int]:
    if step["action_type"] != "edit":
        raise FullTraceScorerV2Error("Edit line set requires an edit step")
    start = int(step["target_start_line_0idx"]) - radius
    end = int(step["target_end_line_0idx"]) + radius
    return set(range(start, end + 1))


def _edit_step_iou(step_a: dict[str, Any], step_b: dict[str, Any], *, radius: int) -> float:
    a_set = _edit_step_line_set(step_a, radius=radius)
    b_set = _edit_step_line_set(step_b, radius=radius)
    union = a_set | b_set
    if not union:
        return 0.0
    return len(a_set & b_set) / len(union)


def _aligned_step_pairs(
    predicted_steps: list[dict[str, Any]],
    observed_steps: list[dict[str, Any]],
    *,
    radius: int,
) -> tuple[float, list[tuple[int, int, float]]]:
    pred = _pre_submit_steps(predicted_steps)
    obs = _pre_submit_steps(observed_steps)
    if not pred and not obs:
        return 1.0, []
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
                pair_score = _edit_step_iou(pred[i - 1], obs[j - 1], radius=radius)
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
                pair_score = _edit_step_iou(pred[i - 1], obs[j - 1], radius=radius)
            else:
                raise FullTraceScorerV2Error("Unexpected aligned pair during backtrack")
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
    return dp[-1][-1] / normalizer, aligned_pairs


def _trajectory_region_overlap(
    predicted_steps: list[dict[str, Any]],
    observed_steps: list[dict[str, Any]],
    *,
    radius: int,
) -> float:
    """Unordered line-set IoU of all pre-submit edit regions.

    If the DP alignment breaks (different step counts, different
    orderings) the alignment score can collapse to zero even when both
    sides are poking at the same regions.  The region-overlap view is
    order-insensitive: union all edited line windows on each side and
    take IoU.  This is the "did you touch the same place, regardless of
    exactly when" view.
    """

    pred = [s for s in _pre_submit_steps(predicted_steps) if s["action_type"] == "edit"]
    obs = [s for s in _pre_submit_steps(observed_steps) if s["action_type"] == "edit"]
    if not pred and not obs:
        return 1.0
    pred_lines: set[int] = set()
    for step in pred:
        pred_lines |= _edit_step_line_set(step, radius=radius)
    obs_lines: set[int] = set()
    for step in obs:
        obs_lines |= _edit_step_line_set(step, radius=radius)
    union = pred_lines | obs_lines
    if not union:
        return 0.0
    return len(pred_lines & obs_lines) / len(union)


def _local_run_count_agreement(
    predicted_pre: list[dict[str, Any]], observed_pre: list[dict[str, Any]]
) -> float:
    """Graded count match in ``[0, 1]``.

    ``1 - |p - o| / max(p, o, 1)`` is 1 on equality, 0.5 when the
    prediction is double/half the truth, and floors at 0 for the
    worst-case ratio.  Replaces the binary v1 version which gave no
    credit for "close" counts.
    """

    predicted_runs = sum(1 for step in predicted_pre if step["action_type"] == "local_run")
    observed_runs = sum(1 for step in observed_pre if step["action_type"] == "local_run")
    denominator = max(predicted_runs, observed_runs, 1)
    return 1.0 - abs(predicted_runs - observed_runs) / denominator


def _trajectory_metrics(
    *,
    predicted_steps: list[dict[str, Any]],
    observed_steps: list[dict[str, Any]],
    edit_window_radius: int,
) -> dict[str, float]:
    alignment_score, aligned_pairs = _aligned_step_pairs(
        predicted_steps,
        observed_steps,
        radius=edit_window_radius,
    )
    predicted_pre = _pre_submit_steps(predicted_steps)
    observed_pre = _pre_submit_steps(observed_steps)
    edit_overlaps = [
        score
        for pred_index, obs_index, score in aligned_pairs
        if predicted_pre[pred_index]["action_type"]
        == observed_pre[obs_index]["action_type"]
        == "edit"
    ]
    edit_span_overlap = sum(edit_overlaps) / len(edit_overlaps) if edit_overlaps else 0.0
    predicted_run_count = sum(1 for step in predicted_pre if step["action_type"] == "local_run")
    observed_run_count = sum(1 for step in observed_pre if step["action_type"] == "local_run")
    return {
        "trajectory_alignment_score": alignment_score,
        "edit_span_overlap": edit_span_overlap,
        "edit_region_overlap_unordered": _trajectory_region_overlap(
            predicted_steps,
            observed_steps,
            radius=edit_window_radius,
        ),
        "local_run_presence_match": float((predicted_run_count > 0) == (observed_run_count > 0)),
        "local_run_count_agreement": _local_run_count_agreement(predicted_pre, observed_pre),
    }


# ---------------------------------------------------------------------------
# Full-code family
# ---------------------------------------------------------------------------


def _structural_lift_over_copy(
    *,
    before_code: str,
    predicted_code: str,
    observed_code: str,
) -> float:
    """Lift of whole-file token-sequence ratio over the stay-put baseline.

    ``full_code_structural_similarity`` pegs near 1.0 on every row
    because boilerplate dominates the token sequence.  Subtracting the
    attempt_n-vs-observed baseline removes that ceiling and exposes the
    incremental similarity the prediction actually buys over doing
    nothing.  Can be positive, zero, or negative.
    """

    before_tokens = _tokenize_all(narrow_normalize_code_for_match(before_code))
    predicted_tokens = _tokenize_all(narrow_normalize_code_for_match(predicted_code))
    observed_tokens = _tokenize_all(narrow_normalize_code_for_match(observed_code))
    if not observed_tokens:
        return 0.0
    predicted_ratio = SequenceMatcher(a=predicted_tokens, b=observed_tokens, autojunk=False).ratio()
    baseline_ratio = SequenceMatcher(a=before_tokens, b=observed_tokens, autojunk=False).ratio()
    return predicted_ratio - baseline_ratio


def _full_code_metrics(
    *,
    before_code: str,
    predicted_code: str,
    observed_code: str,
) -> dict[str, float]:
    predicted_norm = narrow_normalize_code_for_match(predicted_code)
    observed_norm = narrow_normalize_code_for_match(observed_code)
    structural_similarity_diagnostic = SequenceMatcher(
        a=_tokenize_all(predicted_norm),
        b=_tokenize_all(observed_norm),
        autojunk=False,
    ).ratio()
    lift = _structural_lift_over_copy(
        before_code=before_code,
        predicted_code=predicted_code,
        observed_code=observed_code,
    )
    bounded_gain = _code_gain_bounded(
        before_code=before_code,
        predicted_code=predicted_code,
        observed_code=observed_code,
    )
    return {
        "exact_next_code_match": float(predicted_norm == observed_norm),
        "structural_lift_over_copy": lift,
        "full_code_structural_similarity_diagnostic": structural_similarity_diagnostic,
        "worse_than_copy_rate": float(bounded_gain < 0.0),
    }


# ---------------------------------------------------------------------------
# Aggregation across the 3 hypotheses
# ---------------------------------------------------------------------------


RANK_WEIGHTS: tuple[float, float, float] = (0.5, 0.3, 0.2)


def _aggregate_metric(
    *,
    hypothesis_scores: list[float],
    probabilities: list[float],
) -> dict[str, float | int]:
    """Produce all four v1 views plus the new rank-weighted view.

    The rank-weighted view does not use the model's self-reported
    probabilities at all; it assigns fixed weights to the probability-
    sorted hypotheses.  That is what you want when you are unsure the
    model's ``estimated_probability`` is calibrated, which is always
    true in this setting.
    """

    if len(hypothesis_scores) != 3 or len(probabilities) != 3:
        raise FullTraceScorerV2Error("Expected exactly 3 hypothesis scores and 3 probabilities")
    best_score = max(hypothesis_scores)
    top1_index = max(range(3), key=lambda index: probabilities[index])
    ranked_indices = sorted(range(3), key=lambda index: probabilities[index], reverse=True)
    best_indices = [i for i, s in enumerate(hypothesis_scores) if s == best_score]
    best_rank = min(ranked_indices.index(i) + 1 for i in best_indices)
    expected = sum(probabilities[i] * hypothesis_scores[i] for i in range(3))
    rank_weighted = sum(
        RANK_WEIGHTS[rank_pos] * hypothesis_scores[ranked_indices[rank_pos]]
        for rank_pos in range(3)
    )
    return {
        "oracle_at_3": best_score,
        "expected": expected,
        "rank_weighted": rank_weighted,
        "top_1": hypothesis_scores[top1_index],
        "best_hypothesis_rank": best_rank,
    }


# ---------------------------------------------------------------------------
# Observed payload validation (reuses v1 schema versions)
# ---------------------------------------------------------------------------


def _validate_observed_repair_target(target: dict[str, Any]) -> None:
    if target.get("schema_version") != "v6_2_observed_next_repair_target_v1":
        raise FullTraceScorerV2Error("Observed repair target schema_version is invalid")
    if not isinstance(target.get("attempt_n"), dict):
        raise FullTraceScorerV2Error("Observed repair target missing attempt_n")
    if not isinstance(target.get("attempt_n1"), dict):
        raise FullTraceScorerV2Error("Observed repair target missing attempt_n1")


def _validate_observed_coarse_path(path: dict[str, Any]) -> None:
    if path.get("schema_version") != "v6_2_observed_coarse_path_v1":
        raise FullTraceScorerV2Error("Observed coarse path schema_version is invalid")
    attempt_n1 = path.get("attempt_n1")
    if not isinstance(attempt_n1, dict):
        raise FullTraceScorerV2Error("Observed coarse path missing attempt_n1")
    steps = attempt_n1.get("coarse_path_steps")
    if not isinstance(steps, list) or not steps:
        raise FullTraceScorerV2Error("Observed coarse path missing coarse_path_steps")
    if steps[-1].get("action_type") != "submit":
        raise FullTraceScorerV2Error("Observed coarse path must end in submit")
    if not any(isinstance(step, dict) and step.get("action_type") == "edit" for step in steps[:-1]):
        raise FullTraceScorerV2Error(
            "Observed coarse path must include at least one pre-submit edit"
        )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


SCHEMA_VERSION = "v6_2_full_trace_scored_prediction_v2"

DEFAULT_FOOTPRINT_WINDOW_RADIUS = 1
DEFAULT_EDIT_STEP_WINDOW_RADIUS = 1


def _hypothesis_to_dict(hypothesis: FullTraceHypothesis) -> dict[str, Any]:
    return hypothesis.model_dump(mode="json")


def score_full_trace_prediction_v2(
    *,
    response_payload: dict[str, Any],
    observed_repair_target: dict[str, Any],
    observed_coarse_path: dict[str, Any],
    footprint_window_radius: int = DEFAULT_FOOTPRINT_WINDOW_RADIUS,
    edit_step_window_radius: int = DEFAULT_EDIT_STEP_WINDOW_RADIUS,
) -> dict[str, Any]:
    """Score a validated 3-hypothesis response against observed targets.

    Returns a ``v6_2_full_trace_scored_prediction_v2`` object with
    per-hypothesis metrics for every family and the five aggregation
    views per metric.
    """

    _validate_observed_repair_target(observed_repair_target)
    _validate_observed_coarse_path(observed_coarse_path)
    response = FullTracePredictionResponse.model_validate(response_payload)

    attempt_n_code = str(observed_repair_target["attempt_n"]["code"])
    observed_code = str(observed_repair_target["attempt_n1"]["code"])
    observed_steps_raw = observed_coarse_path["attempt_n1"]["coarse_path_steps"]
    if not isinstance(observed_steps_raw, list):
        raise FullTraceScorerV2Error("Observed coarse path steps must be a list")
    observed_steps: list[dict[str, Any]] = list(observed_steps_raw)

    repair_metric_names = [
        "strict_footprint_precision",
        "strict_footprint_recall",
        "strict_footprint_f1",
        "windowed_footprint_precision",
        "windowed_footprint_recall",
        "windowed_footprint_f1",
        "windowed_footprint_mean_iou",
        "aligned_content_f1",
        "aligned_content_structural_f1",
        "aligned_content_identifier_f1",
        "code_gain_over_copy_raw",
        "code_gain_over_copy_bounded",
    ]
    trajectory_metric_names = [
        "trajectory_alignment_score",
        "edit_span_overlap",
        "edit_region_overlap_unordered",
        "local_run_presence_match",
        "local_run_count_agreement",
    ]
    full_code_metric_names = [
        "exact_next_code_match",
        "structural_lift_over_copy",
        "full_code_structural_similarity_diagnostic",
        "worse_than_copy_rate",
    ]

    per_hypothesis: list[dict[str, Any]] = []
    for hypothesis in response.hypotheses:
        predicted_steps = [
            step.model_dump(mode="json") for step in hypothesis.predicted_next_trajectory
        ]
        repair = _repair_metrics(
            before_code=attempt_n_code,
            predicted_code=hypothesis.predicted_next_code,
            observed_code=observed_code,
            window_radius=footprint_window_radius,
        )
        trajectory = _trajectory_metrics(
            predicted_steps=predicted_steps,
            observed_steps=observed_steps,
            edit_window_radius=edit_step_window_radius,
        )
        full_code = _full_code_metrics(
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
                "repair": repair,
                "trajectory": trajectory,
                "full_code": full_code,
                "raw_hypothesis": _hypothesis_to_dict(hypothesis),
            }
        )

    probabilities = [float(item["estimated_probability"]) for item in per_hypothesis]

    def aggregate_family(metric_names: list[str], family_key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric_name in metric_names:
            out[metric_name] = _aggregate_metric(
                hypothesis_scores=[float(item[family_key][metric_name]) for item in per_hypothesis],
                probabilities=probabilities,
            )
        return out

    return {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "footprint_window_radius": footprint_window_radius,
            "edit_step_window_radius": edit_step_window_radius,
            "rank_weights": list(RANK_WEIGHTS),
        },
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


__all__ = [
    "FullTraceScorerV2Error",
    "SCHEMA_VERSION",
    "DEFAULT_FOOTPRINT_WINDOW_RADIUS",
    "DEFAULT_EDIT_STEP_WINDOW_RADIUS",
    "RANK_WEIGHTS",
    "score_full_trace_prediction_v2",
]
