"""Automated cognitive-accuracy proxies for v6.1 student_state_summary analysis.

Produces:
  - data/v61_condition_comparisons/cognitive_accuracy_tests.json
  - data/v61_condition_comparisons/cognitive_accuracy_tests.md

Run with:
    uv run python -m identity_perturbation.narrative_audit.v61_cognitive_accuracy_tests
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy import stats as scipy_stats

from identity_perturbation.codebench_support.openai_batch import _save_json, _utc_now
from identity_perturbation.narrative_audit.v61_eval.helpers import first_change_line, top_hypothesis
from identity_perturbation.narrative_audit.v61_eval.loader import build_examples, load_hydrated_predictions

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = ROOT / "data" / "v61_condition_comparisons"
BATCH_ROOT = ROOT / "data" / "v61_batch_runs"

CONDITION_RUNS = {
    "full": BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_v6",
    "no_trace": BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_no_trace_v1",
    "trace_shuffled": BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1",
}

CONDITION_PAIRS = [
    ("full", "no_trace"),
    ("full", "trace_shuffled"),
]

COGNITIVE_VERBS = (
    "notice",
    "believe",
    "understand",
    "see",
    "realize",
    "recognize",
    "think",
    "assume",
    "expect",
)

ACTION_PATTERNS = {
    "saida_testar": (
        r"\brun\b",
        r"\bruns\b",
        r"\btest\b",
        r"\btests\b",
        r"\btesting\b",
        r"\brerun\b",
        r"\breruns\b",
    ),
    "change": (
        r"\bfix\b",
        r"\bfixes\b",
        r"\bedit\b",
        r"\bedits\b",
        r"\brewrite\b",
        r"\brewrites\b",
        r"\bchange\b",
        r"\bchanges\b",
        r"\bmodify\b",
        r"\bmodifies\b",
        r"\bpatch\b",
        r"\bpatches\b",
        r"\bdelete\b",
        r"\bdeletes\b",
        r"\badd\b",
        r"\badds\b",
        r"\binsert\b",
        r"\binserts\b",
        r"\bremove\b",
        r"\bremoves\b",
        r"\bindent\b",
        r"\bindentation\b",
    ),
    "submit": (
        r"\bsubmit\b",
        r"\bsubmits\b",
        r"\bresubmit\b",
        r"\bresubmits\b",
    ),
}

LINE_RANGE_RE = re.compile(
    r"\blines?\s*-?\s*(\d+)\s*(?:-|–|to|through)\s*(\d+)\b",
    re.IGNORECASE,
)
LINE_SINGLE_RE = re.compile(r"\bline\s*-?\s*(\d+)\b", re.IGNORECASE)
LAST_LINE_RE = re.compile(r"\blast line\b", re.IGNORECASE)
BACKTICK_RE = re.compile(r"`([^`]+)`")
CALL_LIKE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
COGNITIVE_VERB_RE = re.compile(
    r"\b(?:notic(?:e|es|ed|ing)|believ(?:e|es|ed|ing)|understand(?:s|ing|stood)?|"
    r"sees?|seeing|saw|realiz(?:e|es|ed|ing)|recogniz(?:e|es|ed|ing)|"
    r"think(?:s|ing|thought)?|assum(?:e|es|ed|ing)|expect(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SummaryRecord:
    custom_id: str
    class_id: str
    assessment_id: str
    exercise_id: str
    student_id: str
    transition_index_0idx: int
    summary: str
    attempt_n_code: str
    attempt_n_identifier_set: frozenset[str]
    observed_event_types: tuple[str, ...]
    observed_first_change_line_0idx: int | None
    top1_jaccard: float
    top1_similarity: float
    composite_accuracy: float
    line_mentions_0idx: tuple[int, ...]
    line_match_within_2: float | None
    action_types: tuple[str, ...]
    action_alignment_score: float | None
    word_count: int
    code_reference_count: int
    cognitive_verb_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_attempt_n_code(payload_path: Path) -> str:
    payload = _load_json(payload_path)
    attempt_n = payload.get("attempt_n")
    if not isinstance(attempt_n, dict) or "code" not in attempt_n:
        raise ValueError(f"Attempt-n code missing from payload: {payload_path}")
    code = attempt_n["code"]
    if not isinstance(code, str):
        raise ValueError(f"Attempt-n code is not a string in payload: {payload_path}")
    return code


def _extract_code_identifiers(code: str) -> frozenset[str]:
    identifiers = {match.group(0).lower() for match in IDENTIFIER_RE.finditer(code)}
    return frozenset(identifiers)


def _line_count(code: str) -> int:
    return len(code.splitlines()) if code else 0


def _normalize_human_line_number(line_number_1idx: int) -> int:
    if line_number_1idx <= 0:
        return 0
    return line_number_1idx - 1


def _extract_line_mentions(summary: str, code: str) -> tuple[int, ...]:
    lines: set[int] = set()
    for match in LINE_RANGE_RE.finditer(summary):
        start = int(match.group(1))
        end = int(match.group(2))
        lo, hi = sorted((start, end))
        lines.update(range(_normalize_human_line_number(lo), _normalize_human_line_number(hi) + 1))
    for match in LINE_SINGLE_RE.finditer(summary):
        lines.add(_normalize_human_line_number(int(match.group(1))))
    if LAST_LINE_RE.search(summary):
        n_lines = _line_count(code)
        if n_lines > 0:
            lines.add(n_lines - 1)
    return tuple(sorted(line for line in lines if line >= 0))


def _line_match_within_tolerance(
    line_mentions_0idx: tuple[int, ...],
    observed_line_0idx: int | None,
    tolerance: int = 2,
) -> float | None:
    if not line_mentions_0idx or observed_line_0idx is None:
        return None
    return float(any(abs(line - observed_line_0idx) <= tolerance for line in line_mentions_0idx))


def _extract_action_types(summary: str) -> tuple[str, ...]:
    lowered = summary.lower()
    action_types = []
    for event_type, patterns in ACTION_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            action_types.append(event_type)
    return tuple(sorted(action_types))


def _action_alignment_score(
    predicted_action_types: tuple[str, ...],
    observed_event_types: tuple[str, ...],
) -> float | None:
    if not predicted_action_types:
        return None
    observed_set = set(observed_event_types)
    overlap = sum(1 for event_type in predicted_action_types if event_type in observed_set)
    return overlap / len(predicted_action_types)


def _word_count(summary: str) -> int:
    return len(summary.split())


def _extract_code_like_references(summary: str, code_identifiers: frozenset[str]) -> int:
    count = 0
    count += len(LINE_RANGE_RE.findall(summary))
    count += len(LINE_SINGLE_RE.findall(summary))
    count += len(LAST_LINE_RE.findall(summary))
    count += len(BACKTICK_RE.findall(summary))
    count += len(CALL_LIKE_RE.findall(summary))
    summary_identifiers = {
        match.group(0).lower()
        for match in IDENTIFIER_RE.finditer(summary)
        if match.group(0).lower() in code_identifiers
    }
    count += len(summary_identifiers)
    return count


def _cognitive_verb_count(summary: str) -> int:
    return len(COGNITIVE_VERB_RE.findall(summary))


def _safe_float(value: Any) -> float:
    return float(value)


def _index_evaluation_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload["scores"]["rows"]
    return {str(row["custom_id"]): row for row in rows}


def _load_condition_records(run_dir: Path) -> dict[str, SummaryRecord]:
    hydrated_path = run_dir / "hydrated_predictions.json"
    evaluation_path = run_dir / "evaluation.json"

    hydrated_payload = load_hydrated_predictions(hydrated_path)
    examples = build_examples(hydrated_payload=hydrated_payload, prefix_k=1)
    example_by_id = {example.custom_id: example for example in examples}
    raw_predictions = {str(row["custom_id"]): row for row in hydrated_payload["predictions"]}
    evaluation_payload = _load_json(evaluation_path)
    evaluation_by_id = _index_evaluation_rows(evaluation_payload)

    available_ids = sorted(set(example_by_id) & set(raw_predictions) & set(evaluation_by_id))
    if not available_ids:
        raise ValueError(f"No shared custom_ids across hydrated/evaluation artifacts in {run_dir}")

    records: dict[str, SummaryRecord] = {}
    for custom_id in available_ids:
        example = example_by_id[custom_id]
        raw_prediction = raw_predictions[custom_id]
        eval_row = evaluation_by_id[custom_id]
        top = top_hypothesis(example)

        payload_path = Path(str(raw_prediction["payload_path"]))
        attempt_n_code = _extract_attempt_n_code(payload_path)
        identifier_set = _extract_code_identifiers(attempt_n_code)

        observed_first_change_line_0idx = first_change_line(example.observed_events)
        eval_observed_first_change = eval_row["metrics"]["first_change_line"]["observed"]
        if observed_first_change_line_0idx != eval_observed_first_change:
            raise ValueError(
                "Observed first-change line mismatch for "
                f"{custom_id}: helper={observed_first_change_line_0idx}, "
                f"evaluation={eval_observed_first_change}"
            )

        line_mentions_0idx = _extract_line_mentions(top.student_state_summary, attempt_n_code)
        action_types = _extract_action_types(top.student_state_summary)
        observed_event_types = tuple(str(event["event_type"]) for event in example.observed_events)
        top1_jaccard = _safe_float(eval_row["metrics"]["event_type_overlap"]["top1_jaccard"])
        top1_similarity = _safe_float(
            eval_row["metrics"]["event_type_edit_similarity"]["top1_similarity"]
        )

        record = SummaryRecord(
            custom_id=custom_id,
            class_id=example.class_id,
            assessment_id=example.assessment_id,
            exercise_id=example.exercise_id,
            student_id=example.student_id,
            transition_index_0idx=example.transition_index_0idx,
            summary=top.student_state_summary,
            attempt_n_code=attempt_n_code,
            attempt_n_identifier_set=identifier_set,
            observed_event_types=observed_event_types,
            observed_first_change_line_0idx=observed_first_change_line_0idx,
            top1_jaccard=top1_jaccard,
            top1_similarity=top1_similarity,
            composite_accuracy=(top1_jaccard + top1_similarity) / 2.0,
            line_mentions_0idx=line_mentions_0idx,
            line_match_within_2=_line_match_within_tolerance(
                line_mentions_0idx,
                observed_first_change_line_0idx,
            ),
            action_types=action_types,
            action_alignment_score=_action_alignment_score(action_types, observed_event_types),
            word_count=_word_count(top.student_state_summary),
            code_reference_count=_extract_code_like_references(
                top.student_state_summary,
                identifier_set,
            ),
            cognitive_verb_count=_cognitive_verb_count(top.student_state_summary),
        )
        records[custom_id] = record

    return records


def _condition_metadata(run_dir: Path, records: dict[str, SummaryRecord]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir.resolve()),
        "hydrated_predictions_path": str((run_dir / "hydrated_predictions.json").resolve()),
        "evaluation_path": str((run_dir / "evaluation.json").resolve()),
        "n_rows": len(records),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _spearman(values_x: list[float], values_y: list[float]) -> dict[str, Any]:
    if len(values_x) != len(values_y):
        raise ValueError("Spearman inputs must have equal length")
    if len(values_x) < 2:
        return {"n": len(values_x), "rho": None, "p_value": None}
    result = scipy_stats.spearmanr(values_x, values_y)
    rho = None if math.isnan(float(result.statistic)) else float(result.statistic)
    p_value = None if math.isnan(float(result.pvalue)) else float(result.pvalue)
    return {"n": len(values_x), "rho": rho, "p_value": p_value}


def _paired_non_null_values(
    left: dict[str, SummaryRecord],
    right: dict[str, SummaryRecord],
    field_name: str,
) -> tuple[list[float], list[float], list[str]]:
    shared_ids = sorted(set(left) & set(right))
    vals_left: list[float] = []
    vals_right: list[float] = []
    used_ids: list[str] = []
    for custom_id in shared_ids:
        left_val = getattr(left[custom_id], field_name)
        right_val = getattr(right[custom_id], field_name)
        if left_val is None or right_val is None:
            continue
        vals_left.append(float(left_val))
        vals_right.append(float(right_val))
        used_ids.append(custom_id)
    return vals_left, vals_right, used_ids


def _paired_wilcoxon(
    left: dict[str, SummaryRecord],
    right: dict[str, SummaryRecord],
    field_name: str,
) -> dict[str, Any]:
    vals_left, vals_right, used_ids = _paired_non_null_values(left, right, field_name)
    if not used_ids:
        return {
            "n_paired": 0,
            "mean_left": None,
            "mean_right": None,
            "mean_delta": None,
            "wilcoxon_p": None,
        }
    diffs = [lval - rval for lval, rval in zip(vals_left, vals_right, strict=True)]
    if all(diff == 0.0 for diff in diffs):
        p_value = 1.0
    else:
        try:
            p_value = float(scipy_stats.wilcoxon(diffs, alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
    return {
        "n_paired": len(used_ids),
        "mean_left": _mean(vals_left),
        "mean_right": _mean(vals_right),
        "mean_delta": _mean(diffs),
        "wilcoxon_p": p_value,
    }


def _test1_line_matching(condition_records: dict[str, dict[str, SummaryRecord]]) -> dict[str, Any]:
    per_condition: dict[str, Any] = {}
    for label, records in condition_records.items():
        rows = list(records.values())
        mention_count = sum(1 for row in rows if row.line_match_within_2 is not None)
        hits = [row.line_match_within_2 for row in rows if row.line_match_within_2 is not None]
        per_condition[label] = {
            "n_rows": len(rows),
            "n_line_mentions": mention_count,
            "line_mention_rate": mention_count / len(rows),
            "hit_rate_within_2": _mean([float(hit) for hit in hits]),
        }

    pairwise = {}
    for left_label, right_label in CONDITION_PAIRS:
        pairwise[f"{left_label}_vs_{right_label}"] = _paired_wilcoxon(
            condition_records[left_label],
            condition_records[right_label],
            "line_match_within_2",
        )

    return {
        "description": "Whether explicit line mentions are close to the observed first changed line.",
        "per_condition": per_condition,
        "pairwise": pairwise,
    }


def _test2_length_accuracy_correlation(
    condition_records: dict[str, dict[str, SummaryRecord]],
) -> dict[str, Any]:
    per_condition = {}
    for label, records in condition_records.items():
        rows = list(records.values())
        word_counts = [float(row.word_count) for row in rows]
        top1_jaccard = [row.top1_jaccard for row in rows]
        top1_similarity = [row.top1_similarity for row in rows]
        per_condition[label] = {
            "summary_word_count_vs_top1_jaccard": _spearman(word_counts, top1_jaccard),
            "summary_word_count_vs_top1_similarity": _spearman(word_counts, top1_similarity),
        }
    return {
        "description": "Whether longer top summaries correlate with better top-1 prediction accuracy.",
        "per_condition": per_condition,
    }


def _test3_action_alignment(
    condition_records: dict[str, dict[str, SummaryRecord]],
) -> dict[str, Any]:
    per_condition: dict[str, Any] = {}
    for label, records in condition_records.items():
        rows = list(records.values())
        scores = [
            row.action_alignment_score for row in rows if row.action_alignment_score is not None
        ]
        coverage = len(scores)
        per_condition[label] = {
            "n_rows": len(rows),
            "n_action_extractable": coverage,
            "action_extraction_rate": coverage / len(rows),
            "mean_alignment_score": _mean([float(score) for score in scores]),
        }

    pairwise = {}
    for left_label, right_label in CONDITION_PAIRS:
        pairwise[f"{left_label}_vs_{right_label}"] = _paired_wilcoxon(
            condition_records[left_label],
            condition_records[right_label],
            "action_alignment_score",
        )

    return {
        "description": "Whether action types implied by the summary appear in the observed next episode.",
        "per_condition": per_condition,
        "pairwise": pairwise,
    }


def _test4_specificity_accuracy_correlation(
    condition_records: dict[str, dict[str, SummaryRecord]],
) -> dict[str, Any]:
    per_condition = {}
    for label, records in condition_records.items():
        rows = list(records.values())
        composite_accuracy = [row.composite_accuracy for row in rows]
        per_condition[label] = {
            "word_count_vs_composite_accuracy": _spearman(
                [float(row.word_count) for row in rows],
                composite_accuracy,
            ),
            "code_reference_count_vs_composite_accuracy": _spearman(
                [float(row.code_reference_count) for row in rows],
                composite_accuracy,
            ),
            "cognitive_verb_count_vs_composite_accuracy": _spearman(
                [float(row.cognitive_verb_count) for row in rows],
                composite_accuracy,
            ),
        }
    return {
        "description": "Whether more specific summaries correlate with higher composite trajectory accuracy.",
        "per_condition": per_condition,
        "composite_accuracy": "(event_type_overlap.top1_jaccard + event_type_edit_similarity.top1_similarity) / 2",
    }


def run_analysis() -> dict[str, Any]:
    condition_records = {
        label: _load_condition_records(run_dir) for label, run_dir in CONDITION_RUNS.items()
    }

    return {
        "schema_version": "v6_1_cognitive_accuracy_tests_v1",
        "created_at": _utc_now(),
        "conditions": {
            label: _condition_metadata(CONDITION_RUNS[label], records)
            for label, records in condition_records.items()
        },
        "test1_line_matching": _test1_line_matching(condition_records),
        "test2_length_accuracy_correlation": _test2_length_accuracy_correlation(condition_records),
        "test3_action_alignment": _test3_action_alignment(condition_records),
        "test4_specificity_accuracy_correlation": _test4_specificity_accuracy_correlation(
            condition_records
        ),
    }


def _format_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _takeaway_from_pair(pair_payload: dict[str, Any], left_label: str, right_label: str) -> str:
    delta = pair_payload["mean_delta"]
    if delta is None:
        return f"No paired comparison available for `{left_label}` vs `{right_label}`."
    direction = left_label if delta > 0 else right_label if delta < 0 else "neither"
    if direction == "neither":
        return f"`{left_label}` and `{right_label}` are tied on the paired subset."
    return (
        f"`{direction}` is higher on the paired subset "
        f"(delta `{_format_number(delta)}`, p `{_format_number(pair_payload['wilcoxon_p'])}`)."
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cognitive Accuracy Tests",
        "",
        f"- Created at: `{report['created_at']}`",
        "",
        "## Conditions",
        "",
        "| Condition | Rows | Run Dir |",
        "| --- | ---: | --- |",
    ]
    for label, payload in report["conditions"].items():
        lines.append(f"| {label} | {payload['n_rows']} | `{payload['run_dir']}` |")

    test1 = report["test1_line_matching"]
    lines.extend(
        [
            "",
            "## Test 1: Line Mention Matching",
            "",
            test1["description"],
            "",
            "| Condition | Line mentions | Mention rate | Hit rate within +-2 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label, payload in test1["per_condition"].items():
        lines.append(
            f"| {label} | {payload['n_line_mentions']} | {_format_number(payload['line_mention_rate'])} | "
            f"{_format_number(payload['hit_rate_within_2'])} |"
        )
    for pair_name, payload in test1["pairwise"].items():
        left_label, right_label = pair_name.split("_vs_")
        lines.extend(
            [
                "",
                f"- `{left_label}` vs `{right_label}` paired n: `{payload['n_paired']}`",
                f"- Paired means: `{_format_number(payload['mean_left'])}` vs `{_format_number(payload['mean_right'])}`",
                f"- Wilcoxon p: `{_format_number(payload['wilcoxon_p'])}`",
                f"- Takeaway: {_takeaway_from_pair(payload, left_label, right_label)}",
            ]
        )

    test2 = report["test2_length_accuracy_correlation"]
    lines.extend(
        [
            "",
            "## Test 2: Length as Cognitive Proxy",
            "",
            test2["description"],
            "",
            "| Condition | Word count vs Jaccard rho | p | Word count vs edit similarity rho | p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, payload in test2["per_condition"].items():
        jaccard = payload["summary_word_count_vs_top1_jaccard"]
        similarity = payload["summary_word_count_vs_top1_similarity"]
        lines.append(
            f"| {label} | {_format_number(jaccard['rho'])} | {_format_number(jaccard['p_value'])} | "
            f"{_format_number(similarity['rho'])} | {_format_number(similarity['p_value'])} |"
        )

    test3 = report["test3_action_alignment"]
    lines.extend(
        [
            "",
            "## Test 3: Action-Type Alignment",
            "",
            test3["description"],
            "",
            "| Condition | Action extraction n | Extraction rate | Mean alignment |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label, payload in test3["per_condition"].items():
        lines.append(
            f"| {label} | {payload['n_action_extractable']} | "
            f"{_format_number(payload['action_extraction_rate'])} | "
            f"{_format_number(payload['mean_alignment_score'])} |"
        )
    for pair_name, payload in test3["pairwise"].items():
        left_label, right_label = pair_name.split("_vs_")
        lines.extend(
            [
                "",
                f"- `{left_label}` vs `{right_label}` paired n: `{payload['n_paired']}`",
                f"- Paired means: `{_format_number(payload['mean_left'])}` vs `{_format_number(payload['mean_right'])}`",
                f"- Wilcoxon p: `{_format_number(payload['wilcoxon_p'])}`",
                f"- Takeaway: {_takeaway_from_pair(payload, left_label, right_label)}",
            ]
        )

    test4 = report["test4_specificity_accuracy_correlation"]
    lines.extend(
        [
            "",
            "## Test 4: Specificity-Accuracy Correlation",
            "",
            test4["description"],
            "",
            f"- Composite accuracy: `{test4['composite_accuracy']}`",
            "",
            "| Condition | Word count rho | p | Code refs rho | p | Cognitive verbs rho | p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, payload in test4["per_condition"].items():
        word = payload["word_count_vs_composite_accuracy"]
        code = payload["code_reference_count_vs_composite_accuracy"]
        verbs = payload["cognitive_verb_count_vs_composite_accuracy"]
        lines.append(
            f"| {label} | {_format_number(word['rho'])} | {_format_number(word['p_value'])} | "
            f"{_format_number(code['rho'])} | {_format_number(code['p_value'])} | "
            f"{_format_number(verbs['rho'])} | {_format_number(verbs['p_value'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = run_analysis()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "cognitive_accuracy_tests.json"
    md_path = out_root / "cognitive_accuracy_tests.md"
    _save_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
