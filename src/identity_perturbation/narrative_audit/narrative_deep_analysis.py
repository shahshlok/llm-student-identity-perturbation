from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_PATH = (
    REPO_ROOT
    / "data/narrative_audit/full_trace/hydrated_predictions.json"
)
NO_TRACE_PATH = (
    REPO_ROOT
    / "data/narrative_audit/no_trace/hydrated_predictions.json"
)
PROMPT_PATH = REPO_ROOT / "src/identity_perturbation/narrative_audit/v61_prompting.py"
OUTPUT_PATH = REPO_ROOT / "data/narrative_audit/comparisons/narrative_deep_analysis.json"


WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_'-]*")
SENTENCE_SPLIT_RE = re.compile(r"[;,\.]|\band\b", flags=re.IGNORECASE)
LINE_REF_RE = re.compile(r"\bline[- ]?(\d+)\b", flags=re.IGNORECASE)
OPENING_TARGET = ("the", "student", "likely")

CATEGORY_KEYWORDS = {
    "cognitive_attribution": {
        "notice",
        "notices",
        "noticed",
        "believe",
        "believes",
        "thinks",
        "think",
        "understands",
        "understand",
        "realizes",
        "realize",
        "sees",
        "see",
        "recognizes",
        "recognize",
        "suspects",
        "suspect",
        "infers",
        "infer",
        "considers",
        "consider",
    },
    "behavioral_description": {
        "edit",
        "edits",
        "change",
        "changes",
        "fix",
        "fixes",
        "patch",
        "patches",
        "rewrite",
        "rewrites",
        "delete",
        "deletes",
        "add",
        "adds",
        "test",
        "tests",
        "run",
        "runs",
        "submit",
        "submits",
        "rerun",
        "resubmit",
        "restore",
        "replace",
        "removing",
        "remove",
    },
    "evidence_reference": {
        "line",
        "lines",
        "output",
        "test",
        "tests",
        "result",
        "results",
        "error",
        "errors",
        "code",
        "trace",
        "prior",
        "attempt",
        "loop",
        "variable",
        "print",
        "if",
        "for",
        "version",
        "assignment",
        "condition",
        "bound",
    },
    "hedging": {
        "likely",
        "may",
        "probably",
        "appears",
        "seems",
        "might",
        "possibly",
    },
}

TRACE_WORDS = {
    "pause",
    "paused",
    "idle",
    "delay",
    "timing",
    "keystroke",
    "keystrokes",
    "edit",
    "edits",
    "delete",
    "deletes",
    "insert",
    "inserts",
    "revert",
    "reverts",
    "backtrack",
    "backtracks",
    "revisit",
    "revisits",
    "continue",
    "continuing",
}

CODE_REFERENCE_PATTERNS = [
    r"\bfor loop\b",
    r"\bwhile loop\b",
    r"\binner loop\b",
    r"\bouter loop\b",
    r"\bif condition\b",
    r"\belif\b",
    r"\belse branch\b",
    r"\bprint statement\b",
    r"\bprint\b",
    r"\bloop bound\b",
    r"\bcondition\b",
    r"\bvariable\b",
    r"\bfunction\b",
    r"\bindex\b",
    r"\bassignment\b",
    r"\bexpression\b",
    r"\bdecrement\b",
    r"\bincrement\b",
    r"\boutput line\b",
    r"\bline[- ]?\d+\b",
]

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "both",
    "but",
    "by",
    "can",
    "close",
    "did",
    "do",
    "does",
    "done",
    "each",
    "exact",
    "for",
    "from",
    "given",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "i",
    "if",
    "immediate",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "last",
    "latest",
    "leave",
    "like",
    "likely",
    "look",
    "looks",
    "made",
    "make",
    "makes",
    "may",
    "might",
    "more",
    "mostly",
    "now",
    "of",
    "on",
    "only",
    "or",
    "other",
    "output",
    "problem",
    "rather",
    "really",
    "same",
    "seems",
    "short",
    "so",
    "solution",
    "still",
    "student",
    "student's",
    "studently",
    "students",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "try",
    "up",
    "use",
    "using",
    "very",
    "visible",
    "we",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
}

STRUCTURAL_FILLER_WORDS = {
    "appears",
    "close",
    "immediate",
    "mostly",
    "now",
    "only",
    "visible",
    "likely",
    "seems",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def text_ngrams(text: str, min_n: int, max_n: int) -> set[str]:
    tokens = tokenize(text)
    ngrams: set[str] = set()
    for n in range(min_n, max_n + 1):
        for index in range(len(tokens) - n + 1):
            chunk = tokens[index : index + n]
            if len(chunk) != n:
                continue
            if all(token in STOPWORDS for token in chunk):
                continue
            ngrams.add(" ".join(chunk))
    return ngrams


def leading_ngrams(text: str, min_n: int = 3, max_n: int = 6) -> list[str]:
    tokens = tokenize(text)
    phrases: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(tokens) >= n:
            phrases.append(" ".join(tokens[:n]))
    return phrases


def average_ranks(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    order = np.argsort(arr)
    ranks = np.zeros(len(arr), dtype=float)
    index = 0
    while index < len(arr):
        start = index
        value = arr[order[index]]
        while index < len(arr) and arr[order[index]] == value:
            index += 1
        avg_rank = (start + 1 + index) / 2.0
        for pos in range(start, index):
            ranks[order[pos]] = avg_rank
    return ranks


def wilcoxon_signed_rank(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int | None]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    raw_diffs = x_arr - y_arr
    nonzero_mask = raw_diffs != 0
    diffs = raw_diffs[nonzero_mask]
    if diffs.size == 0:
        return {
            "n_nonzero": 0,
            "w_positive": 0.0,
            "w_negative": 0.0,
            "z": 0.0,
            "p_value_two_sided": 1.0,
            "median_difference": 0.0,
            "mean_difference": 0.0,
        }
    abs_diffs = np.abs(diffs)
    ranks = average_ranks(abs_diffs)
    w_positive = float(ranks[diffs > 0].sum())
    w_negative = float(ranks[diffs < 0].sum())
    n = int(diffs.size)
    tie_counts = Counter(abs_diffs.tolist())
    tie_correction = sum(count**3 - count for count in tie_counts.values())
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction) / 24.0
    mean_w = n * (n + 1) / 4.0
    if variance <= 0:
        z = 0.0
        p_value = 1.0
    else:
        continuity = 0.5 if w_positive > mean_w else -0.5 if w_positive < mean_w else 0.0
        z = (w_positive - mean_w - continuity) / math.sqrt(variance)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "n_nonzero": n,
        "w_positive": w_positive,
        "w_negative": w_negative,
        "z": z,
        "p_value_two_sided": p_value,
        "median_difference": float(np.median(raw_diffs)),
        "mean_difference": float(np.mean(raw_diffs)),
    }


def parse_clause_counts(summary: str) -> dict[str, int]:
    counts = dict.fromkeys(CATEGORY_KEYWORDS, 0)
    counts["other"] = 0
    clauses = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(summary) if chunk.strip()]
    for clause in clauses:
        tokens = set(tokenize(clause))
        matched = False
        for category, keywords in CATEGORY_KEYWORDS.items():
            if tokens & keywords:
                counts[category] += 1
                matched = True
        if not matched:
            counts["other"] += 1
    counts["total_clauses"] = len(clauses)
    return counts


def extract_code_references(summary: str) -> dict[str, object]:
    lowered = summary.lower()
    line_refs = [f"line {match.group(1)}" for match in LINE_REF_RE.finditer(summary)]
    matched_patterns: list[str] = []
    for pattern in CODE_REFERENCE_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            matched_patterns.append(match.group(0).lower())

    code_context_tokens = set()
    tokens = tokenize(summary)
    for index, token in enumerate(tokens):
        if "_" in token or any(char.isdigit() for char in token):
            code_context_tokens.add(token)
            continue
        if token in STOPWORDS or len(token) < 2:
            continue
        left = tokens[index - 1] if index > 0 else ""
        if left in {"line", "loop", "loops", "variable", "variables", "function", "functions"}:
            code_context_tokens.add(token)

    trace_hits = [token for token in tokens if token in TRACE_WORDS]
    specific_refs = set(line_refs) | set(matched_patterns) | code_context_tokens
    return {
        "specific_references": sorted(specific_refs),
        "specific_reference_count": len(specific_refs),
        "line_reference_count": len(line_refs),
        "trace_language_count": len(trace_hits),
        "token_count": len(tokens),
        "trace_hits": trace_hits,
    }


def extract_prompt_example(prompt_source: str) -> str | None:
    match = re.search(r'"student_state_summary"\s*:\s*"([^"]+)"', prompt_source)
    return match.group(1) if match else None


def extract_prompt_strings(prompt_source: str) -> list[str]:
    tree = ast.parse(prompt_source)
    strings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if value:
                strings.append(value)
    return strings


def load_predictions(path: Path) -> list[dict[str, object]]:
    data = read_json(path)
    predictions = data["predictions"]
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        hypothesis = prediction["response"]["next_episode_hypotheses"][0]
        rows.append(
            {
                "custom_id": prediction["custom_id"],
                "class_id": prediction["class_id"],
                "assessment_id": prediction["assessment_id"],
                "exercise_id": prediction["exercise_id"],
                "student_id": prediction["student_id"],
                "transition_index_0idx": prediction["transition_index_0idx"],
                "summary": hypothesis["student_state_summary"],
            }
        )
    return rows


def pair_predictions(
    full_rows: list[dict[str, object]], no_trace_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    full_map = {row["custom_id"]: row for row in full_rows}
    no_map = {row["custom_id"]: row for row in no_trace_rows}
    if set(full_map) != set(no_map):
        missing_in_full = sorted(set(no_map) - set(full_map))
        missing_in_no = sorted(set(full_map) - set(no_map))
        raise ValueError(
            f"Prediction keys do not align. Missing in full={missing_in_full[:5]}, missing in no_trace={missing_in_no[:5]}"
        )
    pairs = []
    for custom_id in sorted(full_map):
        full_row = full_map[custom_id]
        no_row = no_map[custom_id]
        pairs.append(
            {
                "custom_id": custom_id,
                "exercise_id": full_row["exercise_id"],
                "student_id": full_row["student_id"],
                "transition_index_0idx": full_row["transition_index_0idx"],
                "full_summary": full_row["summary"],
                "no_trace_summary": no_row["summary"],
            }
        )
    return pairs


def analysis_template_origin(
    prompt_source: str,
    prompt_strings: list[str],
    full_rows: list[dict[str, object]],
    no_trace_rows: list[dict[str, object]],
) -> dict[str, object]:
    prompt_example = extract_prompt_example(prompt_source)
    prompt_ngrams: set[str] = set()
    for fragment in prompt_strings:
        prompt_ngrams |= text_ngrams(fragment, 3, 6)

    all_rows = [("full", row["summary"]) for row in full_rows] + [
        ("no_trace", row["summary"]) for row in no_trace_rows
    ]
    outputs_reusing_prompt_phrases = []
    literal_example_reuse_count = 0
    for condition, summary in all_rows:
        if prompt_example and prompt_example.lower() in summary.lower():
            literal_example_reuse_count += 1
        overlap = sorted(prompt_ngrams & text_ngrams(summary, 3, 6))
        if overlap:
            outputs_reusing_prompt_phrases.append(
                {
                    "condition": condition,
                    "summary": summary,
                    "matched_prompt_phrases": overlap[:10],
                }
            )

    leading_counter = Counter()
    leading_sources = defaultdict(set)
    for condition, summary in all_rows:
        for phrase in leading_ngrams(summary):
            leading_counter[phrase] += 1
            leading_sources[phrase].add(condition)

    common_leads = {phrase: count for phrase, count in leading_counter.items() if count >= 5}
    prompt_derived = {
        phrase: count for phrase, count in common_leads.items() if phrase in prompt_ngrams
    }
    model_induced = {
        phrase: count for phrase, count in common_leads.items() if phrase not in prompt_ngrams
    }
    total_template_mass = sum(common_leads.values()) or 1

    return {
        "prompt_example_text": prompt_example,
        "literal_example_reuse_count": literal_example_reuse_count,
        "outputs_with_any_prompt_phrase_reuse_count": len(outputs_reusing_prompt_phrases),
        "outputs_with_any_prompt_phrase_reuse_fraction": len(outputs_reusing_prompt_phrases)
        / len(all_rows),
        "prompt_phrase_reuse_examples": outputs_reusing_prompt_phrases[:10],
        "common_output_leading_templates": [
            {
                "phrase": phrase,
                "count": count,
                "conditions": sorted(leading_sources[phrase]),
                "prompt_derived": phrase in prompt_ngrams,
            }
            for phrase, count in sorted(
                common_leads.items(), key=lambda item: (-item[1], -len(item[0]), item[0])
            )[:20]
        ],
        "template_language_fraction_prompt_derived": sum(prompt_derived.values())
        / total_template_mass,
        "template_language_fraction_model_induced": sum(model_induced.values())
        / total_template_mass,
        "prompt_derived_template_phrases": sorted(
            prompt_derived.items(), key=lambda item: (-item[1], item[0])
        )[:20],
        "model_induced_template_phrases": sorted(
            model_induced.items(), key=lambda item: (-item[1], item[0])
        )[:20],
    }


def analysis_paired_structure(pairs: list[dict[str, object]]) -> dict[str, object]:
    metrics = {
        "cognitive_attribution": {"full": [], "no_trace": []},
        "behavioral_description": {"full": [], "no_trace": []},
        "evidence_reference": {"full": [], "no_trace": []},
        "hedging": {"full": [], "no_trace": []},
        "other": {"full": [], "no_trace": []},
        "total_clauses": {"full": [], "no_trace": []},
    }
    per_pair: list[dict[str, object]] = []

    for pair in pairs:
        full_counts = parse_clause_counts(pair["full_summary"])
        no_counts = parse_clause_counts(pair["no_trace_summary"])
        for metric in metrics:
            metrics[metric]["full"].append(full_counts[metric])
            metrics[metric]["no_trace"].append(no_counts[metric])
        per_pair.append(
            {
                "custom_id": pair["custom_id"],
                "full": full_counts,
                "no_trace": no_counts,
            }
        )

    summary = {}
    for metric, values in metrics.items():
        full_arr = np.asarray(values["full"], dtype=float)
        no_arr = np.asarray(values["no_trace"], dtype=float)
        summary[metric] = {
            "full_mean": float(full_arr.mean()),
            "no_trace_mean": float(no_arr.mean()),
            "full_median": float(np.median(full_arr)),
            "no_trace_median": float(np.median(no_arr)),
            "paired_wilcoxon": wilcoxon_signed_rank(full_arr, no_arr),
        }

    return {
        "paired_transition_count": len(pairs),
        "aggregate": summary,
        "sample_pairs": per_pair[:10],
    }


def analysis_information_density(pairs: list[dict[str, object]]) -> dict[str, object]:
    metrics = {
        "specific_reference_count": {"full": [], "no_trace": []},
        "line_reference_count": {"full": [], "no_trace": []},
        "trace_language_count": {"full": [], "no_trace": []},
        "token_count": {"full": [], "no_trace": []},
        "specific_reference_density_per_100_tokens": {"full": [], "no_trace": []},
        "line_reference_density_per_100_tokens": {"full": [], "no_trace": []},
        "trace_language_density_per_100_tokens": {"full": [], "no_trace": []},
    }

    per_pair: list[dict[str, object]] = []
    pairs_with_new_refs = 0
    pairs_with_new_line_refs = 0
    pairs_with_new_trace_terms = 0

    for pair in pairs:
        full_info = extract_code_references(pair["full_summary"])
        no_info = extract_code_references(pair["no_trace_summary"])
        full_tokens = max(1, full_info["token_count"])
        no_tokens = max(1, no_info["token_count"])

        metrics["specific_reference_count"]["full"].append(full_info["specific_reference_count"])
        metrics["specific_reference_count"]["no_trace"].append(no_info["specific_reference_count"])
        metrics["line_reference_count"]["full"].append(full_info["line_reference_count"])
        metrics["line_reference_count"]["no_trace"].append(no_info["line_reference_count"])
        metrics["trace_language_count"]["full"].append(full_info["trace_language_count"])
        metrics["trace_language_count"]["no_trace"].append(no_info["trace_language_count"])
        metrics["token_count"]["full"].append(full_tokens)
        metrics["token_count"]["no_trace"].append(no_tokens)
        metrics["specific_reference_density_per_100_tokens"]["full"].append(
            100.0 * full_info["specific_reference_count"] / full_tokens
        )
        metrics["specific_reference_density_per_100_tokens"]["no_trace"].append(
            100.0 * no_info["specific_reference_count"] / no_tokens
        )
        metrics["line_reference_density_per_100_tokens"]["full"].append(
            100.0 * full_info["line_reference_count"] / full_tokens
        )
        metrics["line_reference_density_per_100_tokens"]["no_trace"].append(
            100.0 * no_info["line_reference_count"] / no_tokens
        )
        metrics["trace_language_density_per_100_tokens"]["full"].append(
            100.0 * full_info["trace_language_count"] / full_tokens
        )
        metrics["trace_language_density_per_100_tokens"]["no_trace"].append(
            100.0 * no_info["trace_language_count"] / no_tokens
        )

        full_refs = set(full_info["specific_references"])
        no_refs = set(no_info["specific_references"])
        if full_refs - no_refs:
            pairs_with_new_refs += 1
        if full_info["line_reference_count"] > no_info["line_reference_count"]:
            pairs_with_new_line_refs += 1
        if full_info["trace_language_count"] > no_info["trace_language_count"]:
            pairs_with_new_trace_terms += 1

        per_pair.append(
            {
                "custom_id": pair["custom_id"],
                "full": full_info,
                "no_trace": no_info,
                "delta": {
                    "specific_reference_count": full_info["specific_reference_count"]
                    - no_info["specific_reference_count"],
                    "line_reference_count": full_info["line_reference_count"]
                    - no_info["line_reference_count"],
                    "trace_language_count": full_info["trace_language_count"]
                    - no_info["trace_language_count"],
                    "token_count": full_tokens - no_tokens,
                },
                "new_full_only_references": sorted(full_refs - no_refs),
            }
        )

    aggregates = {}
    for metric, values in metrics.items():
        full_arr = np.asarray(values["full"], dtype=float)
        no_arr = np.asarray(values["no_trace"], dtype=float)
        aggregates[metric] = {
            "full_mean": float(full_arr.mean()),
            "no_trace_mean": float(no_arr.mean()),
            "full_median": float(np.median(full_arr)),
            "no_trace_median": float(np.median(no_arr)),
            "paired_wilcoxon": wilcoxon_signed_rank(full_arr, no_arr),
        }

    qualitative_difference = (
        aggregates["specific_reference_density_per_100_tokens"]["paired_wilcoxon"][
            "mean_difference"
        ]
        > 0
        or aggregates["line_reference_density_per_100_tokens"]["paired_wilcoxon"]["mean_difference"]
        > 0
        or aggregates["trace_language_density_per_100_tokens"]["paired_wilcoxon"]["mean_difference"]
        > 0
    )

    return {
        "aggregate": aggregates,
        "pairs_with_new_specific_references": pairs_with_new_refs,
        "pairs_with_new_specific_references_fraction": pairs_with_new_refs / len(pairs),
        "pairs_with_more_line_references_in_full": pairs_with_new_line_refs,
        "pairs_with_more_trace_language_in_full": pairs_with_new_trace_terms,
        "qualitative_difference_signal": qualitative_difference,
        "interpretation": (
            "Full summaries add qualitatively different detail rather than only length."
            if qualitative_difference
            else "The main difference looks closer to verbosity than to denser references."
        ),
        "sample_pairs": per_pair[:10],
    }


def summary_presence_ngrams(summary: str, min_n: int = 3, max_n: int = 6) -> set[str]:
    return text_ngrams(summary, min_n, max_n)


def prune_template_phrases(phrases: dict[str, int]) -> list[str]:
    kept: list[str] = []
    for phrase, _count in sorted(
        phrases.items(), key=lambda item: (-len(item[0].split()), -item[1], item[0])
    ):
        if any(f" {phrase} " in f" {existing} " for existing in kept):
            continue
        kept.append(phrase)
    return sorted(kept, key=lambda phrase: (-len(phrase.split()), phrase))


def template_token_coverage(tokens: list[str], template_phrases: list[str]) -> int:
    covered = np.zeros(len(tokens), dtype=bool)
    phrase_tokens = [phrase.split() for phrase in template_phrases]
    phrase_tokens.sort(key=len, reverse=True)
    for chunk in phrase_tokens:
        n = len(chunk)
        for index in range(len(tokens) - n + 1):
            if tokens[index : index + n] == chunk:
                covered[index : index + n] = True
    return int(covered.sum())


def analysis_slot_filling(full_rows: list[dict[str, object]]) -> dict[str, object]:
    by_exercise: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in full_rows:
        by_exercise[row["exercise_id"]].append(row)

    exercise_results = []
    for exercise_id, rows in sorted(by_exercise.items()):
        unique_students = {row["student_id"] for row in rows}
        if len(unique_students) < 3:
            continue
        ngram_counter = Counter()
        summary_ngrams = []
        for row in rows:
            grams = summary_presence_ngrams(row["summary"])
            summary_ngrams.append((row, grams))
            ngram_counter.update(grams)
        threshold = math.ceil(len(rows) * 0.5)
        template_candidates = {
            phrase: count
            for phrase, count in ngram_counter.items()
            if count >= threshold and any(token not in STOPWORDS for token in phrase.split())
        }
        template_phrases = prune_template_phrases(template_candidates)
        summary_breakdown = []
        template_fractions = []
        for row, _grams in summary_ngrams:
            tokens = tokenize(row["summary"])
            if not tokens:
                template_fraction = 0.0
                slot_fraction = 0.0
            else:
                covered_tokens = template_token_coverage(tokens, template_phrases)
                template_fraction = covered_tokens / len(tokens)
                slot_fraction = 1.0 - template_fraction
            template_fractions.append(template_fraction)
            summary_breakdown.append(
                {
                    "custom_id": row["custom_id"],
                    "student_id": row["student_id"],
                    "summary": row["summary"],
                    "template_fraction": template_fraction,
                    "slot_fraction": slot_fraction,
                }
            )
        mean_template_fraction = float(np.mean(template_fractions)) if template_fractions else 0.0
        exercise_results.append(
            {
                "exercise_id": exercise_id,
                "summary_count": len(rows),
                "unique_student_count": len(unique_students),
                "shared_phrase_threshold": threshold,
                "template_phrases": template_phrases[:20],
                "mean_template_fraction": mean_template_fraction,
                "mean_slot_fraction": 1.0 - mean_template_fraction,
                "mad_libs_risk": mean_template_fraction >= 0.6,
                "summary_breakdown": summary_breakdown[:15],
            }
        )

    overall_template_fractions = [item["mean_template_fraction"] for item in exercise_results]
    return {
        "exercise_count_analyzed": len(exercise_results),
        "overall_mean_template_fraction": float(np.mean(overall_template_fractions))
        if overall_template_fractions
        else 0.0,
        "overall_mean_slot_fraction": 1.0
        - (float(np.mean(overall_template_fractions)) if overall_template_fractions else 0.0),
        "exercises": exercise_results,
    }


def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def analysis_opening_entropy(rows: list[dict[str, object]]) -> dict[str, object]:
    fourth_words = Counter()
    total = 0
    for row in rows:
        tokens = tokenize(row["summary"])
        if len(tokens) >= 4 and tuple(tokens[:3]) == OPENING_TARGET:
            fourth_words[tokens[3]] += 1
            total += 1
    entropy = entropy_from_counter(fourth_words)
    return {
        "opening": "The student likely",
        "summary_count_with_opening": total,
        "fourth_word_entropy_bits": entropy,
        "fourth_word_perplexity": 2**entropy,
        "unique_fourth_words": len(fourth_words),
        "fourth_word_distribution": fourth_words.most_common(20),
    }


def phrase_candidates(summary: str) -> set[str]:
    tokens = tokenize(summary)
    phrases: set[str] = set()
    for n in (1, 2, 3):
        for index in range(len(tokens) - n + 1):
            chunk = tokens[index : index + n]
            if all(token in STOPWORDS for token in chunk):
                continue
            if n == 1 and (chunk[0] in STOPWORDS or len(chunk[0]) <= 2):
                continue
            phrases.add(" ".join(chunk))
    return phrases


def classify_full_only_phrase(phrase: str) -> str:
    tokens = set(phrase.split())
    if tokens & CATEGORY_KEYWORDS["cognitive_attribution"]:
        return "cognitive_attribution"
    if tokens & CATEGORY_KEYWORDS["behavioral_description"] or tokens & TRACE_WORDS:
        return "behavioral_observation"
    if tokens & STRUCTURAL_FILLER_WORDS:
        return "structural_filler"
    return "structural_filler"


def analysis_full_only_content(
    full_rows: list[dict[str, object]], no_trace_rows: list[dict[str, object]]
) -> dict[str, object]:
    full_counter = Counter()
    no_counter = Counter()
    for row in full_rows:
        full_counter.update(phrase_candidates(row["summary"]))
    for row in no_trace_rows:
        no_counter.update(phrase_candidates(row["summary"]))

    exclusive = []
    for phrase, count in full_counter.items():
        if count >= 5 and no_counter.get(phrase, 0) == 0:
            exclusive.append(
                {
                    "phrase": phrase,
                    "full_summary_count": count,
                    "category": classify_full_only_phrase(phrase),
                }
            )
    exclusive.sort(
        key=lambda item: (-item["full_summary_count"], -len(item["phrase"].split()), item["phrase"])
    )
    by_category = defaultdict(list)
    for item in exclusive:
        by_category[item["category"]].append(item)
    return {
        "phrase_count": len(exclusive),
        "exclusive_phrases": exclusive[:100],
        "by_category": {category: items[:30] for category, items in sorted(by_category.items())},
    }


def build_results() -> dict[str, object]:
    prompt_source = PROMPT_PATH.read_text()
    prompt_strings = extract_prompt_strings(prompt_source)
    full_rows = load_predictions(FULL_PATH)
    no_trace_rows = load_predictions(NO_TRACE_PATH)
    pairs = pair_predictions(full_rows, no_trace_rows)

    return {
        "metadata": {
            "full_path": str(FULL_PATH),
            "no_trace_path": str(NO_TRACE_PATH),
            "prompt_path": str(PROMPT_PATH),
            "output_path": str(OUTPUT_PATH),
            "pair_key": "custom_id",
            "pair_count": len(pairs),
            "full_prediction_count": len(full_rows),
            "no_trace_prediction_count": len(no_trace_rows),
        },
        "analysis_1_prompt_vs_model_templates": analysis_template_origin(
            prompt_source, prompt_strings, full_rows, no_trace_rows
        ),
        "analysis_2_paired_structural_comparison": analysis_paired_structure(pairs),
        "analysis_3_information_density": analysis_information_density(pairs),
        "analysis_4_slot_filling": analysis_slot_filling(full_rows),
        "analysis_5_conditional_entropy_of_openings": {
            "full": analysis_opening_entropy(full_rows),
            "no_trace": analysis_opening_entropy(no_trace_rows),
        },
        "analysis_6_full_only_content": analysis_full_only_content(full_rows, no_trace_rows),
    }


def print_summary(results: dict[str, object]) -> None:
    a1 = results["analysis_1_prompt_vs_model_templates"]
    a2 = results["analysis_2_paired_structural_comparison"]["aggregate"]
    a3 = results["analysis_3_information_density"]
    a4 = results["analysis_4_slot_filling"]
    a5 = results["analysis_5_conditional_entropy_of_openings"]
    a6 = results["analysis_6_full_only_content"]

    print("Narrative Deep Analysis: The Personalization Illusion")
    print("=" * 60)
    print(f"Paired transitions: {results['metadata']['pair_count']} (matched by custom_id)")
    print()

    print("1. Prompt-induced vs model-induced templates")
    print(f"   Prompt example text: {a1['prompt_example_text']!r}")
    print(f"   Literal reuse of exact example: {a1['literal_example_reuse_count']}")
    print(
        "   Any prompt phrase reused: "
        f"{a1['outputs_with_any_prompt_phrase_reuse_count']} / "
        f"{results['metadata']['full_prediction_count'] + results['metadata']['no_trace_prediction_count']}"
    )
    print(
        "   Template language mass: "
        f"{a1['template_language_fraction_prompt_derived']:.3f} prompt-derived, "
        f"{a1['template_language_fraction_model_induced']:.3f} model-induced"
    )
    for item in a1["common_output_leading_templates"][:5]:
        label = "prompt" if item["prompt_derived"] else "model"
        print(f"   - {item['phrase']} ({item['count']}, {label})")
    print()

    print("2. Paired structural comparison")
    for metric in (
        "cognitive_attribution",
        "behavioral_description",
        "evidence_reference",
        "hedging",
    ):
        stats = a2[metric]
        print(
            f"   {metric}: full mean {stats['full_mean']:.3f}, "
            f"no_trace mean {stats['no_trace_mean']:.3f}, "
            f"Wilcoxon p={stats['paired_wilcoxon']['p_value_two_sided']:.4g}"
        )
    print()

    print("3. Information density")
    for metric in (
        "specific_reference_count",
        "line_reference_count",
        "trace_language_count",
        "token_count",
        "specific_reference_density_per_100_tokens",
    ):
        stats = a3["aggregate"][metric]
        print(
            f"   {metric}: full mean {stats['full_mean']:.3f}, "
            f"no_trace mean {stats['no_trace_mean']:.3f}, "
            f"delta {stats['paired_wilcoxon']['mean_difference']:.3f}, "
            f"p={stats['paired_wilcoxon']['p_value_two_sided']:.4g}"
        )
    print(
        f"   Pairs where full adds new specific references: "
        f"{a3['pairs_with_new_specific_references']} / {results['metadata']['pair_count']}"
    )
    print(f"   Interpretation: {a3['interpretation']}")
    print()

    print("4. Slot-filling test")
    print(
        f"   Exercises analyzed: {a4['exercise_count_analyzed']}, "
        f"overall template fraction {a4['overall_mean_template_fraction']:.3f}, "
        f"slot fraction {a4['overall_mean_slot_fraction']:.3f}"
    )
    for item in sorted(
        a4["exercises"], key=lambda row: (-row["mean_template_fraction"], row["exercise_id"])
    )[:5]:
        print(
            f"   - exercise {item['exercise_id']}: template {item['mean_template_fraction']:.3f}, "
            f"students {item['unique_student_count']}, summaries {item['summary_count']}"
        )
    print()

    print("5. Conditional entropy of openings")
    for condition in ("full", "no_trace"):
        stats = a5[condition]
        print(
            f"   {condition}: 'The student likely' count {stats['summary_count_with_opening']}, "
            f"4th-word entropy {stats['fourth_word_entropy_bits']:.3f} bits, "
            f"perplexity {stats['fourth_word_perplexity']:.3f}"
        )
        print(f"   - top continuations: {stats['fourth_word_distribution'][:5]}")
    print()

    print("6. Full-only content analysis")
    print(f"   Exclusive full-only phrases appearing in 5+ summaries: {a6['phrase_count']}")
    for category, items in a6["by_category"].items():
        sample = ", ".join(item["phrase"] for item in items[:5])
        print(f"   - {category}: {sample}")
    print()
    print(f"Saved JSON results to {OUTPUT_PATH}")


def main() -> None:
    results = build_results()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print_summary(results)


if __name__ == "__main__":
    main()
