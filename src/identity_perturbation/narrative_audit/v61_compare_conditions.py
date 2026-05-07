from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

CONDITIONS = {
    "full": ROOT / "data" / "v61_batch_runs" / "gpt54_medium_v61_clean126_buildable117_v6",
    "no_trace": ROOT
    / "data"
    / "v61_batch_runs"
    / "gpt54_medium_v61_clean126_buildable117_no_trace_v1",
    "trace_shuffled": ROOT
    / "data"
    / "v61_batch_runs"
    / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1",
}

OUTPUT_DIR = ROOT / "data" / "v61_condition_comparisons"
MARKDOWN_PATH = OUTPUT_DIR / "student_state_summary_comparison.md"
STATS_PATH = OUTPUT_DIR / "student_state_summary_stats.json"

TRACE_TERM_PATTERNS = {
    "testing_family": r"\b(?:test|tests|testing|tested|rerun|reruns|rerunning|reran)\b",
    "editing_family": r"\b(?:edit|edits|editing|edited|rewrite|rewrites|rewriting|rewrote|patch|patches|patched|patching)\b",
    "previous_attempts": r"\b(?:previous|prior|earlier)\s+attempt(?:s)?\b",
    "delete_family": r"\bdelet(?:e|es|ed|ing|ion|ions)\b",
    "cursor": r"\bcursor\b",
    "keystroke_family": r"\bkeystroke(?:s)?\b",
    "typing_family": r"\btyp(?:e|es|ed|ing)\b",
    "debug_family": r"\bdebug(?:ging|ged|s)?\b",
}

TOKEN_RE = re.compile(r"\b\w+\b")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SummaryRecord:
    custom_id: str
    summary: str


def discover_result_file(condition_dir: Path) -> Path:
    if not condition_dir.exists():
        raise FileNotFoundError(f"Condition directory does not exist: {condition_dir}")

    candidates: list[Path] = []
    for path in condition_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".jsonl"}:
            continue
        name = path.name.lower()
        if "output" in name or ("batch" in name and "result" in name):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No batch result or output JSON/JSONL file found in {condition_dir}"
        )

    candidates.sort(
        key=lambda path: (
            0 if path.name == "output.jsonl" else 1,
            0 if path.suffix == ".jsonl" else 1,
            path.name,
        )
    )
    return candidates[0]


def extract_completion_json_text(line_obj: dict[str, Any]) -> str:
    response_body = line_obj.get("response", {}).get("body", {})
    outputs = response_body.get("output", [])
    for output_item in outputs:
        if output_item.get("type") != "message":
            continue
        for content_item in output_item.get("content", []):
            if content_item.get("type") != "output_text":
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                return text
    raise ValueError(
        f"Could not find output_text content for custom_id={line_obj.get('custom_id')}"
    )


def parse_top1_student_state_summary(line_obj: dict[str, Any]) -> str:
    completion_text = extract_completion_json_text(line_obj)
    completion_obj = json.loads(completion_text)
    hypotheses = completion_obj.get("next_episode_hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError(f"No hypotheses found for custom_id={line_obj.get('custom_id')}")
    summary = hypotheses[0].get("student_state_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(
            f"Missing top-1 student_state_summary for custom_id={line_obj.get('custom_id')}"
        )
    return normalize_text(summary)


def load_condition_summaries(result_path: Path) -> dict[str, SummaryRecord]:
    records: dict[str, SummaryRecord] = {}
    with result_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise ValueError(f"Missing custom_id at {result_path}:{line_number}")
            if custom_id in records:
                raise ValueError(f"Duplicate custom_id={custom_id} in {result_path}")
            records[custom_id] = SummaryRecord(
                custom_id=custom_id,
                summary=parse_top1_student_state_summary(row),
            )
    if not records:
        raise ValueError(f"No rows parsed from {result_path}")
    return records


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_idf(documents: list[str]) -> dict[str, float]:
    document_count = len(documents)
    doc_freq: Counter[str] = Counter()
    for document in documents:
        doc_freq.update(set(tokenize(document)))
    return {
        term: math.log((1 + document_count) / (1 + frequency)) + 1.0
        for term, frequency in doc_freq.items()
    }


def tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    term_counts = Counter(tokenize(text))
    if not term_counts:
        return {}
    total_terms = sum(term_counts.values())
    return {
        term: (count / total_terms) * idf[term]
        for term, count in term_counts.items()
        if term in idf
    }


def cosine_similarity(vector_a: dict[str, float], vector_b: dict[str, float]) -> float:
    if not vector_a or not vector_b:
        return 0.0
    overlap = set(vector_a) & set(vector_b)
    numerator = sum(vector_a[term] * vector_b[term] for term in overlap)
    norm_a = math.sqrt(sum(value * value for value in vector_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vector_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def levenshtein_distance(text_a: str, text_b: str) -> int:
    if text_a == text_b:
        return 0
    if not text_a:
        return len(text_b)
    if not text_b:
        return len(text_a)
    if len(text_a) < len(text_b):
        text_a, text_b = text_b, text_a

    previous_row = list(range(len(text_b) + 1))
    for index_a, char_a in enumerate(text_a, start=1):
        current_row = [index_a]
        for index_b, char_b in enumerate(text_b, start=1):
            insert_cost = current_row[index_b - 1] + 1
            delete_cost = previous_row[index_b] + 1
            replace_cost = previous_row[index_b - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def levenshtein_ratio(text_a: str, text_b: str) -> float:
    total_length = len(text_a) + len(text_b)
    if total_length == 0:
        return 1.0
    distance = levenshtein_distance(text_a, text_b)
    return (total_length - distance) / total_length


def summarize_metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def count_trace_terms(summaries: list[str]) -> dict[str, Any]:
    term_counts = {
        label: sum(len(re.findall(pattern, summary, flags=re.IGNORECASE)) for summary in summaries)
        for label, pattern in TRACE_TERM_PATTERNS.items()
    }
    summary_totals = []
    summaries_with_any = 0
    for summary in summaries:
        total = sum(
            len(re.findall(pattern, summary, flags=re.IGNORECASE))
            for pattern in TRACE_TERM_PATTERNS.values()
        )
        summary_totals.append(total)
        if total > 0:
            summaries_with_any += 1

    token_total = sum(len(tokenize(summary)) for summary in summaries)
    return {
        "total_occurrences": sum(term_counts.values()),
        "summaries_with_any_trace_terms": summaries_with_any,
        "mean_occurrences_per_summary": statistics.mean(summary_totals),
        "occurrences_per_1000_tokens": 0.0
        if token_total == 0
        else (sum(term_counts.values()) / token_total) * 1000.0,
        "term_counts": term_counts,
    }


def format_float(value: float) -> str:
    return f"{value:.3f}"


def build_markdown(
    examples: list[dict[str, Any]],
    matched_count: int,
    result_files: dict[str, str],
) -> str:
    lines = [
        "# Student State Summary Comparison",
        "",
        f"- Matched transitions across all three conditions: `{matched_count}`",
        "- Example selection rule: `5` lowest overall similarity transitions and `5` highest overall similarity transitions.",
        "- Pairwise similarity metrics below use corpus-level TF-IDF cosine similarity and normalized Levenshtein ratio.",
        "",
        "## Result Files",
        "",
    ]
    for label, path in result_files.items():
        lines.append(f"- `{label}`: `{path}`")

    lines.extend(["", "## Example Transitions", ""])
    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"### {index}. `{example['custom_id']}` ({example['selection_bucket']})",
                "",
                f"- Overall mean similarity: `{format_float(example['overall_mean_similarity'])}`",
                f"- `full` vs `no_trace`: TF-IDF `{format_float(example['pairwise']['full_vs_no_trace']['tfidf_cosine'])}`, Levenshtein `{format_float(example['pairwise']['full_vs_no_trace']['levenshtein_ratio'])}`",
                f"- `full` vs `trace_shuffled`: TF-IDF `{format_float(example['pairwise']['full_vs_trace_shuffled']['tfidf_cosine'])}`, Levenshtein `{format_float(example['pairwise']['full_vs_trace_shuffled']['levenshtein_ratio'])}`",
                f"- `no_trace` vs `trace_shuffled`: TF-IDF `{format_float(example['pairwise']['no_trace_vs_trace_shuffled']['tfidf_cosine'])}`, Levenshtein `{format_float(example['pairwise']['no_trace_vs_trace_shuffled']['levenshtein_ratio'])}`",
                "",
                "| Condition | Top-1 `student_state_summary` |",
                "| --- | --- |",
                f"| `full` | {example['summaries']['full']} |",
                f"| `no_trace` | {example['summaries']['no_trace']} |",
                f"| `trace_shuffled` | {example['summaries']['trace_shuffled']} |",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    result_files = {
        label: str(discover_result_file(condition_dir))
        for label, condition_dir in CONDITIONS.items()
    }

    condition_records = {
        label: load_condition_summaries(Path(result_file))
        for label, result_file in result_files.items()
    }

    matched_ids = sorted(
        set(condition_records["full"])
        & set(condition_records["no_trace"])
        & set(condition_records["trace_shuffled"])
    )
    if not matched_ids:
        raise SystemExit("No matched custom_id values across the three conditions")

    all_summaries = [
        condition_records[label][custom_id].summary
        for custom_id in matched_ids
        for label in ("full", "no_trace", "trace_shuffled")
    ]
    idf = build_idf(all_summaries)

    pairwise_metrics = {
        "full_vs_no_trace": {"tfidf_cosine": [], "levenshtein_ratio": []},
        "full_vs_trace_shuffled": {"tfidf_cosine": [], "levenshtein_ratio": []},
        "no_trace_vs_trace_shuffled": {"tfidf_cosine": [], "levenshtein_ratio": []},
    }
    per_transition_examples: list[dict[str, Any]] = []

    for custom_id in matched_ids:
        full_summary = condition_records["full"][custom_id].summary
        no_trace_summary = condition_records["no_trace"][custom_id].summary
        shuffled_summary = condition_records["trace_shuffled"][custom_id].summary

        full_vector = tfidf_vector(full_summary, idf)
        no_trace_vector = tfidf_vector(no_trace_summary, idf)
        shuffled_vector = tfidf_vector(shuffled_summary, idf)

        full_no_trace_tfidf = cosine_similarity(full_vector, no_trace_vector)
        full_shuffled_tfidf = cosine_similarity(full_vector, shuffled_vector)
        no_trace_shuffled_tfidf = cosine_similarity(no_trace_vector, shuffled_vector)

        full_no_trace_lev = levenshtein_ratio(full_summary, no_trace_summary)
        full_shuffled_lev = levenshtein_ratio(full_summary, shuffled_summary)
        no_trace_shuffled_lev = levenshtein_ratio(no_trace_summary, shuffled_summary)

        pairwise_metrics["full_vs_no_trace"]["tfidf_cosine"].append(full_no_trace_tfidf)
        pairwise_metrics["full_vs_no_trace"]["levenshtein_ratio"].append(full_no_trace_lev)
        pairwise_metrics["full_vs_trace_shuffled"]["tfidf_cosine"].append(full_shuffled_tfidf)
        pairwise_metrics["full_vs_trace_shuffled"]["levenshtein_ratio"].append(full_shuffled_lev)
        pairwise_metrics["no_trace_vs_trace_shuffled"]["tfidf_cosine"].append(
            no_trace_shuffled_tfidf
        )
        pairwise_metrics["no_trace_vs_trace_shuffled"]["levenshtein_ratio"].append(
            no_trace_shuffled_lev
        )

        overall_mean_similarity = statistics.mean(
            [
                full_no_trace_tfidf,
                full_shuffled_tfidf,
                no_trace_shuffled_tfidf,
                full_no_trace_lev,
                full_shuffled_lev,
                no_trace_shuffled_lev,
            ]
        )

        per_transition_examples.append(
            {
                "custom_id": custom_id,
                "overall_mean_similarity": overall_mean_similarity,
                "pairwise": {
                    "full_vs_no_trace": {
                        "tfidf_cosine": full_no_trace_tfidf,
                        "levenshtein_ratio": full_no_trace_lev,
                    },
                    "full_vs_trace_shuffled": {
                        "tfidf_cosine": full_shuffled_tfidf,
                        "levenshtein_ratio": full_shuffled_lev,
                    },
                    "no_trace_vs_trace_shuffled": {
                        "tfidf_cosine": no_trace_shuffled_tfidf,
                        "levenshtein_ratio": no_trace_shuffled_lev,
                    },
                },
                "summaries": {
                    "full": full_summary,
                    "no_trace": no_trace_summary,
                    "trace_shuffled": shuffled_summary,
                },
            }
        )

    sorted_by_divergence = sorted(
        per_transition_examples, key=lambda row: row["overall_mean_similarity"]
    )
    low_examples = []
    high_examples = []
    for example in sorted_by_divergence[:5]:
        low_example = dict(example)
        low_example["selection_bucket"] = "low similarity"
        low_examples.append(low_example)
    for example in sorted_by_divergence[-5:]:
        high_example = dict(example)
        high_example["selection_bucket"] = "high similarity"
        high_examples.append(high_example)
    selected_examples = low_examples + list(reversed(high_examples))

    similarity_summary = {
        pair_label: {
            metric_name: summarize_metric(metric_values)
            for metric_name, metric_values in metrics.items()
        }
        for pair_label, metrics in pairwise_metrics.items()
    }

    trace_language = {
        label: count_trace_terms(
            [condition_records[label][custom_id].summary for custom_id in matched_ids]
        )
        for label in ("full", "no_trace", "trace_shuffled")
    }

    full_no_trace_tfidf = pairwise_metrics["full_vs_no_trace"]["tfidf_cosine"]
    full_no_trace_lev = pairwise_metrics["full_vs_no_trace"]["levenshtein_ratio"]
    substantially_different_counts = {
        "tfidf_cosine_below_0_5": sum(value < 0.5 for value in full_no_trace_tfidf),
        "levenshtein_ratio_below_0_5": sum(value < 0.5 for value in full_no_trace_lev),
        "both_metrics_below_0_5": sum(
            tfidf_value < 0.5 and lev_value < 0.5
            for tfidf_value, lev_value in zip(full_no_trace_tfidf, full_no_trace_lev, strict=True)
        ),
    }

    stats_payload = {
        "matched_transition_count": len(matched_ids),
        "condition_result_files": result_files,
        "similarity_metrics": similarity_summary,
        "full_vs_no_trace_substantially_different_counts": substantially_different_counts,
        "trace_related_language": {
            "term_patterns": TRACE_TERM_PATTERNS,
            "by_condition": trace_language,
        },
        "selected_example_transition_ids": [example["custom_id"] for example in selected_examples],
        "metric_notes": {
            "tfidf_cosine": "Cosine similarity over corpus-level TF-IDF vectors built from all matched summaries across all three conditions.",
            "levenshtein_ratio": "Normalized character-level ratio computed as (len(a) + len(b) - distance) / (len(a) + len(b)).",
            "std": "Population standard deviation across matched transitions.",
        },
    }

    markdown = build_markdown(
        examples=selected_examples,
        matched_count=len(matched_ids),
        result_files=result_files,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_PATH.write_text(markdown + "\n", encoding="utf-8")
    STATS_PATH.write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(f"Matched transitions: {len(matched_ids)}")
    print(f"Markdown output: {MARKDOWN_PATH}")
    print(f"Stats output: {STATS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
