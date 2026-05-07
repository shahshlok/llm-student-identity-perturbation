from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[3]

FULL_PATH = (
    ROOT
    / "data"
    / "narrative_audit"
    / "full_trace"
    / "hydrated_predictions.json"
)
NO_TRACE_PATH = (
    ROOT
    / "data"
    / "narrative_audit"
    / "no_trace"
    / "hydrated_predictions.json"
)
OUTPUT_PATH = ROOT / "data" / "narrative_audit" / "comparisons" / "narrative_pattern_analysis.json"

TOKEN_RE = re.compile(r"\b\w+\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "let",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

SUMMARY_START_PATTERNS = {
    "the_student_likely": r"^the student likely\b",
    "the_student_may": r"^the student may\b",
    "the_student_appears": r"^the student appears\b",
    "the_student_seems": r"^the student seems\b",
    "student_likely": r"^student likely\b",
    "student_may": r"^student may\b",
    "they_likely": r"^they likely\b",
    "they_may": r"^they may\b",
}

SENTENCE_STRUCTURE_PATTERNS = {
    "likely_notices_that": r"\blikely notices that\b",
    "likely_notices_the": r"\blikely notices the\b",
    "may_notice": r"\bmay notice\b",
    "likely_believes": r"\blikely believes?\b",
    "likely_sees": r"\blikely sees\b",
    "appears_close_to": r"\bappears close to\b",
    "focused_fix": r"\b(?:make|makes)\s+(?:a\s+)?(?:quick|small|minimal|one-line|focused)\b",
    "verify_or_resubmit": r"\b(?:verify|verifies|resubmit|resubmits|submit|submits)\b",
}


@dataclass(frozen=True)
class SummaryRecord:
    custom_id: str
    exercise_id: str
    student_id: str
    summary: str


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def alpha_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token.isalpha()]


def split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in SENTENCE_RE.split(text.strip()) if segment.strip()]


def leading_phrase(text: str, word_count: int) -> str:
    tokens = tokenize(text)
    return " ".join(tokens[:word_count])


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_records(path: Path) -> list[SummaryRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"Missing predictions array in {path}")

    records: list[SummaryRecord] = []
    for index, prediction in enumerate(predictions):
        hypotheses = prediction.get("response", {}).get("next_episode_hypotheses")
        if not isinstance(hypotheses, list) or not hypotheses:
            raise ValueError(f"Missing next_episode_hypotheses at index {index} in {path}")
        summary = hypotheses[0].get("student_state_summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"Missing student_state_summary at index {index} in {path}")

        custom_id = prediction.get("custom_id")
        exercise_id = prediction.get("exercise_id")
        student_id = prediction.get("student_id")
        if custom_id is None or exercise_id is None or student_id is None:
            raise ValueError(f"Missing identifiers at index {index} in {path}")

        records.append(
            SummaryRecord(
                custom_id=str(custom_id),
                exercise_id=str(exercise_id),
                student_id=str(student_id),
                summary=normalize_text(summary),
            )
        )
    return records


def summarize_values(values: list[float | int]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_manual_tfidf_similarity(summaries: list[str]) -> tuple[np.ndarray, int]:
    document_count = len(summaries)
    tokenized = [tokenize(summary) for summary in summaries]
    vocabulary = sorted({token for tokens in tokenized for token in tokens})
    if not vocabulary:
        return np.zeros((document_count, document_count), dtype=float), 0

    vocabulary_index = {token: index for index, token in enumerate(vocabulary)}
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    idf = {
        token: math.log((1 + document_count) / (1 + document_frequency[token])) + 1.0
        for token in vocabulary
    }

    matrix = np.zeros((document_count, len(vocabulary)), dtype=float)
    for row_index, tokens in enumerate(tokenized):
        if not tokens:
            continue
        counts = Counter(tokens)
        total_terms = sum(counts.values())
        for token, count in counts.items():
            column_index = vocabulary_index[token]
            matrix[row_index, column_index] = (count / total_terms) * idf[token]

    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0.0] = 1.0
    normalized_matrix = matrix / row_norms
    similarities = normalized_matrix @ normalized_matrix.T
    return similarities, len(vocabulary)


def pairwise_similarity_analysis(records: list[SummaryRecord]) -> dict[str, Any]:
    summaries = [record.summary for record in records]
    if SKLEARN_AVAILABLE:
        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
        matrix = vectorizer.fit_transform(summaries)
        similarities = cosine_similarity(matrix)
        vocabulary_size = int(len(vectorizer.vocabulary_))
    else:
        similarities, vocabulary_size = build_manual_tfidf_similarity(summaries)

    overall_pairs: list[float] = []
    within_exercise_pairs: list[float] = []
    across_exercise_pairs: list[float] = []
    within_exercise_by_exercise: dict[str, list[float]] = defaultdict(list)

    for left_index in range(len(records)):
        left = records[left_index]
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            score = float(similarities[left_index, right_index])
            overall_pairs.append(score)
            if left.exercise_id == right.exercise_id and left.student_id != right.student_id:
                within_exercise_pairs.append(score)
                within_exercise_by_exercise[left.exercise_id].append(score)
            elif left.exercise_id != right.exercise_id:
                across_exercise_pairs.append(score)

    per_exercise_stats = {
        exercise_id: summarize_values(scores)
        for exercise_id, scores in sorted(
            within_exercise_by_exercise.items(), key=lambda item: item[0]
        )
        if scores
    }
    per_exercise_mean_ranking = sorted(
        (
            {
                "exercise_id": exercise_id,
                "pair_count": stats["count"],
                "mean": stats["mean"],
                "median": stats["median"],
            }
            for exercise_id, stats in per_exercise_stats.items()
            if stats is not None
        ),
        key=lambda item: item["mean"],
        reverse=True,
    )

    return {
        "document_count": len(records),
        "vocabulary_size": vocabulary_size,
        "overall_pairwise_similarity": summarize_values(overall_pairs),
        "within_exercise_pairwise_similarity": summarize_values(within_exercise_pairs),
        "across_exercise_pairwise_similarity": summarize_values(across_exercise_pairs),
        "within_exercise_pair_count_by_exercise": {
            exercise_id: int(len(scores))
            for exercise_id, scores in sorted(
                within_exercise_by_exercise.items(), key=lambda item: item[0]
            )
        },
        "within_exercise_pairwise_similarity_by_exercise": per_exercise_stats,
        "within_exercise_mean_similarity_ranking": per_exercise_mean_ranking,
    }


def compare_similarity_sections(
    full_stats: dict[str, Any],
    no_trace_stats: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for section_name in (
        "overall_pairwise_similarity",
        "within_exercise_pairwise_similarity",
        "across_exercise_pairwise_similarity",
    ):
        full_section = full_stats.get(section_name)
        no_trace_section = no_trace_stats.get(section_name)
        if not full_section or not no_trace_section:
            comparison[section_name] = None
            continue
        comparison[section_name] = {
            "full_mean": full_section["mean"],
            "no_trace_mean": no_trace_section["mean"],
            "mean_delta_full_minus_no_trace": full_section["mean"] - no_trace_section["mean"],
            "full_median": full_section["median"],
            "no_trace_median": no_trace_section["median"],
            "median_delta_full_minus_no_trace": full_section["median"] - no_trace_section["median"],
            "full_std": full_section["std"],
            "no_trace_std": no_trace_section["std"],
            "std_delta_full_minus_no_trace": full_section["std"] - no_trace_section["std"],
        }
    return comparison


def structural_template_analysis(records: list[SummaryRecord]) -> dict[str, Any]:
    summaries = [record.summary for record in records]
    sentences = [sentence for summary in summaries for sentence in split_sentences(summary)]

    summary_start_pattern_counts = {
        label: int(
            sum(1 for summary in summaries if re.search(pattern, summary, flags=re.IGNORECASE))
        )
        for label, pattern in SUMMARY_START_PATTERNS.items()
    }
    sentence_structure_pattern_counts = {
        label: int(
            sum(1 for sentence in sentences if re.search(pattern, sentence, flags=re.IGNORECASE))
        )
        for label, pattern in SENTENCE_STRUCTURE_PATTERNS.items()
    }

    top_summary_start_trigrams = Counter(
        leading_phrase(summary, 3) for summary in summaries if summary
    )
    top_sentence_start_trigrams = Counter(
        leading_phrase(sentence, 3) for sentence in sentences if sentence
    )
    top_first_five_word_openings = Counter(
        leading_phrase(summary, 5) for summary in summaries if summary
    )

    return {
        "summary_count": len(summaries),
        "sentence_count": len(sentences),
        "summary_start_pattern_counts": summary_start_pattern_counts,
        "top_summary_start_trigrams": [
            {"phrase": phrase, "count": count}
            for phrase, count in top_summary_start_trigrams.most_common(15)
        ],
        "top_first_five_word_openings": [
            {"phrase": phrase, "count": count}
            for phrase, count in top_first_five_word_openings.most_common(15)
        ],
        "top_sentence_start_trigrams": [
            {"phrase": phrase, "count": count}
            for phrase, count in top_sentence_start_trigrams.most_common(15)
        ],
        "sentence_structure_pattern_counts": sentence_structure_pattern_counts,
    }


def lexical_diversity_analysis(records: list[SummaryRecord]) -> dict[str, Any]:
    summaries = [record.summary for record in records]
    per_summary_ttr: list[float] = []
    overall_vocabulary: set[str] = set()
    content_word_counts: Counter[str] = Counter()

    for summary in summaries:
        tokens = alpha_tokens(summary)
        if tokens:
            per_summary_ttr.append(len(set(tokens)) / len(tokens))
        else:
            per_summary_ttr.append(0.0)
        overall_vocabulary.update(tokens)
        content_word_counts.update(
            token for token in tokens if token not in STOPWORDS and len(token) > 1
        )

    return {
        "type_token_ratio_per_summary": summarize_values(per_summary_ttr),
        "unique_vocabulary_size": int(len(overall_vocabulary)),
        "most_frequent_content_words": [
            {"word": word, "count": count} for word, count in content_word_counts.most_common(25)
        ],
    }


def length_analysis(records: list[SummaryRecord]) -> dict[str, Any]:
    word_counts = [len(alpha_tokens(record.summary)) for record in records]
    return {
        "word_count_distribution": summarize_values(word_counts),
        "raw_word_counts": word_counts,
    }


def build_ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def ngram_template_analysis(records: list[SummaryRecord]) -> dict[str, Any]:
    bigrams: Counter[str] = Counter()
    trigrams: Counter[str] = Counter()
    for record in records:
        tokens = alpha_tokens(record.summary)
        bigrams.update(build_ngrams(tokens, 2))
        trigrams.update(build_ngrams(tokens, 3))
    return {
        "top_bigrams": [
            {"ngram": ngram, "count": count} for ngram, count in bigrams.most_common(25)
        ],
        "top_trigrams": [
            {"ngram": ngram, "count": count} for ngram, count in trigrams.most_common(25)
        ],
    }


def compare_lengths(
    full_length_stats: dict[str, Any],
    no_trace_length_stats: dict[str, Any],
) -> dict[str, Any]:
    full_distribution = full_length_stats["word_count_distribution"]
    no_trace_distribution = no_trace_length_stats["word_count_distribution"]
    return {
        "mean_delta_full_minus_no_trace": full_distribution["mean"] - no_trace_distribution["mean"],
        "median_delta_full_minus_no_trace": full_distribution["median"]
        - no_trace_distribution["median"],
        "std_delta_full_minus_no_trace": full_distribution["std"] - no_trace_distribution["std"],
        "full_has_longer_mean": bool(full_distribution["mean"] > no_trace_distribution["mean"]),
        "full_has_higher_variability_std": bool(
            full_distribution["std"] > no_trace_distribution["std"]
        ),
    }


def format_stats(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "n=0"
    return (
        f"n={stats['count']}, mean={stats['mean']:.4f}, median={stats['median']:.4f}, "
        f"std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}"
    )


def format_top_phrase_entries(entries: list[dict[str, Any]], key_name: str) -> str:
    if not entries:
        return "none"
    return ", ".join(f"'{entry[key_name]}' ({entry['count']})" for entry in entries[:5])


def print_report(results: dict[str, Any]) -> None:
    analysis_1 = results["analysis_1_full_condition_similarity"]
    analysis_2 = results["analysis_2_no_trace_condition_similarity"]
    analysis_3 = results["analysis_3_structural_template_detection"]
    analysis_4 = results["analysis_4_lexical_diversity"]
    analysis_5 = results["analysis_5_length_analysis"]
    analysis_6 = results["analysis_6_full_within_vs_across_exercise_similarity"]
    analysis_7 = results["analysis_7_full_condition_ngram_template_detection"]

    print("Narrative Pattern Analysis")
    print("==========================")
    print()

    print("1. Full condition TF-IDF cosine similarity")
    print(f"Overall: {format_stats(analysis_1['overall_pairwise_similarity'])}")
    print(
        "Within exercise (different students): "
        f"{format_stats(analysis_1['within_exercise_pairwise_similarity'])}"
    )
    print(f"Across exercises: {format_stats(analysis_1['across_exercise_pairwise_similarity'])}")
    print()

    print("2. No-trace condition TF-IDF cosine similarity")
    print(f"Overall: {format_stats(analysis_2['overall_pairwise_similarity'])}")
    print(
        "Within exercise (different students): "
        f"{format_stats(analysis_2['within_exercise_pairwise_similarity'])}"
    )
    print(f"Across exercises: {format_stats(analysis_2['across_exercise_pairwise_similarity'])}")
    print(
        "Comparison to full (mean deltas, full - no_trace): "
        f"overall={analysis_2['comparison_to_full']['overall_pairwise_similarity']['mean_delta_full_minus_no_trace']:.4f}, "
        f"within_exercise={analysis_2['comparison_to_full']['within_exercise_pairwise_similarity']['mean_delta_full_minus_no_trace']:.4f}, "
        f"across_exercises={analysis_2['comparison_to_full']['across_exercise_pairwise_similarity']['mean_delta_full_minus_no_trace']:.4f}"
    )
    print()

    print("3. Structural template detection")
    for condition_name, condition_result in analysis_3.items():
        print(f"{condition_name}:")
        print(f"  summary start patterns: {condition_result['summary_start_pattern_counts']}")
        print(
            "  top summary starters: "
            f"{format_top_phrase_entries(condition_result['top_summary_start_trigrams'], 'phrase')}"
        )
        print(
            "  top first 5-word openings: "
            f"{format_top_phrase_entries(condition_result['top_first_five_word_openings'], 'phrase')}"
        )
        print(
            "  sentence structure regex counts: "
            f"{condition_result['sentence_structure_pattern_counts']}"
        )
    print()

    print("4. Lexical diversity")
    for condition_name, condition_result in analysis_4.items():
        top_words = ", ".join(
            f"{entry['word']} ({entry['count']})"
            for entry in condition_result["most_frequent_content_words"][:10]
        )
        print(
            f"{condition_name}: TTR {format_stats(condition_result['type_token_ratio_per_summary'])}, "
            f"unique vocabulary={condition_result['unique_vocabulary_size']}"
        )
        print(f"  top content words: {top_words}")
    print()

    print("5. Length analysis")
    for condition_name in ("full", "no_trace"):
        condition_result = analysis_5[condition_name]
        print(f"{condition_name}: {format_stats(condition_result['word_count_distribution'])}")
    print(
        "Length comparison (full - no_trace): "
        f"mean={analysis_5['comparison']['mean_delta_full_minus_no_trace']:.4f}, "
        f"median={analysis_5['comparison']['median_delta_full_minus_no_trace']:.4f}, "
        f"std={analysis_5['comparison']['std_delta_full_minus_no_trace']:.4f}, "
        f"full_longer_mean={analysis_5['comparison']['full_has_longer_mean']}, "
        f"full_more_variable={analysis_5['comparison']['full_has_higher_variability_std']}"
    )
    print()

    print("6. Full-condition within-exercise vs across-exercise similarity")
    print(
        "Within exercise (different students): "
        f"{format_stats(analysis_6['within_exercise_pairwise_similarity'])}"
    )
    print(f"Across exercises: {format_stats(analysis_6['across_exercise_pairwise_similarity'])}")
    ranking = analysis_6["within_exercise_mean_similarity_ranking"]
    ranking_preview = ", ".join(
        f"{entry['exercise_id']} ({entry['mean']:.4f})" for entry in ranking[:5]
    )
    print(f"Top exercises by within-exercise mean similarity: {ranking_preview}")
    print()

    print("7. Full-condition n-gram template detection")
    print(f"Top bigrams: {format_top_phrase_entries(analysis_7['top_bigrams'], 'ngram')}")
    print(f"Top trigrams: {format_top_phrase_entries(analysis_7['top_trigrams'], 'ngram')}")
    print()
    print(f"Saved JSON report to {OUTPUT_PATH}")


def main() -> None:
    full_records = load_records(FULL_PATH)
    no_trace_records = load_records(NO_TRACE_PATH)

    full_similarity = pairwise_similarity_analysis(full_records)
    no_trace_similarity = pairwise_similarity_analysis(no_trace_records)

    analysis_results = {
        "metadata": {
            "full_input_path": str(FULL_PATH),
            "no_trace_input_path": str(NO_TRACE_PATH),
            "output_path": str(OUTPUT_PATH),
            "full_summary_count": len(full_records),
            "no_trace_summary_count": len(no_trace_records),
        },
        "analysis_1_full_condition_similarity": full_similarity,
        "analysis_2_no_trace_condition_similarity": {
            **no_trace_similarity,
            "comparison_to_full": compare_similarity_sections(full_similarity, no_trace_similarity),
        },
        "analysis_3_structural_template_detection": {
            "full": structural_template_analysis(full_records),
            "no_trace": structural_template_analysis(no_trace_records),
        },
        "analysis_4_lexical_diversity": {
            "full": lexical_diversity_analysis(full_records),
            "no_trace": lexical_diversity_analysis(no_trace_records),
        },
        "analysis_5_length_analysis": {
            "full": length_analysis(full_records),
            "no_trace": length_analysis(no_trace_records),
        },
        "analysis_6_full_within_vs_across_exercise_similarity": {
            "within_exercise_pairwise_similarity": full_similarity[
                "within_exercise_pairwise_similarity"
            ],
            "across_exercise_pairwise_similarity": full_similarity[
                "across_exercise_pairwise_similarity"
            ],
            "within_exercise_pair_count_by_exercise": full_similarity[
                "within_exercise_pair_count_by_exercise"
            ],
            "within_exercise_pairwise_similarity_by_exercise": full_similarity[
                "within_exercise_pairwise_similarity_by_exercise"
            ],
            "within_exercise_mean_similarity_ranking": full_similarity[
                "within_exercise_mean_similarity_ranking"
            ],
        },
        "analysis_7_full_condition_ngram_template_detection": ngram_template_analysis(full_records),
    }
    analysis_results["analysis_5_length_analysis"]["comparison"] = compare_lengths(
        analysis_results["analysis_5_length_analysis"]["full"],
        analysis_results["analysis_5_length_analysis"]["no_trace"],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(to_serializable(analysis_results), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print_report(analysis_results)


if __name__ == "__main__":
    main()
