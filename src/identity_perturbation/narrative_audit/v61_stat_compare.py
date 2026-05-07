"""Paired statistical comparison of V6.1 condition evaluation results.

Produces:
  1. Paired bootstrap 95% CIs (2000 resamples) on the delta between conditions
  2. Paired Wilcoxon signed-rank tests
  3. Majority-class and random baselines for categorical predictive metrics

Usage:
    python -m identity_perturbation.narrative_audit.v61_stat_compare
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from scipy import stats as scipy_stats

from identity_perturbation.codebench_support.openai_batch import _save_json, _utc_now

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUT_ROOT = ROOT / "data" / "v61_condition_comparisons"

# ---------------------------------------------------------------------------
# Condition paths
# ---------------------------------------------------------------------------
BATCH_ROOT = ROOT / "data" / "v61_batch_runs"

CONDITION_PATHS = {
    "full": BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_v6" / "evaluation.json",
    "no_trace": BATCH_ROOT
    / "gpt54_medium_v61_clean126_buildable117_no_trace_v1"
    / "evaluation.json",
    "trace_shuffled": BATCH_ROOT
    / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1"
    / "evaluation.json",
}

# ---------------------------------------------------------------------------
# Target metrics: (metric_name, summary_key, row_field, kind)
#   kind: "categorical" for top1_match accuracy, "continuous" for sequence similarity
# ---------------------------------------------------------------------------
TARGET_METRICS = [
    ("first_event_type", "top1_accuracy", "top1_match", "categorical"),
    ("run_presence", "top1_accuracy", "top1_match", "categorical"),
    ("episode_motif", "top1_accuracy", "top1_match", "categorical"),
    ("event_type_overlap", "mean_top1_jaccard", "top1_jaccard", "continuous"),
    ("event_type_edit_similarity", "mean_top1_similarity", "top1_similarity", "continuous"),
    ("event_family_lcss", "mean_top1_lcss_ratio", "top1_lcss_ratio", "continuous"),
    ("event_family_dtw", "mean_top1_similarity", "top1_similarity", "continuous"),
]

CONDITION_PAIRS = [
    ("full", "no_trace"),
    ("full", "trace_shuffled"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_evaluation(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema_version") != "v6_1_batch_evaluation_v2":
        raise ValueError(f"Unexpected schema in {path}: {payload.get('schema_version')}")
    return payload


def _index_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload["scores"]["rows"]
    return {str(row["custom_id"]): row for row in rows}


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def _paired_values(
    rows_a: dict[str, dict[str, Any]],
    rows_b: dict[str, dict[str, Any]],
    metric_name: str,
    row_field: str,
) -> tuple[list[float], list[float], list[str]]:
    """Return aligned value arrays for all shared custom_ids."""
    common_ids = sorted(set(rows_a) & set(rows_b))
    vals_a = []
    vals_b = []
    for cid in common_ids:
        vals_a.append(float(rows_a[cid]["metrics"][metric_name][row_field]))
        vals_b.append(float(rows_b[cid]["metrics"][metric_name][row_field]))
    return vals_a, vals_b, common_ids


# ---------------------------------------------------------------------------
# Bootstrap CI on paired delta
# ---------------------------------------------------------------------------


def _bootstrap_paired_delta_ci(
    vals_a: list[float],
    vals_b: list[float],
    n_samples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap 95% CI on mean(A - B) via paired resampling by transition."""
    n = len(vals_a)
    assert n == len(vals_b), "Paired arrays must have equal length"
    deltas = [vals_a[i] - vals_b[i] for i in range(n)]
    observed_mean_delta = sum(deltas) / n

    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(n_samples):
        sampled = [deltas[rng.randrange(n)] for _ in range(n)]
        bootstrap_means.append(sum(sampled) / n)
    bootstrap_means.sort()

    lo_idx = int(0.025 * (n_samples - 1))
    hi_idx = int(0.975 * (n_samples - 1))
    return {
        "mean_delta": observed_mean_delta,
        "ci95_lower": bootstrap_means[lo_idx],
        "ci95_upper": bootstrap_means[hi_idx],
    }


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------


def _wilcoxon_p(vals_a: list[float], vals_b: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank test p-value on paired differences."""
    diffs = [a - b for a, b in zip(vals_a, vals_b, strict=False)]
    # If all diffs are zero, Wilcoxon is undefined
    if all(d == 0.0 for d in diffs):
        return 1.0
    try:
        result = scipy_stats.wilcoxon(diffs, alternative="two-sided")
        return float(result.pvalue)
    except ValueError:
        # Too few non-zero differences
        return 1.0


# ---------------------------------------------------------------------------
# Baselines (computed on the full condition's observed labels)
# ---------------------------------------------------------------------------


def _compute_baselines(
    rows: dict[str, dict[str, Any]],
    metric_name: str,
    row_field: str,
    kind: str,
) -> dict[str, float | None]:
    """Compute majority-class and random baselines.

    For categorical metrics: majority-class accuracy, weighted-random accuracy.
    For continuous metrics: baselines are not well-defined, return None.
    """
    if kind != "categorical":
        return {"majority_class": None, "weighted_random": None}

    observed_vals = [str(r["metrics"][metric_name]["observed"]) for r in rows.values()]
    counts = Counter(observed_vals)
    total = len(observed_vals)
    majority_class_accuracy = counts.most_common(1)[0][1] / total
    weighted_random_accuracy = sum((c / total) ** 2 for c in counts.values())
    return {
        "majority_class": majority_class_accuracy,
        "weighted_random": weighted_random_accuracy,
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def run_comparison(
    condition_paths: dict[str, Path],
    target_metrics: list[tuple[str, str, str, str]],
    condition_pairs: list[tuple[str, str]],
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    # Load all conditions
    payloads: dict[str, dict[str, Any]] = {}
    indexed_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in condition_paths.items():
        payloads[label] = _load_evaluation(path)
        indexed_rows[label] = _index_rows(payloads[label])

    results: list[dict[str, Any]] = []

    for metric_name, summary_key, row_field, kind in target_metrics:
        # Baselines (from full condition)
        baselines = _compute_baselines(indexed_rows["full"], metric_name, row_field, kind)

        # Per-condition means
        condition_means: dict[str, float] = {}
        for label, rows in indexed_rows.items():
            vals = [float(r["metrics"][metric_name][row_field]) for r in rows.values()]
            condition_means[label] = sum(vals) / len(vals)

        pair_results: list[dict[str, Any]] = []
        for label_a, label_b in condition_pairs:
            vals_a, vals_b, common_ids = _paired_values(
                indexed_rows[label_a],
                indexed_rows[label_b],
                metric_name,
                row_field,
            )
            ci = _bootstrap_paired_delta_ci(vals_a, vals_b, bootstrap_samples, seed)
            p_value = _wilcoxon_p(vals_a, vals_b)
            pair_results.append(
                {
                    "pair": f"{label_a} - {label_b}",
                    "n_paired": len(common_ids),
                    "mean_a": sum(vals_a) / len(vals_a),
                    "mean_b": sum(vals_b) / len(vals_b),
                    **ci,
                    "wilcoxon_p": p_value,
                }
            )

        results.append(
            {
                "metric_name": metric_name,
                "summary_key": summary_key,
                "kind": kind,
                "condition_means": condition_means,
                "baselines": baselines,
                "pair_comparisons": pair_results,
            }
        )

    return {
        "schema_version": "v6_1_stat_comparison_v1",
        "created_at": _utc_now(),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "conditions": {
            label: {
                "path": str(path.resolve()),
                "run_name": payloads[label]["run_name"],
                "n_rows": len(indexed_rows[label]),
            }
            for label, path in condition_paths.items()
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def _print_report(comparison: dict[str, Any]) -> None:
    print("=" * 100)
    print("V6.1 PAIRED STATISTICAL COMPARISON")
    print("=" * 100)
    print()

    # Conditions summary
    print("CONDITIONS:")
    for label, info in comparison["conditions"].items():
        print(f"  {label:20s}  n={info['n_rows']}  run={info['run_name']}")
    print(f"\nBootstrap samples: {comparison['bootstrap_samples']}, seed: {comparison['seed']}")
    print()

    # Results table
    header = (
        f"{'Metric':<35s} "
        f"{'Pair':<25s} "
        f"{'Mean A':>8s} "
        f"{'Mean B':>8s} "
        f"{'Delta':>8s} "
        f"{'95% CI':>20s} "
        f"{'Wilcoxon p':>12s} "
        f"{'Majority':>10s} "
        f"{'Rand':>8s}"
    )
    print(header)
    print("-" * len(header))

    for result in comparison["results"]:
        metric_name = result["metric_name"]
        summary_key = result["summary_key"]
        baselines = result["baselines"]
        display_name = f"{metric_name} ({summary_key})"

        majority_str = (
            f"{baselines['majority_class']:.3f}"
            if baselines["majority_class"] is not None
            else "n/a"
        )
        rand_str = (
            f"{baselines['weighted_random']:.3f}"
            if baselines["weighted_random"] is not None
            else "n/a"
        )

        for i, pair in enumerate(result["pair_comparisons"]):
            ci_str = f"[{pair['ci95_lower']:+.4f}, {pair['ci95_upper']:+.4f}]"
            sig = ""
            if pair["wilcoxon_p"] < 0.001:
                sig = " ***"
            elif pair["wilcoxon_p"] < 0.01:
                sig = " **"
            elif pair["wilcoxon_p"] < 0.05:
                sig = " *"

            # Only show baselines on the first pair row
            if i == 0:
                print(
                    f"{display_name:<35s} "
                    f"{pair['pair']:<25s} "
                    f"{pair['mean_a']:>8.4f} "
                    f"{pair['mean_b']:>8.4f} "
                    f"{pair['mean_delta']:>+8.4f} "
                    f"{ci_str:>20s} "
                    f"{pair['wilcoxon_p']:>10.4f}{sig:>2s} "
                    f"{majority_str:>10s} "
                    f"{rand_str:>8s}"
                )
            else:
                print(
                    f"{'':35s} "
                    f"{pair['pair']:<25s} "
                    f"{pair['mean_a']:>8.4f} "
                    f"{pair['mean_b']:>8.4f} "
                    f"{pair['mean_delta']:>+8.4f} "
                    f"{ci_str:>20s} "
                    f"{pair['wilcoxon_p']:>10.4f}{sig:>2s} "
                    f"{'':>10s} "
                    f"{'':>8s}"
                )

    print()
    print("Legend: Delta = Mean_A - Mean_B (positive means A > B)")
    print("        * p < 0.05, ** p < 0.01, *** p < 0.001 (Wilcoxon signed-rank, two-sided)")
    print("        Majority = majority-class baseline accuracy")
    print("        Rand = weighted-random (sum p_i^2) baseline accuracy")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _write_markdown(comparison: dict[str, Any], path: Path) -> None:
    lines = [
        "# V6.1 Paired Statistical Comparison",
        "",
        "## Conditions",
        "",
        "| Label | Run name | N |",
        "| --- | --- | --- |",
    ]
    for label, info in comparison["conditions"].items():
        lines.append(f"| {label} | `{info['run_name']}` | {info['n_rows']} |")

    lines.extend(
        [
            "",
            f"Bootstrap samples: {comparison['bootstrap_samples']}, seed: {comparison['seed']}",
            "",
            "## Results",
            "",
            "| Metric | Pair | N paired | Mean A | Mean B | Delta | 95% CI | Wilcoxon p | Majority | Rand |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for result in comparison["results"]:
        metric_label = f"`{result['metric_name']}` ({result['summary_key']})"
        baselines = result["baselines"]
        majority_str = (
            f"{baselines['majority_class']:.3f}"
            if baselines["majority_class"] is not None
            else "n/a"
        )
        rand_str = (
            f"{baselines['weighted_random']:.3f}"
            if baselines["weighted_random"] is not None
            else "n/a"
        )

        for i, pair in enumerate(result["pair_comparisons"]):
            ci_str = f"[{pair['ci95_lower']:+.4f}, {pair['ci95_upper']:+.4f}]"
            p_str = f"{pair['wilcoxon_p']:.4f}"
            if pair["wilcoxon_p"] < 0.001:
                p_str += " ***"
            elif pair["wilcoxon_p"] < 0.01:
                p_str += " **"
            elif pair["wilcoxon_p"] < 0.05:
                p_str += " *"

            base_m = majority_str if i == 0 else ""
            base_r = rand_str if i == 0 else ""
            lines.append(
                f"| {metric_label if i == 0 else ''} "
                f"| {pair['pair']} "
                f"| {pair['n_paired']} "
                f"| {pair['mean_a']:.4f} "
                f"| {pair['mean_b']:.4f} "
                f"| {pair['mean_delta']:+.4f} "
                f"| {ci_str} "
                f"| {p_str} "
                f"| {base_m} "
                f"| {base_r} |"
            )

    lines.extend(
        [
            "",
            "**Legend**: Delta = Mean_A - Mean_B (positive means A > B). "
            "\\* p < 0.05, \\*\\* p < 0.01, \\*\\*\\* p < 0.001 (Wilcoxon signed-rank, two-sided). "
            "Majority = majority-class baseline accuracy. "
            "Rand = weighted-random (sum p_i^2) baseline accuracy.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6.1 paired statistical comparison.")
    parser.add_argument("--name", default="stat_comparison_v1", help="Output directory name.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    comparison = run_comparison(
        condition_paths=CONDITION_PATHS,
        target_metrics=TARGET_METRICS,
        condition_pairs=CONDITION_PAIRS,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    _print_report(comparison)

    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(out_dir / "stat_comparison.json", comparison)
    _write_markdown(comparison, out_dir / "stat_comparison.md")
    print(f"\nSaved JSON: {out_dir / 'stat_comparison.json'}")
    print(f"Saved Markdown: {out_dir / 'stat_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
