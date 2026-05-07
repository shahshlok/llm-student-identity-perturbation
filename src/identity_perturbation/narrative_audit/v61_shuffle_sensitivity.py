from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.openai_batch import _save_json, _utc_now
from identity_perturbation.narrative_audit.v61_stat_compare import (
    TARGET_METRICS,
    _bootstrap_paired_delta_ci,
    _load_evaluation,
    _paired_values,
    _wilcoxon_p,
)

ROOT = Path(__file__).resolve().parents[3]
BATCH_ROOT = ROOT / "data" / "v61_batch_runs"
DEFAULT_OUT_ROOT = ROOT / "data" / "v61_condition_comparisons"
DEFAULT_OUT_NAME = "shuffle_sensitivity_v1"
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 42

DEFAULT_FULL_RUN_DIR = BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_v6"
DEFAULT_DETERMINISTIC_RUN_DIR = (
    BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1"
)
DEFAULT_WITHIN_CLASS_RUN_DIRS = [
    BATCH_ROOT
    / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_within_class_random_seed101",
    BATCH_ROOT
    / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_within_class_random_seed102",
    BATCH_ROOT
    / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_within_class_random_seed103",
    BATCH_ROOT
    / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_within_class_random_seed104",
    BATCH_ROOT
    / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_within_class_random_seed105",
]
DEFAULT_CROSS_CLASS_RUN_DIRS = [
    BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_cross_class_random_seed201",
    BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_cross_class_random_seed202",
    BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_cross_class_random_seed203",
    BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_cross_class_random_seed204",
    BATCH_ROOT / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_cross_class_random_seed205",
]

EVALUATION_FILENAME = "evaluation.json"
SCHEMA_VERSION = "v6_1_shuffle_sensitivity_v1"


@dataclass(frozen=True)
class RunSpec:
    label: str
    strategy: str
    run_dir: Path
    seed: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate shuffle sensitivity across deterministic and randomized donor draws."
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--name", default=DEFAULT_OUT_NAME)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap resamples for draw-level confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--full-run-dir",
        type=Path,
        default=DEFAULT_FULL_RUN_DIR,
        help="Directory containing the full evaluation artifacts.",
    )
    parser.add_argument(
        "--deterministic-run-dir",
        type=Path,
        default=DEFAULT_DETERMINISTIC_RUN_DIR,
        help="Directory containing the original deterministic shuffle evaluation.",
    )
    parser.add_argument(
        "--within-class-run-dir",
        action="append",
        type=Path,
        dest="within_class_run_dirs",
        help="Repeatable path to a within-class randomized shuffle run directory.",
    )
    parser.add_argument(
        "--cross-class-run-dir",
        action="append",
        type=Path,
        dest="cross_class_run_dirs",
        help="Repeatable path to a cross-class randomized shuffle run directory.",
    )
    return parser.parse_args()


def _resolve_evaluation_path(run_dir: Path) -> Path:
    evaluation_path = (
        run_dir if run_dir.name == EVALUATION_FILENAME else run_dir / EVALUATION_FILENAME
    )
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation JSON not found: {evaluation_path}")
    return evaluation_path


def _strict_index_rows(payload: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"Evaluation missing scores object for {label}")
    rows = scores.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Evaluation missing non-empty scores.rows for {label}")

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Malformed row in evaluation for {label}: {row!r}")
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"Row missing custom_id in evaluation for {label}: {row!r}")
        if custom_id in indexed:
            raise ValueError(f"Duplicate custom_id={custom_id} in evaluation for {label}")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Row missing metrics object for custom_id={custom_id} in {label}")
        indexed[custom_id] = row
    return indexed


def _load_run(run_spec: RunSpec) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    evaluation_path = _resolve_evaluation_path(run_spec.run_dir)
    payload = _load_evaluation(evaluation_path)
    if payload.get("run_name") is None:
        raise ValueError(f"Evaluation missing run_name for {run_spec.label}: {evaluation_path}")
    rows = _strict_index_rows(payload, label=run_spec.label)
    return payload, rows, evaluation_path


def _bootstrap_scalar_ci(
    values: list[float],
    *,
    n_samples: int,
    seed: int,
) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty value list")
    if n_samples < 1:
        raise ValueError(f"bootstrap samples must be positive, got {n_samples}")

    if len(values) == 1:
        value = float(values[0])
        return {"mean": value, "ci95_lower": value, "ci95_upper": value}

    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    n = len(values)
    for _ in range(n_samples):
        sampled = [values[rng.randrange(n)] for _ in range(n)]
        bootstrap_means.append(sum(sampled) / n)
    bootstrap_means.sort()
    lo_idx = int(0.025 * (n_samples - 1))
    hi_idx = int(0.975 * (n_samples - 1))
    return {
        "mean": statistics.mean(values),
        "ci95_lower": bootstrap_means[lo_idx],
        "ci95_upper": bootstrap_means[hi_idx],
    }


def _shared_custom_ids(run_rows: list[dict[str, dict[str, Any]]]) -> list[str]:
    if not run_rows:
        raise ValueError("At least one run is required to compute shared custom_ids")
    shared_ids = set(run_rows[0])
    for rows in run_rows[1:]:
        shared_ids &= set(rows)
    if not shared_ids:
        raise ValueError("No shared custom_ids across the provided runs")
    return sorted(shared_ids)


def _build_run_specs(
    args: argparse.Namespace,
) -> tuple[RunSpec, RunSpec, list[RunSpec], list[RunSpec]]:
    within_dirs = args.within_class_run_dirs or DEFAULT_WITHIN_CLASS_RUN_DIRS
    cross_dirs = args.cross_class_run_dirs or DEFAULT_CROSS_CLASS_RUN_DIRS

    if len(within_dirs) != 5:
        raise ValueError(f"Expected exactly 5 within-class run directories, got {len(within_dirs)}")
    if len(cross_dirs) != 5:
        raise ValueError(f"Expected exactly 5 cross-class run directories, got {len(cross_dirs)}")

    full_run = RunSpec(label="full", strategy="full", run_dir=args.full_run_dir, seed=None)
    deterministic_run = RunSpec(
        label="deterministic",
        strategy="deterministic",
        run_dir=args.deterministic_run_dir,
        seed=None,
    )

    within_runs = [
        RunSpec(
            label=f"within_class_random_seed{101 + idx}",
            strategy="within_class_random",
            run_dir=run_dir,
            seed=101 + idx,
        )
        for idx, run_dir in enumerate(within_dirs)
    ]
    cross_runs = [
        RunSpec(
            label=f"cross_class_random_seed{201 + idx}",
            strategy="cross_class_random",
            run_dir=run_dir,
            seed=201 + idx,
        )
        for idx, run_dir in enumerate(cross_dirs)
    ]
    return full_run, deterministic_run, within_runs, cross_runs


def _summarize_strategy_draws(
    *,
    strategy_name: str,
    full_rows: dict[str, dict[str, Any]],
    draw_rows: list[tuple[RunSpec, dict[str, dict[str, Any]]]],
    run_names: dict[str, str],
    shared_ids: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    filtered_full_rows = {custom_id: full_rows[custom_id] for custom_id in shared_ids}
    strategy_result: dict[str, Any] = {
        "strategy": strategy_name,
        "n_draws": len(draw_rows),
        "shared_custom_ids": shared_ids,
        "shared_n_custom_ids": len(shared_ids),
        "metrics": [],
    }

    for metric_index, (metric_name, summary_key, row_field, kind) in enumerate(TARGET_METRICS):
        metric_draws: list[dict[str, Any]] = []
        draw_mean_deltas: list[float] = []
        draw_p_values: list[float] = []
        draw_sig_any = 0
        draw_sig_supporting = 0
        draw_sig_reversals = 0

        for draw_index, (run_spec, rows) in enumerate(draw_rows):
            filtered_rows = {custom_id: rows[custom_id] for custom_id in shared_ids}
            vals_full, vals_draw, paired_ids = _paired_values(
                filtered_full_rows,
                filtered_rows,
                metric_name,
                row_field,
            )
            if len(paired_ids) != len(shared_ids):
                raise ValueError(
                    f"Pairing mismatch for strategy {strategy_name}, metric {metric_name}, "
                    f"run {run_spec.label}: expected {len(shared_ids)} shared ids, "
                    f"got {len(paired_ids)}"
                )
            if not paired_ids:
                raise ValueError(
                    f"No paired ids available for strategy {strategy_name}, metric {metric_name}"
                )

            ci = _bootstrap_paired_delta_ci(
                vals_full,
                vals_draw,
                n_samples=bootstrap_samples,
                seed=bootstrap_seed + metric_index * 1000 + draw_index,
            )
            p_value = _wilcoxon_p(vals_full, vals_draw)
            mean_full = sum(vals_full) / len(vals_full)
            mean_draw = sum(vals_draw) / len(vals_draw)
            mean_delta = ci["mean_delta"]

            metric_draws.append(
                {
                    "label": run_spec.label,
                    "run_name": run_names[run_spec.label],
                    "run_dir": str(run_spec.run_dir.resolve()),
                    "seed": run_spec.seed,
                    "n_paired": len(paired_ids),
                    "mean_full": mean_full,
                    "mean_shuffle": mean_draw,
                    "mean_delta": mean_delta,
                    "ci95_lower": ci["ci95_lower"],
                    "ci95_upper": ci["ci95_upper"],
                    "wilcoxon_p": p_value,
                }
            )
            draw_mean_deltas.append(mean_delta)
            draw_p_values.append(p_value)
            if p_value < 0.05:
                draw_sig_any += 1
                if mean_delta > 0:
                    draw_sig_supporting += 1
                elif mean_delta < 0:
                    draw_sig_reversals += 1

        draw_distribution = _bootstrap_scalar_ci(
            draw_mean_deltas,
            n_samples=bootstrap_samples,
            seed=bootstrap_seed + metric_index,
        )
        draw_distribution["std"] = (
            statistics.pstdev(draw_mean_deltas) if len(draw_mean_deltas) > 1 else 0.0
        )
        draw_distribution["min"] = min(draw_mean_deltas)
        draw_distribution["max"] = max(draw_mean_deltas)
        draw_distribution["n_draws"] = len(draw_mean_deltas)

        strategy_result["metrics"].append(
            {
                "metric_name": metric_name,
                "summary_key": summary_key,
                "row_field": row_field,
                "kind": kind,
                "draws": metric_draws,
                "draw_mean_delta_distribution": draw_distribution,
                "draw_p_values": draw_p_values,
                "n_draws": len(metric_draws),
                "n_significant_draws_any_direction": draw_sig_any,
                "n_significant_draws_supporting_full": draw_sig_supporting,
                "n_significant_draws_reversal": draw_sig_reversals,
                "significant_draw_fraction_any_direction": draw_sig_any / len(metric_draws),
                "significant_draw_fraction_supporting_full": draw_sig_supporting
                / len(metric_draws),
            }
        )

    return strategy_result


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    full_run, deterministic_run, within_runs, cross_runs = _build_run_specs(args)
    all_runs = [full_run, deterministic_run, *within_runs, *cross_runs]

    loaded_runs: dict[str, dict[str, Any]] = {}
    run_rows: dict[str, dict[str, dict[str, Any]]] = {}
    run_paths: dict[str, str] = {}

    for run_spec in all_runs:
        payload, rows, evaluation_path = _load_run(run_spec)
        loaded_runs[run_spec.label] = payload
        run_rows[run_spec.label] = rows
        run_paths[run_spec.label] = str(evaluation_path.resolve())

    shared_ids = _shared_custom_ids([run_rows[spec.label] for spec in all_runs])

    strategies = {
        "deterministic": [deterministic_run],
        "within_class_random": within_runs,
        "cross_class_random": cross_runs,
    }

    strategy_results: dict[str, Any] = {}
    for strategy_name, runs in strategies.items():
        strategy_results[strategy_name] = _summarize_strategy_draws(
            strategy_name=strategy_name,
            full_rows=run_rows[full_run.label],
            draw_rows=[(run_spec, run_rows[run_spec.label]) for run_spec in runs],
            run_names={
                run_spec.label: loaded_runs[run_spec.label]["run_name"] for run_spec in runs
            },
            shared_ids=shared_ids,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )

    deterministic_metric_p_values = {
        metric["metric_name"]: metric["draws"][0]["wilcoxon_p"]
        for metric in strategy_results["deterministic"]["metrics"]
    }

    newly_sensitive_metrics = []
    unstable_metrics = []
    for metric_name, _summary_key, _, _ in TARGET_METRICS:
        deterministic_p = deterministic_metric_p_values[metric_name]
        within_metric = next(
            metric
            for metric in strategy_results["within_class_random"]["metrics"]
            if metric["metric_name"] == metric_name
        )
        cross_metric = next(
            metric
            for metric in strategy_results["cross_class_random"]["metrics"]
            if metric["metric_name"] == metric_name
        )
        if deterministic_p >= 0.05 and (
            within_metric["n_significant_draws_supporting_full"] > 0
            or cross_metric["n_significant_draws_supporting_full"] > 0
        ):
            newly_sensitive_metrics.append(metric_name)
        if deterministic_p < 0.05 and (
            within_metric["n_significant_draws_supporting_full"] < within_metric["n_draws"]
            or cross_metric["n_significant_draws_supporting_full"] < cross_metric["n_draws"]
        ):
            unstable_metrics.append(metric_name)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "full_condition": {
            "label": full_run.label,
            "run_name": loaded_runs[full_run.label]["run_name"],
            "run_dir": str(full_run.run_dir.resolve()),
            "evaluation_path": run_paths[full_run.label],
            "n_rows": len(run_rows[full_run.label]),
        },
        "deterministic_condition": {
            "label": deterministic_run.label,
            "run_name": loaded_runs[deterministic_run.label]["run_name"],
            "run_dir": str(deterministic_run.run_dir.resolve()),
            "evaluation_path": run_paths[deterministic_run.label],
            "n_rows": len(run_rows[deterministic_run.label]),
        },
        "randomized_conditions": {
            "within_class_random": [
                {
                    "label": run_spec.label,
                    "run_name": loaded_runs[run_spec.label]["run_name"],
                    "run_dir": str(run_spec.run_dir.resolve()),
                    "evaluation_path": run_paths[run_spec.label],
                    "seed": run_spec.seed,
                    "n_rows": len(run_rows[run_spec.label]),
                }
                for run_spec in within_runs
            ],
            "cross_class_random": [
                {
                    "label": run_spec.label,
                    "run_name": loaded_runs[run_spec.label]["run_name"],
                    "run_dir": str(run_spec.run_dir.resolve()),
                    "evaluation_path": run_paths[run_spec.label],
                    "seed": run_spec.seed,
                    "n_rows": len(run_rows[run_spec.label]),
                }
                for run_spec in cross_runs
            ],
        },
        "shared_custom_ids": shared_ids,
        "shared_n_custom_ids": len(shared_ids),
        "target_metrics": [
            {
                "metric_name": metric_name,
                "summary_key": summary_key,
                "row_field": row_field,
                "kind": kind,
            }
            for metric_name, summary_key, row_field, kind in TARGET_METRICS
        ],
        "strategies": strategy_results,
        "question_answers": {
            "within_class_random_stability": {
                "deterministic_p_values": deterministic_metric_p_values,
                "newly_sensitive_metrics": newly_sensitive_metrics,
                "unstable_metrics": unstable_metrics,
            },
            "cross_class_pattern": {
                "within_class_supporting_draws": {
                    metric["metric_name"]: metric["n_significant_draws_supporting_full"]
                    for metric in strategy_results["within_class_random"]["metrics"]
                },
                "cross_class_supporting_draws": {
                    metric["metric_name"]: metric["n_significant_draws_supporting_full"]
                    for metric in strategy_results["cross_class_random"]["metrics"]
                },
            },
            "identity_insensitive_metrics_becoming_sensitive": newly_sensitive_metrics,
        },
    }


def _format_number(value: float) -> str:
    return f"{value:.4f}"


def _format_ci(ci_lower: float, ci_upper: float) -> str:
    return f"[{ci_lower:+.4f}, {ci_upper:+.4f}]"


def _format_p_value(value: float) -> str:
    if value < 0.001:
        return "<0.001***"
    if value < 0.01:
        return f"{value:.4f}**"
    if value < 0.05:
        return f"{value:.4f}*"
    return f"{value:.4f}"


def _format_draw_p_values(draws: list[dict[str, Any]]) -> str:
    return ", ".join(_format_p_value(float(draw["wilcoxon_p"])) for draw in draws)


def _strategy_narrative(
    *,
    metric_name: str,
    deterministic_metric: dict[str, Any],
    within_metric: dict[str, Any],
    cross_metric: dict[str, Any],
) -> str:
    deterministic_p = deterministic_metric["draws"][0]["wilcoxon_p"]
    within_support = within_metric["n_significant_draws_supporting_full"]
    cross_support = cross_metric["n_significant_draws_supporting_full"]
    within_any = within_metric["n_significant_draws_any_direction"]
    cross_any = cross_metric["n_significant_draws_any_direction"]
    if deterministic_p < 0.05:
        return (
            f"Deterministic shuffle is significant; within-class retains support in "
            f"{within_support}/{within_metric['n_draws']} draws and cross-class in "
            f"{cross_support}/{cross_metric['n_draws']} draws."
        )
    if within_support > 0 or cross_support > 0:
        return (
            f"Deterministic shuffle is not significant, but randomized draws make it significant in "
            f"{within_support}/{within_metric['n_draws']} within-class draws and "
            f"{cross_support}/{cross_metric['n_draws']} cross-class draws."
        )
    if within_any > 0 or cross_any > 0:
        return (
            "Deterministic shuffle is not significant, and randomized draws cross the threshold only "
            "in the opposite direction or not at all."
        )
    return "No randomized draw crosses p < 0.05."


def _write_markdown(comparison: dict[str, Any], path: Path) -> None:
    full = comparison["full_condition"]
    deterministic = comparison["deterministic_condition"]
    strategies = comparison["strategies"]
    question_answers = comparison["question_answers"]

    lines = [
        "# Shuffle Sensitivity",
        "",
        f"- Full run: `{full['run_name']}`",
        f"- Deterministic shuffle run: `{deterministic['run_name']}`",
        f"- Shared bundles across all runs: `{comparison['shared_n_custom_ids']}`",
        "",
        "## Short Answers",
        "",
        "### 1. Deterministic shuffle under randomized within-class donors",
    ]
    if question_answers["within_class_random_stability"]["newly_sensitive_metrics"]:
        lines.append(
            "- Some metrics that were not significant in the deterministic shuffle become significant under at least one within-class random draw: "
            + ", ".join(
                f"`{metric}`"
                for metric in question_answers["within_class_random_stability"][
                    "newly_sensitive_metrics"
                ]
            )
            + "."
        )
    else:
        lines.append(
            "- No metric that was non-significant in the deterministic shuffle becomes significant under the within-class random draws."
        )
    if question_answers["within_class_random_stability"]["unstable_metrics"]:
        lines.append(
            "- Deterministic-significant metrics that lose support in at least one within-class draw: "
            + ", ".join(
                f"`{metric}`"
                for metric in question_answers["within_class_random_stability"]["unstable_metrics"]
            )
            + "."
        )
    else:
        lines.append(
            "- Every metric that is significant in the deterministic shuffle stays significant across all within-class draws."
        )

    lines.append("### 2. Cross-class pattern")
    within_support_total = sum(
        metric["n_significant_draws_supporting_full"]
        for metric in strategies["within_class_random"]["metrics"]
    )
    cross_support_total = sum(
        metric["n_significant_draws_supporting_full"]
        for metric in strategies["cross_class_random"]["metrics"]
    )
    if cross_support_total > within_support_total:
        lines.append(
            f"- Cross-class random draws produce more significant full > shuffle metric-draws than within-class draws ({cross_support_total} vs {within_support_total})."
        )
    elif cross_support_total < within_support_total:
        lines.append(
            f"- Cross-class random draws produce fewer significant full > shuffle metric-draws than within-class draws ({cross_support_total} vs {within_support_total})."
        )
    else:
        lines.append(
            f"- Cross-class and within-class random draws produce the same number of significant full > shuffle metric-draws ({cross_support_total})."
        )

    lines.append("### 3. Newly sensitive metrics")
    if question_answers["identity_insensitive_metrics_becoming_sensitive"]:
        lines.append(
            "- Metrics that are non-significant in the deterministic shuffle but significant in at least one randomized draw: "
            + ", ".join(
                f"`{metric}`"
                for metric in question_answers["identity_insensitive_metrics_becoming_sensitive"]
            )
            + "."
        )
    else:
        lines.append(
            "- No metric that is non-significant in the deterministic shuffle becomes significant in the randomized draws."
        )

    lines.extend(
        [
            "",
            "## Run Summary",
            "",
            "| Strategy | Run(s) | Shared bundles | Draws | Sig any | Sig supporting full |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for strategy_name in ("deterministic", "within_class_random", "cross_class_random"):
        strategy = strategies[strategy_name]
        if strategy_name == "deterministic":
            run_names = f"`{strategy['metrics'][0]['draws'][0]['run_name']}`"
        else:
            run_names = ", ".join(
                f"`{draw['run_name']}`" for draw in strategy["metrics"][0]["draws"]
            )
        draw_count = strategy["metrics"][0]["n_draws"]
        sig_any = sum(metric["n_significant_draws_any_direction"] for metric in strategy["metrics"])
        sig_support = sum(
            metric["n_significant_draws_supporting_full"] for metric in strategy["metrics"]
        )
        lines.append(
            f"| `{strategy_name}` | {run_names} | {strategy['shared_n_custom_ids']} | {draw_count} | "
            f"{sig_any} | {sig_support} |"
        )

    lines.extend(["", "## Metric Details", ""])

    for metric_index, (metric_name, summary_key, _, _) in enumerate(TARGET_METRICS):
        det_metric = strategies["deterministic"]["metrics"][metric_index]
        within_metric = strategies["within_class_random"]["metrics"][metric_index]
        cross_metric = strategies["cross_class_random"]["metrics"][metric_index]

        lines.extend(
            [
                f"### `{metric_name}` ({summary_key})",
                "",
                _strategy_narrative(
                    metric_name=metric_name,
                    deterministic_metric=det_metric,
                    within_metric=within_metric,
                    cross_metric=cross_metric,
                ),
                "",
                "| Strategy | Draws | Shared bundles | Mean of draw mean deltas | Std | 95% CI across draws | Sig any | Sig supporting full | Draw p-values |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )

        for strategy_name in ("deterministic", "within_class_random", "cross_class_random"):
            metric = strategies[strategy_name]["metrics"][metric_index]
            dist = metric["draw_mean_delta_distribution"]
            lines.append(
                f"| `{strategy_name}` | {metric['n_draws']} | {metric['draws'][0]['n_paired']} | "
                f"{_format_number(dist['mean'])} | {_format_number(dist['std'])} | "
                f"{_format_ci(dist['ci95_lower'], dist['ci95_upper'])} | "
                f"{metric['n_significant_draws_any_direction']}/{metric['n_draws']} | "
                f"{metric['n_significant_draws_supporting_full']}/{metric['n_draws']} | "
                f"{_format_draw_p_values(metric['draws'])} |"
            )

        lines.extend(["", "#### Per-draw details", ""])
        for strategy_name in ("deterministic", "within_class_random", "cross_class_random"):
            metric = strategies[strategy_name]["metrics"][metric_index]
            lines.append(f"- `{strategy_name}`")
            for draw in metric["draws"]:
                lines.append(
                    f"  - `{draw['label']}` (`{draw['run_name']}`): "
                    f"n={draw['n_paired']}, mean delta={_format_number(draw['mean_delta'])}, "
                    f"CI={_format_ci(draw['ci95_lower'], draw['ci95_upper'])}, "
                    f"Wilcoxon p={_format_p_value(draw['wilcoxon_p'])}"
                )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Delta is always `full - shuffle`, so positive values mean the full condition is higher.",
            "- Wilcoxon p-values are two-sided and are reported per draw.",
            "- The draw-level confidence intervals are bootstrap percentile intervals over the paired deltas within each draw.",
            "- The across-draw confidence intervals are bootstrap percentile intervals over the per-draw mean deltas.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    comparison = run_analysis(args)

    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "sensitivity.json"
    md_path = out_dir / "sensitivity.md"
    _save_json(json_path, comparison)
    _write_markdown(comparison, md_path)

    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
