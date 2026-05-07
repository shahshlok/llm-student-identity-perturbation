from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.runner import ROOT, write_json, write_text

DEFAULT_OUT_ROOT = ROOT / "data" / "v6_condition_comparisons"


class V6ConditionComparisonError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired v6 evaluation runs on the same transitions."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Path to baseline v6 evaluation.json",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=Path,
        help="Path to candidate v6 evaluation.json; may be repeated",
    )
    parser.add_argument(
        "--comparison-name",
        required=True,
        help="Subdirectory name for the comparison artifact bundle",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Artifact root for condition-comparison outputs",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for confidence intervals",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap resampling",
    )
    parser.add_argument(
        "--resample-unit",
        choices=("transition", "student"),
        default="transition",
        help="Unit for paired bootstrap resampling",
    )
    return parser.parse_args()


def _load_evaluation(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise V6ConditionComparisonError(f"Evaluation file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v6_batch_evaluation_v1":
        raise V6ConditionComparisonError(
            f"Unexpected evaluation schema in {path}: {payload.get('schema_version')}"
        )
    return payload


def _index_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise V6ConditionComparisonError("Evaluation must contain a non-empty rows list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        custom_id = str(row["custom_id"])
        if custom_id in indexed:
            raise V6ConditionComparisonError(f"Duplicate custom_id in evaluation: {custom_id}")
        indexed[custom_id] = row
    return indexed


def _paired_rows(
    *,
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_rows = _index_rows(baseline_payload)
    candidate_rows = _index_rows(candidate_payload)
    baseline_ids = set(baseline_rows)
    candidate_ids = set(candidate_rows)
    if baseline_ids != candidate_ids:
        missing_from_candidate = sorted(baseline_ids - candidate_ids)
        missing_from_baseline = sorted(candidate_ids - baseline_ids)
        raise V6ConditionComparisonError(
            "Evaluation row sets do not match exactly; "
            f"missing_from_candidate={missing_from_candidate[:5]} "
            f"missing_from_baseline={missing_from_baseline[:5]}"
        )
    paired = []
    for custom_id in sorted(baseline_ids):
        baseline_row = baseline_rows[custom_id]
        candidate_row = candidate_rows[custom_id]
        if baseline_row["observed_heads"] != candidate_row["observed_heads"]:
            raise V6ConditionComparisonError(
                f"Observed labels differ for custom_id {custom_id}; cannot compare conditions"
            )
        paired.append(
            {
                "custom_id": custom_id,
                "student_id": str(baseline_row["student_id"]),
                "baseline": baseline_row,
                "candidate": candidate_row,
            }
        )
    return paired


def _extract_numeric_metrics(row: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "joint.truth_mass": float(row["joint"]["truth_probability_mass"]),
        "joint.top1": float(bool(row["joint"]["top1_match"])),
        "joint.topk": float(bool(row["joint"]["topk_hit"])),
    }
    by_head = row["by_head"]
    for head_key, head_row in by_head.items():
        metrics[f"{head_key}.truth_mass"] = float(head_row["truth_probability_mass"])
        metrics[f"{head_key}.top1"] = float(bool(head_row["top1_match"]))
        metrics[f"{head_key}.topk"] = float(bool(head_row["topk_hit"]))
        metrics[f"{head_key}.brier"] = float(head_row["brier_score"])
    return metrics


def _mean_metric(rows: list[dict[str, Any]], side: str, metric_key: str) -> float:
    return sum(_extract_numeric_metrics(row[side])[metric_key] for row in rows) / len(rows)


def _mean_delta(rows: list[dict[str, Any]], metric_key: str) -> float:
    return sum(
        _extract_numeric_metrics(row["candidate"])[metric_key]
        - _extract_numeric_metrics(row["baseline"])[metric_key]
        for row in rows
    ) / len(rows)


def _bootstrap_delta_ci(
    *,
    rows: list[dict[str, Any]],
    metric_key: str,
    bootstrap_samples: int,
    seed: int,
    resample_unit: str,
) -> dict[str, float]:
    if bootstrap_samples < 1:
        raise V6ConditionComparisonError(
            f"--bootstrap-samples must be positive; got {bootstrap_samples}"
        )
    rng = random.Random(seed)
    samples: list[float] = []

    if resample_unit == "transition":
        for _ in range(bootstrap_samples):
            sampled_rows = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
            samples.append(_mean_delta(sampled_rows, metric_key))
    elif resample_unit == "student":
        rows_by_student: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            rows_by_student.setdefault(row["student_id"], []).append(row)
        student_ids = sorted(rows_by_student)
        for _ in range(bootstrap_samples):
            sampled_rows: list[dict[str, Any]] = []
            for _ in range(len(student_ids)):
                sampled_rows.extend(rows_by_student[student_ids[rng.randrange(len(student_ids))]])
            samples.append(_mean_delta(sampled_rows, metric_key))
    else:
        raise V6ConditionComparisonError(f"Unsupported resample unit: {resample_unit}")

    samples.sort()
    lower_index = int(0.025 * (bootstrap_samples - 1))
    upper_index = int(0.975 * (bootstrap_samples - 1))
    return {
        "delta_mean": _mean_delta(rows, metric_key),
        "ci95_lower": samples[lower_index],
        "ci95_upper": samples[upper_index],
    }


def _metric_keys(example_row: dict[str, Any]) -> list[str]:
    metrics = _extract_numeric_metrics(example_row)
    preferred_order = [
        "joint.truth_mass",
        "joint.top1",
        "joint.topk",
        "likely_first_repair_region.truth_mass",
        "likely_first_repair_region.top1",
        "likely_first_repair_region.topk",
        "likely_first_repair_region.brier",
        "likely_edit_scope.truth_mass",
        "likely_edit_scope.top1",
        "likely_edit_scope.topk",
        "likely_edit_scope.brier",
        "likely_next_test_outcome.truth_mass",
        "likely_next_test_outcome.top1",
        "likely_next_test_outcome.topk",
        "likely_next_test_outcome.brier",
    ]
    return [key for key in preferred_order if key in metrics]


def compare_conditions(
    *,
    baseline_path: Path,
    candidate_paths: list[Path],
    comparison_name: str,
    out_root: Path,
    bootstrap_samples: int,
    seed: int,
    resample_unit: str,
) -> dict[str, Any]:
    baseline_payload = _load_evaluation(baseline_path)
    out_dir = out_root / comparison_name
    out_dir.mkdir(parents=True, exist_ok=False)

    candidate_reports: list[dict[str, Any]] = []
    for candidate_path in candidate_paths:
        candidate_payload = _load_evaluation(candidate_path)
        paired = _paired_rows(
            baseline_payload=baseline_payload,
            candidate_payload=candidate_payload,
        )
        metric_keys = _metric_keys(paired[0]["baseline"])
        comparison_rows: list[dict[str, Any]] = []
        for metric_key in metric_keys:
            baseline_mean = _mean_metric(paired, "baseline", metric_key)
            candidate_mean = _mean_metric(paired, "candidate", metric_key)
            ci = _bootstrap_delta_ci(
                rows=paired,
                metric_key=metric_key,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                resample_unit=resample_unit,
            )
            comparison_rows.append(
                {
                    "metric": metric_key,
                    "baseline_mean": baseline_mean,
                    "candidate_mean": candidate_mean,
                    **ci,
                }
            )

        candidate_reports.append(
            {
                "candidate_run_name": str(candidate_payload["run_name"]),
                "candidate_path": str(candidate_path.resolve()),
                "n_paired_rows": len(paired),
                "metrics": comparison_rows,
            }
        )

    result = {
        "schema_version": "v6_condition_comparison_v1",
        "baseline_run_name": str(baseline_payload["run_name"]),
        "baseline_path": str(baseline_path.resolve()),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "resample_unit": resample_unit,
        "candidates": candidate_reports,
    }
    write_json(out_dir / "comparison.json", result)

    lines = [
        "# V6 Condition Comparison",
        "",
        f"- Baseline: `{baseline_payload['run_name']}`",
        f"- Bootstrap samples: `{bootstrap_samples}`",
        f"- Resample unit: `{resample_unit}`",
        "",
    ]
    for candidate in candidate_reports:
        lines.extend(
            [
                f"## {candidate['candidate_run_name']}",
                "",
                "| Metric | Baseline | Candidate | Delta | 95% CI |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for metric in candidate["metrics"]:
            lines.append(
                f"| {metric['metric']} | {metric['baseline_mean']:.3f} | "
                f"{metric['candidate_mean']:.3f} | {metric['delta_mean']:.3f} | "
                f"[{metric['ci95_lower']:.3f}, {metric['ci95_upper']:.3f}] |"
            )
        lines.append("")
    write_text(out_dir / "comparison.md", "\n".join(lines) + "\n")
    return {
        "result": result,
        "paths": {
            "comparison_dir": str(out_dir.resolve()),
            "comparison_json": str((out_dir / "comparison.json").resolve()),
            "comparison_md": str((out_dir / "comparison.md").resolve()),
        },
    }


def main() -> int:
    args = parse_args()
    try:
        outcome = compare_conditions(
            baseline_path=args.baseline,
            candidate_paths=list(args.candidate),
            comparison_name=args.comparison_name,
            out_root=args.out_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            resample_unit=args.resample_unit,
        )
    except V6ConditionComparisonError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(outcome["result"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
