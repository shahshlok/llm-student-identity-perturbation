"""Reusable report driver for a whole identity-perturbation prediction run.

Reads the run manifest, pairs each bundle with one line of the batch
output JSONL (by ``custom_id``), scores every bundle with the v2
scorer, then emits a single consolidated report JSON containing:

    * per-row scored predictions
    * run-level aggregate (mean / median per family / metric / view)
    * per-exercise-scope majority baselines and paired model lift
    * identity discrimination (A-vs-B) on code metrics
    * pretty ASCII summary on stdout

Run with uv:

    uv run python -m identity_perturbation.prediction_audit.report_full_trace_run_v2 \
        --run-manifest  data/prediction_audit/final_full_trace/manifest.json \
        --output-jsonl  data/prediction_audit/openai_batch_outputs/batch_69e8290b6d78819098933a8a0e8e5a11_output.jsonl \
        --out           data/prediction_audit/final_full_trace/scores_v2/report.json

The driver is intentionally fail-fast: the output JSONL must match the
frozen manifest exactly. Any missing or unexpected ``custom_id`` means
the condition denominator drifted, so the report exits instead of
silently trimming rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from identity_perturbation.prediction_audit.full_trace_scorer_v2 import score_full_trace_prediction_v2
from identity_perturbation.prediction_audit.full_trace_suite_v2 import (
    RowKey,
    ScoredRow,
    aggregate_rows,
    compute_identity_discrimination,
    compute_reality_divergence,
    compute_twin_prediction_similarity,
    model_lift_over_baseline,
    score_majority_baselines,
)
from identity_perturbation.prediction_audit.score_full_trace_bundle import (
    _load_bundle_manifest,
    _load_json_strict,
    _prediction_from_batch_output_item,
    _resolve_bundle_artifact_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the identity-perturbation scoring suite across an entire batch run."
    )
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--condition",
        default="full",
        help="Condition label stamped onto every row (full / no_trace / trace_shuffled).",
    )
    parser.add_argument(
        "--l2b-thresholds",
        default="0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help=(
            "Comma-separated L2B normalized-code-distance thresholds for post-hoc "
            "L2A∩L2B sensitivity analyses. Pass an empty string to disable."
        ),
    )
    return parser.parse_args()


def _parse_l2b_thresholds(raw: str) -> tuple[float, ...]:
    if not raw.strip():
        return ()
    values: list[float] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            value = float(stripped)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid L2B threshold {stripped!r}; expected a comma-separated float list"
            ) from exc
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"L2B threshold must be between 0 and 1, got {value}")
        values.append(value)
    return tuple(sorted(set(values)))


def _l2b_threshold_key(threshold: float) -> str:
    return format(threshold, ".15g")


def _load_output_jsonl_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id")
            if not isinstance(custom_id, str):
                raise SystemExit(f"Line {line_num} missing string custom_id: {path}")
            if custom_id in index:
                raise SystemExit(f"Duplicate custom_id {custom_id!r} at line {line_num}")
            index[custom_id] = item
    return index


def _score_one_bundle(
    *,
    bundle_dir: Path,
    output_item: dict[str, Any],
    condition: str,
) -> ScoredRow:
    manifest = _load_bundle_manifest(bundle_dir)
    custom_id = str(manifest["custom_id"])
    observed_repair_target_path = _resolve_bundle_artifact_path(
        bundle_dir=bundle_dir,
        manifest_value=manifest["observed_next_repair_target_path"],
        expected_filename="observed_next_repair_target.json",
    )
    observed_coarse_path_path = _resolve_bundle_artifact_path(
        bundle_dir=bundle_dir,
        manifest_value=manifest["observed_next_coarse_path_path"],
        expected_filename="observed_next_coarse_path.json",
    )
    observed_repair_target = _load_json_strict(observed_repair_target_path)
    observed_coarse_path = _load_json_strict(observed_coarse_path_path)
    attempt_n_normalized_code = manifest.get("attempt_n_normalized_code")
    if not isinstance(attempt_n_normalized_code, str) or not attempt_n_normalized_code:
        raise SystemExit(
            f"Bundle {custom_id} is missing non-empty attempt_n_normalized_code in manifest"
        )
    response_payload, _source_meta = _prediction_from_batch_output_item(
        item=output_item,
        expected_custom_id=custom_id,
    )
    scored = score_full_trace_prediction_v2(
        response_payload=response_payload,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )
    return ScoredRow(
        custom_id=custom_id,
        key=RowKey.from_custom_id(custom_id),
        condition=condition,
        scored=scored,
        response_payload=response_payload,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
        attempt_n_pass_fail_vector=tuple(
            bool(value) for value in manifest["attempt_n_pass_fail_vector"]
        ),
        attempt_n_pass_vector_signature=str(manifest["attempt_n_pass_vector_signature"]),
        attempt_n_normalized_code=attempt_n_normalized_code,
    )


CODE_LIFT_METRICS: tuple[tuple[str, str, str], ...] = (
    ("repair", "windowed_footprint_f1", "top_1"),
    ("repair", "aligned_content_f1", "top_1"),
    ("repair", "aligned_content_identifier_f1", "top_1"),
    ("repair", "code_gain_over_copy_bounded", "top_1"),
    ("full_code", "exact_next_code_match", "top_1"),
    ("full_code", "structural_lift_over_copy", "top_1"),
    ("trajectory", "edit_region_overlap_unordered", "top_1"),
    ("trajectory", "trajectory_alignment_score", "top_1"),
)

IDENTITY_METRICS: tuple[tuple[str, str], ...] = (
    ("repair", "windowed_footprint_f1"),
    ("repair", "aligned_content_f1"),
    ("repair", "aligned_content_identifier_f1"),
    ("repair", "code_gain_over_copy_bounded"),
    ("trajectory", "trajectory_alignment_score"),
    ("trajectory", "edit_region_overlap_unordered"),
    ("trajectory", "local_run_count_agreement"),
    ("full_code", "exact_next_code_match"),
    ("full_code", "structural_lift_over_copy"),
)

IDENTITY_VIEWS: tuple[str, ...] = ("top_1", "expected", "rank_weighted")

TWIN_METRICS: tuple[tuple[str, str], ...] = (
    ("trajectory", "trajectory_alignment_score"),
    ("trajectory", "edit_region_overlap_unordered"),
    ("trajectory", "local_run_count_agreement"),
    ("full_code", "exact_next_code_match"),
    ("full_code", "full_code_structural_similarity_diagnostic"),
)

REALITY_METRICS: tuple[tuple[str, str], ...] = IDENTITY_METRICS


def _log_progress(message: str) -> None:
    print(f"[report] {message}", file=sys.stderr, flush=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _fmt_float(value: float, *, width: int = 7) -> str:
    if value != value:
        return "   nan".rjust(width)
    return f"{value:+.3f}".rjust(width) if value < 0 else f" {value:.3f}".rjust(width)


def _print_ascii_summary(
    *,
    rows: list[ScoredRow],
    aggregate: dict[str, Any],
    lifts: list[dict[str, Any]],
    discriminations: list[dict[str, Any]],
    realities: list[dict[str, Any]],
    twins: list[dict[str, Any]],
    l2b_identity: dict[str, list[dict[str, Any]]],
) -> None:
    print()
    print(f"  scored rows: {len(rows)}")
    unique_scopes = {r.key.exercise_scope for r in rows}
    students_per_scope = {
        scope: len({r.key.student_id for r in rows if r.key.exercise_scope == scope})
        for scope in unique_scopes
    }
    print(f"  exercise scopes: {len(unique_scopes)}")
    print(
        "  scopes with >=2 distinct students: "
        + str(sum(1 for n in students_per_scope.values() if n >= 2))
    )
    print()
    print("  ── run-level aggregate (top_1 view, mean across rows) ──")
    print("  family       metric                              mean  median")
    print("  ─────────── ──────────────────────────────────  ──────  ──────")
    for family in ("repair", "trajectory", "full_code"):
        for metric, view_data in aggregate["families"][family].items():
            top1 = view_data["top_1"]
            print(
                f"  {family:<11} {metric:<35} "
                f"{_fmt_float(top1['mean'])}  {_fmt_float(top1['median'])}"
            )
        print()
    print("  ── lift over per-exercise majority baseline (top_1) ──")
    print("  family       metric                              meanΔ   win%   sign p  n_unique")
    print("  ─────────── ──────────────────────────────────  ──────  ─────  ───────  ────────")
    for lift in lifts:
        wr = lift["win_rate"]
        wr_s = "  nan" if wr != wr else f"{int(round(wr * 100)):4d}%"
        sp = lift["sign_test_p_two_sided"]
        sp_s = "    nan" if sp != sp else f"{sp:7.3f}"
        print(
            f"  {lift['family']:<11} {lift['metric']:<35} "
            f"{_fmt_float(lift['mean_delta'])}  {wr_s}  {sp_s}   {lift['n_unique_scope_rows']:>4}/{lift['n_rows']}"
        )
    print()
    if discriminations:
        print("  ── identity discrimination (A-vs-B within frozen L2A groups) ──")
        print(
            "  family       metric                              view         mean_self  mean_other   AUC   n_pairs"
        )
        print(
            "  ─────────── ──────────────────────────────────  ───────────  ─────────  ──────────  ─────  ───────"
        )
        for d in discriminations:
            auc = d["discrim_auc_self_gt_other"]
            auc_s = "  nan" if auc != auc else f"{auc:5.3f}"
            print(
                f"  {d['family']:<11} {d['metric']:<35} "
                f"{d['view']:<11}  "
                f"{_fmt_float(d['mean_self'])}   {_fmt_float(d['mean_other'])}   "
                f"{auc_s}   {d['n_pairs']:>4}"
            )
        print()
    else:
        print("  ── identity discrimination: skipped (no scope has >=2 students) ──")
        print()
    if realities:
        print("  ── peer reality similarity (A truth vs B truth within frozen L2A groups) ──")
        print("  family       metric                              mean_sim  median   n_pairs")
        print("  ─────────── ──────────────────────────────────  ────────  ──────  ───────")
        for reality in realities:
            print(
                f"  {reality['family']:<11} {reality['metric']:<35} "
                f"{_fmt_float(reality['mean_reality_similarity'])}  "
                f"{_fmt_float(reality['median_reality_similarity'])}   {reality['n_pairs']:>4}"
            )
        print()
    else:
        print("  ── peer reality similarity: skipped (no L2A group has >=2 students) ──")
        print()
    if twins:
        print("  ── twin prediction similarity (A-vs-B top-1 within L2A groups) ──")
        print("  family       metric                           mean_sim  median   n_pairs")
        print("  ─────────── ────────────────────────────────  ────────  ──────  ───────")
        for twin in twins:
            print(
                f"  {twin['family']:<11} {twin['metric']:<33} "
                f"{_fmt_float(twin['mean_twin_similarity'])}  "
                f"{_fmt_float(twin['median_twin_similarity'])}   {twin['n_pairs']:>4}"
            )
        print()
    else:
        print("  ── twin prediction similarity: skipped (no L2A group has >=2 students) ──")
        print()
    if l2b_identity:
        print("  ── L2B sensitivity (within frozen L2A groups) ──")
        print("  threshold  directed pairs  match groups")
        print("  ─────────  ─────────────  ───────────")
        for threshold_key in sorted(l2b_identity):
            sample = next(
                (
                    item
                    for item in l2b_identity[threshold_key]
                    if item["family"] == "full_code"
                    and item["metric"] == "exact_next_code_match"
                    and item["view"] == "top_1"
                ),
                l2b_identity[threshold_key][0] if l2b_identity[threshold_key] else None,
            )
            if sample is None:
                continue
            print(
                f"  {threshold_key:<9} {sample['n_pairs']:>13}  {sample['n_match_groups_with_pairs']:>11}"
            )
        print()


def main() -> int:
    args = parse_args()
    l2b_thresholds = _parse_l2b_thresholds(args.l2b_thresholds)
    run_manifest_path: Path = args.run_manifest
    output_jsonl_path: Path = args.output_jsonl
    out_path: Path = args.out

    if not run_manifest_path.exists():
        raise SystemExit(f"Run manifest not found: {run_manifest_path}")
    if not output_jsonl_path.exists():
        raise SystemExit(f"Output JSONL not found: {output_jsonl_path}")

    _log_progress(f"loading manifest: {run_manifest_path}")
    run_manifest = _load_json_strict(run_manifest_path)
    bundle_map = run_manifest.get("bundle_map")
    if not isinstance(bundle_map, dict):
        raise SystemExit("Run manifest missing bundle_map dict")
    _log_progress(f"loading batch output index: {output_jsonl_path}")
    output_by_id = _load_output_jsonl_by_custom_id(output_jsonl_path)

    rows: list[ScoredRow] = []
    manifest_ids = set(bundle_map)
    output_ids = set(output_by_id)
    missing_output_ids = sorted(manifest_ids - output_ids)
    unexpected_output_ids = sorted(output_ids - manifest_ids)
    if missing_output_ids or unexpected_output_ids:
        details: list[str] = []
        if missing_output_ids:
            preview = ", ".join(missing_output_ids[:5])
            suffix = (
                ""
                if len(missing_output_ids) <= 5
                else f", ... (+{len(missing_output_ids) - 5} more)"
            )
            details.append(
                f"missing outputs for {len(missing_output_ids)} manifest rows: {preview}{suffix}"
            )
        if unexpected_output_ids:
            preview = ", ".join(unexpected_output_ids[:5])
            suffix = (
                ""
                if len(unexpected_output_ids) <= 5
                else f", ... (+{len(unexpected_output_ids) - 5} more)"
            )
            details.append(
                f"unexpected outputs for {len(unexpected_output_ids)} rows: {preview}{suffix}"
            )
        raise SystemExit("Batch output does not match frozen manifest; " + "; ".join(details))

    total_rows = len(bundle_map)
    _log_progress(f"scoring {total_rows} frozen rows")
    for row_idx, (custom_id, bundle_entry) in enumerate(bundle_map.items(), start=1):
        bundle_dir = Path(bundle_entry["bundle_dir"])
        if not bundle_dir.exists():
            raise SystemExit(f"Bundle directory missing for {custom_id}: {bundle_dir}")
        try:
            row = _score_one_bundle(
                bundle_dir=bundle_dir,
                output_item=output_by_id[custom_id],
                condition=args.condition,
            )
        except Exception as exc:
            raise SystemExit(f"Failed scoring {custom_id}: {exc}") from exc
        rows.append(row)
        if row_idx == 1 or row_idx % 10 == 0 or row_idx == total_rows:
            _log_progress(f"scored rows: {row_idx}/{total_rows}")

    if not rows:
        raise SystemExit("No rows successfully scored; nothing to report.")

    _log_progress("aggregating row-level views")
    aggregate = aggregate_rows(rows)
    _log_progress("scoring per-scope majority baselines")
    baselines = score_majority_baselines(rows)
    lifts: list[dict[str, Any]] = []
    for family, metric, view in CODE_LIFT_METRICS:
        _log_progress(f"baseline lift: {family}/{metric}/{view}")
        lifts.append(
            model_lift_over_baseline(
                rows,
                baselines,
                family=family,
                metric=metric,
                view=view,
            )
        )

    discriminations: list[dict[str, Any]] = []
    realities: list[dict[str, Any]] = []
    twins: list[dict[str, Any]] = []
    l2b_discriminations: dict[str, list[dict[str, Any]]] = {}
    l2b_realities: dict[str, list[dict[str, Any]]] = {}
    l2b_twins: dict[str, list[dict[str, Any]]] = {}
    for family, metric in IDENTITY_METRICS:
        for view in IDENTITY_VIEWS:
            _log_progress(f"identity discrimination: {family}/{metric}/{view}")
            discrimination = compute_identity_discrimination(
                rows,
                family=family,
                metric=metric,
                view=view,
            )
            if discrimination["n_pairs"] > 0:
                discriminations.append(discrimination)
            for threshold in l2b_thresholds:
                threshold_key = _l2b_threshold_key(threshold)
                discrimination_l2b = compute_identity_discrimination(
                    rows,
                    family=family,
                    metric=metric,
                    view=view,
                    l2b_threshold=threshold,
                )
                if discrimination_l2b["n_pairs"] > 0:
                    l2b_discriminations.setdefault(threshold_key, []).append(discrimination_l2b)
    for family, metric in REALITY_METRICS:
        _log_progress(f"peer reality similarity: {family}/{metric}")
        reality = compute_reality_divergence(rows, family=family, metric=metric)
        if reality["n_pairs"] > 0:
            realities.append(reality)
        for threshold in l2b_thresholds:
            threshold_key = _l2b_threshold_key(threshold)
            reality_l2b = compute_reality_divergence(
                rows,
                family=family,
                metric=metric,
                l2b_threshold=threshold,
            )
            if reality_l2b["n_pairs"] > 0:
                l2b_realities.setdefault(threshold_key, []).append(reality_l2b)
    for family, metric in TWIN_METRICS:
        _log_progress(f"twin prediction similarity: {family}/{metric}")
        twin = compute_twin_prediction_similarity(rows, family=family, metric=metric)
        if twin["n_pairs"] > 0:
            twins.append(twin)
        for threshold in l2b_thresholds:
            threshold_key = _l2b_threshold_key(threshold)
            twin_l2b = compute_twin_prediction_similarity(
                rows,
                family=family,
                metric=metric,
                l2b_threshold=threshold,
            )
            if twin_l2b["n_pairs"] > 0:
                l2b_twins.setdefault(threshold_key, []).append(twin_l2b)

    report = {
        "schema_version": "v6_2_full_trace_run_report_v2",
        "run_manifest_path": str(run_manifest_path.resolve()),
        "output_jsonl_path": str(output_jsonl_path.resolve()),
        "condition": args.condition,
        "n_rows_scored": len(rows),
        "skipped": [],
        "aggregate": aggregate,
        "majority_baselines": {
            scope: {
                "unique_scope": baselines[scope]["unique_scope"],
                "n_rows_in_scope": baselines[scope]["n_rows_in_scope"],
                "views": baselines[scope]["scored"]["views"],
            }
            for scope in baselines
        },
        "lift_over_baseline": lifts,
        "l2b_thresholds": list(l2b_thresholds),
        "identity_discrimination": discriminations,
        "identity_discrimination_l2b": l2b_discriminations,
        "reality_peer_similarity": realities,
        "reality_peer_similarity_l2b": l2b_realities,
        "twin_prediction_similarity": twins,
        "twin_prediction_similarity_l2b": l2b_twins,
        "rows": [
            {
                "custom_id": row.custom_id,
                "key": {
                    "class_id": row.key.class_id,
                    "student_id": row.key.student_id,
                    "exercise_id": row.key.exercise_id,
                    "task_id": row.key.task_id,
                    "attempt_n": row.key.attempt_n,
                    "exercise_scope": row.key.exercise_scope,
                },
                "attempt_n_pass_fail_vector": list(row.attempt_n_pass_fail_vector),
                "attempt_n_pass_vector_signature": row.attempt_n_pass_vector_signature,
                "attempt_n_normalized_code": row.attempt_n_normalized_code,
                "views": row.scored["views"],
            }
            for row in rows
        ],
    }

    _log_progress(f"writing report atomically: {out_path}")
    _atomic_write_json(out_path, report)
    _print_ascii_summary(
        rows=rows,
        aggregate=aggregate,
        lifts=lifts,
        discriminations=discriminations,
        realities=realities,
        twins=twins,
        l2b_identity=l2b_discriminations,
    )
    print(f"  wrote report: {out_path.resolve()}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
