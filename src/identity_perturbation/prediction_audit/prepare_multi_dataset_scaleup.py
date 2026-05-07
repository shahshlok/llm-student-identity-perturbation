from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.executions import ExecutionParseError, parse_execution_log_text
from identity_perturbation.prediction_audit.match_policy import narrow_normalize_code_for_match
from identity_perturbation.prediction_audit.pair_matching import normalized_code_distance
from identity_perturbation.prediction_audit.raw_same_task_family_audit import V62AuditError, parse_assessment_data
import identity_perturbation.prediction_audit.select_branching_probe as selector

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOTS = ("2022-2", "2023-1", "2023-2", "2024-1")
DEFAULT_CONDITIONS = ("full", "no_trace")
DEFAULT_VISIBLE_ATTEMPT_COUNT = 3
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_RUN_DATE = datetime.now().strftime("%Y%m%d")


class MultiDatasetScaleupError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a combined multi-dataset v6.2 scale-up run by reconstructing "
            "scope metadata from raw logs and then reusing the live selector/bundle pipeline."
        )
    )
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=list(DEFAULT_DATA_ROOTS),
        help="Dataset roots to merge, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Conditions to emit. Defaults to full and no_trace.",
    )
    parser.add_argument(
        "--visible-attempt-count",
        type=int,
        default=DEFAULT_VISIBLE_ATTEMPT_COUNT,
        help="Visible attempt count n. The model then predicts attempt n+1.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument(
        "--run-date",
        default=DEFAULT_RUN_DATE,
        help="Datestamp suffix for output directories, e.g. 20260421.",
    )
    parser.add_argument(
        "--prep-root",
        type=Path,
        default=REPO_ROOT / "data/v62/prep",
        help="Root directory for shared merged-dataset prep artifacts.",
    )
    parser.add_argument(
        "--batch-runs-root",
        type=Path,
        default=REPO_ROOT / "data/v62/batch_runs",
        help="Root directory for emitted batch runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate prep / run directories if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts and manifests without emitting bundles.",
    )
    parser.add_argument(
        "--skip-mean-pairwise",
        action="store_true",
        help=(
            "Skip exact per-scope mean pairwise distance reconstruction and write 0.0 instead. "
            "This preserves stage-1 canonical filtering and stage-3/4/5 row selection, but "
            "makes tertile/diversity metadata non-informative."
        ),
    )
    return parser.parse_args()


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _safe_recreate_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise MultiDatasetScaleupError(
                f"Path already exists; rerun with --force to replace it: {path}"
            )
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=False)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _iter_scope_rows(
    data_root: Path,
    *,
    skip_mean_pairwise: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scope_rows: list[dict[str, Any]] = []
    bad_assessment_files: list[dict[str, str]] = []
    missing_users_roots: Counter[str] = Counter()
    assessment_file_count = 0

    print(f"[prep] scanning dataset {data_root}", flush=True)
    for assessment_path in sorted(data_root.glob("*/assessments/*.data")):
        assessment_file_count += 1
        try:
            assessment = parse_assessment_data(assessment_path)
        except V62AuditError as exc:
            bad_assessment_files.append(
                {"path": str(assessment_path.resolve()), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        users_root = data_root / assessment.class_id / "users"
        if not users_root.exists():
            missing_users_roots[str(users_root.resolve())] += 1
            continue

        for exercise_id in assessment.exercise_ids:
            normalized_final_codes: list[str] = []
            parse_failures = 0
            for user_dir in sorted(path for path in users_root.iterdir() if path.is_dir()):
                execution_path = (
                    user_dir / "executions" / f"{assessment.assessment_id}_{exercise_id}.log"
                )
                if not execution_path.exists():
                    continue
                try:
                    attempts = parse_execution_log_text(
                        execution_path.read_text(encoding="utf-8")
                    )
                except ExecutionParseError:
                    parse_failures += 1
                    continue
                if not attempts:
                    continue
                normalized_final_codes.append(narrow_normalize_code_for_match(attempts[-1].code))

            if not normalized_final_codes:
                continue

            code_counts = Counter(normalized_final_codes)
            top_code_count = code_counts.most_common(1)[0][1]
            if skip_mean_pairwise:
                mean_pairwise_dist = 0.0
            else:
                distances: list[float] = []
                for left_index in range(len(normalized_final_codes)):
                    left_code = normalized_final_codes[left_index]
                    for right_code in normalized_final_codes[left_index + 1 :]:
                        distances.append(normalized_code_distance(left_code, right_code))
                mean_pairwise_dist = sum(distances) / len(distances) if distances else 0.0

            scope_rows.append(
                {
                    "class_id": assessment.class_id,
                    "assessment_id": assessment.assessment_id,
                    "exercise_id": exercise_id,
                    "n_students": len(normalized_final_codes),
                    "mean_pairwise_dist": mean_pairwise_dist,
                    "distinct_codes": len(code_counts),
                    "canonical_80": (top_code_count / len(normalized_final_codes)) >= 0.8,
                    "dataset_root": str(data_root.resolve()),
                    "parse_failures_ignored": parse_failures,
                }
            )

    audit = {
        "dataset_root": str(data_root.resolve()),
        "assessment_files_seen": assessment_file_count,
        "bad_assessment_files_skipped": bad_assessment_files,
        "missing_users_roots": [
            {"path": path, "assessment_file_count": count}
            for path, count in sorted(missing_users_roots.items())
        ],
        "scope_records_emitted": len(scope_rows),
        "skip_mean_pairwise": skip_mean_pairwise,
    }
    print(
        "[prep] finished dataset "
        f"{data_root.name}: scopes={len(scope_rows)}, "
        f"bad_assessments={len(bad_assessment_files)}, "
        f"missing_users_roots={len(missing_users_roots)}",
        flush=True,
    )
    return scope_rows, audit


def _build_merged_root(data_roots: list[Path], merged_root: Path) -> dict[str, str]:
    class_to_dataset: dict[str, str] = {}
    print("[prep] building merged dataset root", flush=True)
    for data_root in data_roots:
        for class_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
            target = merged_root / class_dir.name
            if class_dir.name in class_to_dataset:
                previous = class_to_dataset[class_dir.name]
                raise MultiDatasetScaleupError(
                    f"Class id collision in merged root for {class_dir.name}: {previous} vs {data_root}"
                )
            os.symlink(class_dir.resolve(), target, target_is_directory=True)
            class_to_dataset[class_dir.name] = str(data_root.resolve())
    print(f"[prep] merged dataset root ready with {len(class_to_dataset)} classes", flush=True)
    return class_to_dataset


def _compute_selected_scope_count(
    *,
    scopes_json: Path,
    dataset_root: Path,
    condition: str,
    visible_attempt_count: int,
) -> dict[str, Any]:
    print(
        f"[selector] computing surviving scope ceiling for n={visible_attempt_count} on {dataset_root}",
        flush=True,
    )
    scope_records = selector._load_scope_records(scopes_json)
    retained_scopes, stage1 = selector.stage1_canonical_exclusion(scope_records)
    scoped_assignments, _stage2 = selector.stage2_assign_tertiles(retained_scopes)
    scope_candidates, stage3 = selector.stage3_enumerate_scope_candidates(scoped_assignments, dataset_root)
    validated_by_scope, stage4 = selector.stage4_validate_attempt_depth(
        scoped_assignments, scope_candidates
    )
    selected_pairs, stage5 = selector.stage5_select_student_pairs(
        scoped_assignments,
        validated_by_scope,
        seed=42,
        dataset_root=dataset_root,
        condition=condition,
    )
    return {
        "selected_scope_count": len(selected_pairs),
        "selected_row_count": stage5["selected_row_count"],
        "stage1": stage1,
        "stage3": stage3,
        "stage4": stage4,
        "stage5": stage5,
        "visible_attempt_count": visible_attempt_count,
    }


def _configure_selector_for_visible_attempts(visible_attempt_count: int) -> None:
    if visible_attempt_count < 2:
        raise MultiDatasetScaleupError(
            f"visible-attempt-count must be >= 2, got {visible_attempt_count}"
        )
    selector.FIXED_VISIBLE_ATTEMPT_COUNT = visible_attempt_count
    selector.FIXED_INTERIOR_N = visible_attempt_count - 1
    selector.MIN_TOTAL_ATTEMPTS_FOR_FIXED_SELECTION = visible_attempt_count + 1


def main() -> int:
    args = parse_args()
    data_roots = [_resolve_repo_path(path) for path in args.data_roots]
    for data_root in data_roots:
        if not data_root.exists():
            raise MultiDatasetScaleupError(f"Dataset root does not exist: {data_root}")

    visible_attempt_count = int(args.visible_attempt_count)
    _configure_selector_for_visible_attempts(visible_attempt_count)
    combined_label = (
        f"multi_dataset_n{visible_attempt_count}_{len(data_roots)}roots_{args.run_date}"
    )
    prep_dir = args.prep_root.resolve() / combined_label
    merged_root = prep_dir / "merged_root"
    scopes_json = prep_dir / "scopes.json"
    prep_report_path = prep_dir / "prep_report.json"
    _safe_recreate_dir(prep_dir, force=args.force)
    merged_root.mkdir(parents=True, exist_ok=False)
    print(f"[prep] writing shared artifacts under {prep_dir}", flush=True)

    combined_scope_rows: list[dict[str, Any]] = []
    dataset_audits: list[dict[str, Any]] = []
    for data_root in data_roots:
        scope_rows, audit = _iter_scope_rows(
            data_root,
            skip_mean_pairwise=bool(args.skip_mean_pairwise),
        )
        combined_scope_rows.extend(scope_rows)
        dataset_audits.append(audit)

    class_to_dataset = _build_merged_root(data_roots, merged_root)
    _write_json(scopes_json, combined_scope_rows)
    print(
        f"[prep] combined scopes written: {len(combined_scope_rows)} records -> {scopes_json}",
        flush=True,
    )

    prep_report = {
        "schema_version": "v6_2_multi_dataset_scaleup_prep_v1",
        "visible_attempt_count": visible_attempt_count,
        "interior_n": visible_attempt_count - 1,
        "minimum_total_attempt_count": visible_attempt_count + 1,
        "data_roots": [str(path) for path in data_roots],
        "merged_root": str(merged_root),
        "scopes_json": str(scopes_json),
        "dataset_audits": dataset_audits,
        "class_to_dataset_root": class_to_dataset,
        "combined_scope_record_count": len(combined_scope_rows),
        "skip_mean_pairwise": bool(args.skip_mean_pairwise),
    }
    _write_json(prep_report_path, prep_report)

    selector_summary = _compute_selected_scope_count(
        scopes_json=scopes_json,
        dataset_root=merged_root,
        condition="full",
        visible_attempt_count=visible_attempt_count,
    )
    target_scopes = int(selector_summary["selected_scope_count"])
    if target_scopes <= 0:
        raise MultiDatasetScaleupError(
            "No scopes survived stage 5; refusing to emit an empty run"
        )
    print(
        f"[selector] selected scope ceiling for n={visible_attempt_count}: "
        f"scopes={target_scopes}, rows={selector_summary['selected_row_count']}",
        flush=True,
    )

    run_summaries: list[dict[str, Any]] = []
    for condition in args.conditions:
        run_name = (
            f"scaleup_l2a_n{visible_attempt_count}_{args.model.replace('.', '')}_"
            f"{args.reasoning_effort}_{target_scopes}scope_{condition}_"
            f"{len(data_roots)}dataset_{args.run_date}"
        )
        run_dir = args.batch_runs_root.resolve() / run_name
        if run_dir.exists() and args.force:
            shutil.rmtree(run_dir)
        elif run_dir.exists():
            raise MultiDatasetScaleupError(
                f"Run directory already exists; rerun with --force to replace it: {run_dir}"
            )

        print(
            f"[run] emitting condition={condition} dry_run={bool(args.dry_run)} -> {run_dir}",
            flush=True,
        )
        report = selector.run_selector(
            scopes_json=scopes_json,
            dataset_root=merged_root,
            out_dir=run_dir,
            condition=condition,
            target_scopes=target_scopes,
            seed=42,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            dry_run=bool(args.dry_run),
        )
        run_summaries.append(
            {
                "condition": condition,
                "run_dir": str(run_dir),
                "final_scope_total": report["summary"]["final_scope_total"],
                "final_bundle_total": report["summary"]["final_bundle_total"],
                "final_scope_counts": report["summary"]["final_scope_counts"],
                "dry_run": bool(args.dry_run),
            }
        )
        print(
            f"[run] completed condition={condition}: "
            f"scopes={report['summary']['final_scope_total']}, "
            f"rows={report['summary']['final_bundle_total']}",
            flush=True,
        )

    final_report = {
        "prep_report_path": str(prep_report_path),
        "selector_summary": {
            "selected_scope_count": selector_summary["selected_scope_count"],
            "selected_row_count": selector_summary["selected_row_count"],
            "stage1_dropped_count": selector_summary["stage1"]["dropped_count"],
            "stage3_candidates_with_both_logs": selector_summary["stage3"][
                "total_candidates_with_both_logs"
            ],
            "stage4_retained_students": selector_summary["stage4"]["total_retained_students"],
        },
        "run_summaries": run_summaries,
    }
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
