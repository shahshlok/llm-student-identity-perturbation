"""Scope-cluster bootstrap comparison for prediction-condition reports.

This consumes two existing ``report_full_trace_run_v2`` JSON artifacts
and re-estimates the condition contrast at the scope level rather than
pretending the directed peer pairs are independent.

Two report families are emitted:

1. Test 2A identity discrimination contrasts
   ``mean_discrim_delta(left) - mean_discrim_delta(right)``

2. Peer-reality-anchored Test 2B over-collapse contrasts
   ``(mean_twin_similarity - mean_reality_similarity)_left - same_right``

Usage:

    uv run python -m identity_perturbation.prediction_audit.compare_condition_reports_v2 \
        --left-report  data/prediction_audit/final_full_trace/scores_v2/report.json \
        --right-report data/prediction_audit/final_no_trace/scores_v2/report.json \
        --out          data/prediction_audit/final_condition_comparison/report.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from identity_perturbation.prediction_audit.full_trace_suite_v2 import (
    FullTraceSuiteV2Error,
    cluster_bootstrap_identity_condition_contrast,
    cluster_bootstrap_overcollapse_condition_contrast,
)
from identity_perturbation.prediction_audit.report_full_trace_run_v2 import REALITY_METRICS, TWIN_METRICS

EXPECTED_SCHEMA_VERSION = "v6_2_full_trace_run_report_v2"
EXPECTED_TEST2B_KEYS = tuple(sorted(set(TWIN_METRICS) & set(REALITY_METRICS)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scope-cluster bootstrap comparison of two prediction-condition reports."
    )
    parser.add_argument("--left-report", required=True, type=Path)
    parser.add_argument("--right-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of scope-bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic bootstrap seed.",
    )
    parser.add_argument(
        "--identity-views",
        default="top_1",
        help="Comma-separated identity-discrimination views to compare.",
    )
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise SystemExit(
            f"Unexpected report schema in {path}: expected {EXPECTED_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    return payload


def _require_condition_label(report: dict[str, Any], *, report_label: str) -> str:
    condition = report.get("condition")
    if not isinstance(condition, str) or not condition:
        raise SystemExit(f"{report_label} report must carry a non-empty string condition")
    return condition


def _require_report_section(
    report: dict[str, Any],
    *,
    key: str,
    report_label: str,
) -> list[dict[str, Any]]:
    if key not in report:
        raise SystemExit(f"{report_label} report is missing required section {key!r}")
    section = report[key]
    if not isinstance(section, list):
        raise SystemExit(f"{report_label} report section {key!r} must be a list")
    return section


def _parse_identity_views(raw: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        value = part.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    values = tuple(values)
    if not values:
        raise SystemExit("identity-views must contain at least one non-empty view")
    return values


def _require_requested_identity_views_present(
    identity_index: dict[tuple[Any, ...], dict[str, Any]],
    *,
    label: str,
    identity_views: tuple[str, ...],
) -> None:
    available_views = {str(key[2]) for key in identity_index}
    missing_requested = [view for view in identity_views if view not in available_views]
    if missing_requested:
        raise SystemExit(
            f"{label} is missing requested identity views: " + ", ".join(missing_requested)
        )


def _index_entries(
    entries: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    label: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit(f"{label} entries must be dicts")
        for field in key_fields:
            if field not in entry:
                raise SystemExit(f"{label} entry is missing required field {field!r}")
            value = entry[field]
            if field == "l2b_threshold":
                if value is not None and not isinstance(value, (int, float)):
                    raise SystemExit(
                        f"{label} entry field {field!r} must be null or numeric, got {value!r}"
                    )
                continue
            if not isinstance(value, str) or not value:
                raise SystemExit(
                    f"{label} entry field {field!r} must be a non-empty string, got {value!r}"
                )
        key = tuple(entry.get(field) for field in key_fields)
        if key in index:
            raise SystemExit(f"Duplicate {label} entry for key {key!r}")
        index[key] = entry
    return index


def _require_same_keys(
    left_index: dict[tuple[Any, ...], dict[str, Any]],
    right_index: dict[tuple[Any, ...], dict[str, Any]],
    *,
    label: str,
) -> list[tuple[Any, ...]]:
    left_keys = set(left_index)
    right_keys = set(right_index)
    if left_keys != right_keys:
        missing = sorted(left_keys - right_keys)
        extra = sorted(right_keys - left_keys)
        details: list[str] = []
        if missing:
            details.append(
                f"missing keys {missing[:3]!r}"
                + ("" if len(missing) <= 3 else f" (+{len(missing) - 3} more)")
            )
        if extra:
            details.append(
                f"unexpected keys {extra[:3]!r}"
                + ("" if len(extra) <= 3 else f" (+{len(extra) - 3} more)")
            )
        raise SystemExit(f"{label} key mismatch: " + "; ".join(details))
    return sorted(left_keys)


def _shared_keys_or_die(
    left_index: dict[tuple[Any, ...], dict[str, Any]],
    right_index: dict[tuple[Any, ...], dict[str, Any]],
    *,
    label: str,
) -> list[tuple[Any, ...]]:
    shared = sorted(set(left_index) & set(right_index))
    if not shared:
        raise SystemExit(f"{label} has no shared keys to compare")
    return shared


def _filter_identity_index_by_views(
    identity_index: dict[tuple[Any, ...], dict[str, Any]],
    identity_views: tuple[str, ...],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {key: entry for key, entry in identity_index.items() if str(key[2]) in identity_views}


def _normalize_base_analysis_entry(
    entry: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(entry)
    match_mode = normalized.get("match_mode", "l2a")
    if not isinstance(match_mode, str) or not match_mode:
        raise SystemExit(
            f"analysis entry match_mode must be a non-empty string, got {match_mode!r}"
        )
    if match_mode != "l2a":
        raise SystemExit(
            f"analysis entry match_mode must be 'l2a' for base reports, got {match_mode!r}"
        )
    normalized["match_mode"] = match_mode
    l2b_threshold = normalized.get("l2b_threshold", None)
    if l2b_threshold is not None and not isinstance(l2b_threshold, (int, float)):
        raise SystemExit(
            f"analysis entry l2b_threshold must be null or numeric, got {l2b_threshold!r}"
        )
    if l2b_threshold is not None:
        raise SystemExit(
            f"analysis entry l2b_threshold must be null for base reports, got {l2b_threshold!r}"
        )
    normalized["l2b_threshold"] = l2b_threshold
    return normalized


def _normalize_l2b_analysis_entry(
    entry: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    normalized = dict(entry)
    match_mode = normalized.get("match_mode")
    if not isinstance(match_mode, str) or not match_mode:
        raise SystemExit(
            f"analysis entry match_mode must be a non-empty string, got {match_mode!r}"
        )
    if match_mode != "l2a_and_l2b":
        raise SystemExit(
            f"analysis entry match_mode must be 'l2a_and_l2b' for L2B reports, got {match_mode!r}"
        )
    normalized["match_mode"] = match_mode
    l2b_threshold = normalized.get("l2b_threshold")
    if not isinstance(l2b_threshold, (int, float)):
        raise SystemExit(
            f"analysis entry l2b_threshold must be numeric for L2B reports, got {l2b_threshold!r}"
        )
    if float(l2b_threshold) != float(threshold):
        raise SystemExit(
            f"analysis entry l2b_threshold {l2b_threshold!r} does not match expected threshold {threshold!r}"
        )
    normalized["l2b_threshold"] = float(l2b_threshold)
    return normalized


def _require_nonempty_view_field(
    entry: dict[str, Any],
    *,
    label: str,
) -> None:
    if "view" not in entry:
        raise SystemExit(f"{label} entry is missing required field 'view'")
    view = entry["view"]
    if not isinstance(view, str) or not view:
        raise SystemExit(f"{label} entry field 'view' must be a non-empty string, got {view!r}")


def _l2b_threshold_key(value: float) -> str:
    return format(float(value), "g")


def _require_l2b_thresholds(
    report: dict[str, Any],
    *,
    report_label: str,
) -> tuple[float, ...]:
    raw = report.get("l2b_thresholds")
    if not isinstance(raw, list):
        raise SystemExit(f"{report_label} report section 'l2b_thresholds' must be a list")
    thresholds: list[float] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, (int, float)):
            raise SystemExit(
                f"{report_label} report L2B threshold entries must be numeric, got {item!r}"
            )
        value = float(item)
        key = _l2b_threshold_key(value)
        if key in seen:
            raise SystemExit(
                f"{report_label} report L2B thresholds contain duplicate value {value!r}"
            )
        seen.add(key)
        thresholds.append(value)
    if not thresholds:
        raise SystemExit(f"{report_label} report must carry at least one L2B threshold")
    return tuple(thresholds)


def _require_matching_l2b_thresholds(
    left_thresholds: tuple[float, ...],
    right_thresholds: tuple[float, ...],
) -> tuple[float, ...]:
    left_keys = tuple(_l2b_threshold_key(value) for value in left_thresholds)
    right_keys = tuple(_l2b_threshold_key(value) for value in right_thresholds)
    if left_keys != right_keys:
        raise SystemExit(
            f"L2B threshold mismatch: left={list(left_keys)!r} right={list(right_keys)!r}"
        )
    return left_thresholds


def _require_l2b_section(
    report: dict[str, Any],
    *,
    key: str,
    report_label: str,
    expected_thresholds: tuple[float, ...],
) -> dict[str, list[dict[str, Any]]]:
    if key not in report:
        raise SystemExit(f"{report_label} report is missing required section {key!r}")
    section = report[key]
    if not isinstance(section, dict):
        raise SystemExit(f"{report_label} report section {key!r} must be a dict")
    expected_keys = {_l2b_threshold_key(value) for value in expected_thresholds}
    actual_keys = set(section)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        details: list[str] = []
        if missing:
            details.append(f"missing thresholds {missing!r}")
        if extra:
            details.append(f"unexpected thresholds {extra!r}")
        raise SystemExit(
            f"{report_label} report section {key!r} threshold mismatch: " + "; ".join(details)
        )
    validated: dict[str, list[dict[str, Any]]] = {}
    for threshold_key, entries in section.items():
        if not isinstance(entries, list):
            raise SystemExit(
                f"{report_label} report section {key!r}[{threshold_key!r}] must be a list"
            )
        validated[threshold_key] = entries
    return validated


def _require_expected_test2b_keys_present(
    index: dict[tuple[Any, ...], dict[str, Any]],
    *,
    label: str,
) -> None:
    missing = [key for key in EXPECTED_TEST2B_KEYS if key not in index]
    if missing:
        raise SystemExit(f"{label} is missing required Test 2B metrics: {missing!r}")


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


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def _is_too_few_scope_clusters_error(exc: FullTraceSuiteV2Error) -> bool:
    return "Cluster bootstrap requires at least 2 scope clusters" in str(exc)


def main() -> int:
    args = parse_args()
    identity_views = _parse_identity_views(args.identity_views)
    if args.bootstrap_samples <= 0:
        raise SystemExit(f"bootstrap-samples must be positive, got {args.bootstrap_samples}")

    left_report = _load_report(args.left_report)
    right_report = _load_report(args.right_report)
    left_condition = _require_condition_label(left_report, report_label="left")
    right_condition = _require_condition_label(right_report, report_label="right")
    l2b_thresholds = _require_matching_l2b_thresholds(
        _require_l2b_thresholds(left_report, report_label="left"),
        _require_l2b_thresholds(right_report, report_label="right"),
    )

    identity_key_fields = ("family", "metric", "view")
    twin_key_fields = ("family", "metric")
    reality_key_fields = ("family", "metric")

    left_identity_index = _index_entries(
        _require_report_section(
            left_report,
            key="identity_discrimination",
            report_label="left",
        ),
        key_fields=identity_key_fields,
        label="left identity_discrimination",
    )
    right_identity_index = _index_entries(
        _require_report_section(
            right_report,
            key="identity_discrimination",
            report_label="right",
        ),
        key_fields=identity_key_fields,
        label="right identity_discrimination",
    )
    left_twin_index = _index_entries(
        _require_report_section(
            left_report,
            key="twin_prediction_similarity",
            report_label="left",
        ),
        key_fields=twin_key_fields,
        label="left twin_prediction_similarity",
    )
    right_twin_index = _index_entries(
        _require_report_section(
            right_report,
            key="twin_prediction_similarity",
            report_label="right",
        ),
        key_fields=twin_key_fields,
        label="right twin_prediction_similarity",
    )
    left_reality_index = _index_entries(
        _require_report_section(
            left_report,
            key="reality_peer_similarity",
            report_label="left",
        ),
        key_fields=reality_key_fields,
        label="left reality_peer_similarity",
    )
    right_reality_index = _index_entries(
        _require_report_section(
            right_report,
            key="reality_peer_similarity",
            report_label="right",
        ),
        key_fields=reality_key_fields,
        label="right reality_peer_similarity",
    )
    left_identity_l2b_sections = _require_l2b_section(
        left_report,
        key="identity_discrimination_l2b",
        report_label="left",
        expected_thresholds=l2b_thresholds,
    )
    right_identity_l2b_sections = _require_l2b_section(
        right_report,
        key="identity_discrimination_l2b",
        report_label="right",
        expected_thresholds=l2b_thresholds,
    )
    left_twin_l2b_sections = _require_l2b_section(
        left_report,
        key="twin_prediction_similarity_l2b",
        report_label="left",
        expected_thresholds=l2b_thresholds,
    )
    right_twin_l2b_sections = _require_l2b_section(
        right_report,
        key="twin_prediction_similarity_l2b",
        report_label="right",
        expected_thresholds=l2b_thresholds,
    )
    left_reality_l2b_sections = _require_l2b_section(
        left_report,
        key="reality_peer_similarity_l2b",
        report_label="left",
        expected_thresholds=l2b_thresholds,
    )
    right_reality_l2b_sections = _require_l2b_section(
        right_report,
        key="reality_peer_similarity_l2b",
        report_label="right",
        expected_thresholds=l2b_thresholds,
    )
    for key, entry in left_twin_index.items():
        _require_nonempty_view_field(entry, label=f"left twin_prediction_similarity {key!r}")
    for key, entry in right_twin_index.items():
        _require_nonempty_view_field(entry, label=f"right twin_prediction_similarity {key!r}")
    for key, entry in left_reality_index.items():
        _require_nonempty_view_field(entry, label=f"left reality_peer_similarity {key!r}")
    for key, entry in right_reality_index.items():
        _require_nonempty_view_field(entry, label=f"right reality_peer_similarity {key!r}")

    _require_requested_identity_views_present(
        left_identity_index,
        label="left identity_discrimination",
        identity_views=identity_views,
    )
    _require_requested_identity_views_present(
        right_identity_index,
        label="right identity_discrimination",
        identity_views=identity_views,
    )
    filtered_left_identity_index = _filter_identity_index_by_views(
        left_identity_index,
        identity_views,
    )
    filtered_right_identity_index = _filter_identity_index_by_views(
        right_identity_index,
        identity_views,
    )
    identity_keys = _require_same_keys(
        filtered_left_identity_index,
        filtered_right_identity_index,
        label="identity discrimination",
    )
    twin_keys = _require_same_keys(
        left_twin_index,
        right_twin_index,
        label="twin prediction similarity",
    )
    reality_keys = _require_same_keys(
        left_reality_index,
        right_reality_index,
        label="reality peer similarity",
    )
    if not twin_keys or not reality_keys:
        raise SystemExit("No peer-reality-anchored over-collapse metrics are available to compare")
    _require_expected_test2b_keys_present(
        left_twin_index,
        label="left twin_prediction_similarity",
    )
    _require_expected_test2b_keys_present(
        right_twin_index,
        label="right twin_prediction_similarity",
    )
    _require_expected_test2b_keys_present(
        left_reality_index,
        label="left reality_peer_similarity",
    )
    _require_expected_test2b_keys_present(
        right_reality_index,
        label="right reality_peer_similarity",
    )

    identity_results: list[dict[str, Any]] = []
    for key in identity_keys:
        try:
            contrast = cluster_bootstrap_identity_condition_contrast(
                _normalize_base_analysis_entry(filtered_left_identity_index[key]),
                _normalize_base_analysis_entry(filtered_right_identity_index[key]),
                left_label=left_condition,
                right_label=right_condition,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        except FullTraceSuiteV2Error as exc:
            raise SystemExit(f"Identity cluster bootstrap failed for key {key!r}: {exc}") from exc
        identity_results.append(contrast)
    if not identity_results:
        raise SystemExit("identity-views filtered out every shared identity discrimination entry")

    overcollapse_results: list[dict[str, Any]] = []
    for key in EXPECTED_TEST2B_KEYS:
        try:
            contrast = cluster_bootstrap_overcollapse_condition_contrast(
                _normalize_base_analysis_entry(left_twin_index[key]),
                _normalize_base_analysis_entry(right_twin_index[key]),
                _normalize_base_analysis_entry(left_reality_index[key]),
                _normalize_base_analysis_entry(right_reality_index[key]),
                left_label=left_condition,
                right_label=right_condition,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        except FullTraceSuiteV2Error as exc:
            raise SystemExit(
                f"Over-collapse cluster bootstrap failed for key {key!r}: {exc}"
            ) from exc
        overcollapse_results.append(contrast)
    if not overcollapse_results:
        raise SystemExit("No peer-reality-anchored over-collapse metrics are available to compare")

    identity_results_l2b: dict[str, list[dict[str, Any]]] = {}
    overcollapse_results_l2b: dict[str, list[dict[str, Any]]] = {}
    skipped_l2b: list[dict[str, Any]] = []
    for threshold in l2b_thresholds:
        threshold_key = _l2b_threshold_key(threshold)
        left_identity_l2b_index = _index_entries(
            left_identity_l2b_sections[threshold_key],
            key_fields=identity_key_fields,
            label=f"left identity_discrimination_l2b[{threshold_key}]",
        )
        right_identity_l2b_index = _index_entries(
            right_identity_l2b_sections[threshold_key],
            key_fields=identity_key_fields,
            label=f"right identity_discrimination_l2b[{threshold_key}]",
        )
        left_twin_l2b_index = _index_entries(
            left_twin_l2b_sections[threshold_key],
            key_fields=twin_key_fields,
            label=f"left twin_prediction_similarity_l2b[{threshold_key}]",
        )
        right_twin_l2b_index = _index_entries(
            right_twin_l2b_sections[threshold_key],
            key_fields=twin_key_fields,
            label=f"right twin_prediction_similarity_l2b[{threshold_key}]",
        )
        left_reality_l2b_index = _index_entries(
            left_reality_l2b_sections[threshold_key],
            key_fields=reality_key_fields,
            label=f"left reality_peer_similarity_l2b[{threshold_key}]",
        )
        right_reality_l2b_index = _index_entries(
            right_reality_l2b_sections[threshold_key],
            key_fields=reality_key_fields,
            label=f"right reality_peer_similarity_l2b[{threshold_key}]",
        )
        for key, entry in left_twin_l2b_index.items():
            _require_nonempty_view_field(
                entry,
                label=f"left twin_prediction_similarity_l2b[{threshold_key}] {key!r}",
            )
        for key, entry in right_twin_l2b_index.items():
            _require_nonempty_view_field(
                entry,
                label=f"right twin_prediction_similarity_l2b[{threshold_key}] {key!r}",
            )
        for key, entry in left_reality_l2b_index.items():
            _require_nonempty_view_field(
                entry,
                label=f"left reality_peer_similarity_l2b[{threshold_key}] {key!r}",
            )
        for key, entry in right_reality_l2b_index.items():
            _require_nonempty_view_field(
                entry,
                label=f"right reality_peer_similarity_l2b[{threshold_key}] {key!r}",
            )

        _require_requested_identity_views_present(
            left_identity_l2b_index,
            label=f"left identity_discrimination_l2b[{threshold_key}]",
            identity_views=identity_views,
        )
        _require_requested_identity_views_present(
            right_identity_l2b_index,
            label=f"right identity_discrimination_l2b[{threshold_key}]",
            identity_views=identity_views,
        )
        filtered_left_identity_l2b_index = _filter_identity_index_by_views(
            left_identity_l2b_index,
            identity_views,
        )
        filtered_right_identity_l2b_index = _filter_identity_index_by_views(
            right_identity_l2b_index,
            identity_views,
        )
        identity_l2b_keys = _require_same_keys(
            filtered_left_identity_l2b_index,
            filtered_right_identity_l2b_index,
            label=f"identity discrimination L2B[{threshold_key}]",
        )
        twin_l2b_keys = _require_same_keys(
            left_twin_l2b_index,
            right_twin_l2b_index,
            label=f"twin prediction similarity L2B[{threshold_key}]",
        )
        reality_l2b_keys = _require_same_keys(
            left_reality_l2b_index,
            right_reality_l2b_index,
            label=f"reality peer similarity L2B[{threshold_key}]",
        )
        if not twin_l2b_keys or not reality_l2b_keys:
            raise SystemExit(
                f"No peer-reality-anchored over-collapse L2B[{threshold_key}] metrics are available to compare"
            )
        _require_expected_test2b_keys_present(
            left_twin_l2b_index,
            label=f"left twin_prediction_similarity_l2b[{threshold_key}]",
        )
        _require_expected_test2b_keys_present(
            right_twin_l2b_index,
            label=f"right twin_prediction_similarity_l2b[{threshold_key}]",
        )
        _require_expected_test2b_keys_present(
            left_reality_l2b_index,
            label=f"left reality_peer_similarity_l2b[{threshold_key}]",
        )
        _require_expected_test2b_keys_present(
            right_reality_l2b_index,
            label=f"right reality_peer_similarity_l2b[{threshold_key}]",
        )

        threshold_identity_results: list[dict[str, Any]] = []
        for key in identity_l2b_keys:
            try:
                contrast = cluster_bootstrap_identity_condition_contrast(
                    _normalize_l2b_analysis_entry(
                        filtered_left_identity_l2b_index[key],
                        threshold=threshold,
                    ),
                    _normalize_l2b_analysis_entry(
                        filtered_right_identity_l2b_index[key],
                        threshold=threshold,
                    ),
                    left_label=left_condition,
                    right_label=right_condition,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed,
                )
            except FullTraceSuiteV2Error as exc:
                if _is_too_few_scope_clusters_error(exc):
                    skipped_l2b.append(
                        {
                            "threshold": threshold,
                            "threshold_key": threshold_key,
                            "analysis_family": "identity_discrimination",
                            "metric_key": list(key),
                            "reason": str(exc),
                        }
                    )
                    continue
                raise SystemExit(
                    f"Identity cluster bootstrap failed for L2B[{threshold_key}] key {key!r}: {exc}"
                ) from exc
            threshold_identity_results.append(contrast)
        if threshold_identity_results:
            identity_results_l2b[threshold_key] = threshold_identity_results

        threshold_overcollapse_results: list[dict[str, Any]] = []
        for key in EXPECTED_TEST2B_KEYS:
            try:
                contrast = cluster_bootstrap_overcollapse_condition_contrast(
                    _normalize_l2b_analysis_entry(left_twin_l2b_index[key], threshold=threshold),
                    _normalize_l2b_analysis_entry(right_twin_l2b_index[key], threshold=threshold),
                    _normalize_l2b_analysis_entry(left_reality_l2b_index[key], threshold=threshold),
                    _normalize_l2b_analysis_entry(
                        right_reality_l2b_index[key], threshold=threshold
                    ),
                    left_label=left_condition,
                    right_label=right_condition,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed,
                )
            except FullTraceSuiteV2Error as exc:
                if _is_too_few_scope_clusters_error(exc):
                    skipped_l2b.append(
                        {
                            "threshold": threshold,
                            "threshold_key": threshold_key,
                            "analysis_family": "peer_reality_anchored_overcollapse",
                            "metric_key": list(key),
                            "reason": str(exc),
                        }
                    )
                    continue
                raise SystemExit(
                    f"Over-collapse cluster bootstrap failed for L2B[{threshold_key}] key {key!r}: {exc}"
                ) from exc
            threshold_overcollapse_results.append(contrast)
        if threshold_overcollapse_results:
            overcollapse_results_l2b[threshold_key] = threshold_overcollapse_results

    report = {
        "schema_version": "v6_2_condition_cluster_bootstrap_report_v1",
        "cluster_unit": "scope",
        "left_report_path": str(args.left_report.resolve()),
        "right_report_path": str(args.right_report.resolve()),
        "left_condition": left_condition,
        "right_condition": right_condition,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "identity_views": list(identity_views),
        "l2b_thresholds": list(l2b_thresholds),
        "identity_discrimination_cluster_bootstrap": identity_results,
        "identity_discrimination_l2b_cluster_bootstrap": identity_results_l2b,
        "peer_reality_anchored_overcollapse_cluster_bootstrap": overcollapse_results,
        "peer_reality_anchored_overcollapse_l2b_cluster_bootstrap": overcollapse_results_l2b,
        "skipped_l2b": skipped_l2b,
    }
    _atomic_write_json(args.out, report)

    print()
    print("  ── Test 2A scope-cluster bootstrap ──")
    print("  metric                              view      delta      95% CI              p")
    print("  ──────────────────────────────────  ────────  ─────────  ───────────────────  ─────")
    for entry in identity_results:
        print(
            f"  {entry['metric']:<34} {entry['view']:<8} "
            f"{_fmt(entry['delta_of_deltas']):>9}  "
            f"[{_fmt(entry['ci95_lower'])}, {_fmt(entry['ci95_upper'])}]  "
            f"{entry['scope_flip_randomization_p_two_sided_against_zero']:.4f}"
        )
    print()
    print("  ── Test 2B peer-reality-anchored over-collapse bootstrap ──")
    print("  metric                              delta      95% CI              p")
    print("  ──────────────────────────────────  ─────────  ───────────────────  ─────")
    for entry in overcollapse_results:
        print(
            f"  {entry['metric']:<34} "
            f"{_fmt(entry['delta_of_overcollapse']):>9}  "
            f"[{_fmt(entry['ci95_lower'])}, {_fmt(entry['ci95_upper'])}]  "
            f"{entry['scope_flip_randomization_p_two_sided_against_zero']:.4f}"
        )
    for threshold in l2b_thresholds:
        threshold_key = _l2b_threshold_key(threshold)
        threshold_identity_entries = identity_results_l2b.get(threshold_key, [])
        threshold_overcollapse_entries = overcollapse_results_l2b.get(threshold_key, [])
        threshold_skips = [
            item for item in skipped_l2b if item.get("threshold_key") == threshold_key
        ]
        print()
        print(f"  ── Test 2A scope-cluster bootstrap (L2B {threshold_key}) ──")
        if threshold_identity_entries:
            print("  metric                              view      delta      95% CI              p")
            print(
                "  ──────────────────────────────────  ────────  ─────────  ───────────────────  ─────"
            )
            for entry in threshold_identity_entries:
                print(
                    f"  {entry['metric']:<34} {entry['view']:<8} "
                    f"{_fmt(entry['delta_of_deltas']):>9}  "
                    f"[{_fmt(entry['ci95_lower'])}, {_fmt(entry['ci95_upper'])}]  "
                    f"{entry['scope_flip_randomization_p_two_sided_against_zero']:.4f}"
                )
        else:
            print("  skipped: no identity metric at this threshold had >=2 scope clusters")
        print()
        print(
            f"  ── Test 2B peer-reality-anchored over-collapse bootstrap (L2B {threshold_key}) ──"
        )
        if threshold_overcollapse_entries:
            print("  metric                              delta      95% CI              p")
            print("  ──────────────────────────────────  ─────────  ───────────────────  ─────")
            for entry in threshold_overcollapse_entries:
                print(
                    f"  {entry['metric']:<34} "
                    f"{_fmt(entry['delta_of_overcollapse']):>9}  "
                    f"[{_fmt(entry['ci95_lower'])}, {_fmt(entry['ci95_upper'])}]  "
                    f"{entry['scope_flip_randomization_p_two_sided_against_zero']:.4f}"
                )
        else:
            print("  skipped: no over-collapse metric at this threshold had >=2 scope clusters")
        if threshold_skips:
            print("  skipped metric entries:")
            for item in threshold_skips:
                print(
                    f"    {item['analysis_family']} {tuple(item['metric_key'])!r}: {item['reason']}"
                )
    print()
    print(f"  wrote report: {args.out.resolve()}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
