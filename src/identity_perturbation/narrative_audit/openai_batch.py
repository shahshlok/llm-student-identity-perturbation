from __future__ import annotations

import argparse
import json
from collections import Counter
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING, Any

from identity_perturbation.codebench_support.openai_batch import (
    _client,
    _display_path,
    _extract_response_text,
    _load_json,
    _save_json,
    _strip_json_fence,
    _structured_text_format,
    _update_batch_snapshot,
    _utc_now,
)

from .labels import EVALUATED_HEAD_SPECS
from .prediction_schema import V6BatchResponse
from .prompting import build_system_prompt
from .runner import ROOT

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_BATCH_ROOT = ROOT / "data" / "v6_batch_runs"


class V6BatchError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v6 transition prompting through OpenAI Batch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Build a v6 Batch JSONL request file from one or more existing transition bundles",
    )
    prepare.add_argument(
        "--bundle-dir",
        action="append",
        default=[],
        type=Path,
        help="Path to a v6 transition bundle directory; may be repeated",
    )
    prepare.add_argument(
        "--bundle-manifest",
        action="append",
        default=[],
        type=Path,
        help="Path to a v6 transition bundle manifest JSON; may be repeated",
    )
    prepare.add_argument("--run-name", default=None)
    prepare.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)

    submit = subparsers.add_parser("submit", help="Upload the request JSONL and create a batch")
    submit.add_argument("--manifest", required=True, type=Path)

    status = subparsers.add_parser("status", help="Check batch status and update the manifest")
    status.add_argument("--manifest", required=True, type=Path)

    download = subparsers.add_parser("download", help="Download output and error files for a batch")
    download.add_argument("--manifest", required=True, type=Path)

    hydrate = subparsers.add_parser(
        "hydrate",
        help="Parse downloaded batch output into strict typed v6 prediction artifacts",
    )
    hydrate.add_argument("--manifest", required=True, type=Path)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Score hydrated v6 predictions against held-out observed labels",
    )
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Rank cutoff for shortlist hit-rate metrics",
    )

    return parser.parse_args()


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def _default_run_name(model: str) -> str:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v6_{_slug(model)}_{stamp}"


def _run_dir(batch_root: Path, run_name: str) -> Path:
    return batch_root / run_name


def _request_custom_id(bundle_manifest: dict[str, Any]) -> str:
    return (
        f"{bundle_manifest['class_id']}:{bundle_manifest['assessment_id']}:"
        f"{bundle_manifest['exercise_id']}:{bundle_manifest['student_id']}:"
        f"{bundle_manifest['transition_index_0idx']}"
    )


def _build_request_body(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": _structured_text_format(V6BatchResponse),
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _save_json(path, manifest)


def _parse_response_payload(payload_text: str) -> dict[str, Any]:
    payload = json.loads(_strip_json_fence(payload_text))
    return V6BatchResponse.model_validate(payload).model_dump()


def _load_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    user_prompt_path = bundle_dir / "user_prompt.txt"
    payload_path = bundle_dir / "payload.json"
    observed_labels_path = bundle_dir / "observed_labels.json"
    if not manifest_path.exists():
        raise V6BatchError(f"Bundle manifest missing: {manifest_path}")
    if not user_prompt_path.exists():
        raise V6BatchError(f"Bundle user prompt missing: {user_prompt_path}")
    if not payload_path.exists():
        raise V6BatchError(f"Bundle payload missing: {payload_path}")
    if not observed_labels_path.exists():
        raise V6BatchError(f"Bundle observed labels missing: {observed_labels_path}")

    manifest = _load_json(manifest_path)
    required_manifest_keys = {
        "class_id",
        "assessment_id",
        "exercise_id",
        "student_id",
        "transition_index_0idx",
    }
    missing = required_manifest_keys - set(manifest)
    if missing:
        raise V6BatchError(
            f"Bundle manifest missing required keys {sorted(missing)}: {manifest_path}"
        )

    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "manifest": manifest,
        "user_prompt_path": str(user_prompt_path.resolve()),
        "payload_path": str(payload_path.resolve()),
        "observed_labels_path": str(observed_labels_path.resolve()),
    }


def _bundle_dirs_from_manifest(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        raise V6BatchError(f"Bundle manifest not found: {path}")
    payload = _load_json(path)
    if payload.get("schema_version") != "v6_transition_bundle_manifest_v1":
        raise V6BatchError(
            f"Unexpected bundle manifest schema in {path}: {payload.get('schema_version')}"
        )
    bundles = payload.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise V6BatchError(f"Bundle manifest must contain a non-empty bundles list: {path}")
    resolved_dirs: list[Path] = []
    for row in bundles:
        if "bundle_dir" not in row:
            raise V6BatchError(f"Bundle manifest row missing bundle_dir: {row}")
        resolved_dirs.append(Path(str(row["bundle_dir"])))
    return tuple(resolved_dirs)


def _resolve_prepare_bundle_dirs(args: argparse.Namespace) -> tuple[Path, ...]:
    explicit_dirs = [Path(path) for path in args.bundle_dir]
    manifest_dirs: list[Path] = []
    for manifest_path in args.bundle_manifest:
        manifest_dirs.extend(_bundle_dirs_from_manifest(manifest_path))
    bundle_dirs = [*explicit_dirs, *manifest_dirs]
    if not bundle_dirs:
        raise V6BatchError("prepare requires at least one --bundle-dir or --bundle-manifest")
    return tuple(bundle_dirs)


def _prepare(args: argparse.Namespace) -> int:
    run_name = args.run_name or _default_run_name(args.model)
    run_dir = _run_dir(args.batch_root, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    requests_path = run_dir / "requests.jsonl"
    output_path = run_dir / "output.jsonl"
    error_path = run_dir / "errors.jsonl"
    manifest_path = run_dir / "manifest.json"
    system_prompt_path = run_dir / "system_prompt.txt"
    response_schema_path = run_dir / "response_schema.json"
    hydrated_path = run_dir / "hydrated_predictions.json"
    hydration_failures_path = run_dir / "hydration_failures.json"
    evaluation_path = run_dir / "evaluation.json"
    evaluation_md_path = run_dir / "evaluation.md"

    system_prompt = build_system_prompt()
    system_prompt_path.write_text(system_prompt, encoding="utf-8")
    _save_json(response_schema_path, V6BatchResponse.model_json_schema())

    ordered_custom_ids: list[str] = []
    bundle_map: dict[str, dict[str, Any]] = {}
    seen_custom_ids: set[str] = set()
    bundle_dirs = _resolve_prepare_bundle_dirs(args)

    with requests_path.open("w", encoding="utf-8") as handle:
        for bundle_dir in bundle_dirs:
            loaded = _load_bundle(bundle_dir)
            bundle_manifest = loaded["manifest"]
            custom_id = _request_custom_id(bundle_manifest)
            if custom_id in seen_custom_ids:
                raise V6BatchError(
                    f"Duplicate custom_id detected while preparing batch: {custom_id}"
                )
            seen_custom_ids.add(custom_id)

            user_prompt = Path(loaded["user_prompt_path"]).read_text(encoding="utf-8")
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": _build_request_body(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                ),
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            ordered_custom_ids.append(custom_id)
            bundle_map[custom_id] = {
                "bundle_dir": loaded["bundle_dir"],
                "manifest_path": str((Path(bundle_dir) / "manifest.json").resolve()),
                "payload_path": loaded["payload_path"],
                "observed_labels_path": loaded["observed_labels_path"],
                "user_prompt_path": loaded["user_prompt_path"],
                "condition": bundle_manifest.get("condition", "full_v6"),
                "class_id": bundle_manifest["class_id"],
                "assessment_id": bundle_manifest["assessment_id"],
                "exercise_id": bundle_manifest["exercise_id"],
                "student_id": bundle_manifest["student_id"],
                "transition_index_0idx": bundle_manifest["transition_index_0idx"],
            }

    manifest = {
        "created_at": _utc_now(),
        "run_name": run_name,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "requests_jsonl": str(requests_path.resolve()),
            "output_jsonl": str(output_path.resolve()),
            "error_jsonl": str(error_path.resolve()),
            "system_prompt_txt": str(system_prompt_path.resolve()),
            "response_schema_json": str(response_schema_path.resolve()),
            "hydrated_predictions_json": str(hydrated_path.resolve()),
            "hydration_failures_json": str(hydration_failures_path.resolve()),
            "evaluation_json": str(evaluation_path.resolve()),
            "evaluation_md": str(evaluation_md_path.resolve()),
        },
        "counts": {
            "bundles": len(ordered_custom_ids),
            "requests_created": len(ordered_custom_ids),
        },
        "conditions": sorted({meta["condition"] for meta in bundle_map.values()}),
        "batch": {
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "input_file_id": None,
            "batch_id": None,
            "status": "prepared",
            "output_file_id": None,
            "error_file_id": None,
        },
        "ordered_custom_ids": ordered_custom_ids,
        "bundle_map": bundle_map,
    }
    _save_json(manifest_path, manifest)

    print(f"Prepared run: {run_name}")
    print(f"Requests: {manifest['counts']['requests_created']}")
    print(f"Manifest: {manifest_path}")
    print(f"Requests JSONL: {requests_path}")
    return 0


def _submit(args: argparse.Namespace) -> int:
    manifest_path = args.manifest
    manifest = _load_json(manifest_path)
    requests_path = Path(manifest["paths"]["requests_jsonl"])

    client = _client()
    with requests_path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "run_name": manifest["run_name"],
            "model": manifest["model"],
        },
    )
    manifest["batch"]["input_file_id"] = file_obj.id
    _update_batch_snapshot(manifest, batch)
    _save_manifest(manifest_path, manifest)

    print(f"Uploaded input file: {file_obj.id}")
    print(f"Created batch: {batch.id}")
    print(f"Status: {batch.status}")
    return 0


def _status(args: argparse.Namespace) -> int:
    manifest_path = args.manifest
    manifest = _load_json(manifest_path)
    batch_id = manifest["batch"].get("batch_id")
    if not batch_id:
        raise SystemExit("Manifest does not contain a batch_id. Run submit first.")

    client = _client()
    batch = client.batches.retrieve(batch_id)
    _update_batch_snapshot(manifest, batch)
    _save_manifest(manifest_path, manifest)

    print(json.dumps(manifest["batch"], indent=2))
    return 0


def _download_file(client: OpenAI, file_id: str, path: Path) -> None:
    content = client.files.content(file_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.read())


def _download(args: argparse.Namespace) -> int:
    manifest_path = args.manifest
    manifest = _load_json(manifest_path)
    batch = manifest["batch"]

    client = _client()
    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")
    output_path = Path(manifest["paths"]["output_jsonl"])
    error_path = Path(manifest["paths"]["error_jsonl"])

    if output_file_id:
        _download_file(client, output_file_id, output_path)
        print(f"Downloaded output file to: {output_path}")
    else:
        print("No output_file_id present on the batch yet.")

    if error_file_id:
        _download_file(client, error_file_id, error_path)
        print(f"Downloaded error file to: {error_path}")
    else:
        print("No error_file_id present on the batch.")
    return 0


def _hydrate(args: argparse.Namespace) -> int:
    manifest_path = args.manifest
    manifest = _load_json(manifest_path)
    output_path = Path(manifest["paths"]["output_jsonl"])
    error_path = Path(manifest["paths"]["error_jsonl"])
    hydrated_path = Path(manifest["paths"]["hydrated_predictions_json"])
    hydration_failures_path = Path(manifest["paths"]["hydration_failures_json"])

    if not output_path.exists():
        raise SystemExit(f"Output JSONL not found: {output_path}")

    bundle_map = manifest["bundle_map"]
    ordered_custom_ids = manifest["ordered_custom_ids"]
    responded_custom_ids: set[str] = set()
    predictions_by_custom_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    error_file_entries = 0

    with output_path.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id")
            if not custom_id:
                failures.append(
                    {
                        "line_num": line_num,
                        "stage": "batch_output",
                        "error": "missing custom_id in batch output row",
                    }
                )
                continue
            if custom_id not in bundle_map:
                failures.append(
                    {
                        "custom_id": custom_id,
                        "line_num": line_num,
                        "stage": "unexpected_custom_id",
                        "error": "custom_id not present in prepared bundle_map",
                    }
                )
                continue
            if custom_id in responded_custom_ids:
                failures.append(
                    {
                        "custom_id": custom_id,
                        "line_num": line_num,
                        "stage": "duplicate_output",
                        "error": "duplicate custom_id encountered in batch output",
                    }
                )
                continue

            responded_custom_ids.add(custom_id)
            meta = bundle_map[custom_id]
            batch_error = item.get("error")
            if batch_error:
                failures.append(
                    {
                        "custom_id": custom_id,
                        "line_num": line_num,
                        "stage": "batch_output",
                        "error": batch_error,
                    }
                )
                continue

            response = item.get("response") or {}
            status_code = response.get("status_code")
            body = response.get("body") or {}
            if status_code != 200:
                failures.append(
                    {
                        "custom_id": custom_id,
                        "line_num": line_num,
                        "stage": "http",
                        "status_code": status_code,
                    }
                )
                continue

            try:
                output_text = _extract_response_text(body)
                parsed = _parse_response_payload(output_text)
            except Exception as exc:
                failures.append(
                    {
                        "custom_id": custom_id,
                        "line_num": line_num,
                        "stage": "parse",
                        "error": str(exc),
                    }
                )
                continue

            predictions_by_custom_id[custom_id] = {
                "custom_id": custom_id,
                "class_id": str(meta["class_id"]),
                "assessment_id": str(meta["assessment_id"]),
                "exercise_id": str(meta["exercise_id"]),
                "student_id": str(meta["student_id"]),
                "transition_index_0idx": int(meta["transition_index_0idx"]),
                "bundle_dir": str(meta["bundle_dir"]),
                "bundle_manifest_path": str(meta["manifest_path"]),
                "payload_path": str(meta["payload_path"]),
                "observed_labels_path": str(meta["observed_labels_path"]),
                "user_prompt_path": str(meta["user_prompt_path"]),
                "condition": str(meta["condition"]),
                "response": parsed,
                "raw_response_id": body.get("id"),
            }

    if error_path.exists():
        with error_path.open("r", encoding="utf-8") as handle:
            for line_num, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                item = json.loads(line)
                custom_id = item.get("custom_id")
                if not custom_id:
                    failures.append(
                        {
                            "line_num": line_num,
                            "stage": "batch_error_file",
                            "error": "missing custom_id in batch error row",
                        }
                    )
                    error_file_entries += 1
                    continue
                if custom_id not in bundle_map:
                    failures.append(
                        {
                            "custom_id": custom_id,
                            "line_num": line_num,
                            "stage": "unexpected_error_custom_id",
                            "error": "custom_id from batch error file not present in prepared bundle_map",
                        }
                    )
                    error_file_entries += 1
                    continue
                failures.append(
                    {
                        "custom_id": custom_id,
                        "line_num": line_num,
                        "stage": "batch_error_file",
                        "error": item.get("error", "batch error file entry"),
                    }
                )
                error_file_entries += 1

    missing_outputs = 0
    for custom_id in ordered_custom_ids:
        if custom_id in responded_custom_ids:
            continue
        failures.append(
            {
                "custom_id": custom_id,
                "stage": "missing_output",
                "error": "missing from batch output",
            }
        )
        missing_outputs += 1

    ordered_predictions = [
        predictions_by_custom_id[custom_id]
        for custom_id in ordered_custom_ids
        if custom_id in predictions_by_custom_id
    ]
    hydrated_payload = {
        "schema_version": "v6_batch_hydrated_predictions_v1",
        "run_name": manifest["run_name"],
        "model": manifest["model"],
        "reasoning_effort": manifest["reasoning_effort"],
        "summary": {
            "requested_predictions": len(ordered_custom_ids),
            "parsed_predictions": len(ordered_predictions),
            "missing_outputs": missing_outputs,
            "failure_count": len(failures),
            "batch_error_file_entries": error_file_entries,
        },
        "predictions": ordered_predictions,
    }
    _save_json(hydrated_path, hydrated_payload)
    _save_json(hydration_failures_path, {"failures": failures})

    manifest["hydration"] = {
        "hydrated_at": _utc_now(),
        "requested_predictions": len(ordered_custom_ids),
        "parsed_predictions": len(ordered_predictions),
        "missing_outputs": missing_outputs,
        "failure_count": len(failures),
    }
    _save_manifest(manifest_path, manifest)

    print(f"Hydrated predictions: {_display_path(hydrated_path)}")
    print(f"Hydration failures: {_display_path(hydration_failures_path)}")
    print(f"Parsed predictions: {len(ordered_predictions)}")
    print(f"Missing outputs: {missing_outputs}")

    if failures:
        raise SystemExit(
            f"Hydration encountered {len(failures)} failure(s); see {_display_path(hydration_failures_path)}"
        )
    return 0


def _ranked_items(distribution: dict[Any, float]) -> list[tuple[Any, float]]:
    return sorted(distribution.items(), key=lambda item: (-item[1], str(item[0])))


def _marginal_distributions(response: dict[str, Any]) -> dict[str, dict[str, float]]:
    distributions: dict[str, dict[str, float]] = {}
    for head_key, head_spec in EVALUATED_HEAD_SPECS.items():
        label_space = tuple(str(label) for label in head_spec["label_space"])
        distributions[head_key] = dict.fromkeys(label_space, 0.0)

    for hypothesis in response["next_move_hypotheses"]:
        probability = float(hypothesis["estimated_probability"])
        for head_key in EVALUATED_HEAD_SPECS:
            label = str(hypothesis[head_key])
            distributions[head_key][label] += probability

    for head_key, distribution in distributions.items():
        total = sum(distribution.values())
        if not isclose(total, 1.0, abs_tol=1e-6):
            raise V6BatchError(
                f"Predicted marginal distribution for {head_key} must sum to 1.0 exactly; got {total}"
            )
    return distributions


def _joint_distribution(response: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    head_order = tuple(EVALUATED_HEAD_SPECS.keys())
    distribution: dict[tuple[str, str, str], float] = {}
    for hypothesis in response["next_move_hypotheses"]:
        key = tuple(str(hypothesis[head_key]) for head_key in head_order)
        probability = float(hypothesis["estimated_probability"])
        distribution[key] = distribution.get(key, 0.0) + probability
    total = sum(distribution.values())
    if not isclose(total, 1.0, abs_tol=1e-6):
        raise V6BatchError(f"Predicted joint distribution must sum to 1.0 exactly; got {total}")
    return distribution


def _brier_score(distribution: dict[str, float], truth_label: str) -> float:
    return sum(
        (probability - (1.0 if label == truth_label else 0.0)) ** 2
        for label, probability in distribution.items()
    )


def _load_observed_labels(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != "v6_observed_labels_v1":
        raise V6BatchError(
            f"Unexpected observed label schema in {path}: {payload.get('schema_version')}"
        )
    if "observed_heads" not in payload:
        raise V6BatchError(f"Observed labels payload missing observed_heads: {path}")
    observed_heads = payload["observed_heads"]
    if not isinstance(observed_heads, dict):
        raise V6BatchError(f"Observed labels payload has non-object observed_heads: {path}")
    expected_heads = set(EVALUATED_HEAD_SPECS)
    actual_heads = set(observed_heads)
    if actual_heads != expected_heads:
        raise V6BatchError(
            f"Observed labels payload has wrong head keys in {path}: expected {sorted(expected_heads)}, got {sorted(actual_heads)}"
        )
    for head_key, head_spec in EVALUATED_HEAD_SPECS.items():
        value = observed_heads[head_key]
        label_space = {str(label) for label in head_spec["label_space"]}
        if str(value) not in label_space:
            raise V6BatchError(
                f"Observed label {value} is outside the label space for {head_key} in {path}"
            )
    return payload


def _score_rows(
    *,
    rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    if top_k < 1:
        raise V6BatchError(f"--top-k must be positive; got {top_k}")
    if not rows:
        raise V6BatchError("Cannot score an empty v6 evaluation set")

    by_head_accumulators = {
        head_key: {
            "truth_mass_sum": 0.0,
            "support_count": 0,
            "top1_count": 0,
            "topk_count": 0,
            "brier_sum": 0.0,
        }
        for head_key in EVALUATED_HEAD_SPECS
    }
    joint_accumulator = {
        "truth_mass_sum": 0.0,
        "support_count": 0,
        "top1_count": 0,
        "topk_count": 0,
    }
    row_details: list[dict[str, Any]] = []

    for row in rows:
        response = row["response"]
        observed_heads = row["observed_heads"]
        marginal_distributions = _marginal_distributions(response)
        joint_distribution = _joint_distribution(response)
        head_metrics: dict[str, Any] = {}

        for head_key, head_spec in EVALUATED_HEAD_SPECS.items():
            truth_label = str(observed_heads[head_key])
            label_space = tuple(str(label) for label in head_spec["label_space"])
            if truth_label not in label_space:
                raise V6BatchError(
                    f"Observed label {truth_label} not in label space for {head_key}"
                )
            distribution = marginal_distributions[head_key]
            ranked_labels = _ranked_items(distribution)
            truth_mass = distribution[truth_label]
            topk_labels = [str(label) for label, _ in ranked_labels[:top_k]]
            top1_label = str(ranked_labels[0][0])
            support = truth_mass > 0.0
            topk_hit = truth_label in topk_labels
            brier = _brier_score(distribution, truth_label)

            by_head_accumulators[head_key]["truth_mass_sum"] += truth_mass
            by_head_accumulators[head_key]["support_count"] += int(support)
            by_head_accumulators[head_key]["top1_count"] += int(top1_label == truth_label)
            by_head_accumulators[head_key]["topk_count"] += int(topk_hit)
            by_head_accumulators[head_key]["brier_sum"] += brier

            head_metrics[head_key] = {
                "observed_label": truth_label,
                "truth_probability_mass": truth_mass,
                "support": support,
                "top1_prediction": top1_label,
                "top1_match": top1_label == truth_label,
                "topk_labels": topk_labels,
                "topk_hit": topk_hit,
                "brier_score": brier,
                "distribution": distribution,
            }

        truth_joint = tuple(str(observed_heads[head_key]) for head_key in EVALUATED_HEAD_SPECS)
        ranked_joint = _ranked_items(joint_distribution)
        top_joint = tuple(str(value) for value in ranked_joint[0][0])
        topk_joint = [tuple(str(value) for value in labels) for labels, _ in ranked_joint[:top_k]]
        truth_joint_mass = joint_distribution.get(truth_joint, 0.0)
        joint_support = truth_joint_mass > 0.0
        joint_topk_hit = truth_joint in topk_joint

        joint_accumulator["truth_mass_sum"] += truth_joint_mass
        joint_accumulator["support_count"] += int(joint_support)
        joint_accumulator["top1_count"] += int(top_joint == truth_joint)
        joint_accumulator["topk_count"] += int(joint_topk_hit)

        row_details.append(
            {
                "custom_id": row["custom_id"],
                "student_id": row["student_id"],
                "transition_index_0idx": row["transition_index_0idx"],
                "observed_heads": observed_heads,
                "by_head": head_metrics,
                "joint": {
                    "observed_tuple": list(truth_joint),
                    "truth_probability_mass": truth_joint_mass,
                    "support": joint_support,
                    "top1_prediction": list(top_joint),
                    "top1_match": top_joint == truth_joint,
                    "topk_predictions": [list(labels) for labels in topk_joint],
                    "topk_hit": joint_topk_hit,
                },
            }
        )

    n_rows = len(rows)
    by_head_summary = {
        head_key: {
            "mean_truth_probability_mass": accumulator["truth_mass_sum"] / n_rows,
            "support_rate": accumulator["support_count"] / n_rows,
            "top1_accuracy": accumulator["top1_count"] / n_rows,
            f"top{top_k}_hit_rate": accumulator["topk_count"] / n_rows,
            "mean_brier_score": accumulator["brier_sum"] / n_rows,
        }
        for head_key, accumulator in by_head_accumulators.items()
    }
    joint_summary = {
        "mean_truth_probability_mass": joint_accumulator["truth_mass_sum"] / n_rows,
        "support_rate": joint_accumulator["support_count"] / n_rows,
        "top1_accuracy": joint_accumulator["top1_count"] / n_rows,
        f"top{top_k}_hit_rate": joint_accumulator["topk_count"] / n_rows,
    }
    return {
        "n_predictions": n_rows,
        "top_k": top_k,
        "joint": joint_summary,
        "by_head": by_head_summary,
        "rows": row_details,
    }


def _evaluate(args: argparse.Namespace) -> int:
    manifest_path = args.manifest
    manifest = _load_json(manifest_path)
    hydrated_path = Path(manifest["paths"]["hydrated_predictions_json"])
    evaluation_path = Path(manifest["paths"]["evaluation_json"])
    evaluation_md_path = Path(manifest["paths"]["evaluation_md"])
    if not hydrated_path.exists():
        raise SystemExit(f"Hydrated predictions not found: {hydrated_path}")

    hydrated = _load_json(hydrated_path)
    predictions = hydrated["predictions"]
    if not predictions:
        raise V6BatchError("No hydrated predictions available for v6 evaluation")

    rows: list[dict[str, Any]] = []
    observed_outcome_counts: dict[str, Counter[str]] = {
        head_key: Counter() for head_key in EVALUATED_HEAD_SPECS
    }
    for prediction in predictions:
        observed_path = Path(prediction["observed_labels_path"])
        observed_payload = _load_observed_labels(observed_path)
        observed_heads = observed_payload["observed_heads"]
        for head_key in EVALUATED_HEAD_SPECS:
            observed_outcome_counts[head_key][str(observed_heads[head_key])] += 1
        rows.append(
            {
                "custom_id": prediction["custom_id"],
                "student_id": prediction["student_id"],
                "transition_index_0idx": prediction["transition_index_0idx"],
                "response": prediction["response"],
                "observed_heads": observed_heads,
            }
        )

    scores = _score_rows(rows=rows, top_k=args.top_k)
    evaluation = {
        "schema_version": "v6_batch_evaluation_v1",
        "evaluated_at": _utc_now(),
        "run_name": manifest["run_name"],
        "model": manifest["model"],
        "reasoning_effort": manifest["reasoning_effort"],
        "summary": {
            "n_predictions": scores["n_predictions"],
            "top_k": scores["top_k"],
            "evaluated_heads": list(EVALUATED_HEAD_SPECS.keys()),
            "excluded_heads": ["likely_edit_strategy"],
            "observed_label_counts": {
                head_key: dict(sorted(counter.items()))
                for head_key, counter in observed_outcome_counts.items()
            },
        },
        "scores": {
            "joint": scores["joint"],
            "by_head": scores["by_head"],
        },
        "rows": scores["rows"],
    }
    _save_json(evaluation_path, evaluation)

    topk_metric_key = f"top{scores['top_k']}_hit_rate"
    lines = [
        "# V6 Promise-Screen Evaluation",
        "",
        f"- Run name: `{manifest['run_name']}`",
        f"- Model: `{manifest['model']}`",
        f"- Reasoning effort: `{manifest['reasoning_effort']}`",
        f"- Evaluated predictions: `{scores['n_predictions']}`",
        f"- Top-k cutoff: `{scores['top_k']}`",
        "- Evaluated heads: `likely_first_repair_region`, `likely_edit_scope`, `likely_next_test_outcome`",
        "- Excluded automatic head: `likely_edit_strategy`",
        "",
        "## Joint 3-Head Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Mean truth probability mass | {scores['joint']['mean_truth_probability_mass']:.3f} |",
        f"| Support rate | {scores['joint']['support_rate']:.3f} |",
        f"| Top-1 accuracy | {scores['joint']['top1_accuracy']:.3f} |",
        f"| Top-{scores['top_k']} hit rate | {scores['joint'][topk_metric_key]:.3f} |",
        "",
        "## Per-Head Metrics",
        "",
        "| Head | Mean truth mass | Support rate | Top-1 | Top-"
        + str(scores["top_k"])
        + " | Mean Brier |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for head_key, head_scores in scores["by_head"].items():
        lines.append(
            f"| {head_key} | "
            f"{head_scores['mean_truth_probability_mass']:.3f} | "
            f"{head_scores['support_rate']:.3f} | "
            f"{head_scores['top1_accuracy']:.3f} | "
            f"{head_scores[topk_metric_key]:.3f} | "
            f"{head_scores['mean_brier_score']:.3f} |"
        )
    evaluation_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest["evaluation"] = {
        "evaluated_at": evaluation["evaluated_at"],
        "n_predictions": scores["n_predictions"],
        "top_k": scores["top_k"],
        "paths": {
            "evaluation_json": str(evaluation_path.resolve()),
            "evaluation_md": str(evaluation_md_path.resolve()),
        },
    }
    _save_manifest(manifest_path, manifest)

    print(f"Evaluation JSON: {_display_path(evaluation_path)}")
    print(f"Evaluation Markdown: {_display_path(evaluation_md_path)}")
    print(f"Joint top-1 accuracy: {scores['joint']['top1_accuracy']:.3f}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "submit":
        return _submit(args)
    if args.command == "status":
        return _status(args)
    if args.command == "download":
        return _download(args)
    if args.command == "hydrate":
        return _hydrate(args)
    if args.command == "evaluate":
        return _evaluate(args)
    raise SystemExit(f"Unknown command: {args.command}")
