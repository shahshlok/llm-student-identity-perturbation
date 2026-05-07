from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.openai_batch import _extract_response_text, _strip_json_fence
from identity_perturbation.prediction_audit.full_trace_scorer import score_full_trace_prediction
from identity_perturbation.prediction_audit.full_trace_target_schema import FullTracePredictionResponse


class FullTraceBundleScoringError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one identity-perturbation prompt bundle against one model response."
    )
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prediction-json",
        type=Path,
        help="Path to an already-extracted JSON payload matching FullTracePredictionResponse.",
    )
    source.add_argument(
        "--batch-output-item-json",
        type=Path,
        help="Path to one raw OpenAI Batch output JSON object containing custom_id and response.",
    )
    source.add_argument(
        "--batch-output-jsonl",
        type=Path,
        help="Path to a raw OpenAI Batch output JSONL file; the bundle custom_id is used to select one line.",
    )
    return parser.parse_args()


def _load_json_strict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FullTraceBundleScoringError(f"JSON file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FullTraceBundleScoringError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FullTraceBundleScoringError(f"Top-level JSON value must be an object in {path}")
    return payload


def _load_bundle_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = _load_json_strict(manifest_path)
    required_keys = {
        "schema_version",
        "custom_id",
        "observed_next_repair_target_path",
        "observed_next_coarse_path_path",
    }
    missing = required_keys - set(manifest)
    if missing:
        raise FullTraceBundleScoringError(
            f"Bundle manifest missing required keys {sorted(missing)}: {manifest_path}"
        )
    schema_version = manifest["schema_version"]
    if schema_version not in {
        "v6_2_full_trace_prototype_bundle_v4",
        "v6_2_full_trace_prototype_bundle_v5",
        "v6_2_full_trace_prototype_bundle_v6",
    }:
        raise FullTraceBundleScoringError(
            f"Unexpected bundle manifest schema_version in {manifest_path}: {schema_version}"
        )
    if schema_version == "v6_2_full_trace_prototype_bundle_v6":
        condition = manifest.get("condition")
        if not isinstance(condition, str) or not condition:
            raise FullTraceBundleScoringError(
                f"Bundle manifest v6 requires a non-empty condition field: {manifest_path}"
            )
    return manifest


def _resolve_bundle_artifact_path(
    *,
    bundle_dir: Path,
    manifest_value: Any,
    expected_filename: str,
) -> Path:
    if not isinstance(manifest_value, str) or not manifest_value:
        raise FullTraceBundleScoringError(
            f"Manifest artifact path for {expected_filename} must be a non-empty string"
        )
    manifest_path = Path(manifest_value)
    if manifest_path.is_absolute():
        if manifest_path.name != expected_filename:
            raise FullTraceBundleScoringError(
                f"Absolute manifest path for {expected_filename} points to unexpected filename: {manifest_path}"
            )
        resolved = bundle_dir / expected_filename
    else:
        resolved = bundle_dir / manifest_path
    if resolved.name != expected_filename:
        raise FullTraceBundleScoringError(
            f"Resolved bundle artifact path for {expected_filename} has unexpected filename: {resolved}"
        )
    if not resolved.exists():
        raise FullTraceBundleScoringError(f"Bundle-local artifact not found: {resolved}")
    return resolved


def _validate_prediction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    validated = FullTracePredictionResponse.model_validate(payload)
    return validated.model_dump(mode="json")


def _parse_prediction_text(payload_text: str) -> dict[str, Any]:
    stripped = _strip_json_fence(payload_text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise FullTraceBundleScoringError(f"Model output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FullTraceBundleScoringError("Model output JSON must be an object")
    return _validate_prediction_payload(payload)


def _prediction_from_prediction_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json_strict(path)
    validated = _validate_prediction_payload(payload)
    return validated, {
        "source_kind": "prediction_json",
        "source_path": str(path.resolve()),
    }


def _prediction_from_responses_body(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    body_status = body.get("status")
    if body_status is not None and body_status != "completed":
        raise FullTraceBundleScoringError(
            f"Responses body status must be 'completed', got {body_status!r}"
        )
    incomplete_details = body.get("incomplete_details")
    if incomplete_details not in (None, {}):
        raise FullTraceBundleScoringError(
            f"Responses body is incomplete and cannot be scored: {incomplete_details!r}"
        )
    output_text = _extract_response_text(body)
    validated = _parse_prediction_text(output_text)
    source_meta: dict[str, Any] = {
        "source_kind": "responses_body",
        "raw_response_id": body.get("id"),
        "raw_response_status": body_status,
        "raw_output_text_length": len(output_text),
    }
    return validated, source_meta


def _prediction_from_batch_output_item(
    *,
    item: dict[str, Any],
    expected_custom_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    custom_id = item.get("custom_id")
    if custom_id != expected_custom_id:
        raise FullTraceBundleScoringError(
            f"Batch output custom_id mismatch: expected {expected_custom_id!r}, got {custom_id!r}"
        )
    batch_error = item.get("error")
    if batch_error:
        raise FullTraceBundleScoringError(
            f"Batch output item contains error for {expected_custom_id}: {batch_error}"
        )
    response = item.get("response")
    if not isinstance(response, dict):
        raise FullTraceBundleScoringError("Batch output item missing response object")
    status_code = response.get("status_code")
    if status_code != 200:
        raise FullTraceBundleScoringError(
            f"Batch output response status_code must be 200, got {status_code!r}"
        )
    body = response.get("body")
    if not isinstance(body, dict):
        raise FullTraceBundleScoringError("Batch output response missing body object")
    validated, source_meta = _prediction_from_responses_body(body)
    source_meta["source_kind"] = "batch_output_item"
    return validated, source_meta


def _prediction_from_batch_output_item_json(
    *,
    path: Path,
    expected_custom_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = _load_json_strict(path)
    validated, source_meta = _prediction_from_batch_output_item(
        item=item,
        expected_custom_id=expected_custom_id,
    )
    source_meta["source_path"] = str(path.resolve())
    return validated, source_meta


def _prediction_from_batch_output_jsonl(
    *,
    path: Path,
    expected_custom_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise FullTraceBundleScoringError(f"Batch output JSONL not found: {path}")
    matches: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FullTraceBundleScoringError(
                    f"Invalid JSON on line {line_num} in {path}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise FullTraceBundleScoringError(
                    f"Batch output line {line_num} is not a JSON object in {path}"
                )
            if item.get("custom_id") == expected_custom_id:
                matches.append((line_num, item))
    if not matches:
        raise FullTraceBundleScoringError(
            f"No batch output line found for custom_id {expected_custom_id!r} in {path}"
        )
    if len(matches) != 1:
        line_numbers = [line_num for line_num, _ in matches]
        raise FullTraceBundleScoringError(
            f"Expected exactly one batch output line for custom_id {expected_custom_id!r} in {path}, "
            f"found {len(matches)} at lines {line_numbers}"
        )
    line_num, item = matches[0]
    validated, source_meta = _prediction_from_batch_output_item(
        item=item,
        expected_custom_id=expected_custom_id,
    )
    source_meta["source_kind"] = "batch_output_jsonl"
    source_meta["source_path"] = str(path.resolve())
    source_meta["source_line_num"] = line_num
    return validated, source_meta


def main() -> int:
    args = parse_args()
    bundle_dir = args.bundle_dir
    if not bundle_dir.exists():
        raise SystemExit(f"Bundle directory not found: {bundle_dir}")

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

    if args.prediction_json is not None:
        response_payload, source_meta = _prediction_from_prediction_json(args.prediction_json)
    elif args.batch_output_item_json is not None:
        response_payload, source_meta = _prediction_from_batch_output_item_json(
            path=args.batch_output_item_json,
            expected_custom_id=custom_id,
        )
    elif args.batch_output_jsonl is not None:
        response_payload, source_meta = _prediction_from_batch_output_jsonl(
            path=args.batch_output_jsonl,
            expected_custom_id=custom_id,
        )
    else:
        raise FullTraceBundleScoringError("Exactly one response source must be provided")

    scored = score_full_trace_prediction(
        response_payload=response_payload,
        observed_repair_target=observed_repair_target,
        observed_coarse_path=observed_coarse_path,
    )
    artifact = {
        "schema_version": "v6_2_full_trace_scored_bundle_v1",
        "custom_id": custom_id,
        "bundle_dir": str(bundle_dir.resolve()),
        "manifest_path": str((bundle_dir / "manifest.json").resolve()),
        "observed_next_repair_target_path": str(observed_repair_target_path.resolve()),
        "observed_next_coarse_path_path": str(observed_coarse_path_path.resolve()),
        "response_source": source_meta,
        "validated_response": response_payload,
        "scored_prediction": scored,
    }

    out_path = args.out if args.out is not None else bundle_dir / "scored_prediction.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"custom_id": custom_id, "out_path": str(out_path.resolve())}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
