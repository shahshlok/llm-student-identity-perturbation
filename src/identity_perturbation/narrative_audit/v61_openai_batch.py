from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from identity_perturbation.codebench_support.openai_batch import (
    _client,
    _display_path,
    _extract_response_text,
    _load_json,
    _save_json,
    _strip_json_fence,
    _update_batch_snapshot,
    _utc_now,
)

from .v61_eval.engine import evaluate_hydrated_predictions
from .v61_eval.report import write_evaluation_artifacts
from .v61_prediction_schema import V61PredictedEpisodeResponse
from .v61_prompting import build_system_prompt, structured_text_format
from .v61_runner import ROOT

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_BATCH_ROOT = ROOT / "data" / "v61_batch_runs"

BUNDLE_SCHEMA_VERSION = "v6_1_transition_bundle_v1"
BUNDLE_MANIFEST_SCHEMA_VERSION = "v6_1_transition_bundle_manifest_v1"
OBSERVED_EPISODE_SCHEMA_VERSION = "v6_1_observed_next_episode_v1"
HYDRATED_SCHEMA_VERSION = "v6_1_batch_hydrated_predictions_v1"
EVALUATION_SCHEMA_VERSION = "v6_1_batch_evaluation_v1"
EVENT_FAMILY_BY_TYPE = {
    "change": "edit",
    "saida_testar": "run",
    "submit": "submit",
    "kill_program": "runtime",
    "keyHandled": "navigate",
    "tab_click": "navigate",
    "idle_gap": "pause",
}


class V61BatchError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v6.1 next-episode prompting through OpenAI Batch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--bundle-dir", action="append", default=[], type=Path)
    prepare.add_argument("--bundle-manifest", action="append", default=[], type=Path)
    prepare.add_argument("--run-name", default=None)
    prepare.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--manifest", required=True, type=Path)

    status = subparsers.add_parser("status")
    status.add_argument("--manifest", required=True, type=Path)

    download = subparsers.add_parser("download")
    download.add_argument("--manifest", required=True, type=Path)

    hydrate = subparsers.add_parser("hydrate")
    hydrate.add_argument("--manifest", required=True, type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--prefix-k", type=int, default=3)

    return parser.parse_args()


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def _default_run_name(model: str) -> str:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v61_{_slug(model)}_{stamp}"


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
        "text": structured_text_format(),
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _save_json(path, manifest)


def _parse_response_payload(payload_text: str) -> dict[str, Any]:
    payload = json.loads(_strip_json_fence(payload_text))
    return V61PredictedEpisodeResponse.model_validate(payload).model_dump(by_alias=True)


def _load_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    user_prompt_path = bundle_dir / "user_prompt.txt"
    payload_path = bundle_dir / "payload.json"
    observed_path = bundle_dir / "observed_next_episode.json"
    for path, label in (
        (manifest_path, "manifest"),
        (user_prompt_path, "user prompt"),
        (payload_path, "payload"),
        (observed_path, "observed next episode"),
    ):
        if not path.exists():
            raise V61BatchError(f"Bundle {label} missing: {path}")

    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise V61BatchError(
            f"Unexpected bundle manifest schema in {manifest_path}: {manifest.get('schema_version')}"
        )

    required_manifest_keys = {
        "class_id",
        "assessment_id",
        "exercise_id",
        "student_id",
        "transition_index_0idx",
        "observed_next_episode_path",
    }
    missing = required_manifest_keys - set(manifest)
    if missing:
        raise V61BatchError(
            f"Bundle manifest missing required keys {sorted(missing)}: {manifest_path}"
        )

    observed_from_manifest = Path(str(manifest["observed_next_episode_path"]))
    if observed_from_manifest.resolve() != observed_path.resolve():
        raise V61BatchError(
            "Bundle manifest observed_next_episode_path does not match bundle file path: "
            f"{manifest_path}"
        )

    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "manifest": manifest,
        "user_prompt_path": str(user_prompt_path.resolve()),
        "payload_path": str(payload_path.resolve()),
        "observed_next_episode_path": str(observed_path.resolve()),
    }


def _bundle_dirs_from_manifest(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        raise V61BatchError(f"Bundle manifest not found: {path}")
    payload = _load_json(path)
    if payload.get("schema_version") != BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise V61BatchError(
            f"Unexpected bundle manifest schema in {path}: {payload.get('schema_version')}"
        )
    bundles = payload.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise V61BatchError(f"Bundle manifest must contain a non-empty bundles list: {path}")
    return tuple(Path(str(row["bundle_dir"])) for row in bundles)


def _resolve_prepare_bundle_dirs(args: argparse.Namespace) -> tuple[Path, ...]:
    bundle_dirs = [Path(path) for path in args.bundle_dir]
    for manifest_path in args.bundle_manifest:
        bundle_dirs.extend(_bundle_dirs_from_manifest(manifest_path))
    if not bundle_dirs:
        raise V61BatchError("prepare requires at least one --bundle-dir or --bundle-manifest")
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
    _save_json(response_schema_path, V61PredictedEpisodeResponse.model_json_schema())

    ordered_custom_ids: list[str] = []
    bundle_map: dict[str, dict[str, Any]] = {}
    seen_custom_ids: set[str] = set()

    with requests_path.open("w", encoding="utf-8") as handle:
        for bundle_dir in _resolve_prepare_bundle_dirs(args):
            loaded = _load_bundle(bundle_dir)
            bundle_manifest = loaded["manifest"]
            custom_id = _request_custom_id(bundle_manifest)
            if custom_id in seen_custom_ids:
                raise V61BatchError(
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
                "observed_next_episode_path": loaded["observed_next_episode_path"],
                "user_prompt_path": loaded["user_prompt_path"],
                "class_id": bundle_manifest["class_id"],
                "assessment_id": bundle_manifest["assessment_id"],
                "exercise_id": bundle_manifest["exercise_id"],
                "student_id": bundle_manifest["student_id"],
                "transition_index_0idx": bundle_manifest["transition_index_0idx"],
            }

    manifest = {
        "schema_version": "v6_1_batch_manifest_v1",
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
        "counts": {"bundles": len(ordered_custom_ids), "requests_created": len(ordered_custom_ids)},
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
        metadata={"run_name": manifest["run_name"], "model": manifest["model"]},
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
    manifest = _load_json(args.manifest)
    client = _client()
    batch = manifest["batch"]
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
                    {"line_num": line_num, "stage": "batch_output", "error": "missing custom_id"}
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
                "observed_next_episode_path": str(meta["observed_next_episode_path"]),
                "user_prompt_path": str(meta["user_prompt_path"]),
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
                failures.append(
                    {
                        "custom_id": item.get("custom_id"),
                        "line_num": line_num,
                        "stage": "batch_error_file",
                        "error": item.get("error", "batch error file entry"),
                    }
                )
                error_file_entries += 1

    missing_outputs = 0
    for custom_id in ordered_custom_ids:
        if custom_id not in responded_custom_ids:
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
        "schema_version": HYDRATED_SCHEMA_VERSION,
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


def _load_observed_episode(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != OBSERVED_EPISODE_SCHEMA_VERSION:
        raise V61BatchError(
            f"Unexpected observed next episode schema in {path}: {payload.get('schema_version')}"
        )
    events = payload.get("semantic_event_tape")
    if not isinstance(events, list) or not events:
        raise V61BatchError(f"Observed next episode missing non-empty semantic_event_tape: {path}")
    return payload


def _event_types(events: list[dict[str, Any]]) -> list[str]:
    return [str(event["event_type"]) for event in events]


def _first_event_type(events: list[dict[str, Any]]) -> str:
    if not events:
        raise V61BatchError("Observed or predicted event list is empty")
    return str(events[0]["event_type"])


def _event_family(event_type: str) -> str:
    if event_type not in EVENT_FAMILY_BY_TYPE:
        raise V61BatchError(f"Unknown event_type for family mapping: {event_type}")
    return EVENT_FAMILY_BY_TYPE[event_type]


def _first_non_idle_event_type(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        event_type = str(event["event_type"])
        if event_type != "idle_gap":
            return event_type
    return None


def _contains_type(events: list[dict[str, Any]], event_type: str) -> bool:
    return any(str(event["event_type"]) == event_type for event in events)


def _first_change_line(events: list[dict[str, Any]]) -> int | None:
    for event in events:
        if str(event["event_type"]) == "change":
            if "primary_line_0idx" in event:
                line = int(event["primary_line_0idx"])
                if line < 0:
                    return None
                return line
            if "from_line" in event:
                return int(event["from_line"])
            from_position = event.get("from")
            if isinstance(from_position, dict) and "line" in from_position:
                return int(from_position["line"])
            raise V61BatchError("change event missing from_line / from.line")
    return None


def _prefix_match(predicted: list[str], observed: list[str], k: int) -> bool:
    return predicted[:k] == observed[:k]


def _shared_prefix_length(predicted: list[str], observed: list[str]) -> int:
    count = 0
    for left, right in zip(predicted, observed, strict=False):
        if left != right:
            break
        count += 1
    return count


def _multiset_jaccard(left: list[str], right: list[str]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    keys = set(left_counts) | set(right_counts)
    if not keys:
        return 1.0
    intersection = sum(min(left_counts[key], right_counts[key]) for key in keys)
    union = sum(max(left_counts[key], right_counts[key]) for key in keys)
    if union == 0:
        return 1.0
    return intersection / union


def _hypotheses(response: dict[str, Any]) -> list[dict[str, Any]]:
    hypotheses = response.get("next_episode_hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise V61BatchError("Response missing non-empty next_episode_hypotheses")
    return hypotheses


def _top_hypothesis(response: dict[str, Any]) -> dict[str, Any]:
    return max(
        _hypotheses(response),
        key=lambda hypothesis: (float(hypothesis["estimated_probability"]), hypothesis["label"]),
    )


def _truth_mass(
    response: dict[str, Any],
    predicate: Callable[[list[dict[str, Any]]], bool],
) -> float:
    total = 0.0
    for hypothesis in _hypotheses(response):
        if predicate(hypothesis["predicted_event_tape"]):
            total += float(hypothesis["estimated_probability"])
    return total


def _expected_score(
    response: dict[str, Any],
    score_fn: Callable[[list[dict[str, Any]]], float],
) -> float:
    total = 0.0
    for hypothesis in _hypotheses(response):
        total += float(hypothesis["estimated_probability"]) * score_fn(
            hypothesis["predicted_event_tape"]
        )
    return total


def _safe_mean(total: float, count: int) -> float | None:
    if count == 0:
        return None
    return total / count


def _line_within_tolerance(
    predicted_line: int | None,
    observed_line: int | None,
    tolerance: int,
) -> bool | None:
    if observed_line is None:
        return None
    if predicted_line is None:
        return False
    return abs(predicted_line - observed_line) <= tolerance


def _evaluate(args: argparse.Namespace) -> int:
    manifest_path = args.manifest
    manifest = _load_json(manifest_path)
    hydrated_path = Path(manifest["paths"]["hydrated_predictions_json"])
    if not hydrated_path.exists():
        raise SystemExit(f"Hydrated predictions not found: {hydrated_path}")
    evaluation = evaluate_hydrated_predictions(
        hydrated_path=hydrated_path,
        prefix_k=args.prefix_k,
    )
    evaluation["evaluated_at"] = _utc_now()
    evaluation["run_name"] = manifest["run_name"]
    evaluation["model"] = manifest["model"]
    evaluation["reasoning_effort"] = manifest["reasoning_effort"]

    evaluation_path = Path(manifest["paths"]["evaluation_json"])
    evaluation_md_path = Path(manifest["paths"]["evaluation_md"])
    write_evaluation_artifacts(
        evaluation=evaluation,
        evaluation_json_path=evaluation_path,
        evaluation_md_path=evaluation_md_path,
        run_name=manifest["run_name"],
        model=manifest["model"],
        reasoning_effort=manifest["reasoning_effort"],
    )

    manifest["evaluation"] = {
        "evaluated_at": evaluation["evaluated_at"],
        "n_predictions": evaluation["scores"]["n_predictions"],
        "prefix_k": args.prefix_k,
        "paths": {
            "evaluation_json": str(evaluation_path.resolve()),
            "evaluation_md": str(evaluation_md_path.resolve()),
        },
    }
    _save_manifest(manifest_path, manifest)
    print(f"Evaluation JSON: {_display_path(evaluation_path)}")
    print(f"Evaluation Markdown: {_display_path(evaluation_md_path)}")
    print(
        "First active event type top-1: "
        f"{evaluation['scores']['metrics']['first_active_event_type']['top1_accuracy']:.3f}"
    )
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


if __name__ == "__main__":
    raise SystemExit(main())
