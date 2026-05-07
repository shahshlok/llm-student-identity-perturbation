from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from .benchmark_payloads import build_benchmark_payloads
from .prompting import build_system_prompt
from .runner import ROOT, SliceBuildError

if TYPE_CHECKING:
    from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv() -> bool:
        return False


load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_BATCH_MODEL", "gpt-5.4")
DEFAULT_REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")
DEFAULT_BATCH_ROOT = ROOT / "data" / "v5_batch_runs"
DEFAULT_DATA_ROOT = ROOT.parent / "tracer" / "2024-1"
HEAD_LABELS = {
    "first_focus_region_3way": (
        "output_region",
        "conditional_region",
        "loop_region",
    ),
    "lines_touched_bucket_3way": (
        "local_1_to_2_lines",
        "regional_3_to_5_lines",
        "broad_6_plus_lines",
    ),
    "next_test_outcome": (
        "all_fail",
        "mixed",
        "all_pass",
    ),
}


class SupportingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["aggregate", "card"]
    source_id: str
    claim: str


class PredictedTraceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_focus_region_3way: Literal["output_region", "conditional_region", "loop_region"]
    lines_touched_bucket_3way: Literal[
        "local_1_to_2_lines",
        "regional_3_to_5_lines",
        "broad_6_plus_lines",
    ]
    next_test_outcome: Literal["all_fail", "mixed", "all_pass"]


class HypothesisItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    description: str
    estimated_prevalence: float
    supporting_evidence: list[SupportingEvidence]
    predicted_trace_profile: PredictedTraceProfile
    counterevidence: list[str]


class V5BatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cohort_summary: str
    hypotheses: list[HypothesisItem]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5 cohort prompting through OpenAI Batch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Build payloads plus a batch JSONL request file"
    )
    prepare.add_argument("--benchmark-manifest", required=True, type=Path)
    prepare.add_argument("--run-name", default=None)
    prepare.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    prepare.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    prepare.add_argument("--max-cards", type=int, default=10)

    submit = subparsers.add_parser("submit", help="Upload the request JSONL and create a batch")
    submit.add_argument("--manifest", required=True, type=Path)

    status = subparsers.add_parser("status", help="Check batch status and update the manifest")
    status.add_argument("--manifest", required=True, type=Path)

    download = subparsers.add_parser("download", help="Download output and error files for a batch")
    download.add_argument("--manifest", required=True, type=Path)

    hydrate = subparsers.add_parser(
        "hydrate",
        help="Parse downloaded batch output into strict typed v5 prediction artifacts",
    )
    hydrate.add_argument("--manifest", required=True, type=Path)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Score hydrated v5 cohort predictions against held-out observed labels",
    )
    evaluate.add_argument("--manifest", required=True, type=Path)

    return parser.parse_args()


def _client() -> OpenAI:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The openai package is not installed in this environment. Use `uv run ...` or `uv sync` first."
        ) from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def _default_run_name(model: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v5_{_slug(model)}_{stamp}"


def _run_dir(batch_root: Path, run_name: str) -> Path:
    return batch_root / run_name


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _structured_text_format(response_model: type[BaseModel]) -> dict[str, object]:
    name = (response_model.__name__ or "structured_output").strip()[:64]
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": response_model.model_json_schema(),
        }
    }


def _request_custom_id(class_id: str, assessment_id: str, exercise_id: str) -> str:
    return f"{class_id}:{assessment_id}:{exercise_id}"


def _build_request_body(
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
        "text": _structured_text_format(V5BatchResponse),
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def _update_batch_snapshot(manifest: dict[str, Any], batch: Any) -> None:
    batch_state = manifest.setdefault("batch", {})
    request_counts = getattr(batch, "request_counts", None)
    batch_state.update(
        {
            "input_file_id": getattr(batch, "input_file_id", batch_state.get("input_file_id")),
            "batch_id": getattr(batch, "id", batch_state.get("batch_id")),
            "status": getattr(batch, "status", batch_state.get("status")),
            "output_file_id": getattr(batch, "output_file_id", batch_state.get("output_file_id")),
            "error_file_id": getattr(batch, "error_file_id", batch_state.get("error_file_id")),
            "created_at": getattr(batch, "created_at", batch_state.get("created_at")),
            "completed_at": getattr(batch, "completed_at", batch_state.get("completed_at")),
            "failed_at": getattr(batch, "failed_at", batch_state.get("failed_at")),
            "expired_at": getattr(batch, "expired_at", batch_state.get("expired_at")),
            "request_counts": getattr(request_counts, "model_dump", lambda: None)()
            if request_counts is not None
            else batch_state.get("request_counts"),
        }
    )


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _save_json(path, manifest)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_response_text(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = body.get("output")
    if isinstance(output, list):
        text_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                item_type = content_item.get("type")
                if item_type in {"output_text", "text"} and isinstance(
                    content_item.get("text"), str
                ):
                    text_parts.append(content_item["text"])
        joined = "".join(text_parts).strip()
        if joined:
            return joined

    content = body.get("content")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"output_text", "text"} and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        joined = "".join(text_parts).strip()
        if joined:
            return joined

    raise ValueError("Responses payload does not contain extractable output text")


def _parse_response_payload(payload_text: str) -> dict[str, Any]:
    payload = json.loads(_strip_json_fence(payload_text))
    return V5BatchResponse.model_validate(payload).model_dump()


def _argmax_label(distribution: dict[str, float], labels: tuple[str, ...]) -> str:
    return max(labels, key=lambda label: (distribution[label], -labels.index(label)))


def _normalize_distribution(
    distribution: dict[str, float], labels: tuple[str, ...]
) -> dict[str, float]:
    total = sum(float(distribution[label]) for label in labels)
    if total <= 0:
        raise SliceBuildError(f"Distribution mass must be positive, got {total}")
    return {label: float(distribution[label]) / total for label in labels}


def _kl_divergence(p: dict[str, float], q: dict[str, float], labels: tuple[str, ...]) -> float:
    import math

    total = 0.0
    for label in labels:
        p_val = float(p[label])
        q_val = float(q[label])
        if p_val <= 0:
            continue
        if q_val <= 0:
            raise SliceBuildError(f"KL divergence encountered zero mass for label {label}")
        total += p_val * math.log(p_val / q_val, 2)
    return total


def _js_divergence(p: dict[str, float], q: dict[str, float], labels: tuple[str, ...]) -> float:
    midpoint = {label: 0.5 * (float(p[label]) + float(q[label])) for label in labels}
    return 0.5 * _kl_divergence(p, midpoint, labels) + 0.5 * _kl_divergence(q, midpoint, labels)


def _l1_distance(p: dict[str, float], q: dict[str, float], labels: tuple[str, ...]) -> float:
    return sum(abs(float(p[label]) - float(q[label])) for label in labels)


def _predicted_head_distributions(response_payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    hypotheses = response_payload["hypotheses"]
    if not hypotheses:
        raise SliceBuildError("Structured response contains no hypotheses")

    raw_weights: list[float] = []
    for index, hypothesis in enumerate(hypotheses):
        weight = float(hypothesis["estimated_prevalence"])
        if weight < 0:
            raise SliceBuildError(f"Hypothesis prevalence must be non-negative at index {index}")
        raw_weights.append(weight)

    total_weight = sum(raw_weights)
    if total_weight <= 0:
        raise SliceBuildError("Hypothesis prevalence mass sums to zero")

    normalized_weights = [weight / total_weight for weight in raw_weights]
    predicted: dict[str, dict[str, float]] = {
        head: dict.fromkeys(labels, 0.0) for head, labels in HEAD_LABELS.items()
    }
    for weight, hypothesis in zip(normalized_weights, hypotheses, strict=True):
        profile = hypothesis["predicted_trace_profile"]
        for head, labels in HEAD_LABELS.items():
            label = profile[head]
            if label not in labels:
                raise SliceBuildError(f"Unexpected predicted label for {head}: {label}")
            predicted[head][label] += weight
    return predicted


def _leave_one_out_majority_distributions(
    slices: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    if len(slices) < 2:
        raise SliceBuildError(
            "Need at least two hydrated slices for leave-one-out majority baseline"
        )

    total_counts_by_head: dict[str, Counter[str]] = {head: Counter() for head in HEAD_LABELS}
    for row in slices:
        observed = row["observed_labels"]["by_head"]
        for head, labels in HEAD_LABELS.items():
            counts = observed[head]["counts"]
            for label in labels:
                total_counts_by_head[head][label] += int(counts[label])

    baseline: dict[str, dict[str, dict[str, float]]] = {}
    for row in slices:
        custom_id = row["custom_id"]
        observed = row["observed_labels"]["by_head"]
        baseline[custom_id] = {}
        for head, labels in HEAD_LABELS.items():
            held_out_counts = observed[head]["counts"]
            remaining = {
                label: total_counts_by_head[head][label] - int(held_out_counts[label])
                for label in labels
            }
            if sum(remaining.values()) <= 0:
                raise SliceBuildError(
                    f"Leave-one-out majority baseline has no remaining mass for {custom_id} / {head}"
                )
            baseline[custom_id][head] = _normalize_distribution(remaining, labels)
    return baseline


def _rotated_slice_baseline(slices: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    if len(slices) < 2:
        raise SliceBuildError("Need at least two hydrated slices for rotated-slice baseline")

    ordered = sorted(slices, key=lambda row: row["custom_id"])
    baseline: dict[str, dict[str, dict[str, float]]] = {}
    for index, row in enumerate(ordered):
        donor = ordered[(index + 1) % len(ordered)]
        baseline[row["custom_id"]] = {
            head: donor["observed_labels"]["by_head"][head]["distribution"] for head in HEAD_LABELS
        }
    return baseline


def _uniform_baseline() -> dict[str, dict[str, float]]:
    return {
        head: {label: 1.0 / len(labels) for label in labels} for head, labels in HEAD_LABELS.items()
    }


def _score_prediction_set(
    slices: list[dict[str, Any]],
    predicted_by_custom_id: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    per_slice: list[dict[str, Any]] = []
    for row in slices:
        custom_id = row["custom_id"]
        observed = row["observed_labels"]["by_head"]
        predicted = predicted_by_custom_id[custom_id]
        head_scores: dict[str, Any] = {}
        exact_top1 = True
        for head, labels in HEAD_LABELS.items():
            observed_distribution = {
                label: float(observed[head]["distribution"][label]) for label in labels
            }
            predicted_distribution = {label: float(predicted[head][label]) for label in labels}
            observed_top1 = _argmax_label(observed_distribution, labels)
            predicted_top1 = _argmax_label(predicted_distribution, labels)
            top1_match = predicted_top1 == observed_top1
            exact_top1 = exact_top1 and top1_match
            head_scores[head] = {
                "predicted_distribution": predicted_distribution,
                "observed_distribution": observed_distribution,
                "predicted_top1": predicted_top1,
                "observed_top1": observed_top1,
                "top1_match": top1_match,
                "l1_distance": _l1_distance(predicted_distribution, observed_distribution, labels),
                "js_divergence": _js_divergence(
                    predicted_distribution, observed_distribution, labels
                ),
            }
        per_slice.append(
            {
                "custom_id": custom_id,
                "class_id": row["class_id"],
                "assessment_id": row["assessment_id"],
                "exercise_id": row["exercise_id"],
                "n_windows": int(row["observed_labels"]["n_windows"]),
                "exact_top1_match_3head": exact_top1,
                "by_head": head_scores,
            }
        )

    summary_by_head: dict[str, Any] = {}
    for head in HEAD_LABELS:
        summary_by_head[head] = {
            "mean_top1_match": sum(
                1.0 if row["by_head"][head]["top1_match"] else 0.0 for row in per_slice
            )
            / len(per_slice),
            "mean_l1_distance": sum(row["by_head"][head]["l1_distance"] for row in per_slice)
            / len(per_slice),
            "mean_js_divergence": sum(row["by_head"][head]["js_divergence"] for row in per_slice)
            / len(per_slice),
        }

    return {
        "n_slices": len(per_slice),
        "exact_top1_match_3head_rate": sum(
            1.0 if row["exact_top1_match_3head"] else 0.0 for row in per_slice
        )
        / len(per_slice),
        "by_head": summary_by_head,
        "per_slice": per_slice,
    }


def _prepare(args: argparse.Namespace) -> int:
    run_name = args.run_name or _default_run_name(args.model)
    run_dir = _run_dir(args.batch_root, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    payload_root = run_dir / "payloads"
    payload_result = build_benchmark_payloads(
        benchmark_manifest_path=args.benchmark_manifest,
        out_root=payload_root,
        data_root=args.data_root,
        max_cards=args.max_cards,
    )
    payload_manifest = payload_result["manifest"]

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
    _save_json(response_schema_path, V5BatchResponse.model_json_schema())
    request_map: dict[str, dict[str, Any]] = {}
    ordered_custom_ids: list[str] = []

    with requests_path.open("w", encoding="utf-8") as handle:
        for row in payload_manifest["slices"]:
            custom_id = _request_custom_id(
                str(row["class_id"]),
                str(row["assessment_id"]),
                str(row["exercise_id"]),
            )
            user_prompt = Path(row["user_prompt_path"]).read_text(encoding="utf-8")
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
            request_map[custom_id] = row
            ordered_custom_ids.append(custom_id)

    manifest = {
        "created_at": _utc_now(),
        "run_name": run_name,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "benchmark_manifest_path": str(args.benchmark_manifest.resolve()),
        "payload_batch_manifest_path": payload_result["paths"]["batch_manifest"],
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
            "selected_slices": len(payload_manifest["slices"]),
            "requests_created": len(payload_manifest["slices"]),
        },
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
        "request_map": request_map,
    }
    _save_json(manifest_path, manifest)

    print(f"Prepared run: {run_name}")
    print(f"Requests: {manifest['counts']['requests_created']}")
    print(f"Manifest: {_display_path(manifest_path)}")
    print(f"Requests JSONL: {_display_path(requests_path)}")
    print(f"System prompt: {_display_path(system_prompt_path)}")
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
        print(f"Downloaded output: {output_path}")
    else:
        print("No output_file_id yet.")

    if error_file_id:
        _download_file(client, error_file_id, error_path)
        print(f"Downloaded errors: {error_path}")
    else:
        print("No error_file_id yet.")
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

    request_map = manifest["request_map"]
    ordered_custom_ids = manifest["ordered_custom_ids"]
    responded_custom_ids: set[str] = set()
    slices_by_custom_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    error_file_entries = 0

    with output_path.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id")
            if not custom_id or custom_id not in request_map:
                continue

            responded_custom_ids.add(custom_id)
            meta = request_map[custom_id]
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

            observed_labels_path = Path(meta["observed_labels_path"])
            slices_by_custom_id[custom_id] = {
                "custom_id": custom_id,
                "class_id": str(meta["class_id"]),
                "assessment_id": str(meta["assessment_id"]),
                "exercise_id": str(meta["exercise_id"]),
                "included_transitions": int(meta["included_transitions"]),
                "included_students": int(meta["included_students"]),
                "payload_path": str(Path(meta["payload_path"]).resolve()),
                "user_prompt_path": str(Path(meta["user_prompt_path"]).resolve()),
                "observed_labels_path": str(observed_labels_path.resolve()),
                "observed_labels": _load_json(observed_labels_path),
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
                if not custom_id or custom_id not in request_map:
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

    ordered_slices = [
        slices_by_custom_id[custom_id]
        for custom_id in ordered_custom_ids
        if custom_id in slices_by_custom_id
    ]
    hydrated_payload = {
        "schema_version": "v5_batch_hydrated_predictions_v1",
        "run_name": manifest["run_name"],
        "model": manifest["model"],
        "reasoning_effort": manifest["reasoning_effort"],
        "summary": {
            "requested_slices": len(ordered_custom_ids),
            "parsed_slices": len(ordered_slices),
            "missing_outputs": missing_outputs,
            "failure_count": len(failures),
            "batch_error_file_entries": error_file_entries,
        },
        "slices": ordered_slices,
    }
    _save_json(hydrated_path, hydrated_payload)
    _save_json(hydration_failures_path, {"failures": failures})

    manifest["hydration"] = {
        "hydrated_at": _utc_now(),
        "requested_slices": len(ordered_custom_ids),
        "parsed_slices": len(ordered_slices),
        "missing_outputs": missing_outputs,
        "failure_count": len(failures),
    }
    _save_manifest(manifest_path, manifest)

    print(f"Hydrated predictions: {_display_path(hydrated_path)}")
    print(f"Hydration failures: {_display_path(hydration_failures_path)}")
    print(f"Parsed slices: {len(ordered_slices)}")
    print(f"Missing outputs: {missing_outputs}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    manifest_path = args.manifest
    manifest = _load_json(manifest_path)
    hydrated_path = Path(manifest["paths"]["hydrated_predictions_json"])
    evaluation_path = Path(manifest["paths"]["evaluation_json"])
    evaluation_md_path = Path(manifest["paths"]["evaluation_md"])

    if not hydrated_path.exists():
        raise SystemExit(f"Hydrated predictions not found: {hydrated_path}")

    hydrated = _load_json(hydrated_path)
    slices = hydrated["slices"]
    if not slices:
        raise SliceBuildError("No hydrated slices available for evaluation")

    model_predictions = {
        row["custom_id"]: _predicted_head_distributions(row["response"]) for row in slices
    }
    majority_predictions = _leave_one_out_majority_distributions(slices)
    rotated_predictions = _rotated_slice_baseline(slices)
    uniform_predictions = {row["custom_id"]: _uniform_baseline() for row in slices}

    model_scores = _score_prediction_set(slices, model_predictions)
    majority_scores = _score_prediction_set(slices, majority_predictions)
    rotated_scores = _score_prediction_set(slices, rotated_predictions)
    uniform_scores = _score_prediction_set(slices, uniform_predictions)

    evaluation = {
        "schema_version": "v5_batch_evaluation_v1",
        "evaluated_at": _utc_now(),
        "run_name": manifest["run_name"],
        "model": manifest["model"],
        "reasoning_effort": manifest["reasoning_effort"],
        "summary": {
            "n_slices": len(slices),
            "selection_policy": _load_json(Path(manifest["benchmark_manifest_path"]))[
                "selection_policy"
            ],
        },
        "scores": {
            "model": model_scores,
            "leave_one_out_majority": majority_scores,
            "rotated_slice": rotated_scores,
            "uniform": uniform_scores,
        },
    }
    _save_json(evaluation_path, evaluation)

    lines = [
        "# V5 Batch Evaluation",
        "",
        f"- Run name: `{manifest['run_name']}`",
        f"- Model: `{manifest['model']}`",
        f"- Reasoning effort: `{manifest['reasoning_effort']}`",
        f"- Evaluated slices: `{len(slices)}`",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Model | Leave-one-out majority | Rotated slice | Uniform |",
        "| --- | --- | --- | --- | --- |",
        f"| Exact 3-head top1 rate | {model_scores['exact_top1_match_3head_rate']:.3f} | {majority_scores['exact_top1_match_3head_rate']:.3f} | {rotated_scores['exact_top1_match_3head_rate']:.3f} | {uniform_scores['exact_top1_match_3head_rate']:.3f} |",
    ]
    for head in HEAD_LABELS:
        lines.append(
            f"| {head} mean top1 | {model_scores['by_head'][head]['mean_top1_match']:.3f} | "
            f"{majority_scores['by_head'][head]['mean_top1_match']:.3f} | "
            f"{rotated_scores['by_head'][head]['mean_top1_match']:.3f} | "
            f"{uniform_scores['by_head'][head]['mean_top1_match']:.3f} |"
        )
        lines.append(
            f"| {head} mean JS | {model_scores['by_head'][head]['mean_js_divergence']:.3f} | "
            f"{majority_scores['by_head'][head]['mean_js_divergence']:.3f} | "
            f"{rotated_scores['by_head'][head]['mean_js_divergence']:.3f} | "
            f"{uniform_scores['by_head'][head]['mean_js_divergence']:.3f} |"
        )
    evaluation_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest["evaluation"] = {
        "evaluated_at": evaluation["evaluated_at"],
        "n_slices": len(slices),
        "paths": {
            "evaluation_json": str(evaluation_path.resolve()),
            "evaluation_md": str(evaluation_md_path.resolve()),
        },
    }
    _save_manifest(manifest_path, manifest)

    print(f"Evaluation JSON: {_display_path(evaluation_path)}")
    print(f"Evaluation Markdown: {_display_path(evaluation_md_path)}")
    print(f"Exact 3-head top1 rate: {model_scores['exact_top1_match_3head_rate']:.3f}")
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
