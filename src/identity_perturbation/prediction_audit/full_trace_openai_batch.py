from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from identity_perturbation.codebench_support.openai_batch import _client, _save_json, _update_batch_snapshot, _utc_now

if TYPE_CHECKING:
    from openai import OpenAI


SCHEMA_VERSION = "v6_2_full_trace_pilot_batch_manifest_v1"
DEFAULT_ENDPOINT = "/v1/responses"
DEFAULT_COMPLETION_WINDOW = "24h"


class FullTraceBatchError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit, monitor, and download a prepared v6.2 full_trace batch run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--manifest", required=True, type=Path)

    status = subparsers.add_parser("status")
    status.add_argument("--manifest", required=True, type=Path)

    download = subparsers.add_parser("download")
    download.add_argument("--manifest", required=True, type=Path)

    return parser.parse_args()


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FullTraceBatchError(f"JSON file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FullTraceBatchError(f"Top-level JSON is not an object in {path}")
    return payload


def _default_run_name(manifest_path: Path) -> str:
    return manifest_path.resolve().parent.name


def _default_output_path(manifest_path: Path) -> Path:
    return manifest_path.resolve().parent / "output.jsonl"


def _default_error_path(manifest_path: Path) -> Path:
    return manifest_path.resolve().parent / "errors.jsonl"


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise FullTraceBatchError(
            f"Unexpected manifest schema_version in {manifest_path}: {manifest.get('schema_version')!r}"
        )
    paths = manifest.get("paths")
    if not isinstance(paths, dict):
        raise FullTraceBatchError(f"Manifest missing paths object: {manifest_path}")
    requests_jsonl = paths.get("requests_jsonl")
    if not isinstance(requests_jsonl, str) or not requests_jsonl.strip():
        raise FullTraceBatchError(f"Manifest missing paths.requests_jsonl: {manifest_path}")
    requests_path = Path(requests_jsonl)
    if not requests_path.exists():
        raise FullTraceBatchError(f"Requests JSONL not found: {requests_path}")

    paths.setdefault("output_jsonl", str(_default_output_path(manifest_path)))
    paths.setdefault("error_jsonl", str(_default_error_path(manifest_path)))
    manifest.setdefault("run_name", _default_run_name(manifest_path))
    manifest.setdefault("batch", {})
    batch = manifest["batch"]
    if not isinstance(batch, dict):
        raise FullTraceBatchError(f"Manifest batch field is not an object: {manifest_path}")
    batch.setdefault("endpoint", DEFAULT_ENDPOINT)
    batch.setdefault("completion_window", DEFAULT_COMPLETION_WINDOW)
    batch.setdefault("input_file_id", None)
    batch.setdefault("batch_id", None)
    batch.setdefault("status", "prepared")
    batch.setdefault("output_file_id", None)
    batch.setdefault("error_file_id", None)
    return manifest


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _save_json(path, manifest)


def _download_file(client: OpenAI, file_id: str, path: Path) -> None:
    content = client.files.content(file_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.read())


def _submit(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    requests_path = Path(str(manifest["paths"]["requests_jsonl"]))
    batch_state = manifest["batch"]

    if batch_state.get("batch_id"):
        raise FullTraceBatchError(
            f"Manifest already contains batch_id={batch_state['batch_id']!r}. "
            "Refusing to create a second batch for the same manifest."
        )

    client = _client()
    with requests_path.open("rb") as handle:
        file_obj = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=str(batch_state["endpoint"]),
        completion_window=str(batch_state["completion_window"]),
        metadata={
            "run_name": str(manifest["run_name"]),
            "model": str(manifest["model"]),
        },
    )
    batch_state["input_file_id"] = file_obj.id
    _update_batch_snapshot(manifest, batch)
    _save_manifest(manifest_path, manifest)

    print(f"Uploaded input file: {file_obj.id}")
    print(f"Created batch: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Manifest: {manifest_path}")
    return 0


def _status(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    batch_id = manifest["batch"].get("batch_id")
    if not batch_id:
        raise FullTraceBatchError("Manifest does not contain a batch_id. Run submit first.")

    client = _client()
    batch = client.batches.retrieve(str(batch_id))
    _update_batch_snapshot(manifest, batch)
    _save_manifest(manifest_path, manifest)
    print(json.dumps(manifest["batch"], ensure_ascii=False, indent=2))
    return 0


def _download(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    batch = manifest["batch"]
    client = _client()

    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")
    output_path = Path(str(manifest["paths"]["output_jsonl"]))
    error_path = Path(str(manifest["paths"]["error_jsonl"]))

    if output_file_id:
        _download_file(client, str(output_file_id), output_path)
        print(f"Downloaded output: {output_path}")
    else:
        print("No output_file_id present on the batch yet.")

    if error_file_id:
        _download_file(client, str(error_file_id), error_path)
        print(f"Downloaded errors: {error_path}")
    else:
        print("No error_file_id present on the batch.")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "submit":
        return _submit(args)
    if args.command == "status":
        return _status(args)
    if args.command == "download":
        return _download(args)
    raise FullTraceBatchError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
