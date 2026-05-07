from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from tempfile import TemporaryDirectory

from identity_perturbation.codebench_support.runner import ROOT, SliceBuildError

from .runner import DEFAULT_DATA_ROOT, V6BuildError, build_transition_payload_bundle
from .transition_manifest import build_transition_manifest

DEFAULT_BENCHMARK_MANIFEST = (
    ROOT
    / "data"
    / "v5"
    / "corpus_benchmarks"
    / "strong_lab56_min2s3t"
    / "benchmark_manifests"
    / "default"
    / "benchmark_manifest.json"
)


class V6SelectionError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a stable explicit v6 transition set from an audited v5 benchmark manifest."
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=DEFAULT_BENCHMARK_MANIFEST,
        help="Path to a v5 benchmark_manifest.json",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of transitions to select",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "v6" / "transition_manifests",
        help="Artifact root for v6 transition manifests",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Raw dataset root containing 2024-1/<class_id>/...",
    )
    parser.add_argument(
        "--manifest-name",
        required=True,
        help="Subdirectory name for the output manifest bundle",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise V6SelectionError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assessment_audits_root(benchmark_manifest_path: Path) -> Path:
    root = benchmark_manifest_path.parents[2] / "assessment_audits"
    if not root.exists():
        raise V6SelectionError(f"Assessment audits root not found: {root}")
    return root


def _slice_audit_path(
    benchmark_manifest_path: Path,
    selected_slice: dict[str, object],
) -> Path:
    audits_root = _assessment_audits_root(benchmark_manifest_path)
    class_id = str(selected_slice["class_id"])
    assessment_id = str(selected_slice["assessment_id"])
    exercise_id = str(selected_slice["exercise_id"])
    path = (
        audits_root
        / class_id
        / assessment_id
        / "exercises"
        / class_id
        / assessment_id
        / exercise_id
        / "audit.json"
    )
    if not path.exists():
        raise V6SelectionError(f"Slice audit not found: {path}")
    return path


def _included_rows_for_slice(
    benchmark_manifest_path: Path,
    selected_slice: dict[str, object],
) -> list[dict[str, object]]:
    payload = _load_json(_slice_audit_path(benchmark_manifest_path, selected_slice))
    if payload.get("schema_version") != "v5_slice_audit_v1":
        raise V6SelectionError(f"Unexpected slice audit schema: {payload.get('schema_version')}")
    included = [row for row in payload["transitions"] if row["status"] == "included"]
    if not included:
        raise V6SelectionError(
            f"No included transitions in slice audit for "
            f"{selected_slice['class_id']}:{selected_slice['assessment_id']}:{selected_slice['exercise_id']}"
        )
    return included


def _per_slice_student_round_robin(
    benchmark_manifest_path: Path,
    selected_slice: dict[str, object],
) -> list[dict[str, object]]:
    included = _included_rows_for_slice(benchmark_manifest_path, selected_slice)
    rows_by_student: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in included:
        rows_by_student[str(row["student_id"])].append(row)
    for rows in rows_by_student.values():
        rows.sort(key=lambda row: int(row["transition_index_0idx"]))

    student_queues: dict[str, deque[dict[str, object]]] = {
        student_id: deque(rows)
        for student_id, rows in sorted(rows_by_student.items(), key=lambda item: item[0])
    }

    ordered: list[dict[str, object]] = []
    while True:
        advanced = False
        for student_id in sorted(student_queues):
            queue = student_queues[student_id]
            if not queue:
                continue
            advanced = True
            row = queue.popleft()
            ordered.append(
                {
                    "class_id": str(selected_slice["class_id"]),
                    "assessment_id": str(selected_slice["assessment_id"]),
                    "exercise_id": str(selected_slice["exercise_id"]),
                    "student_id": str(row["student_id"]),
                    "transition_index_0idx": int(row["transition_index_0idx"]),
                }
            )
        if not advanced:
            break
    return ordered


def _is_buildable_transition(
    *,
    row: dict[str, object],
    data_root: Path,
) -> bool:
    try:
        with TemporaryDirectory(prefix="v6_select_check_") as temp_dir:
            build_transition_payload_bundle(
                class_id=str(row["class_id"]),
                assessment_id=str(row["assessment_id"]),
                exercise_id=str(row["exercise_id"]),
                student_id=str(row["student_id"]),
                transition_index=int(row["transition_index_0idx"]),
                data_root=data_root,
                out_root=Path(temp_dir),
                model="gpt-5.4",
                reasoning_effort="medium",
            )
    except V6BuildError:
        return False
    return True


def build_selected_transition_specs(
    *,
    benchmark_manifest_path: Path,
    count: int,
    data_root: Path,
) -> tuple[str, ...]:
    if count < 1:
        raise V6SelectionError(f"--count must be positive; got {count}")

    benchmark_manifest = _load_json(benchmark_manifest_path)
    if benchmark_manifest.get("schema_version") != "v5_benchmark_manifest_v1":
        raise V6SelectionError(
            f"Unexpected benchmark manifest schema: {benchmark_manifest.get('schema_version')}"
        )
    selected_slices = benchmark_manifest.get("selected_slices")
    if not isinstance(selected_slices, list) or not selected_slices:
        raise V6SelectionError("Benchmark manifest contains no selected_slices")

    per_slice_queues: list[deque[dict[str, object]]] = [
        deque(_per_slice_student_round_robin(benchmark_manifest_path, selected_slice))
        for selected_slice in selected_slices
    ]
    total_available = sum(len(queue) for queue in per_slice_queues)
    if count > total_available:
        raise V6SelectionError(
            f"Requested {count} transitions but only {total_available} are available"
        )

    selected_rows: list[dict[str, object]] = []
    skipped_unbuildable = 0
    while len(selected_rows) < count:
        advanced = False
        for queue in per_slice_queues:
            if not queue:
                continue
            candidate = queue.popleft()
            advanced = True
            if not _is_buildable_transition(row=candidate, data_root=data_root):
                skipped_unbuildable += 1
                continue
            selected_rows.append(candidate)
            if len(selected_rows) == count:
                break
        if not advanced:
            break

    if len(selected_rows) != count:
        raise V6SelectionError(
            f"Selection terminated early: expected {count}, got {len(selected_rows)} "
            f"after skipping {skipped_unbuildable} unbuildable audited transition(s)"
        )
    return tuple(
        (
            f"{row['class_id']}:{row['assessment_id']}:{row['exercise_id']}:"
            f"{row['student_id']}:{row['transition_index_0idx']}"
        )
        for row in selected_rows
    )


def main() -> int:
    args = parse_args()
    try:
        transition_specs = build_selected_transition_specs(
            benchmark_manifest_path=args.benchmark_manifest,
            count=args.count,
            data_root=args.data_root,
        )
        result = build_transition_manifest(
            transition_specs=transition_specs,
            out_root=args.out_root,
            manifest_name=args.manifest_name,
        )
    except (V6SelectionError, SliceBuildError) as exc:
        raise SystemExit(str(exc)) from exc

    manifest = result["manifest"]
    manifest["selection_policy"] = {
        "mode": "v5_benchmark_round_robin",
        "source_benchmark_manifest": str(args.benchmark_manifest.resolve()),
        "requested_count": args.count,
        "slice_order": "benchmark_manifest_selected_slices_order",
        "within_slice_order": "student_round_robin_then_transition_index_0idx",
    }
    output_path = Path(result["paths"]["manifest_json"])
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
