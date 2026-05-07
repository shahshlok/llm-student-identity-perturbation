from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import CohortWindow
from .payload import build_payload
from .prompting import build_system_prompt, build_user_prompt
from .runner import (
    DEFAULT_DATA_ROOT,
    SliceBuildError,
    build_observed_distributions,
    build_transition_window_record,
    load_student_transition_context,
    parse_assessment_file,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v5 payload/prompt bundles for slices selected in a benchmark manifest."
    )
    parser.add_argument(
        "--benchmark-manifest",
        required=True,
        type=Path,
        help="Path to benchmark_manifest.json",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Output root for generated slice payload bundles",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional raw dataset root override; defaults to runner default",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=10,
        help="Representative cards to include in each payload",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _resolve_assessment_audit_path(
    benchmark_manifest_path: Path,
    class_id: str,
    assessment_id: str,
    source_assessment_audit_paths: tuple[Path, ...],
) -> Path:
    if source_assessment_audit_paths:
        matches = [
            path
            for path in source_assessment_audit_paths
            if path.parent.name == assessment_id and path.parent.parent.name == class_id
        ]
    else:
        matches = [
            path
            for path in (benchmark_manifest_path.parent.parent.parent.parent).rglob(
                "assessment_audit.json"
            )
            if path.parent.name == assessment_id and path.parent.parent.name == class_id
        ]

    if len(matches) != 1:
        raise SliceBuildError(
            f"Expected exactly one assessment_audit.json for class={class_id} assessment={assessment_id}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_slice_audit(
    benchmark_manifest_path: Path,
    selected_row: dict[str, object],
    source_assessment_audit_paths: tuple[Path, ...],
) -> tuple[Path, dict[str, object]]:
    class_id = str(selected_row["class_id"])
    assessment_id = str(selected_row["assessment_id"])
    exercise_id = str(selected_row["exercise_id"])
    audit_path_fragment = Path(str(selected_row["audit_path"]))
    assessment_audit_path = _resolve_assessment_audit_path(
        benchmark_manifest_path=benchmark_manifest_path,
        class_id=class_id,
        assessment_id=assessment_id,
        source_assessment_audit_paths=source_assessment_audit_paths,
    )
    candidate_paths = (
        assessment_audit_path.parent / audit_path_fragment,
        assessment_audit_path.parent
        / "exercises"
        / class_id
        / assessment_id
        / exercise_id
        / "audit.json",
    )
    existing = []
    seen: set[Path] = set()
    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            existing.append(path)
    if len(existing) != 1:
        raise SliceBuildError(
            f"Expected exactly one slice audit path for class={class_id} assessment={assessment_id} "
            f"exercise={exercise_id}, found {len(existing)}"
        )
    slice_audit_path = existing[0]
    return slice_audit_path, json.loads(slice_audit_path.read_text(encoding="utf-8"))


def _included_transition_rows(slice_audit: dict[str, object]) -> list[dict[str, object]]:
    rows = [row for row in slice_audit["transitions"] if row["status"] == "included"]
    if not rows:
        raise SliceBuildError("Slice audit contains no included transitions")
    return rows


def _validate_rebuilt_record(
    rebuilt: dict[str, object],
    audited: dict[str, object],
) -> None:
    keys = (
        "student_id",
        "transition_index_0idx",
        "attempt_n_index_0idx",
        "attempt_n1_index_0idx",
        "alignment_status",
        "first_change_line_0idx",
        "first_focus_region_3way",
        "lines_touched_bucket_3way",
        "next_test_outcome",
    )
    mismatches = [key for key in keys if rebuilt[key] != audited[key]]
    if mismatches:
        mismatch_text = ", ".join(
            f"{key}: rebuilt={rebuilt[key]!r} audited={audited[key]!r}" for key in mismatches
        )
        raise SliceBuildError(f"Rebuilt transition does not match audit record: {mismatch_text}")


def _build_slice_from_audit(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    assessment_title: str,
    slice_audit_path: Path,
    slice_audit: dict[str, object],
    data_root: Path,
    out_root: Path,
    max_cards: int,
) -> dict[str, object]:
    included_rows = _included_transition_rows(slice_audit)
    class_root = data_root / class_id
    if not class_root.exists():
        raise SliceBuildError(f"Class directory not found: {class_root}")

    assessment_meta = parse_assessment_file(class_root / "assessments" / f"{assessment_id}.data")
    if exercise_id not in assessment_meta["exercise_ids"]:
        raise SliceBuildError(
            f"Exercise {exercise_id} not listed in assessment {assessment_id} for class {class_id}"
        )

    rows_by_student: dict[str, list[dict[str, object]]] = {}
    for row in included_rows:
        rows_by_student.setdefault(str(row["student_id"]), []).append(row)

    windows: list[CohortWindow] = []
    window_records: list[dict[str, object]] = []
    for student_id, student_rows in sorted(rows_by_student.items()):
        exec_path = (
            class_root / "users" / student_id / "executions" / f"{assessment_id}_{exercise_id}.log"
        )
        cm_path = (
            class_root / "users" / student_id / "codemirror" / f"{assessment_id}_{exercise_id}.log"
        )
        attempts, trace, transitions = load_student_transition_context(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            exec_path=exec_path,
            cm_path=cm_path,
        )
        if trace is None:
            raise SliceBuildError(f"Included student has no replay trace: {student_id}")

        indexed_transitions = dict(enumerate(transitions))
        for audited_row in sorted(student_rows, key=lambda row: int(row["transition_index_0idx"])):
            transition_index = int(audited_row["transition_index_0idx"])
            if transition_index not in indexed_transitions:
                raise SliceBuildError(
                    f"Included audited transition missing from rebuilt transition list: "
                    f"student={student_id} transition_index_0idx={transition_index}"
                )
            window, rebuilt_record = build_transition_window_record(
                class_id=class_id,
                assessment_id=assessment_id,
                exercise_id=exercise_id,
                student_id=student_id,
                transition_index=transition_index,
                transition=indexed_transitions[transition_index],
                trace=trace,
                allow_partial=False,
            )
            _validate_rebuilt_record(rebuilt_record, audited_row)
            windows.append(window)
            window_records.append(rebuilt_record)

    if not windows:
        raise SliceBuildError("Audited slice rebuild produced zero windows")

    payload = build_payload(tuple(windows), max_cards=max_cards)
    payload["slice_header"]["assessment_title"] = assessment_title
    observed = build_observed_distributions(window_records)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(payload)

    slice_dir = out_root / class_id / assessment_id / exercise_id
    slice_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "v5_version": "v5",
        "class_id": class_id,
        "assessment_id": assessment_id,
        "exercise_id": exercise_id,
        "assessment_title": assessment_title,
        "alignment_policy": "full_match_only",
        "data_root": str(data_root),
        "candidate_student_count": int(slice_audit["slice_summary"]["candidate_student_count"]),
        "included_student_count": len({window.student_id for window in windows}),
        "excluded_student_count": int(slice_audit["slice_summary"]["excluded_student_count"]),
        "window_count": len(windows),
        "artifact_dir": str(slice_dir),
        "source_slice_audit_path": str(slice_audit_path.resolve()),
        "rebuilt_from_audited_included_transitions_only": True,
    }

    write_json(slice_dir / "manifest.json", manifest)
    write_json(slice_dir / "observed_labels.json", observed)
    write_json(slice_dir / "window_records.json", window_records)
    write_json(slice_dir / "payload.json", payload)
    write_text(slice_dir / "system_prompt.txt", system_prompt)
    write_text(slice_dir / "user_prompt.txt", user_prompt)

    return {
        "manifest": manifest,
        "payload": payload,
        "observed_labels": observed,
        "window_records": window_records,
        "paths": {
            "manifest": str(slice_dir / "manifest.json"),
            "observed_labels": str(slice_dir / "observed_labels.json"),
            "window_records": str(slice_dir / "window_records.json"),
            "payload": str(slice_dir / "payload.json"),
            "system_prompt": str(slice_dir / "system_prompt.txt"),
            "user_prompt": str(slice_dir / "user_prompt.txt"),
        },
    }


def build_benchmark_payloads(
    benchmark_manifest_path: Path,
    out_root: Path,
    data_root: Path | None,
    max_cards: int,
) -> dict[str, object]:
    if not benchmark_manifest_path.exists():
        raise SliceBuildError(f"Benchmark manifest not found: {benchmark_manifest_path}")

    manifest_payload = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    selected_slices = manifest_payload["selected_slices"]
    if not selected_slices:
        raise SliceBuildError("Benchmark manifest contains no selected_slices")

    source_assessment_audit_paths = tuple(
        Path(path) for path in manifest_payload.get("source_assessment_audit_paths", [])
    )

    out_root.mkdir(parents=True, exist_ok=False)
    built_rows: list[dict[str, object]] = []
    resolved_data_root = data_root if data_root is not None else DEFAULT_DATA_ROOT
    for row in selected_slices:
        slice_audit_path, slice_audit = _load_slice_audit(
            benchmark_manifest_path=benchmark_manifest_path,
            selected_row=row,
            source_assessment_audit_paths=source_assessment_audit_paths,
        )
        artifacts = _build_slice_from_audit(
            class_id=str(row["class_id"]),
            assessment_id=str(row["assessment_id"]),
            exercise_id=str(row["exercise_id"]),
            assessment_title=str(row["assessment_title"]),
            slice_audit_path=slice_audit_path,
            slice_audit=slice_audit,
            data_root=resolved_data_root,
            out_root=out_root,
            max_cards=max_cards,
        )
        built_rows.append(
            {
                "class_id": row["class_id"],
                "assessment_id": row["assessment_id"],
                "exercise_id": row["exercise_id"],
                "included_transitions": row["included_transitions"],
                "included_students": row["included_students"],
                "payload_path": artifacts["paths"]["payload"],
                "user_prompt_path": artifacts["paths"]["user_prompt"],
                "observed_labels_path": artifacts["paths"]["observed_labels"],
                "source_slice_audit_path": str(slice_audit_path.resolve()),
            }
        )

    batch_manifest = {
        "schema_version": "v5_benchmark_payload_batch_v1",
        "source_benchmark_manifest": str(benchmark_manifest_path),
        "slice_count": len(built_rows),
        "slices": built_rows,
    }
    _write_json(out_root / "batch_manifest.json", batch_manifest)
    return {
        "manifest": batch_manifest,
        "paths": {
            "out_root": str(out_root),
            "batch_manifest": str(out_root / "batch_manifest.json"),
        },
    }


def main() -> int:
    args = parse_args()
    result = build_benchmark_payloads(
        benchmark_manifest_path=args.benchmark_manifest,
        out_root=args.out_root,
        data_root=args.data_root,
        max_cards=args.max_cards,
    )
    print(json.dumps(result["manifest"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
