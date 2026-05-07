from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_perturbation.codebench_support.codemirror import infer_initial_code, parse_codemirror_log
from identity_perturbation.codebench_support.runner import (
    ROOT,
    SliceBuildError,
    load_student_transition_context,
    write_json,
    write_text,
)

from .conditions import (
    FULL_V6_CONDITION,
    TRACE_SHUFFLED_WITHIN_EXERCISE_CONDITION,
    V6ConditionError,
    apply_condition_to_payload,
    validate_condition,
)
from .labels import build_observed_labels
from .payload import V6PayloadError, build_payload
from .prompting import build_system_prompt, build_user_prompt
from .trace_card import V6TraceCardError, build_attempt_trace_cards, trace_card_to_prompt_dict
from .trace_shuffle import V6TraceShuffleError, load_within_exercise_shuffled_trace_cards

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_DATA_ROOT = ROOT.parent / "tracer" / "2024-1"
DEFAULT_OUT_ROOT = ROOT / "data" / "v6" / "transition_payloads"


class V6BuildError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one strict v6 transition payload bundle.")
    parser.add_argument("--class-id", required=True, help="CodeBench class id")
    parser.add_argument("--assessment-id", required=True, help="Assessment id")
    parser.add_argument("--exercise-id", required=True, help="Exercise id")
    parser.add_argument("--student-id", required=True, help="Student id")
    parser.add_argument(
        "--transition-index",
        required=True,
        type=int,
        help="0-indexed execution transition index for attempt_n -> attempt_n+1",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Raw dataset root containing 2024-1/<class_id>/...",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Artifact root for v6 transition payload outputs",
    )
    parser.add_argument(
        "--condition",
        default=FULL_V6_CONDITION,
        help="v6 condition to build, e.g. full_v6 or no_trace",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    return parser.parse_args()


def _resolve_student_paths(
    *,
    data_root: Path,
    class_id: str,
    student_id: str,
    assessment_id: str,
    exercise_id: str,
) -> tuple[Path, Path]:
    class_root = data_root / class_id
    if not class_root.exists():
        raise V6BuildError(f"Class directory not found: {class_root}")

    student_root = class_root / "users" / student_id
    if not student_root.exists():
        raise V6BuildError(f"Student directory not found: {student_root}")

    exec_path = student_root / "executions" / f"{assessment_id}_{exercise_id}.log"
    cm_path = student_root / "codemirror" / f"{assessment_id}_{exercise_id}.log"
    if not exec_path.exists():
        raise V6BuildError(f"Execution log not found: {exec_path}")
    if not cm_path.exists():
        raise V6BuildError(f"CodeMirror log not found: {cm_path}")
    return exec_path, cm_path


def build_transition_payload_bundle(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index: int,
    data_root: Path,
    out_root: Path,
    model: str,
    reasoning_effort: str,
    condition: str = FULL_V6_CONDITION,
) -> dict[str, object]:
    try:
        validated_condition = validate_condition(condition)
    except V6ConditionError as exc:
        raise V6BuildError(str(exc)) from exc
    exec_path, cm_path = _resolve_student_paths(
        data_root=data_root,
        class_id=class_id,
        student_id=student_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
    )

    try:
        attempts, trace, transitions = load_student_transition_context(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            exec_path=exec_path,
            cm_path=cm_path,
        )
    except SliceBuildError as exc:
        raise V6BuildError(str(exc)) from exc

    if trace is None:
        raise V6BuildError("Replay trace is missing; cannot build v6 payload bundle")
    if transition_index < 0 or transition_index >= len(transitions):
        raise V6BuildError(
            f"transition-index must be between 0 and {len(transitions) - 1}, got {transition_index}"
        )

    try:
        cm_events = parse_codemirror_log(cm_path)
        initial_code = infer_initial_code(cm_events, attempts[0].code)
        trace_cards = build_attempt_trace_cards(cm_events, initial_code=initial_code)
    except V6TraceCardError as exc:
        raise V6BuildError(f"v6 trace-card build failed: {exc}") from exc
    if len(trace_cards) != len(trace.snapshots):
        raise V6BuildError(
            "Trace-card count does not match replay snapshot count; strict v6 build aborted"
        )

    transition = transitions[transition_index]
    try:
        base_payload = build_payload(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition=transition,
            trace=trace,
            trace_cards=trace_cards,
        )
        condition_metadata: dict[str, object] | None = None
        shuffled_trace_cards = None
        if validated_condition == TRACE_SHUFFLED_WITHIN_EXERCISE_CONDITION:
            donor_cards, condition_metadata = load_within_exercise_shuffled_trace_cards(
                data_root=data_root,
                class_id=class_id,
                assessment_id=assessment_id,
                exercise_id=exercise_id,
                target_student_id=student_id,
                visible_attempt_count=len(transition.history) + 1,
            )
            shuffled_trace_cards = [trace_card_to_prompt_dict(card) for card in donor_cards]
        payload = apply_condition_to_payload(
            base_payload,
            condition=validated_condition,
            shuffled_trace_cards=shuffled_trace_cards,
            condition_metadata=condition_metadata,
        )
    except (V6PayloadError, V6TraceCardError, V6ConditionError, V6TraceShuffleError) as exc:
        raise V6BuildError(f"v6 payload build failed: {exc}") from exc

    observed_labels = build_observed_labels(
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        student_id=student_id,
        transition_index=transition_index,
        transition=transition,
        trace=trace,
    )
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(payload)

    bundle_root = (
        out_root if validated_condition == FULL_V6_CONDITION else out_root / validated_condition
    )
    bundle_dir = (
        bundle_root / class_id / assessment_id / exercise_id / student_id / str(transition_index)
    )
    bundle_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "v6_version": "v6",
        "condition": validated_condition,
        "class_id": class_id,
        "assessment_id": assessment_id,
        "exercise_id": exercise_id,
        "student_id": student_id,
        "transition_index_0idx": transition_index,
        "attempt_n_index_0idx": transition.attempt_n.attempt_index_0idx,
        "attempt_n1_index_0idx": transition.attempt_n1.attempt_index_0idx,
        "artifact_dir": str(bundle_dir.resolve()),
        "data_root": str(data_root.resolve()),
        "exec_path": str(exec_path.resolve()),
        "codemirror_path": str(cm_path.resolve()),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "batch_endpoint": "/v1/responses",
        "trace_card_count": len(trace_cards),
        "attempt_count": len(attempts),
        "observed_labels_path": str((bundle_dir / "observed_labels.json").resolve()),
    }
    if "condition_metadata" in payload:
        manifest["condition_metadata"] = payload["condition_metadata"]

    write_json(bundle_dir / "manifest.json", manifest)
    write_json(bundle_dir / "payload.json", payload)
    write_json(bundle_dir / "observed_labels.json", observed_labels)
    write_text(bundle_dir / "system_prompt.txt", system_prompt)
    write_text(bundle_dir / "user_prompt.txt", user_prompt)

    return {
        "manifest": manifest,
        "payload": payload,
        "paths": {
            "manifest": str(bundle_dir / "manifest.json"),
            "payload": str(bundle_dir / "payload.json"),
            "observed_labels": str(bundle_dir / "observed_labels.json"),
            "system_prompt": str(bundle_dir / "system_prompt.txt"),
            "user_prompt": str(bundle_dir / "user_prompt.txt"),
        },
    }


def main() -> int:
    args = parse_args()
    artifacts = build_transition_payload_bundle(
        class_id=args.class_id,
        assessment_id=args.assessment_id,
        exercise_id=args.exercise_id,
        student_id=args.student_id,
        transition_index=args.transition_index,
        data_root=args.data_root,
        out_root=args.out_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        condition=args.condition,
    )
    print(json.dumps(artifacts["paths"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
