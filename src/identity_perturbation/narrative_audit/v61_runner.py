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

from .payload import V6PayloadError
from .trace_card import V6TraceCardError, build_attempt_trace_cards
from .v61_conditions import (
    FULL_V61_CONDITION,
    V61ConditionError,
    apply_condition_to_payload,
    condition_output_dirname,
    validate_condition,
)
from .v61_encoding_policy import V61EncodingDecision, V61EncodingStats, decide_encoding
from .v61_labels import V61LabelsError, build_observed_next_episode
from .v61_payload import V61PayloadError, build_v61_payload
from .v61_prompting import build_system_prompt, build_user_prompt

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_DATA_ROOT = ROOT.parent / "tracer" / "2024-1"
DEFAULT_OUT_ROOT = ROOT / "data" / "v61" / "transition_payloads"


class V61BuildError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one strict v6.1 transition payload bundle.")
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--assessment-id", required=True)
    parser.add_argument("--exercise-id", required=True)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--transition-index", required=True, type=int)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument(
        "--condition",
        default=FULL_V61_CONDITION,
        help=(
            "v6.1 condition to build, e.g. full_v61, no_trace, trace_shuffled_within_exercise, "
            "trace_shuffled_within_class_random, or trace_shuffled_cross_class_random"
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--idle-gap-seconds", type=float, default=30.0)
    parser.add_argument("--include-keyhandled", action="store_true")
    parser.add_argument("--exclude-navigation", action="store_true")
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
        raise V61BuildError(f"Class directory not found: {class_root}")
    student_root = class_root / "users" / student_id
    if not student_root.exists():
        raise V61BuildError(f"Student directory not found: {student_root}")
    exec_path = student_root / "executions" / f"{assessment_id}_{exercise_id}.log"
    cm_path = student_root / "codemirror" / f"{assessment_id}_{exercise_id}.log"
    if not exec_path.exists():
        raise V61BuildError(f"Execution log not found: {exec_path}")
    if not cm_path.exists():
        raise V61BuildError(f"CodeMirror log not found: {cm_path}")
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
    condition: str,
    shuffle_seed: int | None,
    idle_gap_seconds: float,
    include_keyhandled: bool,
    include_navigation: bool,
) -> dict[str, object]:
    try:
        validated_condition = validate_condition(condition)
    except V61ConditionError as exc:
        raise V61BuildError(str(exc)) from exc
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
        raise V61BuildError(str(exc)) from exc
    if trace is None:
        raise V61BuildError("Replay trace is missing; cannot build v6.1 payload bundle")
    if transition_index < 0 or transition_index >= len(transitions):
        raise V61BuildError(
            f"transition-index must be between 0 and {len(transitions) - 1}, got {transition_index}"
        )

    try:
        cm_events = parse_codemirror_log(cm_path)
        initial_code = infer_initial_code(cm_events, attempts[0].code)
        trace_cards = build_attempt_trace_cards(cm_events, initial_code=initial_code)
    except V6TraceCardError as exc:
        raise V61BuildError(f"v6.1 trace-card build failed: {exc}") from exc
    if len(trace_cards) != len(trace.snapshots):
        raise V61BuildError(
            "Trace-card count does not match replay snapshot count; strict v6.1 build aborted"
        )

    transition = transitions[transition_index]
    try:
        base_payload = build_v61_payload(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition_index=transition_index,
            transition=transition,
            trace=trace,
            trace_cards=trace_cards,
            codemirror_log_path=cm_path,
            idle_gap_seconds=idle_gap_seconds,
            include_keyhandled=include_keyhandled,
            include_navigation=include_navigation,
        )
    except (V61PayloadError, V6PayloadError, V6TraceCardError) as exc:  # type: ignore[name-defined]
        raise V61BuildError(f"v6.1 payload build failed: {exc}") from exc
    try:
        payload, condition_metadata = apply_condition_to_payload(
            base_payload,
            condition=validated_condition,
            data_root=data_root,
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition_index_0idx=transition_index,
            idle_gap_seconds=idle_gap_seconds,
            include_keyhandled=include_keyhandled,
            include_navigation=include_navigation,
            shuffle_seed=shuffle_seed,
        )
    except V61ConditionError as exc:
        raise V61BuildError(f"v6.1 condition application failed: {exc}") from exc
    try:
        observed_next_episode = build_observed_next_episode(
            class_id=class_id,
            assessment_id=assessment_id,
            exercise_id=exercise_id,
            student_id=student_id,
            transition_index=transition_index,
            transition=transition,
            trace=trace,
            trace_cards=trace_cards,
            codemirror_log_path=cm_path,
            idle_gap_seconds=idle_gap_seconds,
            include_keyhandled=include_keyhandled,
            include_navigation=include_navigation,
        )
    except V61LabelsError as exc:
        raise V61BuildError(f"v6.1 observed next-episode build failed: {exc}") from exc

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(payload)
    semantic_tape = payload["attempt_n"]["semantic_tape"]
    semantic_events = []
    semantic_summary = {
        "raw_interval_event_count": 0,
        "semantic_event_count": 0,
        "change_event_count": 0,
        "saida_testar_count": 0,
        "submit_count": 0,
        "kill_program_count": 0,
        "idle_gap_count": 0,
    }
    if (
        isinstance(semantic_tape, dict)
        and semantic_tape.get("schema_version") == "v6_1_attempt_semantic_tape_v1"
    ):
        semantic_events = semantic_tape["semantic_event_tape"]
        semantic_summary = semantic_tape["semantic_tape_summary"]
    policy_stats = V61EncodingStats(
        prompt_chars=len(user_prompt),
        raw_interval_event_count=int(semantic_summary["raw_interval_event_count"]),
        event_count=int(semantic_summary["semantic_event_count"]),
        change_count=int(semantic_summary["change_event_count"]),
        run_count=int(semantic_summary["saida_testar_count"]),
        submit_count=int(semantic_summary["submit_count"]),
        kill_program_count=int(semantic_summary["kill_program_count"]),
        idle_gap_count=int(semantic_summary["idle_gap_count"]),
        navigation_count=sum(1 for event in semantic_events if event["event_type"] == "tab_click"),
        stdout_lines=sum(
            len(event.get("output_lines", []))
            for event in semantic_events
            if event["event_type"] == "saida_testar"
        ),
        used_full_stdout=all(
            event.get("output_line_limit") is None
            for event in semantic_events
            if event["event_type"] == "saida_testar"
        ),
    )
    if (
        isinstance(semantic_tape, dict)
        and semantic_tape.get("schema_version") == "v6_1_attempt_semantic_tape_v1"
    ):
        policy_decision = decide_encoding(policy_stats)
    else:
        policy_decision = V61EncodingDecision(
            case_type="withheld_trace",
            keep_full_event_log=False,
            flagged_case=False,
            flags=("trace_withheld",),
        )

    condition_dirname = condition_output_dirname(validated_condition, shuffle_seed)
    bundle_root = out_root if condition_dirname is None else out_root / condition_dirname
    bundle_dir = (
        bundle_root / class_id / assessment_id / exercise_id / student_id / str(transition_index)
    )
    bundle_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": "v6_1_transition_bundle_v1",
        "v6_version": "v6.1",
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
        "condition": validated_condition,
        "shuffle_seed": shuffle_seed,
        "idle_gap_seconds": idle_gap_seconds,
        "include_keyhandled": include_keyhandled,
        "include_navigation": include_navigation,
        "batch_endpoint": "/v1/responses",
        "trace_card_count": len(trace_cards),
        "attempt_count": len(attempts),
        "observed_next_episode_path": str((bundle_dir / "observed_next_episode.json").resolve()),
        "encoding_policy": {
            "prompt_chars": policy_stats.prompt_chars,
            "raw_interval_event_count": policy_stats.raw_interval_event_count,
            "event_count": policy_stats.event_count,
            "change_count": policy_stats.change_count,
            "run_count": policy_stats.run_count,
            "submit_count": policy_stats.submit_count,
            "kill_program_count": policy_stats.kill_program_count,
            "idle_gap_count": policy_stats.idle_gap_count,
            "navigation_count": policy_stats.navigation_count,
            "stdout_lines": policy_stats.stdout_lines,
            "used_full_stdout": policy_stats.used_full_stdout,
            "case_type": policy_decision.case_type,
            "keep_full_event_log": policy_decision.keep_full_event_log,
            "flagged_case": policy_decision.flagged_case,
            "flags": list(policy_decision.flags),
        },
    }
    if condition_metadata is not None:
        manifest["condition_metadata"] = condition_metadata

    write_json(bundle_dir / "manifest.json", manifest)
    write_json(bundle_dir / "payload.json", payload)
    write_json(bundle_dir / "observed_next_episode.json", observed_next_episode)
    write_text(bundle_dir / "system_prompt.txt", system_prompt)
    write_text(bundle_dir / "user_prompt.txt", user_prompt)

    return {
        "manifest": manifest,
        "payload": payload,
        "paths": {
            "manifest": str(bundle_dir / "manifest.json"),
            "payload": str(bundle_dir / "payload.json"),
            "observed_next_episode": str(bundle_dir / "observed_next_episode.json"),
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
        shuffle_seed=args.shuffle_seed,
        idle_gap_seconds=args.idle_gap_seconds,
        include_keyhandled=args.include_keyhandled,
        include_navigation=not args.exclude_navigation,
    )
    print(json.dumps(artifacts["paths"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
