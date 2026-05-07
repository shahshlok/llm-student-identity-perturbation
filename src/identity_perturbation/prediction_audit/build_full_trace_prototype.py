from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from identity_perturbation.codebench_support.codemirror import _code_to_lines, apply_change, infer_initial_code
from identity_perturbation.codebench_support.executions import next_test_outcome, parse_execution_log_text
from identity_perturbation.codebench_support.payload import failed_test_indices_0idx, pass_count
from identity_perturbation.prediction_audit.coarse_path import build_observed_coarse_path_artifact
from identity_perturbation.prediction_audit.full_trace_prompting import (
    SUPPORTED_CONDITIONS,
    build_model_visible_payload,
    build_system_prompt,
    build_user_prompt,
    structured_text_format,
    validate_condition,
)
from identity_perturbation.prediction_audit.full_trace_target_schema import FullTracePredictionResponse
from identity_perturbation.prediction_audit.full_trace_targets import (
    build_observed_repair_target,
)
from identity_perturbation.prediction_audit.match_policy import (
    candidate_snapshot_indices_narrow_normalized,
    count_monotonic_solutions,
    matching_indices_after,
    narrow_normalize_code_for_match,
    resolve_unique_monotonic_alignment,
)
from identity_perturbation.prediction_audit.raw_same_task_family_audit import (
    RawTraceEvent,
    cm_events_from_raw_events,
    parse_assessment_data,
    parse_raw_trace_events_with_output,
    read_text_strict,
)

DEFAULT_DATA_ROOT = Path("2024-1")
DEFAULT_OUT_ROOT = Path("data/v62/prototypes/full_trace_v3")
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
ALIGNMENT_POLICY = (
    "visible_narrow_normalized_unique_monotonic__"
    "target_immediate_next_submit_block_narrow_normalized_unique__"
    "coarse_path_requires_real_edit_and_final_submit"
)


class V62FullTracePrototypeError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one exact v6.2 full-trace prototype bundle from raw logs."
    )
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--assessment-id", required=True)
    parser.add_argument("--exercise-id", required=True)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--transition-index", required=True, type=int)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--condition", choices=SUPPORTED_CONDITIONS, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    return parser.parse_args()


def _resolve_paths(
    *,
    data_root: Path,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
) -> tuple[Path, Path, Path]:
    class_root = data_root / class_id
    if not class_root.exists():
        raise V62FullTracePrototypeError(f"Class directory not found: {class_root}")
    assessment_path = class_root / "assessments" / f"{assessment_id}.data"
    if not assessment_path.exists():
        raise V62FullTracePrototypeError(f"Assessment file not found: {assessment_path}")
    user_root = class_root / "users" / student_id
    if not user_root.exists():
        raise V62FullTracePrototypeError(f"Student directory not found: {user_root}")
    execution_path = user_root / "executions" / f"{assessment_id}_{exercise_id}.log"
    codemirror_path = user_root / "codemirror" / f"{assessment_id}_{exercise_id}.log"
    if not execution_path.exists():
        raise V62FullTracePrototypeError(f"Execution log not found: {execution_path}")
    if not codemirror_path.exists():
        raise V62FullTracePrototypeError(f"CodeMirror log not found: {codemirror_path}")
    return assessment_path, execution_path, codemirror_path


def _test_result_to_dict(result: Any) -> dict[str, object]:
    return {
        "test_index_0idx": result.test_index_0idx,
        "passed": result.passed,
        "input": result.input,
        "expected": result.expected,
        "actual": result.actual,
    }


def _attempt_execution_result(attempt: Any) -> dict[str, object]:
    return {
        "timestamp": attempt.timestamp,
        "grade": attempt.grade,
        "pass_count": pass_count(attempt),
        "failed_test_indices_0idx": list(failed_test_indices_0idx(attempt)),
        "test_outcome": next_test_outcome(attempt),
        "test_results": [_test_result_to_dict(result) for result in attempt.test_results],
    }


def _change_event_to_dict(event: RawTraceEvent) -> dict[str, object]:
    if not isinstance(event.payload, dict):
        raise V62FullTracePrototypeError(
            f"Change payload is not a dict at raw line {event.line_number}"
        )
    payload = event.payload
    text = payload["text"]
    removed = payload["removed"]
    origin = payload["origin"]
    if not isinstance(text, list) or not all(isinstance(item, str) for item in text):
        raise V62FullTracePrototypeError(
            f"Change text is not list[str] at raw line {event.line_number}"
        )
    if not isinstance(removed, list) or not all(isinstance(item, str) for item in removed):
        raise V62FullTracePrototypeError(
            f"Change removed is not list[str] at raw line {event.line_number}"
        )
    if not isinstance(origin, str) or not origin:
        raise V62FullTracePrototypeError(
            f"Change origin is missing or invalid at raw line {event.line_number}"
        )
    return {
        "event_type": "change",
        "timestamp": event.timestamp.isoformat(),
        "line_number": event.line_number,
        "from": {
            "line": int(payload["from"]["line"]),
            "ch": int(payload["from"]["ch"]),
        },
        "to": {
            "line": int(payload["to"]["line"]),
            "ch": int(payload["to"]["ch"]),
        },
        "text": text,
        "removed": removed,
        "origin": origin,
    }


def _edit_segment_item(change_events: list[RawTraceEvent]) -> dict[str, object]:
    if not change_events:
        raise V62FullTracePrototypeError("Cannot build an edit segment from zero change events")
    return {
        "item_type": "edit_segment",
        "change_event_count": len(change_events),
        "changes": [_change_event_to_dict(event) for event in change_events],
    }


def _saida_testar_item(event: RawTraceEvent, code_after_anchor: str) -> dict[str, object]:
    if event.raw_type != "saida_testar":
        raise V62FullTracePrototypeError(
            f"Expected saida_testar event, got {event.raw_type!r} at raw line {event.line_number}"
        )
    return {
        "item_type": "saida_testar",
        "timestamp": event.timestamp.isoformat(),
        "line_number": event.line_number,
        "command": str(event.payload),
        "output_lines": event.output_text.splitlines(),
        "code_after_anchor": code_after_anchor,
    }


def _submit_item(event: RawTraceEvent, code_after_anchor: str) -> dict[str, object]:
    if event.raw_type != "submit":
        raise V62FullTracePrototypeError(
            f"Expected submit event, got {event.raw_type!r} at raw line {event.line_number}"
        )
    return {
        "item_type": "submit",
        "timestamp": event.timestamp.isoformat(),
        "line_number": event.line_number,
        "feedback": str(event.payload),
        "code_after_anchor": code_after_anchor,
    }


def _reconstruct_submit_bounded_blocks(
    raw_events: tuple[RawTraceEvent, ...],
    *,
    initial_code: str,
) -> tuple[dict[str, object], ...]:
    lines = _code_to_lines(initial_code)
    attempt_start_code = "\n".join(lines)
    pending_changes: list[RawTraceEvent] = []
    pending_items: list[dict[str, object]] = []
    blocks: list[dict[str, object]] = []
    submit_index = 0

    for event in raw_events:
        if event.raw_type == "change":
            if not isinstance(event.payload, dict):
                raise V62FullTracePrototypeError(
                    f"Change payload is not a dict at raw line {event.line_number}"
                )
            lines = apply_change(lines, event.payload)
            pending_changes.append(event)
            continue
        if event.raw_type == "saida_testar":
            if pending_changes:
                pending_items.append(_edit_segment_item(pending_changes))
                pending_changes = []
            pending_items.append(_saida_testar_item(event, "\n".join(lines)))
            continue
        if event.raw_type == "submit":
            if pending_changes:
                pending_items.append(_edit_segment_item(pending_changes))
                pending_changes = []
            pending_items.append(_submit_item(event, "\n".join(lines)))
            blocks.append(
                {
                    "submit_index_0idx": submit_index,
                    "trace": {
                        "attempt_start_code": attempt_start_code,
                        "items": pending_items,
                    },
                }
            )
            submit_index += 1
            attempt_start_code = "\n".join(lines)
            pending_items = []
            continue

    if not blocks:
        raise V62FullTracePrototypeError("CodeMirror log produced no submit-bounded blocks")
    return tuple(blocks)


def _build_prompt_payload(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index: int,
    data_root: Path,
    condition: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    validated_condition = validate_condition(condition)
    assessment_path, execution_path, codemirror_path = _resolve_paths(
        data_root=data_root,
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        student_id=student_id,
    )
    assessment_spec = parse_assessment_data(assessment_path)
    if exercise_id not in assessment_spec.exercise_ids:
        raise V62FullTracePrototypeError(
            f"Exercise {exercise_id} is not declared in assessment {assessment_id}"
        )

    attempts = parse_execution_log_text(read_text_strict(execution_path))
    if transition_index < 0 or transition_index >= len(attempts) - 1:
        raise V62FullTracePrototypeError(
            f"transition-index must be between 0 and {len(attempts) - 2}, got {transition_index}"
        )

    raw_events = parse_raw_trace_events_with_output(
        read_text_strict(codemirror_path), codemirror_path
    )
    cm_events = cm_events_from_raw_events(raw_events)
    initial_code = infer_initial_code(cm_events, attempts[0].code)
    blocks = _reconstruct_submit_bounded_blocks(raw_events, initial_code=initial_code)
    snapshot_codes = tuple(
        block["trace"]["items"][-1]["code_after_anchor"]  # type: ignore[index]
        for block in blocks
    )
    visible_attempts = attempts[: transition_index + 1]
    candidates = candidate_snapshot_indices_narrow_normalized(
        attempt_codes=tuple(attempt.code for attempt in visible_attempts),
        snapshot_codes=snapshot_codes,
    )
    if not all(candidates):
        raise V62FullTracePrototypeError(
            "At least one visible attempt had no narrow-normalized matching submit snapshot in the CodeMirror replay"
        )
    if count_monotonic_solutions(candidates) != 1:
        raise V62FullTracePrototypeError(
            "Visible attempts did not have a unique narrow-normalized monotonic alignment to submit snapshots"
        )
    alignment = resolve_unique_monotonic_alignment(candidates)

    prompt_attempts: list[dict[str, object]] = []
    for attempt, snapshot_index in zip(visible_attempts, alignment, strict=True):
        block = blocks[snapshot_index]
        submit_item = block["trace"]["items"][-1]  # type: ignore[index]
        if submit_item["item_type"] != "submit":
            raise V62FullTracePrototypeError(
                f"Last item for submit-bounded block {snapshot_index} is not a submit item"
            )
        if narrow_normalize_code_for_match(
            str(submit_item["code_after_anchor"])
        ) != narrow_normalize_code_for_match(attempt.code):
            raise V62FullTracePrototypeError(
                f"Narrow-normalized aligned submit snapshot did not match execution code for attempt {attempt.attempt_index_0idx}"
            )
        prompt_attempts.append(
            {
                "attempt_index_0idx": attempt.attempt_index_0idx,
                "aligned_submit_index_0idx": snapshot_index,
                "execution_result": _attempt_execution_result(attempt),
                "submitted_code": attempt.code,
                "trace": block["trace"],
            }
        )

    target_attempt = attempts[transition_index + 1]
    target_snapshot_index = alignment[-1] + 1
    if target_snapshot_index >= len(blocks):
        raise V62FullTracePrototypeError(
            "Target attempt n+1 had no following submit-bounded CodeMirror block"
        )
    target_block = blocks[target_snapshot_index]
    target_submit_item = target_block["trace"]["items"][-1]  # type: ignore[index]
    if target_submit_item["item_type"] != "submit":
        raise V62FullTracePrototypeError(
            f"Last item for target submit-bounded block {target_snapshot_index} is not a submit item"
        )
    normalized_target_code = narrow_normalize_code_for_match(target_attempt.code)
    target_submit_code = str(target_submit_item["code_after_anchor"])
    if narrow_normalize_code_for_match(target_submit_code) != normalized_target_code:
        raise V62FullTracePrototypeError(
            f"Narrow-normalized aligned submit snapshot did not match execution code for target attempt {target_attempt.attempt_index_0idx}"
        )
    matching_after_prefix = matching_indices_after(
        snapshot_codes=snapshot_codes,
        normalized_target_code=normalized_target_code,
        start_index=alignment[-1],
    )
    if matching_after_prefix != (target_snapshot_index,):
        raise V62FullTracePrototypeError(
            f"Target attempt {target_attempt.attempt_index_0idx} did not have a unique immediate next narrow-normalized submit block"
        )

    raw_prompt_payload = {
        "schema_version": "v6_2_full_trace_prompt_payload_v2",
        "source": {
            "class_id": class_id,
            "assessment_id": assessment_id,
            "assessment_title": assessment_spec.title,
            "exercise_id": exercise_id,
            "student_id": student_id,
            "condition": validated_condition,
            "visible_transition_index_0idx": transition_index,
            "visible_attempt_indices_0idx": [
                attempt.attempt_index_0idx for attempt in visible_attempts
            ],
            "prediction_target_attempt_index_0idx": target_attempt.attempt_index_0idx,
            "alignment_policy": ALIGNMENT_POLICY,
            "execution_log_path": str(execution_path.resolve()),
            "codemirror_log_path": str(codemirror_path.resolve()),
        },
        "visible_attempts": prompt_attempts,
        "prediction_target": {
            "attempt_index_0idx": target_attempt.attempt_index_0idx,
            "task": "Predict the next submitted attempt after the visible history.",
        },
    }
    payload = build_model_visible_payload(raw_prompt_payload)
    observed_next_repair_target = build_observed_repair_target(
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        student_id=student_id,
        transition_index_0idx=transition_index,
        attempt_n=visible_attempts[-1],
        attempt_n1=target_attempt,
    )
    observed_next_coarse_path = build_observed_coarse_path_artifact(
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        student_id=student_id,
        transition_index_0idx=transition_index,
        attempt_index_0idx=target_attempt.attempt_index_0idx,
        aligned_submit_index_0idx=target_snapshot_index,
        trace=target_block["trace"],  # type: ignore[arg-type]
    )
    coarse_steps = observed_next_coarse_path["attempt_n1"]["coarse_path_steps"]  # type: ignore[index]
    if not isinstance(coarse_steps, list) or not coarse_steps:
        raise V62FullTracePrototypeError("Observed next coarse path is missing coarse_path_steps")
    if not any(
        isinstance(step, dict) and step.get("action_type") == "edit" for step in coarse_steps[:-1]
    ):
        raise V62FullTracePrototypeError(
            f"Target attempt {target_attempt.attempt_index_0idx} has no real pre-submit edit in observed coarse path"
        )
    return (
        payload,
        observed_next_repair_target,
        observed_next_coarse_path,
    )


def _build_request_body(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, object]:
    body: dict[str, object] = {
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


def main() -> int:
    args = parse_args()
    (
        payload,
        observed_next_repair_target,
        observed_next_coarse_path,
    ) = _build_prompt_payload(
        class_id=args.class_id,
        assessment_id=args.assessment_id,
        exercise_id=args.exercise_id,
        student_id=args.student_id,
        transition_index=args.transition_index,
        data_root=args.data_root,
        condition=args.condition,
    )

    system_prompt = build_system_prompt(payload["source"]["condition"])
    user_prompt = build_user_prompt(payload)
    response_schema = FullTracePredictionResponse.model_json_schema()
    request_body = _build_request_body(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    custom_id = f"{args.class_id}:{args.assessment_id}:{args.exercise_id}:{args.student_id}:{args.transition_index}"
    batch_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": request_body,
    }

    bundle_dir = (
        args.out_root
        / args.condition
        / args.class_id
        / args.assessment_id
        / args.exercise_id
        / args.student_id
        / str(args.transition_index)
    )
    bundle_dir.mkdir(parents=True, exist_ok=False)

    system_prompt_path = bundle_dir / "system_prompt.txt"
    user_payload_path = bundle_dir / "user_payload.json"
    user_prompt_path = bundle_dir / "user_prompt.txt"
    response_schema_path = bundle_dir / "response_schema.json"
    request_body_path = bundle_dir / "request_body.json"
    batch_request_path = bundle_dir / "batch_request.json"
    batch_request_jsonl_path = bundle_dir / "requests.jsonl"
    observed_next_repair_target_path = bundle_dir / "observed_next_repair_target.json"
    observed_next_coarse_path_path = bundle_dir / "observed_next_coarse_path.json"
    manifest_path = bundle_dir / "manifest.json"

    system_prompt_path.write_text(system_prompt, encoding="utf-8")
    user_payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    user_prompt_path.write_text(user_prompt, encoding="utf-8")
    response_schema_path.write_text(
        json.dumps(response_schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    request_body_path.write_text(
        json.dumps(request_body, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    batch_request_path.write_text(
        json.dumps(batch_request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    batch_request_jsonl_path.write_text(
        json.dumps(batch_request, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    observed_next_repair_target_path.write_text(
        json.dumps(observed_next_repair_target, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    observed_next_coarse_path_path.write_text(
        json.dumps(observed_next_coarse_path, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    def rel(path: Path) -> str:
        return path.relative_to(bundle_dir).as_posix()

    manifest = {
        "schema_version": "v6_2_full_trace_prototype_bundle_v6",
        "custom_id": custom_id,
        "condition": args.condition,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "alignment_policy": ALIGNMENT_POLICY,
        "system_prompt_path": rel(system_prompt_path),
        "user_payload_path": rel(user_payload_path),
        "user_prompt_path": rel(user_prompt_path),
        "response_schema_path": rel(response_schema_path),
        "request_body_path": rel(request_body_path),
        "batch_request_path": rel(batch_request_path),
        "batch_request_jsonl_path": rel(batch_request_jsonl_path),
        "observed_next_repair_target_path": rel(observed_next_repair_target_path),
        "observed_next_coarse_path_path": rel(observed_next_coarse_path_path),
        "prompt_char_count": len(user_prompt),
        "visible_attempt_count": len(payload["visible_attempts"]),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(bundle_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
