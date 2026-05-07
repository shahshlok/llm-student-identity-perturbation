from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Any

from identity_perturbation.prediction_audit.full_trace_target_schema import FullTracePredictionResponse

FULL_TRACE_CONDITION = "full"
NO_TRACE_CONDITION = "no_trace"
SUPPORTED_CONDITIONS = (FULL_TRACE_CONDITION, NO_TRACE_CONDITION)


FULL_SYSTEM_PROMPT_V62 = """You are predicting the next submitted attempt by one beginner Python student on CodeBench.

Your job is to predict what THIS student is likely to submit next, not the best or cleanest fix.

Important rules:
- Work only from the visible evidence up to and including attempt n.
- Use the CodeMirror trace as behavioral evidence.
- The next attempt may still be wrong.
- Preserve the student's apparent naming, style, and strategy when plausible.
- Return exactly 3 hypotheses.
- The hypothesis probabilities must sum to 1.0 exactly.
- Return only valid JSON matching the provided schema.
- Do not use markdown, code fences, comments, or prose outside the JSON object.
"""


NO_TRACE_SYSTEM_PROMPT_V62 = """You are predicting the next submitted attempt by one beginner Python student on CodeBench.

Your job is to predict what THIS student is likely to submit next, not the best or cleanest fix.

Important rules:
- Work only from the visible evidence up to and including attempt n.
- Rely on submitted code and execution results only.
- The next attempt may still be wrong.
- Preserve the student's apparent naming, style, and strategy when plausible.
- Return exactly 3 hypotheses.
- The hypothesis probabilities must sum to 1.0 exactly.
- Return only valid JSON matching the provided schema.
- Do not use markdown, code fences, comments, or prose outside the JSON object.
"""


def build_system_prompt(condition: object) -> str:
    validated_condition = validate_condition(condition)
    if validated_condition == FULL_TRACE_CONDITION:
        return FULL_SYSTEM_PROMPT_V62
    if validated_condition == NO_TRACE_CONDITION:
        return NO_TRACE_SYSTEM_PROMPT_V62
    raise AssertionError(f"Unhandled v6.2 condition: {validated_condition}")


def validate_condition(condition: object) -> str:
    if not isinstance(condition, str):
        raise ValueError(f"Condition must be a string, got {type(condition).__name__}")
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(
            f"Unsupported v6.2 condition {condition!r}; expected one of {list(SUPPORTED_CONDITIONS)}"
        )
    return condition


def _payload_source(payload: dict[str, object]) -> dict[str, Any]:
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("Payload source must be a dict")
    return source


def _payload_condition(payload: dict[str, object]) -> str:
    source = _payload_source(payload)
    return validate_condition(source.get("condition"))


def _conditioned_attempt_common(attempt: dict[str, Any]) -> dict[str, Any]:
    required = ("attempt_index_0idx", "aligned_submit_index_0idx", "execution_result")
    missing = [key for key in required if key not in attempt]
    if missing:
        raise ValueError(f"Visible attempt missing required keys: {missing}")
    execution_result = attempt["execution_result"]
    if not isinstance(execution_result, dict):
        raise ValueError("Visible attempt execution_result must be a dict")
    return {
        "attempt_index_0idx": attempt["attempt_index_0idx"],
        "aligned_submit_index_0idx": attempt["aligned_submit_index_0idx"],
        "execution_result": deepcopy(execution_result),
    }


def build_model_visible_payload(payload: dict[str, object]) -> dict[str, object]:
    """Project a canonical v6.2 payload into the exact model-visible condition view."""
    required_keys = {"schema_version", "source", "visible_attempts", "prediction_target"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Payload missing required keys: {sorted(missing)}")

    source = _payload_source(payload)
    condition = validate_condition(source.get("condition"))
    visible_attempts = payload["visible_attempts"]
    prediction_target = payload["prediction_target"]
    if not isinstance(visible_attempts, list):
        raise ValueError("Payload visible_attempts must be a list")
    if not isinstance(prediction_target, dict):
        raise ValueError("Payload prediction_target must be a dict")

    conditioned_attempts: list[dict[str, Any]] = []
    for attempt in visible_attempts:
        if not isinstance(attempt, dict):
            raise ValueError("Payload visible_attempts entries must be dicts")
        conditioned_attempt = _conditioned_attempt_common(attempt)
        if condition == FULL_TRACE_CONDITION:
            trace = attempt.get("trace")
            if not isinstance(trace, dict):
                raise ValueError("Full-condition visible attempt must include a trace dict")
            conditioned_attempt["trace"] = deepcopy(trace)
        elif condition == NO_TRACE_CONDITION:
            submitted_code = attempt.get("submitted_code")
            if not isinstance(submitted_code, str):
                raise ValueError("No-trace visible attempt must include submitted_code as a string")
            conditioned_attempt["submitted_code"] = submitted_code
        else:
            raise AssertionError(f"Unhandled v6.2 condition: {condition}")
        conditioned_attempts.append(conditioned_attempt)

    return {
        "schema_version": payload["schema_version"],
        "source": deepcopy(source),
        "visible_attempts": conditioned_attempts,
        "prediction_target": deepcopy(prediction_target),
    }


def _parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _attempt_base_timestamp(trace: dict[str, Any]) -> datetime:
    items = trace["items"]
    if not items:
        raise ValueError("Attempt trace has no items")
    first_item = items[0]
    if first_item["item_type"] == "edit_segment":
        changes = first_item["changes"]
        if not changes:
            raise ValueError("Edit segment has no changes")
        return _parse_iso_timestamp(str(changes[0]["timestamp"]))
    return _parse_iso_timestamp(str(first_item["timestamp"]))


def _delta_ms(*, event_timestamp: str, base_timestamp: datetime) -> int:
    return int(
        round((_parse_iso_timestamp(event_timestamp) - base_timestamp).total_seconds() * 1000.0)
    )


def _json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _render_execution_result(result: dict[str, Any]) -> str:
    test_lines: list[str] = []
    for test in result["test_results"]:
        status = "PASS" if test["passed"] else "FAIL"
        test_lines.append(
            "\n".join(
                [
                    f"  - test_index_0idx: {test['test_index_0idx']} [{status}]",
                    f"    input: {_json_cell(test['input'])}",
                    f"    expected: {_json_cell(test['expected'])}",
                    f"    actual: {_json_cell(test['actual'])}",
                ]
            )
        )
    return "\n".join(
        [
            f"- submitted_at: {result['timestamp']}",
            f"- grade: {result['grade']}",
            f"- pass_count: {result['pass_count']}",
            f"- failed_test_indices_0idx: {_json_cell(result['failed_test_indices_0idx'])}",
            f"- test_outcome: {result['test_outcome']}",
            "- test_results:",
            "\n".join(test_lines),
        ]
    )


def _render_code_block(title: str, code: str) -> str:
    return f"""{title}
```python
{code}
```"""


def _render_edit_segment(
    item: dict[str, Any],
    *,
    base_timestamp: datetime,
    segment_index_1idx: int,
) -> str:
    lines = [
        f"Edit Segment {segment_index_1idx}",
        f"- raw_change_event_count: {item['change_event_count']}",
        "TSV columns: event_time_delta_ms\traw_log_line\tfrom_line\tfrom_ch\tto_line\tto_ch\tinserted_text_json\tremoved_text_json\tedit_origin",
        "event_time_delta_ms\traw_log_line\tfrom_line\tfrom_ch\tto_line\tto_ch\tinserted_text_json\tremoved_text_json\tedit_origin",
    ]
    for change in item["changes"]:
        lines.append(
            "\t".join(
                [
                    str(
                        _delta_ms(
                            event_timestamp=str(change["timestamp"]), base_timestamp=base_timestamp
                        )
                    ),
                    str(change["line_number"]),
                    str(change["from"]["line"]),
                    str(change["from"]["ch"]),
                    str(change["to"]["line"]),
                    str(change["to"]["ch"]),
                    _json_cell(change["text"]),
                    _json_cell(change["removed"]),
                    str(change["origin"]),
                ]
            )
        )
    return "\n".join(lines)


def _render_saida_testar(
    item: dict[str, Any],
    *,
    base_timestamp: datetime,
    run_index_1idx: int,
) -> str:
    output_text = "\n".join(item["output_lines"])
    return "\n".join(
        [
            f"Local Run {run_index_1idx}",
            f"- event_time_delta_ms: {_delta_ms(event_timestamp=str(item['timestamp']), base_timestamp=base_timestamp)}",
            f"- raw_log_line: {item['line_number']}",
            f"- command: {_json_cell(item['command'])}",
            _render_code_block("Code After This Local Run", str(item["code_after_anchor"])),
            "Run Output",
            "```text",
            output_text,
            "```",
        ]
    )


def _render_submit(
    item: dict[str, Any],
    *,
    base_timestamp: datetime,
    submit_index_1idx: int,
) -> str:
    return "\n".join(
        [
            f"Submit Anchor {submit_index_1idx}",
            f"- event_time_delta_ms: {_delta_ms(event_timestamp=str(item['timestamp']), base_timestamp=base_timestamp)}",
            f"- raw_log_line: {item['line_number']}",
            f"- feedback: {_json_cell(item['feedback'])}",
            _render_code_block("Code At This Submit", str(item["code_after_anchor"])),
        ]
    )


def _render_full_attempt(attempt: dict[str, Any]) -> str:
    trace = attempt["trace"]
    base_timestamp = _attempt_base_timestamp(trace)
    item_blocks: list[str] = []
    edit_index = 0
    run_index = 0
    submit_index = 0
    for item in trace["items"]:
        item_type = item["item_type"]
        if item_type == "edit_segment":
            edit_index += 1
            item_blocks.append(
                _render_edit_segment(
                    item,
                    base_timestamp=base_timestamp,
                    segment_index_1idx=edit_index,
                )
            )
            continue
        if item_type == "saida_testar":
            run_index += 1
            item_blocks.append(
                _render_saida_testar(
                    item,
                    base_timestamp=base_timestamp,
                    run_index_1idx=run_index,
                )
            )
            continue
        if item_type == "submit":
            submit_index += 1
            item_blocks.append(
                _render_submit(
                    item,
                    base_timestamp=base_timestamp,
                    submit_index_1idx=submit_index,
                )
            )
            continue
        raise ValueError(f"Unsupported trace item type: {item_type}")

    return "\n\n".join(
        [
            f"## Attempt {attempt['attempt_index_0idx']}",
            f"- aligned_submit_index_0idx: {attempt['aligned_submit_index_0idx']}",
            f"- trace_base_timestamp: {base_timestamp.isoformat()}",
            "Execution Result",
            _render_execution_result(attempt["execution_result"]),
            _render_code_block("Attempt Start Code", str(trace["attempt_start_code"])),
            "Trace Timeline",
            "\n\n".join(item_blocks),
        ]
    )


def _render_no_trace_attempt(attempt: dict[str, Any]) -> str:
    if "trace" in attempt:
        raise ValueError("No-trace attempt payload must not include trace")
    submitted_code = attempt.get("submitted_code")
    if not isinstance(submitted_code, str):
        raise ValueError("No-trace attempt payload must include submitted_code")
    return "\n\n".join(
        [
            f"## Attempt {attempt['attempt_index_0idx']}",
            f"- aligned_submit_index_0idx: {attempt['aligned_submit_index_0idx']}",
            "Execution Result",
            _render_execution_result(attempt["execution_result"]),
            _render_code_block("Submitted Code", submitted_code),
        ]
    )


def build_user_prompt(payload: dict[str, object]) -> str:
    required_keys = {"schema_version", "source", "visible_attempts", "prediction_target"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Payload missing required keys: {sorted(missing)}")

    source = payload["source"]
    visible_attempts = payload["visible_attempts"]
    prediction_target = payload["prediction_target"]
    if not isinstance(source, dict):
        raise ValueError("Payload source must be a dict")
    if not isinstance(visible_attempts, list):
        raise ValueError("Payload visible_attempts must be a list")
    if not isinstance(prediction_target, dict):
        raise ValueError("Payload prediction_target must be a dict")
    condition = _payload_condition(payload)

    if condition == FULL_TRACE_CONDITION:
        rendered_attempts = "\n\n".join(
            _render_full_attempt(attempt) for attempt in visible_attempts
        )
        payload_notes = "\n".join(
            [
                "- Each attempt includes the exact code buffer at the start of that attempt block.",
                "- Each edit segment is lossless raw CodeMirror change data rendered as TSV with a legend.",
                "- `event_time_delta_ms` means milliseconds since the first trace event in that attempt.",
                "- `from_line/from_ch/to_line/to_ch` are the raw CodeMirror coordinates.",
                "- `inserted_text_json` and `removed_text_json` keep the exact inserted and removed text as JSON arrays.",
                "- `edit_origin` is the raw CodeMirror origin string.",
                "- Local runs include raw output and the exact code snapshot after the run.",
                "- Submit anchors include raw feedback and the exact code snapshot at submit.",
                "- Some attempts may contain repeated submits with no edits between them. Treat that as real behavior, not as noise.",
            ]
        )
    elif condition == NO_TRACE_CONDITION:
        rendered_attempts = "\n\n".join(
            _render_no_trace_attempt(attempt) for attempt in visible_attempts
        )
        payload_notes = "\n".join(
            [
                "- Each attempt includes the authoritative execution result for that submitted attempt.",
                "- Each attempt includes the submitted code for that attempt.",
            ]
        )
    else:
        raise AssertionError(f"Unhandled v6.2 condition: {condition}")
    if condition == FULL_TRACE_CONDITION:
        return f"""Task: predict this student's next submitted attempt n+1.

How to read the payload:
- Attempts are ordered from oldest to newest.
- The execution result is the authoritative grading result for each submitted attempt.
{payload_notes}

Return exactly 3 hypotheses.

Each hypothesis must include:
- a short label
- a probability
- the full predicted code for the next submitted attempt
- a predicted next trajectory with 1 to 8 coarse steps

The priority order is:
1. get the next submitted code right
2. get the rough next trajectory right

For each predicted trajectory step:
- use `action_type` from this set only: `edit`, `local_run`, `submit`
- for `edit`, set `target_start_line_0idx` and `target_end_line_0idx` to one coarse line span in the predicted next code
- for `local_run` and `submit`, set both target line fields to `-1`
- repeated `edit` and repeated `local_run` steps are allowed
- every trajectory must end with `submit`
- do not try to predict every raw event; this path should stay coarse

Do not invent exercise requirements that are not supported by the visible evidence.

SOURCE
- class_id: {source["class_id"]}
- assessment_id: {source["assessment_id"]}
- assessment_title: {source["assessment_title"]}
- exercise_id: {source["exercise_id"]}
- student_id: {source["student_id"]}
- condition: {condition}
- visible_transition_index_0idx: {source["visible_transition_index_0idx"]}
- visible_attempt_indices_0idx: {_json_cell(source["visible_attempt_indices_0idx"])}
- prediction_target_attempt_index_0idx: {source["prediction_target_attempt_index_0idx"]}
- alignment_policy: {source["alignment_policy"]}

PREDICTION TARGET
- attempt_index_0idx: {prediction_target["attempt_index_0idx"]}
- task: {prediction_target["task"]}

VISIBLE ATTEMPTS
{rendered_attempts}
"""
    return f"""Task: predict this student's next submitted attempt n+1.

How to read the payload:
- Attempts are ordered from oldest to newest.
- The execution result is the authoritative grading result for each submitted attempt.
{payload_notes}

Return exactly 3 hypotheses.

Each hypothesis must include:
- a short label
- a probability
- the full predicted code for the next submitted attempt
- a predicted next trajectory with 1 to 8 coarse steps

The priority order is:
1. get the next submitted code right
2. get the rough next trajectory right

For each predicted trajectory step:
- use `action_type` from this set only: `edit`, `local_run`, `submit`
- for `edit`, set `target_start_line_0idx` and `target_end_line_0idx` to one coarse line span in the predicted next code
- for `local_run` and `submit`, set both target line fields to `-1`
- repeated `edit` and repeated `local_run` steps are allowed
- every trajectory must end with `submit`
- do not try to predict every raw event; this path should stay coarse

Do not invent exercise requirements that are not supported by the visible evidence.

SOURCE
- class_id: {source["class_id"]}
- assessment_id: {source["assessment_id"]}
- assessment_title: {source["assessment_title"]}
- exercise_id: {source["exercise_id"]}
- student_id: {source["student_id"]}
- visible_transition_index_0idx: {source["visible_transition_index_0idx"]}
- visible_attempt_indices_0idx: {_json_cell(source["visible_attempt_indices_0idx"])}
- prediction_target_attempt_index_0idx: {source["prediction_target_attempt_index_0idx"]}
- alignment_policy: {source["alignment_policy"]}

PREDICTION TARGET
- attempt_index_0idx: {prediction_target["attempt_index_0idx"]}
- task: {prediction_target["task"]}

VISIBLE ATTEMPTS
{rendered_attempts}
"""


def structured_text_format() -> dict[str, object]:
    return {
        "format": {
            "type": "json_schema",
            "name": "FullTracePredictionResponse",
            "strict": True,
            "schema": FullTracePredictionResponse.model_json_schema(),
        }
    }
