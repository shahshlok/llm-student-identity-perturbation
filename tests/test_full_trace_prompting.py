from __future__ import annotations

import pytest

from identity_perturbation.prediction_audit.full_trace_prompting import (
    FULL_TRACE_CONDITION,
    NO_TRACE_CONDITION,
    build_model_visible_payload,
    build_system_prompt,
    build_user_prompt,
)


def _execution_result() -> dict[str, object]:
    return {
        "timestamp": "2026-04-16T10:00:00",
        "grade": 0.5,
        "pass_count": 1,
        "failed_test_indices_0idx": [1],
        "test_outcome": "mixed",
        "test_results": [
            {
                "test_index_0idx": 0,
                "passed": True,
                "input": "1",
                "expected": "1",
                "actual": "1",
            },
            {
                "test_index_0idx": 1,
                "passed": False,
                "input": "2",
                "expected": "4",
                "actual": "3",
            },
        ],
    }


def _trace_payload() -> dict[str, object]:
    return {
        "attempt_start_code": "x = 1",
        "items": [
            {
                "item_type": "edit_segment",
                "change_event_count": 1,
                "changes": [
                    {
                        "timestamp": "2026-04-16T10:00:00",
                        "line_number": 1,
                        "from": {"line": 0, "ch": 0},
                        "to": {"line": 0, "ch": 0},
                        "text": ["x = 1"],
                        "removed": [""],
                        "origin": "+input",
                    }
                ],
            },
            {
                "item_type": "submit",
                "timestamp": "2026-04-16T10:00:01",
                "line_number": 2,
                "feedback": "wrong answer",
                "code_after_anchor": "x = 1",
            },
        ],
    }


def _raw_payload(*, condition: str) -> dict[str, object]:
    return {
        "schema_version": "v6_2_full_trace_prompt_payload_v2",
        "source": {
            "class_id": "589",
            "assessment_id": "6353",
            "assessment_title": "Lab",
            "exercise_id": "9735",
            "student_id": "5897",
            "condition": condition,
            "visible_transition_index_0idx": 0,
            "visible_attempt_indices_0idx": [0],
            "prediction_target_attempt_index_0idx": 1,
            "alignment_policy": "policy",
            "execution_log_path": "/tmp/execution.log",
            "codemirror_log_path": "/tmp/codemirror.log",
        },
        "visible_attempts": [
            {
                "attempt_index_0idx": 0,
                "aligned_submit_index_0idx": 0,
                "execution_result": _execution_result(),
                "submitted_code": "x = 1",
                "trace": _trace_payload(),
            }
        ],
        "prediction_target": {
            "attempt_index_0idx": 1,
            "task": "Predict the next submitted attempt after the visible history.",
        },
    }


def test_build_model_visible_payload_keeps_trace_for_full() -> None:
    payload = build_model_visible_payload(_raw_payload(condition=FULL_TRACE_CONDITION))

    attempt = payload["visible_attempts"][0]
    assert payload["source"]["condition"] == FULL_TRACE_CONDITION
    assert "trace" in attempt
    assert "submitted_code" not in attempt


def test_build_model_visible_payload_drops_trace_for_no_trace() -> None:
    payload = build_model_visible_payload(_raw_payload(condition=NO_TRACE_CONDITION))

    attempt = payload["visible_attempts"][0]
    assert payload["source"]["condition"] == NO_TRACE_CONDITION
    assert attempt["submitted_code"] == "x = 1"
    assert "trace" not in attempt


def test_build_user_prompt_renders_full_trace_prompt() -> None:
    payload = build_model_visible_payload(_raw_payload(condition=FULL_TRACE_CONDITION))

    prompt = build_user_prompt(payload)

    assert "condition: full" in prompt
    assert "Trace Timeline" in prompt
    assert "Submitted Code" not in prompt


def test_build_user_prompt_renders_no_trace_prompt() -> None:
    payload = build_model_visible_payload(_raw_payload(condition=NO_TRACE_CONDITION))

    prompt = build_user_prompt(payload)

    assert "CodeMirror" not in prompt
    assert "trace" not in prompt.lower()
    assert "Trace Timeline" not in prompt
    assert "Submitted Code" in prompt
    assert "condition: no_trace" not in prompt


def test_build_system_prompt_is_condition_specific() -> None:
    full_prompt = build_system_prompt(FULL_TRACE_CONDITION)
    no_trace_prompt = build_system_prompt(NO_TRACE_CONDITION)

    assert "CodeMirror trace" in full_prompt
    assert "CodeMirror" not in no_trace_prompt
    assert "trace" not in no_trace_prompt.lower()


def test_build_user_prompt_rejects_missing_condition() -> None:
    payload = _raw_payload(condition=FULL_TRACE_CONDITION)
    del payload["source"]["condition"]

    with pytest.raises(ValueError, match="Condition must be a string"):
        build_user_prompt(payload)


def test_build_user_prompt_rejects_no_trace_payload_with_trace() -> None:
    payload = _raw_payload(condition=NO_TRACE_CONDITION)
    conditioned = build_model_visible_payload(payload)
    conditioned["visible_attempts"][0]["trace"] = _trace_payload()

    with pytest.raises(ValueError, match="must not include trace"):
        build_user_prompt(conditioned)
