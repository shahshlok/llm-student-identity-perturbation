from __future__ import annotations

import json

from .v61_prediction_schema import V61PredictedEpisodeResponse

SYSTEM_PROMPT_V61 = """You are helping an instructor reason about one student's likely next repair episode in beginner CS1 Python work.

Your task is not to claim certainty about the student's hidden mental state.
Your task is to form cautious, evidence-grounded forecasts about what the student is likely to do next.

Important rules:
- Work only from the evidence available up to and including attempt n.
- Use the provided semantic event taxonomy exactly.
- Do not invent future observations that are not plausible consequences of the provided evidence.
- Return 2 to 5 bounded next-episode hypotheses with probabilities that sum to exactly 1.0.
- Return only valid JSON matching the requested schema.
- Do not use markdown, code fences, comments, or prose outside the JSON object.
- Keep every event `detail` short, concrete, and plain-English.
"""


OUTPUT_SCHEMA_V61 = {
    "schema_version": "v6_1_predicted_episode_response_v1",
    "instructor_summary": "short instructor-facing forecast",
    "next_episode_hypotheses": [
        {
            "label": "local patch then rerun",
            "estimated_probability": 0.55,
            "student_state_summary": "short cautious state summary",
            "predicted_event_tape": [
                {
                    "predicted_event_index_1idx": 1,
                    "event_type": "change",
                    "primary_line_0idx": 15,
                    "secondary_line_0idx": 15,
                    "detail": "delete x += x and replace it with a corrected expression",
                }
            ],
        }
    ],
}


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_V61


def build_user_prompt(payload: dict[str, object]) -> str:
    required_keys = {"student_header", "attempt_n", "prior_history"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Payload missing required keys: {sorted(missing)}")

    condition = str(payload.get("condition", "full_v61"))
    student_header_json = json.dumps(payload["student_header"], ensure_ascii=True, indent=2)
    attempt_n_json = json.dumps(payload["attempt_n"], ensure_ascii=True, indent=2)
    prior_history_json = json.dumps(payload["prior_history"], ensure_ascii=True, indent=2)
    output_schema_json = json.dumps(OUTPUT_SCHEMA_V61, ensure_ascii=True, indent=2)

    trace_evidence_line = "- the full observed semantic CodeMirror tape for attempt n"
    extra_note = ""
    if condition == "no_trace":
        trace_evidence_line = "- no CodeMirror log evidence for attempt n in this condition"
        extra_note = (
            "\nAdditional note: do not refer to missing CodeMirror evidence in your explanation."
        )

    return f"""Task: Forecast this student's next repair episode after attempt n.

You are given:
- the current code and test behavior at attempt n
- compact same-exercise prior history
{trace_evidence_line}

Use only the provided evidence.

Your job:
1. produce 2 to 5 plausible next-episode hypotheses
2. make the probabilities sum to 1.0 exactly
   - use short decimals like `0.55`, `0.25`, `0.20`
   - make the final probability the adjusted one so the total is exactly `1.0`
3. use the exact semantic event taxonomy in every predicted event tape
4. keep each predicted episode bounded and plausible as the next immediate repair episode
5. keep each event object simple:
   - `primary_line_0idx` is the main line involved, or `-1` if no single line applies
   - `secondary_line_0idx` is a second line if useful, otherwise `-1`
   - `detail` is one short plain-English description of the event
   - prefer `5-12` words in `detail`
6. event guidance:
   - `change`: say the likely edit in plain English and set line hints to the edited region
   - `saida_testar`: describe the local run or test action
   - `submit`: describe the likely submit action or expected result in plain English
   - `kill_program`: describe stopping a run
   - `keyHandled`: describe the key action
   - `tab_click`: describe the navigation target
   - `idle_gap`: describe the pause and set line hints to `-1`
7. keep the JSON compact and valid; do not include extra fields
8. every hypothesis must include at least one event{extra_note}

STUDENT HEADER
{student_header_json}

ATTEMPT N
{attempt_n_json}

PRIOR SAME-EXERCISE HISTORY
{prior_history_json}

Return JSON matching this schema:
{output_schema_json}
"""


def structured_text_format() -> dict[str, object]:
    return {
        "format": {
            "type": "json_schema",
            "name": "V61PredictedEpisodeResponse",
            "strict": True,
            "schema": V61PredictedEpisodeResponse.model_json_schema(),
        }
    }
