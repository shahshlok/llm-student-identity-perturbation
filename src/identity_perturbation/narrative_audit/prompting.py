from __future__ import annotations

import json

from .prediction_schema import V6BatchResponse

SYSTEM_PROMPT_V6 = """You are helping an instructor reason about one student's likely next repair move in beginner CS1 Python work.

Your task is not to claim access to the student's hidden mental state with certainty.
Your task is to form cautious, evidence-grounded hypotheses about what this student is likely to do next.

Important rules:
- Work only from the provided evidence available up to and including attempt n.
- Do not use any information that would only be known after attempt n is submitted.
- Do not invent exercise requirements beyond what can be inferred from code and test behavior.
- Express uncertainty through multiple hypotheses with explicit probability mass.
- Make the probabilities sum to exactly 1.0.
- Ground each hypothesis in concrete evidence from code, tests, history, and trace cards.
- Return only valid JSON matching the requested schema.
"""


OUTPUT_SCHEMA_V6 = {
    "student_state_summary": "short cautious description of the student's current likely state",
    "instructor_summary": "short instructor-facing note about likely next struggle or intervention target",
    "next_move_hypotheses": [
        {
            "label": "short name for this possible next move",
            "estimated_probability": 0.5,
            "difficulty_hypothesis": "1-2 sentence explanation of what may be going wrong",
            "likely_first_repair_region": "output_region|conditional_region|loop_region",
            "likely_edit_scope": "local_1_to_2_lines|regional_3_to_5_lines|broad_6_plus_lines",
            "likely_edit_strategy": (
                "patch_condition|patch_loop_logic|patch_output_only|"
                "delete_and_rewrite_local_block|revert_previous_change|"
                "format_or_surface_cleanup|no_meaningful_progress"
            ),
            "likely_next_test_outcome": "all_fail|mixed|all_pass",
            "likely_code_delta_summary": "brief summary of the likely next code change",
            "supporting_signals": [
                {
                    "signal_type": "code|tests|history|trace_card",
                    "signal_id": "attempt_n.code|attempt_n.tests|history[0]|trace_card_n",
                    "claim": "brief evidence statement",
                }
            ],
            "counterevidence": ["brief caution or alternative possibility"],
        }
    ],
}


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_V6


def build_user_prompt(payload: dict) -> str:
    required_keys = {
        "student_header",
        "attempt_n",
        "prior_history",
        "trace_cards_pre_n",
    }
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Payload missing required keys: {sorted(missing)}")

    student_header_json = json.dumps(payload["student_header"], ensure_ascii=True, indent=2)
    attempt_n_json = json.dumps(payload["attempt_n"], ensure_ascii=True, indent=2)
    prior_history_json = json.dumps(payload["prior_history"], ensure_ascii=True, indent=2)
    trace_cards_json = json.dumps(payload["trace_cards_pre_n"], ensure_ascii=True, indent=2)
    output_schema_json = json.dumps(OUTPUT_SCHEMA_V6, ensure_ascii=True, indent=2)
    attempt_n = payload["attempt_n"]
    if not isinstance(attempt_n, dict):
        raise ValueError("Payload attempt_n must be an object")
    current_trace_available = "trace_card" in attempt_n
    trace_cards_pre_n = payload["trace_cards_pre_n"]
    if not isinstance(trace_cards_pre_n, list):
        raise ValueError("Payload trace_cards_pre_n must be a list")
    prior_history = payload["prior_history"]
    if not isinstance(prior_history, list):
        raise ValueError("Payload prior_history must be a list")

    evidence_lines = [
        "This includes the current code state, current test behavior, and prior same-exercise history."
    ]
    if current_trace_available or trace_cards_pre_n:
        evidence_lines.append(
            "It also includes pre-submit trace evidence available up to and including attempt n."
        )
    else:
        evidence_lines.append("No pre-submit trace evidence is provided in this condition.")

    grounding_lines = ["5. ground every hypothesis in explicit evidence from the provided payload"]
    if current_trace_available or trace_cards_pre_n:
        grounding_lines.append(
            "6. when trace evidence is present, use it carefully rather than treating it as certainty"
        )
    else:
        grounding_lines.append(
            "6. do not refer to trace-card evidence because this condition withholds it"
        )

    trace_section_label = "TRACE CARDS STRICTLY BEFORE N"
    if not trace_cards_pre_n:
        trace_section_label = "TRACE CARDS STRICTLY BEFORE N (WITHHELD OR EMPTY IN THIS CONDITION)"
    evidence_text = " ".join(evidence_lines)
    grounding_text = "\n".join(grounding_lines)

    return f"""Task: Forecast this student's most likely next repair move after attempt n.

You are given one student's same-exercise evidence available up to and including attempt n.
{evidence_text}

You do NOT have any evidence from attempt n+1 or later.

Your job:
1. infer a cautious summary of the student's current likely state
2. propose 2 to 5 plausible next-move hypotheses
3. distribute probability mass across those hypotheses so the probabilities sum to exactly 1.0
4. predict the likely first repair region, edit scope, edit strategy, and next test outcome for each hypothesis
{grounding_text}

STUDENT HEADER
{student_header_json}

ATTEMPT N
{attempt_n_json}

PRIOR SAME-EXERCISE HISTORY
{prior_history_json}

{trace_section_label}
{trace_cards_json}

Return JSON matching this schema:
{output_schema_json}
"""


def structured_text_format() -> dict[str, object]:
    return {
        "format": {
            "type": "json_schema",
            "name": "V6BatchResponse",
            "strict": True,
            "schema": V6BatchResponse.model_json_schema(),
        }
    }
