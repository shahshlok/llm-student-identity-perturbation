from __future__ import annotations

import json

SYSTEM_PROMPT_V5 = """You are helping an instructor understand cohort-level debugging and repair patterns in beginner CS1 Python work.

Your task is not to diagnose individual students or claim access to hidden mental states.
Your task is to surface cautious, evidence-grounded cohort hypotheses from code and execution history.

Important rules:
- Work only from the provided cohort evidence.
- Do not invent missing exercise requirements beyond what can be inferred from code and test behavior.
- Do not claim certainty.
- Favor recurring cohort patterns over one-off anecdotes.
- Ground every hypothesis in the provided evidence.
- For each hypothesis, predict a small hidden behavioral profile that can later be checked against held-out logs.
- Return only valid JSON matching the requested schema.
"""


OUTPUT_SCHEMA_V5 = {
    "cohort_summary": "short overall description of the dominant cohort repair tendencies",
    "hypotheses": [
        {
            "label": "short pattern name",
            "description": "1-2 sentence cautious cohort-level explanation",
            "estimated_prevalence": 0.0,
            "supporting_evidence": [
                {
                    "source_type": "aggregate|card",
                    "source_id": "aggregate.common_failed_tests[0]|S01|S02",
                    "claim": "brief evidence statement",
                }
            ],
            "predicted_trace_profile": {
                "first_focus_region_3way": "output_region|conditional_region|loop_region",
                "lines_touched_bucket_3way": "local_1_to_2_lines|regional_3_to_5_lines|broad_6_plus_lines",
                "next_test_outcome": "all_fail|mixed|all_pass",
            },
            "counterevidence": ["brief caution or alternative interpretation"],
        }
    ],
}


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_V5


def build_user_prompt(payload: dict) -> str:
    required_keys = {"slice_header", "aggregate_summary", "representative_cards"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Payload missing required keys: {sorted(missing)}")

    slice_header_json = json.dumps(payload["slice_header"], ensure_ascii=True, indent=2)
    aggregate_summary_json = json.dumps(payload["aggregate_summary"], ensure_ascii=True, indent=2)
    representative_cards_json = json.dumps(
        payload["representative_cards"],
        ensure_ascii=True,
        indent=2,
    )
    output_schema_json = json.dumps(OUTPUT_SCHEMA_V5, ensure_ascii=True, indent=2)

    return f"""Task: Surface cohort-level hypotheses for this CodeBench cohort slice.

You are given one cohort slice defined at the level of `class_assessment_exercise`.
The evidence below comes from code snapshots, test outcomes, and prior same-exercise execution history.
You do NOT have editor-trace data.

Your job:
1. Identify the most plausible recurring cohort patterns in how students are approaching or repairing this exercise.
2. Ground each pattern in concrete evidence from the aggregate summary and representative attempt cards.
3. Predict the hidden behavioral profile that would likely be observed later if the pattern is real.
4. Keep claims cautious and cohort-level.

COHORT SLICE HEADER
{slice_header_json}

EXECUTION-DERIVED COHORT SUMMARY
{aggregate_summary_json}

REPRESENTATIVE ATTEMPT CARDS
{representative_cards_json}

Return JSON matching this schema:
{output_schema_json}
"""
