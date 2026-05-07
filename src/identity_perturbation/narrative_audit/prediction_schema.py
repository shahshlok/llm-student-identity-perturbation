from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FirstRepairRegion = Literal["output_region", "conditional_region", "loop_region"]
EditScope = Literal["local_1_to_2_lines", "regional_3_to_5_lines", "broad_6_plus_lines"]
NextTestOutcome = Literal["all_fail", "mixed", "all_pass"]
EditStrategy = Literal[
    "patch_condition",
    "patch_loop_logic",
    "patch_output_only",
    "delete_and_rewrite_local_block",
    "revert_previous_change",
    "format_or_surface_cleanup",
    "no_meaningful_progress",
]


class V6SupportingSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_type: Literal["code", "tests", "history", "trace_card"]
    signal_id: str
    claim: str


class V6NextMoveHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    estimated_probability: float = Field(gt=0.0, le=1.0)
    difficulty_hypothesis: str = Field(min_length=1)
    likely_first_repair_region: FirstRepairRegion
    likely_edit_scope: EditScope
    likely_edit_strategy: EditStrategy
    likely_next_test_outcome: NextTestOutcome
    likely_code_delta_summary: str = Field(min_length=1)
    supporting_signals: list[V6SupportingSignal] = Field(min_length=1)
    counterevidence: list[str] = Field(min_length=1)


class V6BatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_state_summary: str = Field(min_length=1)
    instructor_summary: str = Field(min_length=1)
    next_move_hypotheses: list[V6NextMoveHypothesis] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def validate_probability_mass(self) -> V6BatchResponse:
        total = sum(item.estimated_probability for item in self.next_move_hypotheses)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"next_move_hypotheses estimated_probability must sum to 1.0 exactly; got {total}"
            )
        return self
