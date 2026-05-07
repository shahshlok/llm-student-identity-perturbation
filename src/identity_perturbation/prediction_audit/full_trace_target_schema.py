from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictedCoarsePathStep(_StrictModel):
    action_type: Literal["edit", "local_run", "submit"]
    target_start_line_0idx: int
    target_end_line_0idx: int

    @model_validator(mode="after")
    def validate_step(self) -> PredictedCoarsePathStep:
        if self.action_type == "edit":
            if self.target_start_line_0idx < 0 or self.target_end_line_0idx < 0:
                raise ValueError("edit steps must use non-negative target line indices")
            if self.target_start_line_0idx > self.target_end_line_0idx:
                raise ValueError(
                    "edit steps must satisfy target_start_line_0idx <= target_end_line_0idx"
                )
            return self

        if self.target_start_line_0idx != -1 or self.target_end_line_0idx != -1:
            raise ValueError("local_run and submit steps must set both target line fields to -1")
        return self


class FullTraceHypothesis(_StrictModel):
    label: str = Field(min_length=1)
    estimated_probability: StrictFloat = Field(gt=0.0, le=1.0)
    predicted_next_code: str
    predicted_next_trajectory: list[PredictedCoarsePathStep] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_trajectory(self) -> FullTraceHypothesis:
        if self.predicted_next_trajectory[-1].action_type != "submit":
            raise ValueError("predicted_next_trajectory must end with a submit step")
        return self


class FullTracePredictionResponse(_StrictModel):
    schema_version: Literal["v6_2_full_trace_prediction_v3"]
    hypotheses: list[FullTraceHypothesis] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_probability_mass(self) -> FullTracePredictionResponse:
        total = sum(item.estimated_probability for item in self.hypotheses)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"hypotheses estimated_probability must sum to 1.0; got {total}")
        return self
