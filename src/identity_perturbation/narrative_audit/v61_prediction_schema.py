from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictedSemanticEvent(_StrictModel):
    predicted_event_index_1idx: int = Field(ge=1)
    event_type: Literal[
        "change",
        "saida_testar",
        "submit",
        "kill_program",
        "keyHandled",
        "tab_click",
        "idle_gap",
    ]
    primary_line_0idx: int = Field(ge=-1)
    secondary_line_0idx: int = Field(ge=-1)
    detail: str


class V61EpisodeHypothesis(_StrictModel):
    label: str = Field(min_length=1)
    estimated_probability: float = Field(gt=0.0, le=1.0)
    student_state_summary: str = Field(min_length=1)
    predicted_event_tape: list[PredictedSemanticEvent] = Field(min_length=1, max_length=40)


class V61PredictedEpisodeResponse(_StrictModel):
    schema_version: Literal["v6_1_predicted_episode_response_v1"]
    instructor_summary: str = Field(min_length=1)
    next_episode_hypotheses: list[V61EpisodeHypothesis] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def validate_probability_mass(self) -> V61PredictedEpisodeResponse:
        total = sum(item.estimated_probability for item in self.next_episode_hypotheses)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"next_episode_hypotheses estimated_probability must sum to 1.0; got {total}"
            )
        return self
