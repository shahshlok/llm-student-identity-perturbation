from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CodePosition(_StrictModel):
    line: int = Field(ge=0)
    ch: int = Field(ge=0)


class SemanticEventBase(_StrictModel):
    event_index_1idx: int = Field(ge=1)
    raw_ordinal_in_interval_1idx: int = Field(ge=1)
    seconds_since_previous_semantic_event: float | None = Field(default=None, ge=0.0)
    timestamp: str
    event_type: str


class ChangeSemanticEvent(SemanticEventBase):
    event_type: Literal["change"]
    from_position: CodePosition = Field(alias="from")
    to_position: CodePosition = Field(alias="to")
    inserted_text: str
    removed_text: str
    origin: str = Field(min_length=1)


class RunLocalSemanticEvent(SemanticEventBase):
    event_type: Literal["saida_testar"]
    command: str = Field(min_length=1)
    output_lines: list[str]
    truncated_output_line_count: int = Field(ge=0)
    output_line_limit: int | None = Field(default=None, ge=1)


class SubmitSemanticEvent(SemanticEventBase):
    event_type: Literal["submit"]
    feedback: str = Field(min_length=1)


class KillProgramSemanticEvent(SemanticEventBase):
    event_type: Literal["kill_program"]
    value: bool
    raw_value: str


class KeyHandledSemanticEvent(SemanticEventBase):
    event_type: Literal["keyHandled"]
    key: str = Field(min_length=1)


class TabClickSemanticEvent(SemanticEventBase):
    event_type: Literal["tab_click"]
    target: str = Field(min_length=1)


class IdleGapSemanticEvent(_StrictModel):
    event_index_1idx: int = Field(ge=1)
    raw_ordinal_in_interval_1idx: int = Field(ge=1)
    event_type: Literal["idle_gap"]
    seconds_since_previous_semantic_event: float = Field(ge=0.0)
    starts_at: str
    ends_at: str
    previous_event_type: str | None = None
    next_event_type: str


SemanticEvent = Annotated[
    ChangeSemanticEvent
    | RunLocalSemanticEvent
    | SubmitSemanticEvent
    | KillProgramSemanticEvent
    | KeyHandledSemanticEvent
    | TabClickSemanticEvent
    | IdleGapSemanticEvent,
    Field(discriminator="event_type"),
]


class SemanticTapeSummary(_StrictModel):
    raw_interval_event_count: int = Field(ge=0)
    semantic_event_count: int = Field(ge=0)
    change_event_count: int = Field(ge=0)
    saida_testar_count: int = Field(ge=0)
    submit_count: int = Field(ge=0)
    kill_program_count: int = Field(ge=0)
    idle_gap_count: int = Field(ge=0)


class SemanticTapeTransitionRef(_StrictModel):
    class_id: str = Field(min_length=1)
    assessment_id: str = Field(min_length=1)
    exercise_id: str = Field(min_length=1)
    student_id: str = Field(min_length=1)
    transition_index_0idx: int = Field(ge=0)


class AttemptNSemanticTape(_StrictModel):
    schema_version: Literal["v6_1_attempt_semantic_tape_v1"]
    transition: SemanticTapeTransitionRef
    current_trace_card: dict
    semantic_tape_summary: SemanticTapeSummary
    semantic_event_tape: list[SemanticEvent]


class ObservedNextEpisodeTape(_StrictModel):
    schema_version: Literal["v6_1_observed_next_episode_v1"]
    transition: SemanticTapeTransitionRef
    attempt_n1_index_0idx: int = Field(ge=0)
    semantic_tape_summary: SemanticTapeSummary
    semantic_event_tape: list[SemanticEvent]


class PredictedSemanticTape(_StrictModel):
    schema_version: Literal["v6_1_predicted_semantic_tape_v1"]
    predicted_event_tape: list[SemanticEvent]
