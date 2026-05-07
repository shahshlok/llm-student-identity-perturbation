from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Hypothesis:
    label: str
    probability: float
    student_state_summary: str
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EvaluationExample:
    custom_id: str
    class_id: str
    assessment_id: str
    exercise_id: str
    student_id: str
    transition_index_0idx: int
    prefix_k: int
    observed_events: tuple[dict[str, Any], ...]
    hypotheses: tuple[Hypothesis, ...]
