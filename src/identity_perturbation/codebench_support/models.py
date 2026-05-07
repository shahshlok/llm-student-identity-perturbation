from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TestResult:
    test_index_0idx: int
    passed: bool
    input: str
    expected: str
    actual: str


@dataclass(frozen=True)
class ExecutionAttempt:
    timestamp: str
    attempt_index_0idx: int
    code: str
    grade: float
    test_results: tuple[TestResult, ...]


@dataclass(frozen=True)
class ExecutionTransition:
    attempt_n: ExecutionAttempt
    attempt_n1: ExecutionAttempt
    history: tuple[ExecutionAttempt, ...]
    all_attempts: tuple[ExecutionAttempt, ...]


@dataclass(frozen=True)
class CohortWindow:
    student_id: str
    class_id: str
    assessment_id: str
    exercise_id: str
    transition: ExecutionTransition


@dataclass(frozen=True)
class CmEvent:
    timestamp: datetime
    raw_type: str
    payload: object
    line_number: int


@dataclass(frozen=True)
class SubmitSnapshot:
    submit_event: CmEvent
    code: str
    changes_since_prev: tuple[dict, ...]
    submit_index: int


@dataclass(frozen=True)
class ReplayTrace:
    snapshots: tuple[SubmitSnapshot, ...]
    final_code: str
    trailing_changes: tuple[dict, ...]


@dataclass(frozen=True)
class AlignmentResult:
    status: str
    snap_n_index: int
    snap_n1_index: int | None
    submit_n_timestamp: str
    submit_n1_timestamp: str | None
    changes_between: tuple[dict, ...]
    first_change_line_0idx: int
    lines_touched_0idx: tuple[int, ...]
