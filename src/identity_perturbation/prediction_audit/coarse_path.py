from __future__ import annotations

from difflib import SequenceMatcher
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from identity_perturbation.codebench_support.codemirror import _code_to_lines


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EditCoarsePathStep(_StrictModel):
    action_type: Literal["edit"]
    target_start_line_0idx: int = Field(ge=0)
    target_end_line_0idx: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_line_targets(self) -> EditCoarsePathStep:
        if self.target_start_line_0idx > self.target_end_line_0idx:
            raise ValueError("target_start_line_0idx must be <= target_end_line_0idx")
        return self


class LocalRunCoarsePathStep(_StrictModel):
    action_type: Literal["local_run"]
    target_start_line_0idx: None = Field(...)
    target_end_line_0idx: None = Field(...)


class SubmitCoarsePathStep(_StrictModel):
    action_type: Literal["submit"]
    target_start_line_0idx: None = Field(...)
    target_end_line_0idx: None = Field(...)


CoarsePathStep: TypeAlias = Annotated[
    EditCoarsePathStep | LocalRunCoarsePathStep | SubmitCoarsePathStep,
    Field(discriminator="action_type"),
]


class ObservedCoarsePathError(ValueError):
    pass


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ObservedCoarsePathError(f"Expected {key!r} to be a string")
    return value


def _required_items(trace: dict[str, Any]) -> list[dict[str, Any]]:
    items = trace.get("items")
    if not isinstance(items, list) or not items:
        raise ObservedCoarsePathError("Trace must contain a non-empty items list")
    if not all(isinstance(item, dict) for item in items):
        raise ObservedCoarsePathError("Trace items must all be dict objects")
    return items


def _longest_common_prefix_len(before_lines: list[str], after_lines: list[str]) -> int:
    prefix_len = 0
    limit = min(len(before_lines), len(after_lines))
    while prefix_len < limit and before_lines[prefix_len] == after_lines[prefix_len]:
        prefix_len += 1
    return prefix_len


def _coarse_span_from_code_diff(*, before_code: str, after_code: str) -> tuple[int, int]:
    before_lines = _code_to_lines(before_code)
    after_lines = _code_to_lines(after_code)
    prefix_len = _longest_common_prefix_len(before_lines, after_lines)

    before_tail = len(before_lines) - 1
    after_tail = len(after_lines) - 1
    while (
        before_tail >= prefix_len
        and after_tail >= prefix_len
        and before_lines[before_tail] == after_lines[after_tail]
    ):
        before_tail -= 1
        after_tail -= 1

    if prefix_len <= after_tail:
        return (prefix_len, after_tail)
    if before_code == after_code:
        raise ObservedCoarsePathError(
            "Edit segment produced no net code change; cannot derive a diff-based span"
        )

    anchor_line = prefix_len if prefix_len < len(after_lines) else len(after_lines) - 1
    if anchor_line < 0:
        raise ObservedCoarsePathError("Derived an invalid anchor line for a pure-deletion edit")
    return (anchor_line, anchor_line)


def _required_anchor_code(item: dict[str, Any]) -> str:
    return _required_str(item, "code_after_anchor")


def _project_span_to_final_code(
    *,
    source_code: str,
    final_code: str,
    source_start_line_0idx: int,
    source_end_line_0idx: int,
) -> tuple[int, int]:
    source_lines = _code_to_lines(source_code)
    final_lines = _code_to_lines(final_code)
    if source_start_line_0idx < 0 or source_end_line_0idx >= len(source_lines):
        raise ObservedCoarsePathError(
            "Source span is outside the source code line range during final-frame projection"
        )

    matcher = SequenceMatcher(a=source_lines, b=final_lines, autojunk=False)
    projected_spans: list[tuple[int, int]] = []
    source_start_exclusive = source_end_line_0idx + 1

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        overlap_start = max(source_start_line_0idx, i1)
        overlap_end_exclusive = min(source_start_exclusive, i2)
        if overlap_start >= overlap_end_exclusive:
            continue
        if tag == "equal":
            mapped_start = j1 + (overlap_start - i1)
            mapped_end_exclusive = j1 + (overlap_end_exclusive - i1)
            projected_spans.append((mapped_start, mapped_end_exclusive - 1))
            continue
        if tag == "replace":
            if j1 == j2:
                continue
            projected_spans.append((j1, j2 - 1))
            continue
        if tag == "delete":
            continue
        raise ObservedCoarsePathError(
            f"Unexpected opcode tag during final-frame projection: {tag!r}"
        )

    if not projected_spans:
        raise ObservedCoarsePathError(
            "Could not project edit span into final submitted-code coordinates"
        )

    return (
        min(span[0] for span in projected_spans),
        max(span[1] for span in projected_spans),
    )


def build_observed_coarse_path_steps(trace: dict[str, Any]) -> list[dict[str, object]]:
    attempt_start_code = _required_str(trace, "attempt_start_code")
    items = _required_items(trace)
    if items[-1].get("item_type") != "submit":
        raise ObservedCoarsePathError("Observed coarse path requires a submit-bounded trace")
    final_submitted_code = _required_anchor_code(items[-1])

    current_code = attempt_start_code
    steps: list[dict[str, object]] = []

    for item_index, item in enumerate(items):
        item_type = item.get("item_type")
        if item_type == "edit_segment":
            if item_index + 1 >= len(items):
                raise ObservedCoarsePathError("Edit segment cannot be the final item in a trace")
            next_item = items[item_index + 1]
            next_item_type = next_item.get("item_type")
            if next_item_type not in {"saida_testar", "submit"}:
                raise ObservedCoarsePathError(
                    "Edit segments must be followed immediately by saida_testar or submit"
                )
            next_code = _required_anchor_code(next_item)
            local_start_line, local_end_line = _coarse_span_from_code_diff(
                before_code=current_code,
                after_code=next_code,
            )
            start_line, end_line = _project_span_to_final_code(
                source_code=next_code,
                final_code=final_submitted_code,
                source_start_line_0idx=local_start_line,
                source_end_line_0idx=local_end_line,
            )
            step = EditCoarsePathStep(
                action_type="edit",
                target_start_line_0idx=start_line,
                target_end_line_0idx=end_line,
            )
            steps.append(step.model_dump())
            current_code = next_code
            continue

        if item_type == "saida_testar":
            step = LocalRunCoarsePathStep(
                action_type="local_run",
                target_start_line_0idx=None,
                target_end_line_0idx=None,
            )
            steps.append(step.model_dump())
            current_code = _required_anchor_code(item)
            continue

        if item_type == "submit":
            step = SubmitCoarsePathStep(
                action_type="submit",
                target_start_line_0idx=None,
                target_end_line_0idx=None,
            )
            steps.append(step.model_dump())
            current_code = _required_anchor_code(item)
            continue

        raise ObservedCoarsePathError(f"Unsupported trace item type: {item_type!r}")

    return steps


def build_observed_coarse_path_artifact(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index_0idx: int,
    attempt_index_0idx: int,
    aligned_submit_index_0idx: int,
    trace: dict[str, Any],
) -> dict[str, object]:
    steps = build_observed_coarse_path_steps(trace)
    items = _required_items(trace)
    final_item = items[-1]
    if final_item.get("item_type") != "submit":
        raise ObservedCoarsePathError(
            "Observed coarse path artifact requires the final item to be submit"
        )
    submitted_code = _required_anchor_code(final_item)

    return {
        "schema_version": "v6_2_observed_coarse_path_v1",
        "source": {
            "class_id": class_id,
            "assessment_id": assessment_id,
            "exercise_id": exercise_id,
            "student_id": student_id,
            "transition_index_0idx": transition_index_0idx,
        },
        "attempt_n1": {
            "attempt_index_0idx": attempt_index_0idx,
            "aligned_submit_index_0idx": aligned_submit_index_0idx,
            "submitted_code": submitted_code,
            "coarse_path_steps": steps,
        },
    }
