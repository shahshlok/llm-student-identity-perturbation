from __future__ import annotations

import json

from .codemirror import (
    first_change_line_0idx,
    lines_touched_0idx,
    normalize_code_for_match,
)
from .models import AlignmentResult, ExecutionTransition, SubmitSnapshot


class AlignmentError(ValueError):
    pass


def _candidate_snapshot_indices(
    code: str, snapshots: tuple[SubmitSnapshot, ...]
) -> tuple[int, ...]:
    normalized = normalize_code_for_match(code)
    indices = [
        index
        for index, snapshot in enumerate(snapshots)
        if normalize_code_for_match(snapshot.code) == normalized
    ]
    return tuple(indices)


def _candidate_signature(changes_between: tuple[dict, ...]) -> str:
    return json.dumps(changes_between, ensure_ascii=True, sort_keys=True)


def _choose_unambiguous_candidate(
    candidates: list[tuple[str, AlignmentResult]],
    context: str,
) -> AlignmentResult:
    if not candidates:
        raise AlignmentError(f"No alignment candidates available for {context}")
    unique_signatures = {signature for signature, _ in candidates}
    if len(unique_signatures) > 1:
        raise AlignmentError(f"Ambiguous alignment candidates for {context}")
    return candidates[0][1]


def align_transition(
    transition: ExecutionTransition,
    snapshots: tuple[SubmitSnapshot, ...],
    trailing_code: str | None = None,
    trailing_changes: tuple[dict, ...] = (),
    allow_partial: bool = False,
) -> AlignmentResult:
    candidates_n = _candidate_snapshot_indices(transition.attempt_n.code, snapshots)
    candidates_n1 = _candidate_snapshot_indices(transition.attempt_n1.code, snapshots)

    if not candidates_n:
        raise AlignmentError("attempt_n code not found in any submit snapshot")

    matched_candidates: list[tuple[str, AlignmentResult]] = []
    for snap_n_index in candidates_n:
        for snap_n1_index in candidates_n1:
            if snap_n1_index <= snap_n_index:
                continue
            changes_between = []
            for snapshot in snapshots[snap_n_index + 1 : snap_n1_index + 1]:
                changes_between.extend(snapshot.changes_since_prev)
            if not changes_between:
                continue
            changes_tuple = tuple(changes_between)
            matched_candidates.append(
                (
                    _candidate_signature(changes_tuple),
                    AlignmentResult(
                        status="matched",
                        snap_n_index=snap_n_index,
                        snap_n1_index=snap_n1_index,
                        submit_n_timestamp=snapshots[
                            snap_n_index
                        ].submit_event.timestamp.isoformat(),
                        submit_n1_timestamp=snapshots[
                            snap_n1_index
                        ].submit_event.timestamp.isoformat(),
                        changes_between=changes_tuple,
                        first_change_line_0idx=first_change_line_0idx(changes_tuple),
                        lines_touched_0idx=lines_touched_0idx(changes_tuple),
                    ),
                )
            )

    if matched_candidates:
        return _choose_unambiguous_candidate(
            matched_candidates,
            context="matched snapshot pair",
        )

    if not allow_partial:
        raise AlignmentError("attempt_n1 code not found in later submit snapshots")

    if trailing_code is None:
        raise AlignmentError("attempt_n1 code not found in later submit snapshots")

    normalized_trailing = normalize_code_for_match(trailing_code)
    normalized_target = normalize_code_for_match(transition.attempt_n1.code)
    if normalized_trailing != normalized_target:
        raise AlignmentError("Trailing replay does not reconstruct attempt_n1 code exactly")

    partial_candidates: list[tuple[str, AlignmentResult]] = []
    for snap_n_index in candidates_n:
        changes_between = []
        for snapshot in snapshots[snap_n_index + 1 :]:
            changes_between.extend(snapshot.changes_since_prev)
        changes_between.extend(trailing_changes)
        if not changes_between:
            continue
        changes_tuple = tuple(changes_between)
        partial_candidates.append(
            (
                _candidate_signature(changes_tuple),
                AlignmentResult(
                    status="partial",
                    snap_n_index=snap_n_index,
                    snap_n1_index=None,
                    submit_n_timestamp=snapshots[snap_n_index].submit_event.timestamp.isoformat(),
                    submit_n1_timestamp=None,
                    changes_between=changes_tuple,
                    first_change_line_0idx=first_change_line_0idx(changes_tuple),
                    lines_touched_0idx=lines_touched_0idx(changes_tuple),
                ),
            )
        )

    if not partial_candidates:
        raise AlignmentError("Partial alignment has no trailing changes")
    return _choose_unambiguous_candidate(
        partial_candidates,
        context="partial trailing replay",
    )
