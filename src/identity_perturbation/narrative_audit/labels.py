from __future__ import annotations

from typing import Final

from identity_perturbation.codebench_support.models import ExecutionTransition, ReplayTrace
from identity_perturbation.codebench_support.runner import HEAD_LABELS, build_transition_window_record

EVALUATED_HEAD_SPECS: Final[dict[str, dict[str, object]]] = {
    "likely_first_repair_region": {
        "source_label_key": "first_focus_region_3way",
        "label_space": HEAD_LABELS["first_focus_region_3way"],
    },
    "likely_edit_scope": {
        "source_label_key": "lines_touched_bucket_3way",
        "label_space": HEAD_LABELS["lines_touched_bucket_3way"],
    },
    "likely_next_test_outcome": {
        "source_label_key": "next_test_outcome",
        "label_space": HEAD_LABELS["next_test_outcome"],
    },
}


def build_observed_labels(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    student_id: str,
    transition_index: int,
    transition: ExecutionTransition,
    trace: ReplayTrace,
) -> dict[str, object]:
    _, record = build_transition_window_record(
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        student_id=student_id,
        transition_index=transition_index,
        transition=transition,
        trace=trace,
        allow_partial=False,
    )

    observed_heads = {
        head_key: str(record[str(head_spec["source_label_key"])])
        for head_key, head_spec in EVALUATED_HEAD_SPECS.items()
    }
    source_labels = {
        str(head_spec["source_label_key"]): str(record[str(head_spec["source_label_key"])])
        for head_spec in EVALUATED_HEAD_SPECS.values()
    }
    return {
        "schema_version": "v6_observed_labels_v1",
        "class_id": class_id,
        "assessment_id": assessment_id,
        "exercise_id": exercise_id,
        "student_id": student_id,
        "transition_index_0idx": transition_index,
        "attempt_n_index_0idx": transition.attempt_n.attempt_index_0idx,
        "attempt_n1_index_0idx": transition.attempt_n1.attempt_index_0idx,
        "observed_heads": observed_heads,
        "source_labels": source_labels,
        "alignment": {
            "status": str(record["alignment_status"]),
            "snap_n_index_0idx": int(record["snap_n_index_0idx"]),
            "snap_n1_index_0idx": int(record["snap_n1_index_0idx"]),
            "submit_n_timestamp": str(record["submit_n_timestamp"]),
            "submit_n1_timestamp": str(record["submit_n1_timestamp"]),
            "first_change_line_0idx": int(record["first_change_line_0idx"]),
            "lines_touched_0idx": [int(value) for value in record["lines_touched_0idx"]],
        },
    }
