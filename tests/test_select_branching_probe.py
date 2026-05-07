from __future__ import annotations

import json
from pathlib import Path

import pytest

import identity_perturbation.prediction_audit.select_branching_probe as selector
from identity_perturbation.prediction_audit.select_branching_probe import (
    BranchingProbeSelectionError,
    ScopePairSelection,
    ScopeRecord,
    ScopeWithTertile,
    StudentLogCandidate,
    ValidatedStudentCandidate,
    _resolve_buildable_student_selection,
    stage1_canonical_exclusion,
    stage2_assign_tertiles,
    stage3_enumerate_scope_candidates,
    stage4_validate_attempt_depth,
    stage5_select_student_pairs,
    stage6_stratified_sampling,
    stage7_emit_bundles,
    stage8_self_validate_manifest,
)


def _scope(
    *,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    mean_pairwise_dist: float,
    canonical_80: bool = False,
) -> ScopeRecord:
    return ScopeRecord(
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        n_students=4,
        mean_pairwise_dist=mean_pairwise_dist,
        distinct_codes=3,
        canonical_80=canonical_80,
    )


def _write_text(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _submission_block(index: int) -> str:
    return "\n".join(
        [
            f"== SUBMITION (2024-01-01 10:00:0{index})",
            "-- CODE:",
            f"print({index})",
            "-- GRADE:",
            f"{index}.0",
            "-- TEST CASE 1:",
            "---- input:",
            "1",
            "---- correct output:",
            "1",
            "---- user output:",
            "1",
        ]
    )


def _write_execution_log(path: Path, submission_count: int) -> None:
    _write_text(
        path, "\n".join(_submission_block(index) for index in range(submission_count)) + "\n"
    )


def _pass_vector_signature(vector: tuple[bool, ...]) -> str:
    return "".join("P" if value else "F" for value in vector)


def _validated_student(
    *,
    tmp_path: Path,
    student_id: str,
    exercise_id: str,
    pass_vector: tuple[bool, ...] = (True,),
    total_attempt_count: int = 3,
    interior_n: int = 1,
) -> ValidatedStudentCandidate:
    execution_path = tmp_path / student_id / "executions" / f"201_{exercise_id}.log"
    codemirror_path = tmp_path / student_id / "codemirror" / f"201_{exercise_id}.log"
    _write_execution_log(execution_path, submission_count=total_attempt_count)
    _write_text(codemirror_path)
    failed_indices = tuple(index for index, passed in enumerate(pass_vector) if not passed)
    return ValidatedStudentCandidate(
        class_id="101",
        assessment_id="201",
        exercise_id=exercise_id,
        student_id=student_id,
        execution_path=execution_path,
        codemirror_path=codemirror_path,
        total_attempt_count=total_attempt_count,
        interior_n=interior_n,
        attempt_n_code="print(1)",
        attempt_n_normalized_code="print(1)",
        attempt_n1_code="print(2)",
        attempt_n_pass_fail_vector=pass_vector,
        attempt_n_failed_test_indices_0idx=failed_indices,
        attempt_n_test_count=len(pass_vector),
        attempt_n_pass_vector_signature=_pass_vector_signature(pass_vector),
    )


def _manifest_bundle_entry(
    *,
    tmp_path: Path,
    custom_id: str,
    scope_id: str,
    tertile: str,
    vector: list[bool],
    condition: str = "full",
    transition_index: int = 1,
    visible_attempt_count: int = 2,
    model: str = "gpt-5.4",
    reasoning_effort: str = "medium",
    missing_observed: bool = False,
) -> dict[str, str | int | list[bool] | list[int]]:
    bundle_dir = tmp_path / "bundles" / custom_id.replace(":", "__")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_manifest_path = bundle_dir / "manifest.json"
    request_body_path = bundle_dir / "request_body.json"
    batch_request_path = bundle_dir / "batch_request.json"
    batch_request_jsonl_path = bundle_dir / "requests.jsonl"
    repair_path = bundle_dir / "repair.json"
    coarse_path = bundle_dir / "coarse.json"
    request_body = {
        "model": model,
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "text": {"format": {"type": "json_schema", "name": "stub", "schema": {"type": "object"}}},
        "reasoning": {"effort": reasoning_effort},
    }
    batch_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": request_body,
    }
    _write_text(repair_path, "{}")
    if not missing_observed:
        _write_text(coarse_path, "{}")
    failed_indices = [index for index, passed in enumerate(vector) if not passed]
    bundle_manifest = {
        "schema_version": selector.BUNDLE_SCHEMA_VERSION,
        "custom_id": custom_id,
        "condition": condition,
        "exercise_scope": scope_id,
        "tertile": tertile,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "request_body_path": "request_body.json",
        "batch_request_path": "batch_request.json",
        "batch_request_jsonl_path": "requests.jsonl",
        "observed_next_repair_target_path": "repair.json",
        "observed_next_coarse_path_path": "coarse.json",
        "transition_index_0idx": transition_index,
        "visible_attempt_count": visible_attempt_count,
        "attempt_n_pass_fail_vector": vector,
        "attempt_n_failed_test_indices_0idx": failed_indices,
        "attempt_n_test_count": len(vector),
        "attempt_n_pass_vector_signature": "".join("P" if passed else "F" for passed in vector),
        "attempt_n_normalized_code": "print(1)",
    }
    _write_text(request_body_path, json.dumps(request_body, ensure_ascii=False, indent=2) + "\n")
    _write_text(batch_request_path, json.dumps(batch_request, ensure_ascii=False, indent=2) + "\n")
    _write_text(batch_request_jsonl_path, json.dumps(batch_request, ensure_ascii=False) + "\n")
    _write_text(
        bundle_manifest_path, json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "condition": condition,
        "bundle_dir": str(bundle_dir),
        "bundle_manifest_path": str(bundle_manifest_path),
        "request_body_path": str(request_body_path),
        "exercise_scope": scope_id,
        "tertile": tertile,
        "transition_index_0idx": transition_index,
        "visible_attempt_count": visible_attempt_count,
        "attempt_n_pass_fail_vector": vector,
        "attempt_n_failed_test_indices_0idx": failed_indices,
        "attempt_n_test_count": len(vector),
        "attempt_n_pass_vector_signature": "".join("P" if passed else "F" for passed in vector),
        "attempt_n_normalized_code": "print(1)",
        "observed_next_repair_target_path": str(repair_path),
        "observed_next_coarse_path_path": str(coarse_path),
    }


def _manifest_paths_and_counts(
    *,
    tmp_path: Path,
    ordered_custom_ids: list[str],
    bundle_map: dict[str, dict[str, object]],
) -> tuple[dict[str, str], dict[str, int]]:
    bundles_root = tmp_path / "bundles"
    bundles_root.mkdir(parents=True, exist_ok=True)
    requests_path = tmp_path / "requests.jsonl"
    request_lines: list[str] = []
    for custom_id in ordered_custom_ids:
        request_body_path = Path(str(bundle_map[custom_id]["request_body_path"]))
        request_body = json.loads(request_body_path.read_text(encoding="utf-8"))
        request_lines.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": request_body,
                }
            )
        )
    requests_path.write_text("\n".join(request_lines) + "\n", encoding="utf-8")
    return (
        {
            "bundles_root": str(bundles_root),
            "requests_jsonl": str(requests_path),
            "output_jsonl": str(tmp_path / "output.jsonl"),
            "error_jsonl": str(tmp_path / "errors.jsonl"),
        },
        {
            "scopes": 0,
            "rows": len(ordered_custom_ids),
            "requests_created": len(ordered_custom_ids),
        },
    )


def _documented_scope_row(
    *,
    scope_id: str,
    tertile: str,
    row_count: int,
    signature_counts: dict[str, int],
    buildable_student_ids_before_l2a: list[str],
    excluded_buildable_student_ids: list[str],
) -> dict[str, object]:
    return {
        "scope_id": scope_id,
        "tertile": tertile,
        "row_count": row_count,
        "l2a_group_signature_counts": signature_counts,
        "buildable_student_count_before_l2a": len(buildable_student_ids_before_l2a),
        "buildable_student_ids_before_l2a": buildable_student_ids_before_l2a,
        "excluded_buildable_student_count": len(excluded_buildable_student_ids),
        "excluded_buildable_student_ids": excluded_buildable_student_ids,
        "dropped_singleton_l2a_groups": [],
    }


def _write_stage8_manifest(
    *,
    tmp_path: Path,
    bundle_map: dict[str, dict[str, object]],
    ordered_custom_ids: list[str],
    condition: str = "full",
    model: str = "gpt-5.4",
    reasoning_effort: str = "medium",
    scope_id: str = "101:201:301",
    tertile: str = "T1",
    row_count: int | None = None,
    signature_counts: dict[str, int] | None = None,
    buildable_student_ids_before_l2a: list[str] | None = None,
    excluded_buildable_student_ids: list[str] | None = None,
) -> Path:
    paths, counts = _manifest_paths_and_counts(
        tmp_path=tmp_path,
        ordered_custom_ids=ordered_custom_ids,
        bundle_map=bundle_map,
    )
    counts["scopes"] = 1
    actual_row_count = len(ordered_custom_ids) if row_count is None else row_count
    if signature_counts is None:
        signatures = [
            str(bundle_map[custom_id]["attempt_n_pass_vector_signature"])
            for custom_id in ordered_custom_ids
        ]
        signature_counts = {}
        for signature in signatures:
            signature_counts[signature] = signature_counts.get(signature, 0) + 1
    if buildable_student_ids_before_l2a is None:
        buildable_student_ids_before_l2a = [
            custom_id.split(":")[1] for custom_id in ordered_custom_ids
        ]
    if excluded_buildable_student_ids is None:
        excluded_buildable_student_ids = []
    manifest = {
        "schema_version": "v6_2_full_trace_pilot_batch_manifest_v1",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "condition": condition,
        "counts": counts,
        "paths": paths,
        "ordered_custom_ids": ordered_custom_ids,
        "bundle_map": bundle_map,
        "selection": {
            "condition": condition,
            "target_scopes": 1,
            "fixed_transition_index_0idx": 1,
            "fixed_visible_attempt_count": 2,
            "matched_cohort_preflight_condition": "full",
            "documented_final_scope_counts": {"T1": 1, "T2": 0, "T3": 0},
            "documented_final_row_counts": {"T1": actual_row_count, "T2": 0, "T3": 0},
            "documented_final_scope_rows": [
                _documented_scope_row(
                    scope_id=scope_id,
                    tertile=tertile,
                    row_count=actual_row_count,
                    signature_counts=signature_counts,
                    buildable_student_ids_before_l2a=buildable_student_ids_before_l2a,
                    excluded_buildable_student_ids=excluded_buildable_student_ids,
                )
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_stage1_canonical_exclusion_drops_only_canonical_scopes() -> None:
    retained, report = stage1_canonical_exclusion(
        [
            _scope(class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.11),
            _scope(
                class_id="101",
                assessment_id="201",
                exercise_id="302",
                mean_pairwise_dist=0.22,
                canonical_80=True,
            ),
            _scope(class_id="101", assessment_id="201", exercise_id="303", mean_pairwise_dist=0.33),
        ]
    )

    assert [scope.scope_id for scope in retained] == ["101:201:301", "101:201:303"]
    assert report["input_count"] == 3
    assert report["dropped_count"] == 1
    assert report["retained_count"] == 2
    assert report["dropped_scopes"] == [
        {
            "scope_id": "101:201:302",
            "reason": "canonical_80_true",
            "mean_pairwise_dist": 0.22,
            "n_students": 4,
            "distinct_codes": 3,
        }
    ]


def test_stage2_assign_tertiles_uses_empirical_percentiles() -> None:
    scopes = [
        _scope(class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.10),
        _scope(class_id="101", assessment_id="201", exercise_id="302", mean_pairwise_dist=0.20),
        _scope(class_id="101", assessment_id="201", exercise_id="303", mean_pairwise_dist=0.30),
        _scope(class_id="101", assessment_id="201", exercise_id="304", mean_pairwise_dist=0.40),
        _scope(class_id="101", assessment_id="201", exercise_id="305", mean_pairwise_dist=0.50),
        _scope(class_id="101", assessment_id="201", exercise_id="306", mean_pairwise_dist=0.60),
    ]

    assignments, report = stage2_assign_tertiles(scopes)

    assert pytest.approx(report["p33_mean_pairwise_dist"], rel=0.0, abs=1e-9) == 0.265
    assert pytest.approx(report["p67_mean_pairwise_dist"], rel=0.0, abs=1e-9) == 0.435
    assert report["tertile_counts"] == {"T1": 2, "T2": 2, "T3": 2}
    assert [(item.scope.scope_id, item.tertile) for item in assignments] == [
        ("101:201:301", "T1"),
        ("101:201:302", "T1"),
        ("101:201:303", "T2"),
        ("101:201:304", "T2"),
        ("101:201:305", "T3"),
        ("101:201:306", "T3"),
    ]


def test_stage3_enumerate_scope_candidates_checks_both_log_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    scope = ScopeWithTertile(
        scope=_scope(
            class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
        ),
        tertile="T2",
    )

    users_root = dataset_root / "101" / "users"
    (users_root / "9001").mkdir(parents=True)
    (users_root / "9002").mkdir(parents=True)
    (users_root / "9003").mkdir(parents=True)
    (users_root / "9004").mkdir(parents=True)

    _write_text(users_root / "9001" / "executions" / "201_301.log")
    _write_text(users_root / "9001" / "codemirror" / "201_301.log")
    _write_text(users_root / "9002" / "executions" / "201_301.log")
    _write_text(users_root / "9003" / "codemirror" / "201_301.log")

    scope_candidates, report = stage3_enumerate_scope_candidates([scope], dataset_root)

    assert [item.student_id for item in scope_candidates["101:201:301"]] == ["9001"]
    assert report["total_candidates_with_both_logs"] == 1
    assert report["total_missing_logs"] == 3
    assert report["scope_reports"] == [
        {
            "scope_id": "101:201:301",
            "tertile": "T2",
            "n_users_in_class": 4,
            "n_candidates_with_both_logs": 1,
            "n_missing_logs": 3,
            "missing_log_reason_counts": {
                "codemirror_log_missing": 1,
                "execution_log_missing": 1,
                "execution_log_missing+codemirror_log_missing": 1,
            },
            "candidate_student_ids": ["9001"],
            "dropped_students": [
                {
                    "student_id": "9002",
                    "reason": "codemirror_log_missing",
                    "execution_path": str(users_root / "9002" / "executions" / "201_301.log"),
                    "codemirror_path": str(users_root / "9002" / "codemirror" / "201_301.log"),
                },
                {
                    "student_id": "9003",
                    "reason": "execution_log_missing",
                    "execution_path": str(users_root / "9003" / "executions" / "201_301.log"),
                    "codemirror_path": str(users_root / "9003" / "codemirror" / "201_301.log"),
                },
                {
                    "student_id": "9004",
                    "reason": "execution_log_missing+codemirror_log_missing",
                    "execution_path": str(users_root / "9004" / "executions" / "201_301.log"),
                    "codemirror_path": str(users_root / "9004" / "codemirror" / "201_301.log"),
                },
            ],
        }
    ]


def test_stage4_validate_attempt_depth_locks_to_fixed_n2(tmp_path: Path) -> None:
    execution_path = tmp_path / "9001" / "executions" / "201_301.log"
    codemirror_path = tmp_path / "9001" / "codemirror" / "201_301.log"
    _write_execution_log(execution_path, submission_count=4)
    _write_text(codemirror_path)
    scope = ScopeWithTertile(
        scope=_scope(
            class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
        ),
        tertile="T2",
    )
    scope_candidates = {
        "101:201:301": [
            StudentLogCandidate(
                class_id="101",
                assessment_id="201",
                exercise_id="301",
                student_id="9001",
                execution_path=execution_path,
                codemirror_path=codemirror_path,
            )
        ]
    }

    validated_by_scope, report = stage4_validate_attempt_depth([scope], scope_candidates)

    retained = validated_by_scope["101:201:301"]
    assert len(retained) == 1
    assert retained[0].interior_n == 1
    assert retained[0].visible_attempt_count == 2
    assert retained[0].attempt_n_code == "print(1)"
    assert retained[0].attempt_n_normalized_code == "print(1)"
    assert retained[0].attempt_n1_code == "print(2)"
    assert retained[0].attempt_n_pass_fail_vector == (True,)
    assert retained[0].attempt_n_failed_test_indices_0idx == ()
    assert retained[0].attempt_n_test_count == 1
    assert retained[0].attempt_n_pass_vector_signature == "P"
    assert report["fixed_transition_index_0idx"] == 1
    assert report["fixed_visible_attempt_count"] == 2
    assert report["minimum_total_attempt_count"] == 3
    assert report["scope_reports"][0]["retained_students"][0]["visible_attempt_count"] == 2


def test_stage4_validate_attempt_depth_drops_students_with_fewer_than_three_attempts(
    tmp_path: Path,
) -> None:
    execution_path = tmp_path / "9001" / "executions" / "201_301.log"
    codemirror_path = tmp_path / "9001" / "codemirror" / "201_301.log"
    _write_execution_log(execution_path, submission_count=2)
    _write_text(codemirror_path)
    scope = ScopeWithTertile(
        scope=_scope(
            class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
        ),
        tertile="T2",
    )
    scope_candidates = {
        "101:201:301": [
            StudentLogCandidate(
                class_id="101",
                assessment_id="201",
                exercise_id="301",
                student_id="9001",
                execution_path=execution_path,
                codemirror_path=codemirror_path,
            )
        ]
    }

    validated_by_scope, report = stage4_validate_attempt_depth([scope], scope_candidates)

    assert validated_by_scope["101:201:301"] == []
    assert report["total_dropped_insufficient_attempts"] == 1
    assert report["scope_reports"][0]["dropped_insufficient_attempts_students"] == [
        {
            "student_id": "9001",
            "reason": "insufficient_parseable_submissions",
            "attempt_count": 2,
            "required_attempt_count": 3,
            "execution_path": str(execution_path),
        }
    ]


def test_stage5_select_student_pairs_keeps_all_non_singleton_l2a_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = ScopeWithTertile(
        scope=_scope(
            class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
        ),
        tertile="T2",
    )
    validated_by_scope = {
        "101:201:301": [
            _validated_student(
                tmp_path=tmp_path, student_id="9001", exercise_id="301", pass_vector=(False, True)
            ),
            _validated_student(
                tmp_path=tmp_path, student_id="9002", exercise_id="301", pass_vector=(False, True)
            ),
            _validated_student(
                tmp_path=tmp_path, student_id="9003", exercise_id="301", pass_vector=(True, False)
            ),
            _validated_student(
                tmp_path=tmp_path, student_id="9004", exercise_id="301", pass_vector=(True, False)
            ),
            _validated_student(
                tmp_path=tmp_path, student_id="9005", exercise_id="301", pass_vector=(True, True)
            ),
        ]
    }

    monkeypatch.setattr(
        selector,
        "_resolve_buildable_student_selection",
        lambda student, dataset_root, condition: (student, None),
    )

    selected_scopes, report = stage5_select_student_pairs(
        [scope],
        validated_by_scope,
        seed=42,
        dataset_root=tmp_path,
        condition="full",
    )

    assert len(selected_scopes) == 1
    selected_students = selected_scopes[0].students
    assert [student.student_id for student in selected_students] == ["9001", "9002", "9003", "9004"]
    assert report["selected_scope_count"] == 1
    assert report["selected_row_count"] == 4
    assert report["selected_l2a_group_count"] == 2
    assert report["cohort_preflight_condition"] == "full"
    assert report["dropped_scope_count_no_l2a_match"] == 0
    assert report["selected_scopes"] == [
        {
            "scope_id": "101:201:301",
            "tertile": "T2",
            "buildable_student_count_before_l2a": 5,
            "buildable_student_ids_before_l2a": ["9001", "9002", "9003", "9004", "9005"],
            "selected_row_count": 4,
            "excluded_buildable_student_count": 1,
            "excluded_buildable_student_ids": ["9005"],
            "l2a_group_signature_counts": {"FP": 2, "PF": 2},
            "preflight_failures": [],
            "dropped_singleton_l2a_groups": [
                {
                    "attempt_n_pass_vector_signature": "PP",
                    "attempt_n_pass_fail_vector": [True, True],
                    "student_ids": ["9005"],
                }
            ],
            "students": [
                selector._validated_student_to_dict(student) for student in selected_students
            ],
        }
    ]


def test_stage5_select_student_pairs_freezes_cohort_on_full_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = ScopeWithTertile(
        scope=_scope(
            class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
        ),
        tertile="T2",
    )
    validated_by_scope = {
        "101:201:301": [
            _validated_student(
                tmp_path=tmp_path, student_id="9001", exercise_id="301", pass_vector=(False, True)
            ),
            _validated_student(
                tmp_path=tmp_path, student_id="9002", exercise_id="301", pass_vector=(False, True)
            ),
        ]
    }
    attempted_conditions: list[str] = []

    def fake_resolve(
        student: ValidatedStudentCandidate,
        dataset_root: Path,
        condition: str,
    ) -> tuple[ValidatedStudentCandidate | None, dict[str, object] | None]:
        attempted_conditions.append(condition)
        return student, None

    monkeypatch.setattr(selector, "_resolve_buildable_student_selection", fake_resolve)

    selected_scopes, report = stage5_select_student_pairs(
        [scope],
        validated_by_scope,
        seed=42,
        dataset_root=tmp_path,
        condition="no_trace",
    )

    assert len(selected_scopes) == 1
    assert attempted_conditions == ["full", "full"]
    assert report["cohort_preflight_condition"] == "full"


def test_stage6_stratified_sampling_reports_scope_and_row_counts(tmp_path: Path) -> None:
    selections = [
        ScopePairSelection(
            scope=ScopeWithTertile(
                scope=_scope(
                    class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.10
                ),
                tertile="T1",
            ),
            students=(
                _validated_student(tmp_path=tmp_path, student_id="9001", exercise_id="301"),
                _validated_student(tmp_path=tmp_path, student_id="9002", exercise_id="301"),
            ),
        ),
        ScopePairSelection(
            scope=ScopeWithTertile(
                scope=_scope(
                    class_id="101", assessment_id="201", exercise_id="302", mean_pairwise_dist=0.20
                ),
                tertile="T2",
            ),
            students=(
                _validated_student(tmp_path=tmp_path, student_id="9003", exercise_id="302"),
                _validated_student(tmp_path=tmp_path, student_id="9004", exercise_id="302"),
                _validated_student(tmp_path=tmp_path, student_id="9005", exercise_id="302"),
            ),
        ),
        ScopePairSelection(
            scope=ScopeWithTertile(
                scope=_scope(
                    class_id="101", assessment_id="201", exercise_id="303", mean_pairwise_dist=0.30
                ),
                tertile="T3",
            ),
            students=(
                _validated_student(tmp_path=tmp_path, student_id="9006", exercise_id="303"),
                _validated_student(tmp_path=tmp_path, student_id="9007", exercise_id="303"),
            ),
        ),
    ]

    selected, _remaining, report = stage6_stratified_sampling(selections, target_scopes=3, seed=42)

    assert [item.scope_id for item in selected] == ["101:201:301", "101:201:302", "101:201:303"]
    assert report["available_scope_counts"] == {"T1": 1, "T2": 1, "T3": 1}
    assert report["available_row_counts"] == {"T1": 2, "T2": 3, "T3": 2}
    assert report["final_selected_counts"] == {"T1": 1, "T2": 1, "T3": 1}
    assert report["final_selected_row_counts"] == {"T1": 2, "T2": 3, "T3": 2}
    assert report["reserve_scope_count"] == 0
    assert report["reserve_row_count"] == 0


def test_stage5_select_student_pairs_drops_scope_without_non_singleton_l2a_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = ScopeWithTertile(
        scope=_scope(
            class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
        ),
        tertile="T2",
    )
    validated_by_scope = {
        "101:201:301": [
            _validated_student(
                tmp_path=tmp_path, student_id="9001", exercise_id="301", pass_vector=(False, True)
            ),
            _validated_student(
                tmp_path=tmp_path, student_id="9002", exercise_id="301", pass_vector=(True, False)
            ),
        ]
    }

    monkeypatch.setattr(
        selector,
        "_resolve_buildable_student_selection",
        lambda student, dataset_root, condition: (student, None),
    )

    selected_scopes, report = stage5_select_student_pairs(
        [scope],
        validated_by_scope,
        seed=42,
        dataset_root=tmp_path,
        condition="full",
    )

    assert selected_scopes == []
    assert report["selected_scope_count"] == 0
    assert report["selected_row_count"] == 0
    assert report["dropped_scope_count_no_l2a_match"] == 1
    assert report["dropped_scopes"] == [
        {
            "scope_id": "101:201:301",
            "tertile": "T2",
            "reason": "no_l2a_group_with_at_least_two_buildable_students",
            "buildable_student_ids": ["9001", "9002"],
            "preflight_failures": [],
            "dropped_singleton_l2a_groups": [
                {
                    "attempt_n_pass_vector_signature": "FP",
                    "attempt_n_pass_fail_vector": [False, True],
                    "student_ids": ["9001"],
                },
                {
                    "attempt_n_pass_vector_signature": "PF",
                    "attempt_n_pass_fail_vector": [True, False],
                    "student_ids": ["9002"],
                },
            ],
        }
    ]


def test_resolve_buildable_student_selection_does_not_fallback_to_later_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_path = tmp_path / "9001" / "executions" / "201_301.log"
    codemirror_path = tmp_path / "9001" / "codemirror" / "201_301.log"
    _write_execution_log(execution_path, submission_count=4)
    _write_text(codemirror_path)
    student = ValidatedStudentCandidate(
        class_id="101",
        assessment_id="201",
        exercise_id="301",
        student_id="9001",
        execution_path=execution_path,
        codemirror_path=codemirror_path,
        total_attempt_count=4,
        interior_n=1,
        attempt_n_code="print(1)",
        attempt_n_normalized_code="print(1)",
        attempt_n1_code="print(2)",
    )
    attempted_indices: list[int] = []

    def fake_build_prompt_payload(
        *, transition_index: int, **_: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        attempted_indices.append(transition_index)
        if transition_index == 1:
            raise RuntimeError("fixed transition failed")
        return ({"source": {"condition": "full"}, "visible_attempts": []}, {}, {})

    monkeypatch.setattr(selector, "_build_prompt_payload", fake_build_prompt_payload)

    resolved, failure = _resolve_buildable_student_selection(student, tmp_path, "full")

    assert resolved is None
    assert attempted_indices == [1]
    assert failure == {
        "student_id": "9001",
        "reason": "fixed_transition_not_buildable",
        "attempted_failures": [
            {
                "interior_n": 1,
                "error_type": "RuntimeError",
                "error_message": "fixed transition failed",
            }
        ],
    }


def test_stage7_emit_bundles_fails_loudly_on_first_scope_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    scope_selection = ScopePairSelection(
        scope=ScopeWithTertile(
            scope=_scope(
                class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
            ),
            tertile="T2",
        ),
        students=(
            _validated_student(tmp_path=tmp_path, student_id="9001", exercise_id="301"),
            _validated_student(tmp_path=tmp_path, student_id="9002", exercise_id="301"),
        ),
    )

    def fake_emit_one_bundle(
        *, out_dir: Path, student: ValidatedStudentCandidate, **_: object
    ) -> dict[str, object]:
        bundle_dir = selector._bundle_dir(out_dir, student)
        bundle_dir.mkdir(parents=True, exist_ok=False)
        _write_text(bundle_dir / "partial.txt", "partial")
        raise RuntimeError("emit failed")

    monkeypatch.setattr(selector, "_emit_one_bundle", fake_emit_one_bundle)

    with pytest.raises(
        BranchingProbeSelectionError, match=r"Bundle emission failed for 101:201:301: emit failed"
    ):
        stage7_emit_bundles(
            [scope_selection],
            out_dir=out_dir,
            dataset_root=tmp_path,
            target_scopes=1,
            condition="full",
        )
    assert not selector._bundle_dir(out_dir, scope_selection.students[0]).exists()


def test_stage7_emit_bundles_cleans_previous_scope_outputs_when_later_scope_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def make_student(student_id: str, exercise_id: str) -> ValidatedStudentCandidate:
        return _validated_student(tmp_path=tmp_path, student_id=student_id, exercise_id=exercise_id)

    first_scope = ScopePairSelection(
        scope=ScopeWithTertile(
            scope=_scope(
                class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
            ),
            tertile="T1",
        ),
        students=(make_student("9001", "301"), make_student("9002", "301")),
    )
    second_scope = ScopePairSelection(
        scope=ScopeWithTertile(
            scope=_scope(
                class_id="101", assessment_id="201", exercise_id="302", mean_pairwise_dist=0.35
            ),
            tertile="T2",
        ),
        students=(make_student("9003", "302"), make_student("9004", "302")),
    )
    call_count = 0

    def fake_emit_one_bundle(
        *, out_dir: Path, student: ValidatedStudentCandidate, **_: object
    ) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        bundle_dir = selector._bundle_dir(out_dir, student)
        bundle_dir.mkdir(parents=True, exist_ok=False)
        _write_text(bundle_dir / "partial.txt", "partial")
        if call_count <= 2:
            return {
                "custom_id": student.custom_id,
                "request_jsonl_line": json.dumps({"custom_id": student.custom_id}),
                "bundle_map_entry": {
                    "condition": "full",
                    "exercise_scope": f"{student.class_id}:{student.assessment_id}:{student.exercise_id}",
                    "tertile": "T1",
                    "transition_index_0idx": 1,
                    "visible_attempt_count": 2,
                    "attempt_n_pass_fail_vector": [True],
                    "attempt_n_failed_test_indices_0idx": [],
                    "attempt_n_test_count": 1,
                    "attempt_n_pass_vector_signature": "P",
                    "observed_next_repair_target_path": str(bundle_dir / "repair.json"),
                    "observed_next_coarse_path_path": str(bundle_dir / "coarse.json"),
                },
                "bundle_dir": bundle_dir,
            }
        raise RuntimeError("emit failed on later scope")

    monkeypatch.setattr(selector, "_emit_one_bundle", fake_emit_one_bundle)

    with pytest.raises(
        BranchingProbeSelectionError,
        match=r"Bundle emission failed for 101:201:302: emit failed on later scope",
    ):
        stage7_emit_bundles(
            [first_scope, second_scope],
            out_dir=out_dir,
            dataset_root=tmp_path,
            target_scopes=2,
            condition="full",
        )
    assert not (out_dir / "bundles").exists()


def test_stage7_emit_bundles_rejects_non_fixed_interior_n(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    scope_selection = ScopePairSelection(
        scope=ScopeWithTertile(
            scope=_scope(
                class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
            ),
            tertile="T2",
        ),
        students=(
            _validated_student(
                tmp_path=tmp_path,
                student_id="9001",
                exercise_id="301",
                total_attempt_count=4,
                interior_n=2,
            ),
            _validated_student(
                tmp_path=tmp_path, student_id="9002", exercise_id="301", total_attempt_count=4
            ),
        ),
    )

    with pytest.raises(BranchingProbeSelectionError, match=r"requires fixed interior_n=1"):
        stage7_emit_bundles(
            [scope_selection],
            out_dir=out_dir,
            dataset_root=tmp_path,
            target_scopes=1,
            condition="full",
        )


def test_emit_one_bundle_rejects_payloads_with_wrong_visible_attempt_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    student = _validated_student(
        tmp_path=tmp_path, student_id="9001", exercise_id="301", total_attempt_count=4
    )
    scope = ScopeWithTertile(
        scope=_scope(
            class_id="101", assessment_id="201", exercise_id="301", mean_pairwise_dist=0.25
        ),
        tertile="T2",
    )

    def fake_build_prompt_payload(
        **_: object,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        return (
            {
                "source": {"condition": "full"},
                "visible_attempts": [{}, {}, {}],
            },
            {},
            {},
        )

    monkeypatch.setattr(selector, "_build_prompt_payload", fake_build_prompt_payload)

    with pytest.raises(BranchingProbeSelectionError, match=r"Expected exactly 2 visible attempts"):
        selector._emit_one_bundle(
            out_dir=tmp_path / "out",
            dataset_root=tmp_path,
            scope=scope,
            student=student,
            condition="full",
        )


def test_stage8_self_validation_passes_for_balanced_manifest(tmp_path: Path) -> None:
    ordered_custom_ids = [
        "101:9001:201:301:1",
        "101:9002:201:301:1",
        "101:9003:201:302:1",
        "101:9004:201:302:1",
        "101:9005:201:302:1",
    ]
    bundle_map = {
        custom_id: _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id=custom_id,
            scope_id=scope_id,
            tertile=tertile,
            vector=vector,
        )
        for custom_id, scope_id, tertile, vector in [
            ("101:9001:201:301:1", "101:201:301", "T1", [False, True]),
            ("101:9002:201:301:1", "101:201:301", "T1", [False, True]),
            ("101:9003:201:302:1", "101:201:302", "T2", [True, False]),
            ("101:9004:201:302:1", "101:201:302", "T2", [True, False]),
            ("101:9005:201:302:1", "101:201:302", "T2", [True, False]),
        ]
    }
    paths, counts = _manifest_paths_and_counts(
        tmp_path=tmp_path,
        ordered_custom_ids=ordered_custom_ids,
        bundle_map=bundle_map,
    )
    counts["scopes"] = 2

    manifest = {
        "schema_version": "v6_2_full_trace_pilot_batch_manifest_v1",
        "model": "gpt-5.4",
        "reasoning_effort": "medium",
        "condition": "full",
        "counts": counts,
        "paths": paths,
        "ordered_custom_ids": ordered_custom_ids,
        "bundle_map": bundle_map,
        "selection": {
            "condition": "full",
            "target_scopes": 2,
            "fixed_transition_index_0idx": 1,
            "fixed_visible_attempt_count": 2,
            "matched_cohort_preflight_condition": "full",
            "documented_final_scope_counts": {"T1": 1, "T2": 1, "T3": 0},
            "documented_final_row_counts": {"T1": 2, "T2": 3, "T3": 0},
            "documented_final_scope_rows": [
                _documented_scope_row(
                    scope_id="101:201:301",
                    tertile="T1",
                    row_count=2,
                    signature_counts={"FP": 2},
                    buildable_student_ids_before_l2a=["9001", "9002"],
                    excluded_buildable_student_ids=[],
                ),
                _documented_scope_row(
                    scope_id="101:201:302",
                    tertile="T2",
                    row_count=3,
                    signature_counts={"PF": 3},
                    buildable_student_ids_before_l2a=["9003", "9004", "9005"],
                    excluded_buildable_student_ids=[],
                ),
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    result = stage8_self_validate_manifest(manifest_path)

    assert result == {
        "status": "passed",
        "unique_custom_ids": 5,
        "validated_scope_counts": {"T1": 1, "T2": 1, "T3": 0},
        "validated_bundle_count": 5,
    }


def test_stage8_self_validation_fails_when_observed_file_is_missing(tmp_path: Path) -> None:
    ordered_custom_ids = ["101:9001:201:301:1", "101:9002:201:301:1"]
    bundle_map = {
        custom_id: _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id=custom_id,
            scope_id="101:201:301",
            tertile="T1",
            vector=[True],
            missing_observed=True,
        )
        for custom_id in ordered_custom_ids
    }
    paths, counts = _manifest_paths_and_counts(
        tmp_path=tmp_path,
        ordered_custom_ids=ordered_custom_ids,
        bundle_map=bundle_map,
    )
    counts["scopes"] = 1
    manifest = {
        "schema_version": "v6_2_full_trace_pilot_batch_manifest_v1",
        "model": "gpt-5.4",
        "reasoning_effort": "medium",
        "counts": counts,
        "paths": paths,
        "ordered_custom_ids": ordered_custom_ids,
        "bundle_map": bundle_map,
        "selection": {
            "condition": "full",
            "target_scopes": 1,
            "fixed_transition_index_0idx": 1,
            "fixed_visible_attempt_count": 2,
            "matched_cohort_preflight_condition": "full",
            "documented_final_scope_counts": {"T1": 1, "T2": 0, "T3": 0},
            "documented_final_row_counts": {"T1": 2, "T2": 0, "T3": 0},
            "documented_final_scope_rows": [
                _documented_scope_row(
                    scope_id="101:201:301",
                    tertile="T1",
                    row_count=2,
                    signature_counts={"P": 2},
                    buildable_student_ids_before_l2a=["9001", "9002"],
                    excluded_buildable_student_ids=[],
                )
            ],
        },
        "condition": "full",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BranchingProbeSelectionError, match="Referenced file does not exist"):
        stage8_self_validate_manifest(manifest_path)


def test_stage8_self_validation_fails_when_bundle_condition_mismatches_manifest(
    tmp_path: Path,
) -> None:
    ordered_custom_ids = ["101:9001:201:301:1", "101:9002:201:301:1"]
    bundle_map = {
        custom_id: _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id=custom_id,
            scope_id="101:201:301",
            tertile="T1",
            vector=[True],
            condition="full",
        )
        for custom_id in ordered_custom_ids
    }
    paths, counts = _manifest_paths_and_counts(
        tmp_path=tmp_path,
        ordered_custom_ids=ordered_custom_ids,
        bundle_map=bundle_map,
    )
    counts["scopes"] = 1
    manifest = {
        "schema_version": "v6_2_full_trace_pilot_batch_manifest_v1",
        "model": "gpt-5.4",
        "reasoning_effort": "medium",
        "condition": "no_trace",
        "counts": counts,
        "paths": paths,
        "ordered_custom_ids": ordered_custom_ids,
        "bundle_map": bundle_map,
        "selection": {
            "condition": "no_trace",
            "target_scopes": 1,
            "fixed_transition_index_0idx": 1,
            "fixed_visible_attempt_count": 2,
            "matched_cohort_preflight_condition": "full",
            "documented_final_scope_counts": {"T1": 1, "T2": 0, "T3": 0},
            "documented_final_row_counts": {"T1": 2, "T2": 0, "T3": 0},
            "documented_final_scope_rows": [
                _documented_scope_row(
                    scope_id="101:201:301",
                    tertile="T1",
                    row_count=2,
                    signature_counts={"P": 2},
                    buildable_student_ids_before_l2a=["9001", "9002"],
                    excluded_buildable_student_ids=[],
                )
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BranchingProbeSelectionError, match="condition mismatch"):
        stage8_self_validate_manifest(manifest_path)


def test_stage8_self_validation_fails_when_bundle_depth_metadata_mismatches_selection(
    tmp_path: Path,
) -> None:
    ordered_custom_ids = ["101:9001:201:301:1", "101:9002:201:301:1"]
    bundle_map = {
        custom_id: _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id=custom_id,
            scope_id="101:201:301",
            tertile="T1",
            vector=[True],
            transition_index=2,
            visible_attempt_count=3,
        )
        for custom_id in ordered_custom_ids
    }
    paths, counts = _manifest_paths_and_counts(
        tmp_path=tmp_path,
        ordered_custom_ids=ordered_custom_ids,
        bundle_map=bundle_map,
    )
    counts["scopes"] = 1
    manifest = {
        "schema_version": "v6_2_full_trace_pilot_batch_manifest_v1",
        "model": "gpt-5.4",
        "reasoning_effort": "medium",
        "condition": "full",
        "counts": counts,
        "paths": paths,
        "ordered_custom_ids": ordered_custom_ids,
        "bundle_map": bundle_map,
        "selection": {
            "condition": "full",
            "target_scopes": 1,
            "fixed_transition_index_0idx": 1,
            "fixed_visible_attempt_count": 2,
            "matched_cohort_preflight_condition": "full",
            "documented_final_scope_counts": {"T1": 1, "T2": 0, "T3": 0},
            "documented_final_row_counts": {"T1": 2, "T2": 0, "T3": 0},
            "documented_final_scope_rows": [
                _documented_scope_row(
                    scope_id="101:201:301",
                    tertile="T1",
                    row_count=2,
                    signature_counts={"P": 2},
                    buildable_student_ids_before_l2a=["9001", "9002"],
                    excluded_buildable_student_ids=[],
                )
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BranchingProbeSelectionError, match="transition_index_0idx mismatch"):
        stage8_self_validate_manifest(manifest_path)


def test_stage8_self_validation_fails_when_scope_contains_singleton_l2a_group(
    tmp_path: Path,
) -> None:
    ordered_custom_ids = [
        "101:9001:201:301:1",
        "101:9002:201:301:1",
        "101:9003:201:301:1",
    ]
    bundle_map = {
        "101:9001:201:301:1": _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id="101:9001:201:301:1",
            scope_id="101:201:301",
            tertile="T1",
            vector=[False, True],
        ),
        "101:9002:201:301:1": _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id="101:9002:201:301:1",
            scope_id="101:201:301",
            tertile="T1",
            vector=[False, True],
        ),
        "101:9003:201:301:1": _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id="101:9003:201:301:1",
            scope_id="101:201:301",
            tertile="T1",
            vector=[True, False],
        ),
    }
    paths, counts = _manifest_paths_and_counts(
        tmp_path=tmp_path,
        ordered_custom_ids=ordered_custom_ids,
        bundle_map=bundle_map,
    )
    counts["scopes"] = 1
    manifest = {
        "schema_version": "v6_2_full_trace_pilot_batch_manifest_v1",
        "model": "gpt-5.4",
        "reasoning_effort": "medium",
        "condition": "full",
        "counts": counts,
        "paths": paths,
        "ordered_custom_ids": ordered_custom_ids,
        "bundle_map": bundle_map,
        "selection": {
            "condition": "full",
            "target_scopes": 1,
            "fixed_transition_index_0idx": 1,
            "fixed_visible_attempt_count": 2,
            "matched_cohort_preflight_condition": "full",
            "documented_final_scope_counts": {"T1": 1, "T2": 0, "T3": 0},
            "documented_final_row_counts": {"T1": 3, "T2": 0, "T3": 0},
            "documented_final_scope_rows": [
                {
                    **_documented_scope_row(
                        scope_id="101:201:301",
                        tertile="T1",
                        row_count=3,
                        signature_counts={"FP": 2, "PF": 1},
                        buildable_student_ids_before_l2a=["9001", "9002", "9003"],
                        excluded_buildable_student_ids=[],
                    ),
                }
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BranchingProbeSelectionError, match="invalid L2A signature count"):
        stage8_self_validate_manifest(manifest_path)


def test_stage8_self_validation_fails_when_request_contract_drifts_from_manifest(
    tmp_path: Path,
) -> None:
    ordered_custom_ids = ["101:9001:201:301:1", "101:9002:201:301:1"]
    bundle_map = {
        custom_id: _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id=custom_id,
            scope_id="101:201:301",
            tertile="T1",
            vector=[True],
            model="gpt-5.4",
            reasoning_effort="medium",
        )
        for custom_id in ordered_custom_ids
    }
    paths, counts = _manifest_paths_and_counts(
        tmp_path=tmp_path,
        ordered_custom_ids=ordered_custom_ids,
        bundle_map=bundle_map,
    )
    counts["scopes"] = 1
    manifest = {
        "schema_version": "v6_2_full_trace_pilot_batch_manifest_v1",
        "model": "gpt-5.4-mini",
        "reasoning_effort": "high",
        "condition": "full",
        "counts": counts,
        "paths": paths,
        "ordered_custom_ids": ordered_custom_ids,
        "bundle_map": bundle_map,
        "selection": {
            "condition": "full",
            "target_scopes": 1,
            "fixed_transition_index_0idx": 1,
            "fixed_visible_attempt_count": 2,
            "matched_cohort_preflight_condition": "full",
            "documented_final_scope_counts": {"T1": 1, "T2": 0, "T3": 0},
            "documented_final_row_counts": {"T1": 2, "T2": 0, "T3": 0},
            "documented_final_scope_rows": [
                _documented_scope_row(
                    scope_id="101:201:301",
                    tertile="T1",
                    row_count=2,
                    signature_counts={"P": 2},
                    buildable_student_ids_before_l2a=["9001", "9002"],
                    excluded_buildable_student_ids=[],
                )
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BranchingProbeSelectionError, match="Request body model mismatch"):
        stage8_self_validate_manifest(manifest_path)


def test_stage8_self_validation_fails_when_bundle_request_body_reasoning_drifts(
    tmp_path: Path,
) -> None:
    ordered_custom_ids = ["101:9001:201:301:1", "101:9002:201:301:1"]
    bundle_map = {
        custom_id: _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id=custom_id,
            scope_id="101:201:301",
            tertile="T1",
            vector=[True],
            model="gpt-5.4-mini",
            reasoning_effort="high",
        )
        for custom_id in ordered_custom_ids
    }
    manifest_path = _write_stage8_manifest(
        tmp_path=tmp_path,
        bundle_map=bundle_map,
        ordered_custom_ids=ordered_custom_ids,
        model="gpt-5.4-mini",
        reasoning_effort="high",
    )
    request_body_path = Path(str(bundle_map[ordered_custom_ids[0]]["request_body_path"]))
    payload = json.loads(request_body_path.read_text(encoding="utf-8"))
    payload["reasoning"]["effort"] = "medium"
    request_body_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        BranchingProbeSelectionError, match="Request body reasoning effort mismatch"
    ):
        stage8_self_validate_manifest(manifest_path)


def test_stage8_self_validation_fails_when_bundle_manifest_model_drifts(tmp_path: Path) -> None:
    ordered_custom_ids = ["101:9001:201:301:1", "101:9002:201:301:1"]
    bundle_map = {
        custom_id: _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id=custom_id,
            scope_id="101:201:301",
            tertile="T1",
            vector=[True],
            model="gpt-5.4-mini",
            reasoning_effort="high",
        )
        for custom_id in ordered_custom_ids
    }
    manifest_path = _write_stage8_manifest(
        tmp_path=tmp_path,
        bundle_map=bundle_map,
        ordered_custom_ids=ordered_custom_ids,
        model="gpt-5.4-mini",
        reasoning_effort="high",
    )
    bundle_manifest_path = Path(str(bundle_map[ordered_custom_ids[0]]["bundle_manifest_path"]))
    payload = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    payload["model"] = "gpt-5.4"
    bundle_manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BranchingProbeSelectionError, match="Bundle manifest model mismatch"):
        stage8_self_validate_manifest(manifest_path)


def test_stage8_self_validation_fails_when_bundle_batch_request_drifts(tmp_path: Path) -> None:
    ordered_custom_ids = ["101:9001:201:301:1", "101:9002:201:301:1"]
    bundle_map = {
        custom_id: _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id=custom_id,
            scope_id="101:201:301",
            tertile="T1",
            vector=[True],
            model="gpt-5.4-mini",
            reasoning_effort="high",
        )
        for custom_id in ordered_custom_ids
    }
    manifest_path = _write_stage8_manifest(
        tmp_path=tmp_path,
        bundle_map=bundle_map,
        ordered_custom_ids=ordered_custom_ids,
        model="gpt-5.4-mini",
        reasoning_effort="high",
    )
    batch_request_path = (
        Path(str(bundle_map[ordered_custom_ids[0]]["bundle_dir"])) / "batch_request.json"
    )
    payload = json.loads(batch_request_path.read_text(encoding="utf-8"))
    payload["body"]["model"] = "gpt-5.4"
    batch_request_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        BranchingProbeSelectionError,
        match="Bundle batch_request does not match requests.jsonl entry",
    ):
        stage8_self_validate_manifest(manifest_path)


def test_stage8_self_validation_fails_when_bundle_request_jsonl_drifts(tmp_path: Path) -> None:
    ordered_custom_ids = ["101:9001:201:301:1", "101:9002:201:301:1"]
    bundle_map = {
        custom_id: _manifest_bundle_entry(
            tmp_path=tmp_path,
            custom_id=custom_id,
            scope_id="101:201:301",
            tertile="T1",
            vector=[True],
            model="gpt-5.4-mini",
            reasoning_effort="high",
        )
        for custom_id in ordered_custom_ids
    }
    manifest_path = _write_stage8_manifest(
        tmp_path=tmp_path,
        bundle_map=bundle_map,
        ordered_custom_ids=ordered_custom_ids,
        model="gpt-5.4-mini",
        reasoning_effort="high",
    )
    bundle_request_jsonl_path = (
        Path(str(bundle_map[ordered_custom_ids[0]]["bundle_dir"])) / "requests.jsonl"
    )
    payload = json.loads(bundle_request_jsonl_path.read_text(encoding="utf-8").strip())
    payload["body"]["model"] = "gpt-5.4"
    bundle_request_jsonl_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        BranchingProbeSelectionError,
        match="Bundle requests.jsonl entry does not match requests.jsonl body",
    ):
        stage8_self_validate_manifest(manifest_path)
