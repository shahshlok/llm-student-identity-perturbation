from __future__ import annotations

import random
from pathlib import Path

from identity_perturbation.codebench_support.codemirror import CodeMirrorParseError, infer_initial_code, parse_codemirror_log
from identity_perturbation.codebench_support.executions import ExecutionParseError, parse_execution_log

from .trace_card import V6TraceCardError, build_attempt_trace_cards


class V6TraceShuffleError(ValueError):
    pass


def _same_slice_candidates(
    *,
    data_root: Path,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    excluded_student_id: str,
) -> tuple[tuple[str, str, str], ...]:
    users_root = data_root / class_id / "users"
    if not users_root.exists():
        raise V6TraceShuffleError(f"Users directory not found: {users_root}")

    candidates: list[tuple[str, str, str]] = []
    for user_dir in sorted(users_root.iterdir(), key=lambda path: path.name):
        if not user_dir.is_dir():
            continue
        student_id = user_dir.name
        if student_id == excluded_student_id:
            continue
        exec_path = user_dir / "executions" / f"{assessment_id}_{exercise_id}.log"
        cm_path = user_dir / "codemirror" / f"{assessment_id}_{exercise_id}.log"
        if exec_path.exists() and cm_path.exists():
            candidates.append((class_id, assessment_id, student_id))
    return tuple(candidates)


def _cross_class_exercise_candidates(
    *,
    data_root: Path,
    exercise_id: str,
    excluded_student_id: str,
    excluded_exact_slice: tuple[str, str],
) -> tuple[tuple[str, str, str], ...]:
    candidates: list[tuple[str, str, str]] = []
    for class_root in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not class_root.is_dir():
            continue
        class_id = class_root.name
        users_root = class_root / "users"
        if not users_root.exists():
            continue
        for user_dir in sorted(users_root.iterdir(), key=lambda path: path.name):
            if not user_dir.is_dir():
                continue
            student_id = user_dir.name
            if student_id == excluded_student_id:
                continue
            exec_root = user_dir / "executions"
            cm_root = user_dir / "codemirror"
            if not exec_root.exists() or not cm_root.exists():
                continue
            for exec_path in sorted(
                exec_root.glob(f"*_{exercise_id}.log"), key=lambda path: path.name
            ):
                assessment_id = exec_path.stem.split("_", 1)[0]
                if (class_id, assessment_id) == excluded_exact_slice:
                    continue
                cm_path = cm_root / exec_path.name
                if cm_path.exists():
                    candidates.append((class_id, assessment_id, student_id))
    return tuple(candidates)


def _visible_donor_cards(
    *,
    data_root: Path,
    exercise_id: str,
    donor_class_id: str,
    donor_assessment_id: str,
    donor_student_id: str,
    visible_attempt_count: int,
) -> tuple:
    exec_path = (
        data_root
        / donor_class_id
        / "users"
        / donor_student_id
        / "executions"
        / f"{donor_assessment_id}_{exercise_id}.log"
    )
    cm_path = (
        data_root
        / donor_class_id
        / "users"
        / donor_student_id
        / "codemirror"
        / f"{donor_assessment_id}_{exercise_id}.log"
    )
    try:
        cm_events = parse_codemirror_log(cm_path)
        seed_code = ""
        try:
            attempts = parse_execution_log(exec_path)
        except ExecutionParseError:
            attempts = ()
        if attempts:
            seed_code = attempts[0].code
        initial_code = infer_initial_code(cm_events, seed_code)
        trace_cards = build_attempt_trace_cards(cm_events, initial_code=initial_code)
    except (V6TraceCardError, CodeMirrorParseError) as exc:
        raise V6TraceShuffleError(str(exc)) from exc
    if len(trace_cards) < visible_attempt_count:
        raise V6TraceShuffleError(
            f"Donor student {donor_student_id} has only {len(trace_cards)} trace-card intervals; need {visible_attempt_count}"
        )
    return trace_cards[:visible_attempt_count]


def _trace_shuffle_metadata(
    *,
    donor_class_id: str,
    donor_assessment_id: str,
    donor_student_id: str,
    visible_attempt_count: int,
    scope_name: str,
    rng_seed: int | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "trace_donor_class_id": donor_class_id,
        "trace_donor_assessment_id": donor_assessment_id,
        "trace_donor_student_id": donor_student_id,
        "trace_visible_attempt_count": visible_attempt_count,
        "trace_shuffle_scope": scope_name,
    }
    if rng_seed is not None:
        metadata["trace_shuffle_seed"] = rng_seed
    return metadata


def _load_from_candidates(
    *,
    candidates: tuple[tuple[str, str, str], ...],
    data_root: Path,
    exercise_id: str,
    visible_attempt_count: int,
    scope_name: str,
    rng_seed: int | None = None,
) -> tuple[tuple, dict[str, object]]:
    if not candidates:
        raise V6TraceShuffleError(f"No donor candidates available for scope {scope_name}")

    ordered_candidates = list(candidates)
    if rng_seed is not None:
        rng = random.Random(rng_seed)
        rng.shuffle(ordered_candidates)

    failures: list[str] = []
    for donor_class_id, donor_assessment_id, donor_student_id in ordered_candidates:
        try:
            donor_cards = _visible_donor_cards(
                data_root=data_root,
                exercise_id=exercise_id,
                donor_class_id=donor_class_id,
                donor_assessment_id=donor_assessment_id,
                donor_student_id=donor_student_id,
                visible_attempt_count=visible_attempt_count,
            )
        except V6TraceShuffleError as exc:
            failures.append(f"{donor_class_id}:{donor_assessment_id}:{donor_student_id}: {exc}")
            continue
        return donor_cards, _trace_shuffle_metadata(
            donor_class_id=donor_class_id,
            donor_assessment_id=donor_assessment_id,
            donor_student_id=donor_student_id,
            visible_attempt_count=visible_attempt_count,
            scope_name=scope_name,
            rng_seed=rng_seed,
        )

    raise V6TraceShuffleError(
        f"No valid donor found for scope {scope_name}. "
        f"Tried {len(ordered_candidates)} candidates. Failures: {failures[:5]}"
    )


def load_within_exercise_shuffled_trace_cards(
    *,
    data_root: Path,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    target_student_id: str,
    visible_attempt_count: int,
) -> tuple[tuple, dict[str, object]]:
    primary_candidates = _same_slice_candidates(
        data_root=data_root,
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        excluded_student_id=target_student_id,
    )
    fallback_candidates = _cross_class_exercise_candidates(
        data_root=data_root,
        exercise_id=exercise_id,
        excluded_student_id=target_student_id,
        excluded_exact_slice=(class_id, assessment_id),
    )

    candidate_groups = (
        ("same_class_assessment_exercise", primary_candidates),
        ("same_exercise_across_classes", fallback_candidates),
    )
    for scope_name, candidates in candidate_groups:
        try:
            return _load_from_candidates(
                candidates=candidates,
                data_root=data_root,
                exercise_id=exercise_id,
                visible_attempt_count=visible_attempt_count,
                scope_name=scope_name,
            )
        except V6TraceShuffleError:
            continue

    raise V6TraceShuffleError(
        "No valid same-exercise trace donor found across deterministic candidate groups."
    )


def load_within_class_random_trace_cards(
    *,
    data_root: Path,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    target_student_id: str,
    visible_attempt_count: int,
    rng_seed: int,
) -> tuple[tuple, dict[str, object]]:
    primary_candidates = _same_slice_candidates(
        data_root=data_root,
        class_id=class_id,
        assessment_id=assessment_id,
        exercise_id=exercise_id,
        excluded_student_id=target_student_id,
    )
    return _load_from_candidates(
        candidates=primary_candidates,
        data_root=data_root,
        exercise_id=exercise_id,
        visible_attempt_count=visible_attempt_count,
        scope_name="within_class_random",
        rng_seed=rng_seed,
    )


def load_cross_class_random_trace_cards(
    *,
    data_root: Path,
    class_id: str,
    assessment_id: str,
    exercise_id: str,
    target_student_id: str,
    visible_attempt_count: int,
    rng_seed: int,
) -> tuple[tuple, dict[str, object]]:
    cross_class_candidates = _cross_class_exercise_candidates(
        data_root=data_root,
        exercise_id=exercise_id,
        excluded_student_id=target_student_id,
        excluded_exact_slice=(class_id, assessment_id),
    )
    return _load_from_candidates(
        candidates=cross_class_candidates,
        data_root=data_root,
        exercise_id=exercise_id,
        visible_attempt_count=visible_attempt_count,
        scope_name="cross_class_random",
        rng_seed=rng_seed,
    )
