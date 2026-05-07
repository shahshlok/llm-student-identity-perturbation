from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from identity_perturbation.codebench_support.codemirror import (
    CodeMirrorParseError,
    _code_to_lines,
    apply_change,
    infer_initial_code,
    normalize_code_for_match,
    parse_cm_timestamp,
    replay_trace,
)
from identity_perturbation.codebench_support.executions import (
    ExecutionParseError,
    parse_execution_log_text,
)
from identity_perturbation.codebench_support.models import CmEvent

DEFAULT_DATA_ROOT = Path("2024-1")
DEFAULT_OUT_ROOT = Path("data/v62/raw_same_task_family_audit")
TARGET_ASSESSMENT_TITLES = (
    "Lab 5 - Vetores e Strings",
    "Lab 6 - Repetição por Contagem (for)",
)


class V62AuditError(ValueError):
    pass


@dataclass(frozen=True)
class AssessmentSpec:
    class_id: str
    assessment_id: str
    title: str
    exercise_ids: tuple[str, ...]


@dataclass(frozen=True)
class RawTraceEvent:
    timestamp: datetime
    raw_type: str
    payload: object
    line_number: int
    output_text: str


@dataclass(frozen=True)
class AttemptBlockMetrics:
    submit_index_0idx: int
    change_event_count: int
    run_event_count: int
    anchor_event_count: int


@dataclass(frozen=True)
class CandidateTransition:
    class_id: str
    assessment_id: str
    assessment_title: str
    exercise_id: str
    student_id: str
    transition_index_0idx: int
    visible_attempt_count: int
    total_attempt_count: int
    visible_submit_count: int
    visible_run_count: int
    visible_anchor_count: int
    visible_change_event_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw-only v6.2 audit for the same task family as the current Lab 5 / Lab 6 study."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def read_text_strict(path: Path) -> str:
    if not path.exists():
        raise V62AuditError(f"Path does not exist: {path}")
    return path.read_text(encoding="utf-8")


def parse_assessment_data(path: Path) -> AssessmentSpec:
    lines = read_text_strict(path).splitlines()
    title: str | None = None
    exercise_ids: list[str] = []
    for line in lines:
        if line.startswith("---- assessment title:"):
            title = line.split(":", 1)[1].strip()
            continue
        if line.startswith("---- exercise "):
            exercise_id = line.split(":", 1)[1].strip()
            if not exercise_id:
                raise V62AuditError(f"Empty exercise id in assessment file: {path}")
            exercise_ids.append(exercise_id)
    if title is None:
        raise V62AuditError(f"Assessment title missing in file: {path}")
    if not exercise_ids:
        raise V62AuditError(f"No exercises found in assessment file: {path}")
    return AssessmentSpec(
        class_id=path.parts[-3],
        assessment_id=path.stem,
        title=title,
        exercise_ids=tuple(exercise_ids),
    )


def collect_target_assessments(data_root: Path) -> tuple[AssessmentSpec, ...]:
    if not data_root.exists():
        raise V62AuditError(f"Data root does not exist: {data_root}")
    assessments: list[AssessmentSpec] = []
    for path in sorted(data_root.glob("*/assessments/*.data")):
        spec = parse_assessment_data(path)
        if spec.title in TARGET_ASSESSMENT_TITLES:
            assessments.append(spec)
    if not assessments:
        raise V62AuditError(
            f"No assessments matched target titles {TARGET_ASSESSMENT_TITLES!r} under {data_root}"
        )
    return tuple(assessments)


def parse_raw_trace_line(raw_line: str, line_number: int) -> tuple[object, str, str] | None:
    stripped = raw_line.rstrip("\n")
    if stripped.strip() == "":
        return None
    parts = stripped.split("#", 2)
    if len(parts) < 2:
        return None
    timestamp = parse_cm_timestamp(parts[0])
    raw_type = parts[1]
    payload_raw = parts[2] if len(parts) > 2 else ""
    return (timestamp, raw_type, payload_raw)


def parse_raw_trace_events_with_output(text: str, path: Path) -> tuple[RawTraceEvent, ...]:
    lines = text.splitlines()
    events: list[RawTraceEvent] = []
    index = 0
    while index < len(lines):
        parsed = parse_raw_trace_line(lines[index], index + 1)
        if parsed is None:
            if lines[index].strip() == "":
                index += 1
                continue
            raise V62AuditError(f"Malformed CodeMirror raw line at {path}:{index + 1}")
        timestamp, raw_type, payload_raw = parsed
        payload: object = payload_raw
        if raw_type == "change":
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError as exc:
                raise V62AuditError(f"Malformed change payload at {path}:{index + 1}") from exc
        output_lines: list[str] = []
        if raw_type == "saida_testar":
            lookahead = index + 1
            while lookahead < len(lines):
                next_parsed = parse_raw_trace_line(lines[lookahead], lookahead + 1)
                if next_parsed is not None:
                    break
                output_lines.append(lines[lookahead])
                lookahead += 1
            index = lookahead - 1
        events.append(
            RawTraceEvent(
                timestamp=timestamp,
                raw_type=raw_type,
                payload=payload,
                line_number=index + 1,
                output_text="\n".join(output_lines),
            )
        )
        index += 1
    if not events:
        raise V62AuditError(f"CodeMirror log produced no parseable raw events: {path}")
    return tuple(events)


def cm_events_from_raw_events(raw_events: tuple[RawTraceEvent, ...]) -> tuple[CmEvent, ...]:
    cm_events: list[CmEvent] = []
    for event in raw_events:
        cm_events.append(
            CmEvent(
                timestamp=event.timestamp,
                raw_type=event.raw_type,
                payload=event.payload,
                line_number=event.line_number,
            )
        )
    return tuple(cm_events)


def build_attempt_block_metrics(
    raw_events: tuple[RawTraceEvent, ...],
    initial_code: str,
) -> tuple[AttemptBlockMetrics, ...]:
    lines = _code_to_lines(initial_code)
    submit_index = 0
    block_change_count = 0
    block_run_count = 0
    block_anchor_count = 0
    blocks: list[AttemptBlockMetrics] = []
    for event in raw_events:
        if event.raw_type == "change":
            if not isinstance(event.payload, dict):
                raise V62AuditError(
                    f"Change payload is not a dict at CodeMirror line {event.line_number}"
                )
            lines = apply_change(lines, event.payload)
            block_change_count += 1
            continue
        if event.raw_type == "saida_testar":
            block_run_count += 1
            block_anchor_count += 1
            continue
        if event.raw_type == "submit":
            block_anchor_count += 1
            blocks.append(
                AttemptBlockMetrics(
                    submit_index_0idx=submit_index,
                    change_event_count=block_change_count,
                    run_event_count=block_run_count,
                    anchor_event_count=block_anchor_count,
                )
            )
            submit_index += 1
            block_change_count = 0
            block_run_count = 0
            block_anchor_count = 0
    if not blocks:
        raise V62AuditError("No submit-bounded attempt blocks were reconstructed")
    return tuple(blocks)


def candidate_snapshot_indices(
    attempt_codes: tuple[str, ...],
    snapshot_codes: tuple[str, ...],
) -> tuple[tuple[int, ...], ...]:
    normalized_snapshots = tuple(normalize_code_for_match(code) for code in snapshot_codes)
    candidates: list[tuple[int, ...]] = []
    for code in attempt_codes:
        normalized = normalize_code_for_match(code)
        matched = tuple(
            index
            for index, snapshot_code in enumerate(normalized_snapshots)
            if snapshot_code == normalized
        )
        candidates.append(matched)
    return tuple(candidates)


def count_monotonic_solutions(
    candidates: tuple[tuple[int, ...], ...],
    *,
    limit: int = 2,
) -> int:
    if limit < 1:
        raise V62AuditError(f"Solution-count limit must be positive, got {limit}")
    cache: dict[tuple[int, int], int] = {}

    def walk(position: int, previous_index: int) -> int:
        key = (position, previous_index)
        if key in cache:
            return cache[key]
        if position == len(candidates):
            return 1
        total = 0
        for snapshot_index in candidates[position]:
            if snapshot_index <= previous_index:
                continue
            total += walk(position + 1, snapshot_index)
            if total >= limit:
                cache[key] = limit
                return limit
        cache[key] = total
        return total

    return walk(0, -1)


def classify_transition_prefixes(
    *,
    assessment: AssessmentSpec,
    exercise_id: str,
    student_id: str,
    attempts: tuple[object, ...],
    attempt_blocks: tuple[AttemptBlockMetrics, ...],
    snapshot_codes: tuple[str, ...],
) -> tuple[CandidateTransition, ...]:
    attempt_codes = tuple(attempt.code for attempt in attempts)
    candidates = candidate_snapshot_indices(
        attempt_codes=attempt_codes, snapshot_codes=snapshot_codes
    )
    output: list[CandidateTransition] = []
    for transition_index_0idx in range(len(attempts) - 1):
        prefix_candidates = candidates[: transition_index_0idx + 1]
        if not all(prefix_candidates):
            continue
        if count_monotonic_solutions(prefix_candidates) != 1:
            continue
        visible_blocks = attempt_blocks[: transition_index_0idx + 1]
        output.append(
            CandidateTransition(
                class_id=assessment.class_id,
                assessment_id=assessment.assessment_id,
                assessment_title=assessment.title,
                exercise_id=exercise_id,
                student_id=student_id,
                transition_index_0idx=transition_index_0idx,
                visible_attempt_count=transition_index_0idx + 1,
                total_attempt_count=len(attempts),
                visible_submit_count=transition_index_0idx + 1,
                visible_run_count=sum(block.run_event_count for block in visible_blocks),
                visible_anchor_count=sum(block.anchor_event_count for block in visible_blocks),
                visible_change_event_count=sum(
                    block.change_event_count for block in visible_blocks
                ),
            )
        )
    return tuple(output)


def build_audit_report(data_root: Path) -> dict[str, object]:
    assessments = collect_target_assessments(data_root)
    failure_counts: Counter[str] = Counter()
    failure_examples: dict[str, list[str]] = {}
    candidate_transitions: list[CandidateTransition] = []
    pair_count = 0
    with_exec_and_cm = 0
    min_two_attempt_pairs = 0
    raw_parse_ok_pairs = 0
    replay_ok_pairs = 0
    full_unique_alignment_pairs = 0

    for assessment in assessments:
        class_users_root = data_root / assessment.class_id / "users"
        if not class_users_root.exists():
            raise V62AuditError(f"Users directory missing for class {assessment.class_id}")
        for user_dir in sorted(class_users_root.iterdir()):
            if not user_dir.is_dir():
                continue
            student_id = user_dir.name
            for exercise_id in assessment.exercise_ids:
                pair_count += 1
                exec_path = (
                    user_dir / "executions" / f"{assessment.assessment_id}_{exercise_id}.log"
                )
                cm_path = user_dir / "codemirror" / f"{assessment.assessment_id}_{exercise_id}.log"
                if not exec_path.exists() or not cm_path.exists():
                    failure_counts["missing_log"] += 1
                    failure_examples.setdefault("missing_log", [])
                    if len(failure_examples["missing_log"]) < 10:
                        failure_examples["missing_log"].append(
                            f"{assessment.class_id}:{assessment.assessment_id}:{exercise_id}:{student_id}"
                        )
                    continue
                with_exec_and_cm += 1

                try:
                    attempts = parse_execution_log_text(read_text_strict(exec_path))
                except ExecutionParseError as exc:
                    key = f"execution_parse_error:{type(exc).__name__}"
                    failure_counts[key] += 1
                    failure_examples.setdefault(key, [])
                    if len(failure_examples[key]) < 10:
                        failure_examples[key].append(f"{exec_path} :: {exc}")
                    continue
                if len(attempts) < 2:
                    failure_counts["lt2_attempts"] += 1
                    failure_examples.setdefault("lt2_attempts", [])
                    if len(failure_examples["lt2_attempts"]) < 10:
                        failure_examples["lt2_attempts"].append(str(exec_path))
                    continue
                min_two_attempt_pairs += 1

                try:
                    raw_events = parse_raw_trace_events_with_output(
                        read_text_strict(cm_path), cm_path
                    )
                except (V62AuditError, ValueError) as exc:
                    key = f"raw_codemirror_parse_error:{type(exc).__name__}"
                    failure_counts[key] += 1
                    failure_examples.setdefault(key, [])
                    if len(failure_examples[key]) < 10:
                        failure_examples[key].append(f"{cm_path} :: {exc}")
                    continue
                raw_parse_ok_pairs += 1

                try:
                    cm_events = cm_events_from_raw_events(raw_events)
                    initial_code = infer_initial_code(cm_events, attempts[0].code)
                    trace = replay_trace(cm_events, initial_code=initial_code)
                    attempt_blocks = build_attempt_block_metrics(
                        raw_events, initial_code=initial_code
                    )
                except (CodeMirrorParseError, V62AuditError) as exc:
                    key = f"replay_error:{type(exc).__name__}"
                    failure_counts[key] += 1
                    failure_examples.setdefault(key, [])
                    if len(failure_examples[key]) < 10:
                        failure_examples[key].append(f"{cm_path} :: {exc}")
                    continue
                replay_ok_pairs += 1

                if len(trace.snapshots) != len(attempt_blocks):
                    raise V62AuditError(
                        "Submit snapshot count does not match reconstructed attempt block count for "
                        f"{assessment.class_id}:{assessment.assessment_id}:{exercise_id}:{student_id}"
                    )

                snapshot_codes = tuple(snapshot.code for snapshot in trace.snapshots)
                attempt_codes = tuple(attempt.code for attempt in attempts)
                all_candidates = candidate_snapshot_indices(
                    attempt_codes=attempt_codes,
                    snapshot_codes=snapshot_codes,
                )
                if all(all_candidates) and count_monotonic_solutions(all_candidates) == 1:
                    full_unique_alignment_pairs += 1

                for candidate in classify_transition_prefixes(
                    assessment=assessment,
                    exercise_id=exercise_id,
                    student_id=student_id,
                    attempts=attempts,
                    attempt_blocks=attempt_blocks,
                    snapshot_codes=snapshot_codes,
                ):
                    candidate_transitions.append(candidate)

    class_ids = sorted({assessment.class_id for assessment in assessments})
    assessment_ids = sorted({assessment.assessment_id for assessment in assessments})
    exercise_keys = {
        (assessment.class_id, assessment.assessment_id, exercise_id)
        for assessment in assessments
        for exercise_id in assessment.exercise_ids
    }
    student_keys = {
        (
            candidate.class_id,
            candidate.assessment_id,
            candidate.exercise_id,
            candidate.student_id,
        )
        for candidate in candidate_transitions
    }
    report = {
        "schema_version": "v6_2_raw_same_task_family_audit_v1",
        "scope": {
            "assessment_titles": list(TARGET_ASSESSMENT_TITLES),
            "assessment_count": len(assessments),
            "class_count": len(class_ids),
            "assessment_ids": assessment_ids,
            "exercise_count": len(exercise_keys),
        },
        "summary": {
            "pair_count": pair_count,
            "with_exec_and_cm": with_exec_and_cm,
            "min_two_attempt_pairs": min_two_attempt_pairs,
            "raw_parse_ok_pairs": raw_parse_ok_pairs,
            "replay_ok_pairs": replay_ok_pairs,
            "full_unique_alignment_pairs": full_unique_alignment_pairs,
            "candidate_transition_count": len(candidate_transitions),
            "candidate_student_exercise_pairs": len(student_keys),
            "candidate_student_count": len(
                {candidate.student_id for candidate in candidate_transitions}
            ),
        },
        "assessments": [asdict(assessment) for assessment in assessments],
        "failure_counts": dict(sorted(failure_counts.items())),
        "failure_examples": failure_examples,
        "candidate_transitions": [asdict(candidate) for candidate in candidate_transitions],
    }
    return report


def write_report(report: dict[str, object], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=False)
    json_path = out_root / "report.json"
    md_path = out_root / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = report["summary"]
    scope = report["scope"]
    lines = [
        "# V6.2 Raw Same-Task-Family Audit",
        "",
        f"- Assessment titles: `{', '.join(scope['assessment_titles'])}`",
        f"- Assessment count: `{scope['assessment_count']}`",
        f"- Pair count: `{summary['pair_count']}`",
        f"- Pairs with both logs: `{summary['with_exec_and_cm']}`",
        f"- Pairs with >=2 execution attempts: `{summary['min_two_attempt_pairs']}`",
        f"- Raw CodeMirror parse ok pairs: `{summary['raw_parse_ok_pairs']}`",
        f"- Replay ok pairs: `{summary['replay_ok_pairs']}`",
        f"- Full unique alignment pairs: `{summary['full_unique_alignment_pairs']}`",
        f"- Candidate transitions: `{summary['candidate_transition_count']}`",
        "",
        "## Failure Counts",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
    ]
    for reason, count in report["failure_counts"].items():
        lines.append(f"| {reason} | {count} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_audit_report(args.data_root)
    write_report(report, args.out_root)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
