from __future__ import annotations

from dataclasses import dataclass


class V61EncodingPolicyError(ValueError):
    pass


FULL_LOG_MAX_PROMPT_CHARS = 900_000
FULL_LOG_MAX_EVENT_COUNT = 2_000
FULL_LOG_MAX_RAW_INTERVAL_EVENTS = 8_000


@dataclass(frozen=True)
class V61EncodingStats:
    prompt_chars: int
    raw_interval_event_count: int
    event_count: int
    change_count: int
    run_count: int
    submit_count: int
    kill_program_count: int
    idle_gap_count: int
    navigation_count: int
    stdout_lines: int
    used_full_stdout: bool


@dataclass(frozen=True)
class V61EncodingDecision:
    case_type: str
    keep_full_event_log: bool
    flagged_case: bool
    flags: tuple[str, ...]


def classify_case_type(stats: V61EncodingStats) -> str:
    if stats.event_count <= 1 and stats.submit_count == 1 and stats.change_count == 0:
        return "submit_only"
    if stats.change_count == 0 and stats.run_count >= 1:
        return "run_heavy_no_change"
    if stats.event_count >= 400 or stats.change_count >= 350:
        return "large_dense"
    return "repair_focused"


def decide_encoding(stats: V61EncodingStats) -> V61EncodingDecision:
    if not stats.used_full_stdout:
        raise V61EncodingPolicyError(
            "v6.1 forbids silent stdout truncation; full stdout is required"
        )
    if stats.submit_count != 1:
        raise V61EncodingPolicyError(
            f"v6.1 expects exactly one submit in the current event log, got {stats.submit_count}"
        )
    if stats.prompt_chars > FULL_LOG_MAX_PROMPT_CHARS:
        raise V61EncodingPolicyError(
            "Full v6.1 prompt exceeds the hard character ceiling; "
            f"got {stats.prompt_chars}, max {FULL_LOG_MAX_PROMPT_CHARS}"
        )
    if stats.event_count > FULL_LOG_MAX_EVENT_COUNT:
        raise V61EncodingPolicyError(
            "Full v6.1 event log exceeds the hard event ceiling; "
            f"got {stats.event_count}, max {FULL_LOG_MAX_EVENT_COUNT}"
        )
    if stats.raw_interval_event_count > FULL_LOG_MAX_RAW_INTERVAL_EVENTS:
        raise V61EncodingPolicyError(
            "Raw interval exceeds the hard event ceiling before encoding; "
            f"got {stats.raw_interval_event_count}, max {FULL_LOG_MAX_RAW_INTERVAL_EVENTS}"
        )

    flags: list[str] = []
    if stats.event_count <= 5:
        flags.append("very_small_event_log")
    if stats.change_count == 0:
        flags.append("no_code_edits")
    if stats.run_count >= 2 and stats.change_count < 3:
        flags.append("run_heavy_with_few_edits")
    if stats.idle_gap_count >= 20:
        flags.append("many_idle_gaps")
    if stats.run_count >= 20:
        flags.append("many_local_runs")
    if stats.kill_program_count >= 1:
        flags.append("contains_kill_program")

    return V61EncodingDecision(
        case_type=classify_case_type(stats),
        keep_full_event_log=True,
        flagged_case=bool(flags),
        flags=tuple(flags),
    )
