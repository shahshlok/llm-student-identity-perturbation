from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .models import ExecutionAttempt, ExecutionTransition, TestResult

SECTION_HEADER_RE = re.compile(
    r"^== (?P<kind>SUBMITION|TEST) \((?P<timestamp>[^)]+)\)\s*$",
    re.MULTILINE,
)
TEST_CASE_HEADER_RE = re.compile(r"^-- TEST CASE (\d+):\s*$")


class ExecutionParseError(ValueError):
    pass


def normalize_output(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def parse_timestamp(raw_timestamp: str) -> datetime:
    return datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S")


def is_code_block_stop(line: str) -> bool:
    return (
        line.startswith("-- EXECUTION TIME:")
        or line.startswith("-- TEST CASE ")
        or line.startswith("-- GRADE:")
        or line.startswith("-- OUTPUT:")
        or line.startswith("-- ERROR:")
        or line.startswith("*-*-")
    )


def is_generic_stop(line: str) -> bool:
    return line.startswith("-- ") or line.startswith("*-*-")


def is_test_field_stop(line: str) -> bool:
    return line.startswith("---- ") or is_generic_stop(line)


def collect_block(
    lines: list[str],
    start_index: int,
    stop_predicate: Callable[[str], bool],
) -> tuple[str, int]:
    block: list[str] = []
    index = start_index
    while index < len(lines) and not stop_predicate(lines[index]):
        block.append(lines[index])
        index += 1
    return "\n".join(block).rstrip("\n"), index


def parse_grade(grade_text: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", grade_text)
    if not match:
        raise ExecutionParseError(f"Could not parse grade from: {grade_text!r}")
    return float(match.group(1))


def parse_test_case(
    lines: list[str],
    start_index: int,
    test_index_0idx: int,
) -> tuple[TestResult, int]:
    input_text = ""
    expected_text = ""
    actual_text = ""
    index = start_index
    seen_input = False
    seen_expected = False
    seen_actual = False

    while index < len(lines):
        line = lines[index]
        if line.startswith("-- TEST CASE ") or is_generic_stop(line):
            break
        if line == "---- input:":
            input_text, index = collect_block(lines, index + 1, is_test_field_stop)
            seen_input = True
            continue
        if line == "---- correct output:":
            expected_text, index = collect_block(lines, index + 1, is_test_field_stop)
            seen_expected = True
            continue
        if line == "---- user output:":
            actual_text, index = collect_block(lines, index + 1, is_test_field_stop)
            seen_actual = True
            continue
        index += 1

    if not (seen_input and seen_expected and seen_actual):
        raise ExecutionParseError(f"Malformed test case {test_index_0idx}: missing required field")

    return (
        TestResult(
            test_index_0idx=test_index_0idx,
            passed=normalize_output(expected_text) == normalize_output(actual_text),
            input=input_text,
            expected=expected_text,
            actual=actual_text,
        ),
        index,
    )


def parse_submission_body(body: str) -> tuple[str, float, tuple[TestResult, ...]]:
    lines = body.splitlines()
    code = None
    grade = None
    test_results: list[TestResult] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if line == "-- CODE:":
            code, index = collect_block(lines, index + 1, is_code_block_stop)
            continue
        if line == "-- GRADE:":
            grade_text, index = collect_block(lines, index + 1, is_generic_stop)
            grade = parse_grade(grade_text)
            continue
        match = TEST_CASE_HEADER_RE.match(line)
        if match:
            raw_test_number = int(match.group(1))
            if raw_test_number <= 0:
                raise ExecutionParseError(f"Invalid raw test number: {raw_test_number}")
            test_result, index = parse_test_case(lines, index + 1, raw_test_number - 1)
            test_results.append(test_result)
            continue
        index += 1

    if code is None:
        raise ExecutionParseError("Submission missing -- CODE: block")
    if grade is None:
        raise ExecutionParseError("Submission missing -- GRADE: block")
    if not test_results:
        raise ExecutionParseError("Submission has no parsed test results")
    return code, grade, tuple(test_results)


def parse_execution_log_text(log_text: str) -> tuple[ExecutionAttempt, ...]:
    attempts: list[ExecutionAttempt] = []
    matches = list(SECTION_HEADER_RE.finditer(log_text))
    if not matches:
        raise ExecutionParseError("Execution log contains no section headers")

    parsed_sections: list[tuple[datetime, str, str, float, tuple[TestResult, ...]]] = []
    for index, match in enumerate(matches):
        if match.group("kind") != "SUBMITION":
            continue
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(log_text)
        timestamp = match.group("timestamp")
        sort_key = parse_timestamp(timestamp)
        code, grade, test_results = parse_submission_body(log_text[body_start:body_end])
        parsed_sections.append((sort_key, timestamp, code, grade, test_results))

    if not parsed_sections:
        raise ExecutionParseError("Execution log contains no SUBMITION sections")

    parsed_sections.sort(key=lambda item: item[0])
    for attempt_index_0idx, (_, timestamp, code, grade, test_results) in enumerate(parsed_sections):
        attempts.append(
            ExecutionAttempt(
                timestamp=timestamp,
                attempt_index_0idx=attempt_index_0idx,
                code=code,
                grade=grade,
                test_results=test_results,
            )
        )
    return tuple(attempts)


def parse_execution_log(path: Path) -> tuple[ExecutionAttempt, ...]:
    return parse_execution_log_text(path.read_text(encoding="utf-8", errors="replace"))


def next_test_outcome(attempt: ExecutionAttempt) -> str:
    if not attempt.test_results:
        raise ExecutionParseError("Attempt has empty test_results")
    passed_values = [result.passed for result in attempt.test_results]
    if all(not value for value in passed_values):
        return "all_fail"
    if all(passed_values):
        return "all_pass"
    return "mixed"


def build_transitions(attempts: tuple[ExecutionAttempt, ...]) -> tuple[ExecutionTransition, ...]:
    if len(attempts) < 2:
        raise ExecutionParseError("Need at least two attempts to build transitions")

    transitions: list[ExecutionTransition] = []
    for index in range(len(attempts) - 1):
        transitions.append(
            ExecutionTransition(
                attempt_n=attempts[index],
                attempt_n1=attempts[index + 1],
                history=attempts[:index],
                all_attempts=attempts,
            )
        )
    return tuple(transitions)
