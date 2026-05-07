from __future__ import annotations

from collections.abc import Iterable


class MatchPolicyError(ValueError):
    pass


def narrow_normalize_code_for_match(code: str) -> str:
    text = code.replace("\r\n", "\n").replace("\r", "\n")
    if text.endswith("\n"):
        text = text[:-1]
    return "\n".join(line.rstrip(" \t") for line in text.split("\n"))


def candidate_snapshot_indices_narrow_normalized(
    *,
    attempt_codes: tuple[str, ...],
    snapshot_codes: tuple[str, ...],
) -> tuple[tuple[int, ...], ...]:
    normalized_snapshots = tuple(narrow_normalize_code_for_match(code) for code in snapshot_codes)
    candidates: list[tuple[int, ...]] = []
    for code in attempt_codes:
        normalized = narrow_normalize_code_for_match(code)
        matches = tuple(
            index
            for index, snapshot_code in enumerate(normalized_snapshots)
            if snapshot_code == normalized
        )
        candidates.append(matches)
    return tuple(candidates)


def count_monotonic_solutions(
    candidates: tuple[tuple[int, ...], ...],
    *,
    limit: int = 2,
) -> int:
    if limit < 1:
        raise MatchPolicyError(f"Solution-count limit must be positive, got {limit}")
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


def resolve_unique_monotonic_alignment(
    candidates: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    solutions: list[tuple[int, ...]] = []

    def walk(position: int, previous_index: int, path: list[int]) -> None:
        if len(solutions) > 1:
            return
        if position == len(candidates):
            solutions.append(tuple(path))
            return
        for snapshot_index in candidates[position]:
            if snapshot_index <= previous_index:
                continue
            path.append(snapshot_index)
            walk(position + 1, snapshot_index, path)
            path.pop()

    walk(0, -1, [])
    if len(solutions) != 1:
        raise MatchPolicyError(f"Expected exactly one monotonic alignment, found {len(solutions)}")
    return solutions[0]


def matching_indices_after(
    *,
    snapshot_codes: Iterable[str],
    normalized_target_code: str,
    start_index: int,
) -> tuple[int, ...]:
    matches: list[int] = []
    for index, snapshot_code in enumerate(snapshot_codes):
        if index <= start_index:
            continue
        if narrow_normalize_code_for_match(snapshot_code) == normalized_target_code:
            matches.append(index)
    return tuple(matches)
