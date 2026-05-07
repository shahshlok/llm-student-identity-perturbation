from __future__ import annotations

import ast


class FocusRegionError(ValueError):
    pass


def _zero_index_span(node: ast.AST) -> set[int]:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    if lineno is None or end_lineno is None:
        raise FocusRegionError(f"Node missing line span: {type(node).__name__}")
    return set(range(lineno - 1, end_lineno))


def _is_print_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    value = node.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "print"
    )


def build_region_sets(attempt_n_code: str) -> tuple[set[int], set[int], set[int]]:
    try:
        tree = ast.parse(attempt_n_code)
    except SyntaxError as exc:
        raise FocusRegionError("attempt_n_code does not parse as Python") from exc

    output_lines: set[int] = set()
    conditional_lines: set[int] = set()
    loop_lines: set[int] = set()

    for node in ast.walk(tree):
        if _is_print_call(node):
            output_lines.update(_zero_index_span(node))
        if isinstance(node, ast.If):
            conditional_lines.update(_zero_index_span(node))
            for child in node.body:
                conditional_lines.update(_zero_index_span(child))
            for child in node.orelse:
                conditional_lines.update(_zero_index_span(child))
        if isinstance(node, (ast.For, ast.While)):
            loop_lines.update(_zero_index_span(node))
            for child in node.body:
                loop_lines.update(_zero_index_span(child))
            for child in node.orelse:
                loop_lines.update(_zero_index_span(child))

    return output_lines, conditional_lines, loop_lines


def first_focus_region_3way(attempt_n_code: str, first_change_line_0idx: int) -> str:
    lines = attempt_n_code.splitlines()
    if first_change_line_0idx < 0 or first_change_line_0idx >= len(lines):
        raise FocusRegionError(
            f"first_change_line_0idx out of bounds: {first_change_line_0idx} for {len(lines)} lines"
        )

    output_lines, conditional_lines, loop_lines = build_region_sets(attempt_n_code)
    if first_change_line_0idx in output_lines:
        return "output_region"
    if first_change_line_0idx in conditional_lines:
        return "conditional_region"
    if first_change_line_0idx in loop_lines:
        return "loop_region"
    raise FocusRegionError(
        f"Could not map line {first_change_line_0idx} into output/conditional/loop regions"
    )
