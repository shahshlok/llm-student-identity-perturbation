from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from identity_perturbation.codebench_support.codemirror import infer_initial_code, parse_codemirror_log
from identity_perturbation.codebench_support.runner import ROOT, load_student_transition_context

from .trace_card import build_attempt_trace_cards
from .v61_encoding_policy import V61EncodingStats, decide_encoding
from .v61_payload import build_v61_payload
from .v61_prompting import build_user_prompt

DEFAULT_DATA_ROOT = ROOT.parent / "tracer" / "2024-1"


class V61EncodingReportError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect v6.1 event-log fit over a transition set."
    )
    parser.add_argument("--transition-manifest", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--idle-gap-seconds", type=float, default=30.0)
    parser.add_argument("--include-keyhandled", action="store_true")
    parser.add_argument("--exclude-navigation", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    rows = payload.get("selected_transitions")
    if not isinstance(rows, list) or not rows:
        raise V61EncodingReportError(f"Transition manifest missing selected_transitions: {path}")
    return rows


def build_report(
    *,
    transition_manifest: Path,
    data_root: Path,
    idle_gap_seconds: float,
    include_keyhandled: bool,
    include_navigation: bool,
) -> dict[str, object]:
    rows = _load_rows(transition_manifest)
    records: list[dict[str, object]] = []
    case_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()

    for row in rows:
        cls = str(row["class_id"])
        assess = str(row["assessment_id"])
        ex = str(row["exercise_id"])
        stu = str(row["student_id"])
        idx = int(row["transition_index_0idx"])
        exec_path = data_root / cls / "users" / stu / "executions" / f"{assess}_{ex}.log"
        cm_path = data_root / cls / "users" / stu / "codemirror" / f"{assess}_{ex}.log"
        attempts, trace, transitions = load_student_transition_context(
            class_id=cls,
            assessment_id=assess,
            exercise_id=ex,
            student_id=stu,
            exec_path=exec_path,
            cm_path=cm_path,
        )
        if trace is None:
            raise V61EncodingReportError(
                f"Missing replay trace for {cls}:{assess}:{ex}:{stu}:{idx}"
            )
        cm_events = parse_codemirror_log(cm_path)
        initial_code = infer_initial_code(cm_events, attempts[0].code)
        trace_cards = build_attempt_trace_cards(cm_events, initial_code=initial_code)
        payload = build_v61_payload(
            class_id=cls,
            assessment_id=assess,
            exercise_id=ex,
            student_id=stu,
            transition_index=idx,
            transition=transitions[idx],
            trace=trace,
            trace_cards=trace_cards,
            codemirror_log_path=cm_path,
            idle_gap_seconds=idle_gap_seconds,
            include_keyhandled=include_keyhandled,
            include_navigation=include_navigation,
        )
        user_prompt = build_user_prompt(payload)
        semantic_tape = payload["attempt_n"]["semantic_tape"]
        semantic_events = semantic_tape["semantic_event_tape"]
        summary = semantic_tape["semantic_tape_summary"]
        stats = V61EncodingStats(
            prompt_chars=len(user_prompt),
            raw_interval_event_count=int(summary["raw_interval_event_count"]),
            event_count=int(summary["semantic_event_count"]),
            change_count=int(summary["change_event_count"]),
            run_count=int(summary["saida_testar_count"]),
            submit_count=int(summary["submit_count"]),
            kill_program_count=int(summary["kill_program_count"]),
            idle_gap_count=int(summary["idle_gap_count"]),
            navigation_count=sum(
                1 for event in semantic_events if event["event_type"] == "tab_click"
            ),
            stdout_lines=sum(
                len(event.get("output_lines", []))
                for event in semantic_events
                if event["event_type"] == "saida_testar"
            ),
            used_full_stdout=all(
                event.get("output_line_limit") is None
                for event in semantic_events
                if event["event_type"] == "saida_testar"
            ),
        )
        decision = decide_encoding(stats)
        case_counter[decision.case_type] += 1
        flag_counter.update(decision.flags)
        records.append(
            {
                "id": f"{cls}:{assess}:{ex}:{stu}:{idx}",
                "prompt_chars": stats.prompt_chars,
                "event_count": stats.event_count,
                "stdout_lines": stats.stdout_lines,
                "case_type": decision.case_type,
                "flagged_case": decision.flagged_case,
                "flags": list(decision.flags),
            }
        )

    return {
        "schema_version": "v6_1_encoding_report_v1",
        "transition_manifest": str(transition_manifest.resolve()),
        "n_rows": len(records),
        "case_type_counts": dict(sorted(case_counter.items())),
        "flag_counts": dict(sorted(flag_counter.items())),
        "rows": records,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        transition_manifest=args.transition_manifest,
        data_root=args.data_root,
        idle_gap_seconds=args.idle_gap_seconds,
        include_keyhandled=args.include_keyhandled,
        include_navigation=not args.exclude_navigation,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
