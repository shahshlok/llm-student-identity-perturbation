"""Per-exercise breakdown of v6.1 condition comparison.

For each exercise, compute mean metrics per condition, deltas (full - shuffled),
and paired Wilcoxon tests where N >= 5. Also estimates exercise complexity from
episode motif length (event count in the observed episode).
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]

CONDITIONS = {
    "full": ROOT
    / "data"
    / "v61_batch_runs"
    / "gpt54_medium_v61_clean126_buildable117_v6"
    / "evaluation.json",
    "no_trace": ROOT
    / "data"
    / "v61_batch_runs"
    / "gpt54_medium_v61_clean126_buildable117_no_trace_v1"
    / "evaluation.json",
    "shuffled": ROOT
    / "data"
    / "v61_batch_runs"
    / "gpt54_medium_v61_clean126_buildable117_trace_shuffled_v1"
    / "evaluation.json",
}

OUTPUT_DIR = ROOT / "data" / "v61_condition_comparisons"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["scores"]["rows"]


def extract_metric(row: dict[str, Any], metric_name: str, key: str) -> float | None:
    m = row.get("metrics", {}).get(metric_name)
    if m is None:
        return None
    v = m.get(key)
    if v is None:
        return None
    return float(v)


def episode_event_count(row: dict[str, Any]) -> int | None:
    """Count events in the observed episode motif as a complexity proxy."""
    m = row.get("metrics", {}).get("episode_motif")
    if m is None:
        return None
    observed = m.get("observed")
    if not isinstance(observed, str):
        return None
    return len(observed.split("->"))


def main() -> None:
    # --- Load all rows, keyed by custom_id ---
    condition_rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for label, path in CONDITIONS.items():
        for row in load_rows(path):
            cid = row["custom_id"]
            condition_rows[cid][label] = row

    # --- Group by exercise_id ---
    exercise_data: dict[str, list[str]] = defaultdict(list)  # exercise_id -> [custom_ids]
    for cid, cond_map in condition_rows.items():
        any_row = next(iter(cond_map.values()))
        ex_id = any_row["exercise_id"]
        exercise_data[ex_id].append(cid)

    # --- Metrics to extract ---
    METRICS = [
        ("event_type_overlap", "top1_jaccard", "jaccard"),
        ("event_type_edit_similarity", "top1_similarity", "edit_sim"),
        ("episode_motif", "top1_match", "motif_top1"),
    ]

    # --- Compute per-exercise stats ---
    rows_out: list[dict[str, Any]] = []

    for ex_id, cids in sorted(exercise_data.items(), key=lambda kv: kv[0]):
        n = len(cids)
        # Gather student ids for this exercise
        students = set()
        for cid in cids:
            any_row = next(iter(condition_rows[cid].values()))
            students.add(any_row["student_id"])

        rec: dict[str, Any] = {
            "exercise_id": ex_id,
            "n_transitions": n,
            "n_students": len(students),
        }

        # Complexity proxy: mean episode event count from full condition
        event_counts = []
        for cid in cids:
            r = condition_rows[cid].get("full")
            if r:
                ec = episode_event_count(r)
                if ec is not None:
                    event_counts.append(ec)
        rec["mean_episode_events"] = (
            round(statistics.mean(event_counts), 1) if event_counts else None
        )

        for metric_name, metric_key, short_name in METRICS:
            per_cond_values: dict[str, list[float]] = {}
            for label in CONDITIONS:
                vals = []
                for cid in cids:
                    r = condition_rows[cid].get(label)
                    if r is None:
                        continue
                    v = extract_metric(r, metric_name, metric_key)
                    if v is not None:
                        if isinstance(v, bool):
                            vals.append(1.0 if v else 0.0)
                        else:
                            vals.append(v)
                per_cond_values[label] = vals

            for label in CONDITIONS:
                vals = per_cond_values.get(label, [])
                rec[f"{short_name}_{label}"] = round(statistics.mean(vals), 4) if vals else None

            # Delta & Wilcoxon: full vs shuffled, paired by custom_id
            paired_full = []
            paired_shuf = []
            for cid in cids:
                rf = condition_rows[cid].get("full")
                rs = condition_rows[cid].get("shuffled")
                if rf and rs:
                    vf = extract_metric(rf, metric_name, metric_key)
                    vs = extract_metric(rs, metric_name, metric_key)
                    if vf is not None and vs is not None:
                        if isinstance(vf, bool):
                            vf = 1.0 if vf else 0.0
                        if isinstance(vs, bool):
                            vs = 1.0 if vs else 0.0
                        paired_full.append(vf)
                        paired_shuf.append(vs)

            if paired_full:
                delta = statistics.mean(paired_full) - statistics.mean(paired_shuf)
                rec[f"{short_name}_delta"] = round(delta, 4)
                rec[f"{short_name}_paired_n"] = len(paired_full)

                diffs = [a - b for a, b in zip(paired_full, paired_shuf, strict=False)]
                if any(d != 0 for d in diffs) and len(diffs) >= 5:
                    try:
                        stat, pval = wilcoxon(paired_full, paired_shuf)
                        rec[f"{short_name}_wilcoxon_p"] = round(pval, 4)
                    except Exception:
                        rec[f"{short_name}_wilcoxon_p"] = None
                else:
                    rec[f"{short_name}_wilcoxon_p"] = None
            else:
                rec[f"{short_name}_delta"] = None
                rec[f"{short_name}_paired_n"] = 0
                rec[f"{short_name}_wilcoxon_p"] = None

        rows_out.append(rec)

    # --- Sort by jaccard delta descending ---
    rows_out.sort(key=lambda r: r.get("jaccard_delta") or -999, reverse=True)

    # --- Write markdown report ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / "per_exercise_breakdown.md"
    lines = _build_markdown(rows_out)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")

    # --- Also dump JSON ---
    json_path = OUTPUT_DIR / "per_exercise_breakdown.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, indent=2)
    print(f"Wrote {json_path}")


def _build_markdown(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Per-Exercise Breakdown: v6.1 Condition Comparison",
        "",
        "Sorted by Jaccard delta (full - shuffled), descending.",
        "",
        "- **full**: full student-specific trace",
        "- **shuffled**: traces from a random student on the same exercise",
        "- **no_trace**: no CodeMirror trace at all",
        "- **delta**: full minus shuffled (positive = student-specific trace helps)",
        "- **p**: paired Wilcoxon signed-rank test p-value (full vs shuffled), only for N >= 5",
        "",
    ]

    # --- Main table: Jaccard ---
    lines.extend(
        [
            "## Jaccard (event_type_overlap)",
            "",
            "| Exercise | N trans | N stu | Ep events | full | shuffled | no_trace | delta | Wilcoxon p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['exercise_id']} "
            f"| {r['n_transitions']} "
            f"| {r['n_students']} "
            f"| {_fmt(r.get('mean_episode_events'))} "
            f"| {_fmt(r.get('jaccard_full'))} "
            f"| {_fmt(r.get('jaccard_shuffled'))} "
            f"| {_fmt(r.get('jaccard_no_trace'))} "
            f"| {_fmt_delta(r.get('jaccard_delta'))} "
            f"| {_fmt_p(r.get('jaccard_wilcoxon_p'))} |"
        )

    # --- Edit similarity table ---
    lines.extend(
        [
            "",
            "## Edit Similarity (event_type_edit_similarity)",
            "",
            "| Exercise | N trans | full | shuffled | no_trace | delta | Wilcoxon p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['exercise_id']} "
            f"| {r['n_transitions']} "
            f"| {_fmt(r.get('edit_sim_full'))} "
            f"| {_fmt(r.get('edit_sim_shuffled'))} "
            f"| {_fmt(r.get('edit_sim_no_trace'))} "
            f"| {_fmt_delta(r.get('edit_sim_delta'))} "
            f"| {_fmt_p(r.get('edit_sim_wilcoxon_p'))} |"
        )

    # --- Episode motif table ---
    lines.extend(
        [
            "",
            "## Episode Motif Top-1 Accuracy",
            "",
            "| Exercise | N trans | full | shuffled | no_trace | delta | Wilcoxon p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['exercise_id']} "
            f"| {r['n_transitions']} "
            f"| {_fmt(r.get('motif_top1_full'))} "
            f"| {_fmt(r.get('motif_top1_shuffled'))} "
            f"| {_fmt(r.get('motif_top1_no_trace'))} "
            f"| {_fmt_delta(r.get('motif_top1_delta'))} "
            f"| {_fmt_p(r.get('motif_top1_wilcoxon_p'))} |"
        )

    # --- Summary: which exercises show signal? ---
    lines.extend(
        [
            "",
            "## Summary",
            "",
        ]
    )

    pos_exercises = [r for r in rows if (r.get("jaccard_delta") or 0) > 0]
    neg_exercises = [r for r in rows if (r.get("jaccard_delta") or 0) < 0]
    zero_exercises = [r for r in rows if (r.get("jaccard_delta") or 0) == 0]

    lines.append(
        f"- Exercises with positive Jaccard delta (full > shuffled): **{len(pos_exercises)}**"
    )
    lines.append(
        f"- Exercises with negative Jaccard delta (full < shuffled): **{len(neg_exercises)}**"
    )
    lines.append(f"- Exercises with zero/missing delta: **{len(zero_exercises)}**")
    lines.append("")

    sig_exercises = [
        r
        for r in rows
        if r.get("jaccard_wilcoxon_p") is not None and r["jaccard_wilcoxon_p"] < 0.05
    ]
    lines.append(
        f"- Exercises with Wilcoxon p < 0.05 on Jaccard (full vs shuffled): **{len(sig_exercises)}**"
    )
    if sig_exercises:
        for r in sig_exercises:
            lines.append(
                f"  - Exercise {r['exercise_id']}: delta={_fmt_delta(r.get('jaccard_delta'))}, p={_fmt_p(r.get('jaccard_wilcoxon_p'))}"
            )
    lines.append("")

    # --- Complexity correlation ---
    lines.extend(
        [
            "## Complexity vs. Signal",
            "",
            "Is the trace delta correlated with episode complexity (mean event count)?",
            "",
        ]
    )

    pairs = [
        (r["mean_episode_events"], r["jaccard_delta"])
        for r in rows
        if r.get("mean_episode_events") is not None and r.get("jaccard_delta") is not None
    ]
    if len(pairs) >= 4:
        from scipy.stats import spearmanr

        x_vals, y_vals = zip(*pairs, strict=False)
        rho, sp_p = spearmanr(x_vals, y_vals)
        lines.append(
            f"- Spearman rho (episode events vs Jaccard delta): **{rho:.3f}** (p={sp_p:.4f}, N={len(pairs)} exercises)"
        )
    else:
        lines.append("- Not enough data for correlation.")

    lines.append("")

    # Rank exercises by complexity and show top/bottom
    by_complexity = sorted(
        [r for r in rows if r.get("mean_episode_events") is not None],
        key=lambda r: r["mean_episode_events"],
        reverse=True,
    )
    if by_complexity:
        lines.append("### Top 5 most complex exercises (by mean episode events):")
        lines.append("")
        lines.append("| Exercise | Ep events | N trans | Jaccard delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for r in by_complexity[:5]:
            lines.append(
                f"| {r['exercise_id']} | {_fmt(r['mean_episode_events'])} | {r['n_transitions']} | {_fmt_delta(r.get('jaccard_delta'))} |"
            )
        lines.append("")
        lines.append("### Bottom 5 simplest exercises:")
        lines.append("")
        lines.append("| Exercise | Ep events | N trans | Jaccard delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for r in by_complexity[-5:]:
            lines.append(
                f"| {r['exercise_id']} | {_fmt(r['mean_episode_events'])} | {r['n_transitions']} | {_fmt_delta(r.get('jaccard_delta'))} |"
            )

    return lines


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _fmt_delta(v: Any) -> str:
    if v is None:
        return "-"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.4f}"


def _fmt_p(v: Any) -> str:
    if v is None:
        return "-"
    if v < 0.001:
        return f"**{v:.4f}**"
    if v < 0.05:
        return f"**{v:.4f}**"
    return f"{v:.4f}"


if __name__ == "__main__":
    main()
