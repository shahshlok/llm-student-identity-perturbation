from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import SliceBuildError

HEAD_SUBSETS = {
    "focus_only": ("first_focus_region_3way",),
    "lines_only": ("lines_touched_bucket_3way",),
    "outcome_only": ("next_test_outcome",),
    "focus_plus_lines": ("first_focus_region_3way", "lines_touched_bucket_3way"),
    "focus_plus_outcome": ("first_focus_region_3way", "next_test_outcome"),
    "lines_plus_outcome": ("lines_touched_bucket_3way", "next_test_outcome"),
    "full_3head": (
        "first_focus_region_3way",
        "lines_touched_bucket_3way",
        "next_test_outcome",
    ),
}
BASELINE_NAMES = (
    "leave_one_out_majority",
    "rotated_slice",
    "uniform",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a head-subset ablation report from a v5 evaluation.json file."
    )
    parser.add_argument("--evaluation", required=True, type=Path, help="Path to evaluation.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults to the evaluation file's parent",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    if not path.parent.exists():
        raise SliceBuildError(f"Parent directory does not exist: {path.parent}")
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _subset_rate(per_slice: list[dict[str, object]], heads: tuple[str, ...]) -> float:
    matches = 0
    for row in per_slice:
        if all(bool(row["by_head"][head]["top1_match"]) for head in heads):
            matches += 1
    return matches / len(per_slice)


def build_head_ablation_report(evaluation_path: Path, out_dir: Path | None) -> dict[str, object]:
    if not evaluation_path.exists():
        raise SliceBuildError(f"Evaluation file not found: {evaluation_path}")

    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    scores = payload["scores"]
    run_dir = out_dir if out_dir is not None else evaluation_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    by_subset: dict[str, dict[str, float]] = {}
    for subset_name, heads in HEAD_SUBSETS.items():
        by_subset[subset_name] = {}
        for score_name, score_payload in scores.items():
            by_subset[subset_name][score_name] = _subset_rate(score_payload["per_slice"], heads)

    ranked = []
    for subset_name, subset_scores in by_subset.items():
        best_baseline = max(subset_scores[name] for name in BASELINE_NAMES)
        ranked.append(
            {
                "subset_name": subset_name,
                "heads": list(HEAD_SUBSETS[subset_name]),
                "model_rate": subset_scores["model"],
                "best_baseline_rate": best_baseline,
                "model_minus_best_baseline": subset_scores["model"] - best_baseline,
                "leave_one_out_majority": subset_scores["leave_one_out_majority"],
                "rotated_slice": subset_scores["rotated_slice"],
                "uniform": subset_scores["uniform"],
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["model_minus_best_baseline"],
            -row["model_rate"],
            row["subset_name"],
        )
    )

    report = {
        "schema_version": "v5_head_ablation_v1",
        "source_evaluation": str(evaluation_path.resolve()),
        "run_name": payload["run_name"],
        "model": payload["model"],
        "reasoning_effort": payload["reasoning_effort"],
        "n_slices": payload["summary"]["n_slices"],
        "by_subset": by_subset,
        "ranked_subsets": ranked,
    }

    json_path = run_dir / "head_ablation.json"
    md_path = run_dir / "head_ablation.md"
    _write_json(json_path, report)

    lines = [
        "# V5 Head-Subset Ablation",
        "",
        f"- Run name: `{report['run_name']}`",
        f"- Model: `{report['model']}`",
        f"- Reasoning effort: `{report['reasoning_effort']}`",
        f"- Evaluated slices: `{report['n_slices']}`",
        "",
        "| Subset | Heads | Model | Best baseline | Model - best baseline | Majority | Rotated | Uniform |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ranked:
        lines.append(
            f"| {row['subset_name']} | {', '.join(row['heads'])} | "
            f"{row['model_rate']:.3f} | {row['best_baseline_rate']:.3f} | "
            f"{row['model_minus_best_baseline']:.3f} | "
            f"{row['leave_one_out_majority']:.3f} | {row['rotated_slice']:.3f} | "
            f"{row['uniform']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "report": report,
        "paths": {
            "json": str(json_path.resolve()),
            "md": str(md_path.resolve()),
        },
    }


def main() -> int:
    args = parse_args()
    result = build_head_ablation_report(
        evaluation_path=args.evaluation,
        out_dir=args.out_dir,
    )
    print(json.dumps(result["report"]["ranked_subsets"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
