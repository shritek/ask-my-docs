import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from evaluation.compare_judge_calibration import (
    REQUIRED_METRICS,
    load_ragas_result,
    sha256_file,
    validate_score,
    write_report_file,
)

REPORT_SCHEMA_VERSION = 1


def compare_judge_stability(
    baseline: dict[str, Any],
    repeat: dict[str, Any],
    *,
    baseline_path: Path,
    repeat_path: Path,
    material_delta_threshold: float = 0.25,
) -> dict[str, Any]:
    if not 0.0 < material_delta_threshold <= 1.0:
        raise ValueError("Material delta threshold must be in (0, 1]")

    if baseline["config"] != repeat["config"]:
        raise ValueError("Stability runs use different evaluator configurations")

    baseline_by_case_id = {
        result["case_id"]: result
        for result in baseline["results"]
    }
    repeat_by_case_id = {
        result["case_id"]: result
        for result in repeat["results"]
    }
    if baseline_by_case_id.keys() != repeat_by_case_id.keys():
        raise ValueError("Stability runs contain different case IDs")

    case_comparisons = []
    metric_comparisons = {metric_name: [] for metric_name in REQUIRED_METRICS}
    cases_with_material_change = set()

    for case_id in baseline_by_case_id:
        baseline_metrics = baseline_by_case_id[case_id].get("metrics", {})
        repeat_metrics = repeat_by_case_id[case_id].get("metrics", {})
        missing_metrics = set(REQUIRED_METRICS) - (
            baseline_metrics.keys() & repeat_metrics.keys()
        )
        if missing_metrics:
            raise ValueError(
                f"Stability case {case_id} is missing metrics: "
                f"{sorted(missing_metrics)}"
            )

        metrics = {}
        for metric_name in REQUIRED_METRICS:
            baseline_score = validate_score(
                baseline_metrics[metric_name].get("value"),
                f"Baseline {metric_name} score for {case_id}",
            )
            repeat_score = validate_score(
                repeat_metrics[metric_name].get("value"),
                f"Repeat {metric_name} score for {case_id}",
            )
            signed_delta = repeat_score - baseline_score
            absolute_delta = abs(signed_delta)
            comparison = {
                "baseline": baseline_score,
                "repeat": repeat_score,
                "signed_delta": signed_delta,
                "absolute_delta": absolute_delta,
                "exact_match": math.isclose(
                    baseline_score,
                    repeat_score,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "material_change": absolute_delta >= material_delta_threshold,
            }
            metrics[metric_name] = comparison
            metric_comparisons[metric_name].append(comparison)
            if comparison["material_change"]:
                cases_with_material_change.add(case_id)

        case_comparisons.append({
            "case_id": case_id,
            "metrics": metrics,
        })

    metric_summary = {}
    for metric_name, comparisons in metric_comparisons.items():
        absolute_deltas = [
            comparison["absolute_delta"]
            for comparison in comparisons
        ]
        metric_summary[metric_name] = {
            "mean_absolute_delta": fmean(absolute_deltas),
            "max_absolute_delta": max(absolute_deltas),
            "exact_matches": sum(
                comparison["exact_match"] for comparison in comparisons
            ),
            "material_changes": sum(
                comparison["material_change"] for comparison in comparisons
            ),
        }

    material_case_ids = sorted(cases_with_material_change)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "repeat": {
            "path": str(repeat_path),
            "sha256": sha256_file(repeat_path),
        },
        "config": baseline["config"],
        "thresholds": {
            "material_absolute_delta": material_delta_threshold,
        },
        "summary": {
            "compared_cases": len(case_comparisons),
            "metrics": metric_summary,
            "cases_with_material_change": len(material_case_ids),
            "material_change_case_ids": material_case_ids,
        },
        "cases": case_comparisons,
    }


def default_output_path(repeat_path: Path) -> Path:
    stem = repeat_path.stem.removeprefix("ragas__")
    return repeat_path.with_name(f"judge_stability__{stem}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare repeated Ragas runs for judge-score stability."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="First completed Ragas sidecar",
    )
    parser.add_argument(
        "--repeat",
        type=Path,
        required=True,
        help="Independent repeated Ragas sidecar",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report (default: beside the repeated sidecar)",
    )
    parser.add_argument(
        "--material-delta-threshold",
        type=float,
        default=0.25,
        help="Absolute score change treated as material (default: 0.25)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_ragas_result(args.baseline)
    repeat = load_ragas_result(args.repeat)
    report = compare_judge_stability(
        baseline,
        repeat,
        baseline_path=args.baseline,
        repeat_path=args.repeat,
        material_delta_threshold=args.material_delta_threshold,
    )
    output_path = args.output or default_output_path(args.repeat)
    write_report_file(output_path, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"Judge stability report saved to: {output_path}")


if __name__ == "__main__":
    main()
