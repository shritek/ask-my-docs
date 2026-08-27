import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

DEFAULT_CALIBRATION_PATH = Path("evaluation/judge_calibration_set.json")
NUMERIC_METRICS = ("faithfulness", "context_precision", "context_recall")
REQUIRED_METRICS = (*NUMERIC_METRICS, "answer_relevancy")
REPORT_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report_file(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def validate_score(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{description} must be between 0 and 1")
    return score


def load_calibration(path: Path) -> dict[str, Any]:
    calibration = load_json(path)
    labeling = calibration.get("labeling", {})
    if labeling.get("review_status") != "approved_for_project_calibration":
        raise ValueError("Calibration labels are not approved for project use")

    cases = calibration.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Calibration set must contain at least one case")

    case_ids = set()
    for index, case in enumerate(cases):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Calibration case {index} has no case ID")
        if case_id in case_ids:
            raise ValueError(f"Calibration set contains duplicate case ID {case_id}")
        case_ids.add(case_id)
        if case.get("review_status") != "approved":
            raise ValueError(f"Calibration case {case_id} is not approved")

        judgment = case.get("judgment", {})
        for metric_name in NUMERIC_METRICS:
            metric = judgment.get(metric_name, {})
            score_key = "average_precision" if metric_name == "context_precision" else "score"
            validate_score(
                metric.get(score_key),
                f"Calibration {metric_name} score for {case_id}",
            )
        band = judgment.get("answer_relevancy", {}).get("band")
        if band not in {"low", "medium", "high"}:
            raise ValueError(f"Calibration answer-relevancy band is invalid for {case_id}")

    source_result = calibration.get("source_result", {})
    if not isinstance(source_result.get("sha256"), str):
        raise ValueError("Calibration set has no source-result fingerprint")
    return calibration


def load_ragas_result(path: Path) -> dict[str, Any]:
    ragas_result = load_json(path)
    if ragas_result.get("status") != "completed":
        raise ValueError("Judge calibration requires a completed Ragas result")
    if ragas_result.get("errors"):
        raise ValueError("Judge calibration requires a Ragas result without errors")

    results = ragas_result.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Ragas result must contain at least one case")

    case_ids = set()
    for index, result in enumerate(results):
        case_id = result.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Ragas result {index} has no case ID")
        if case_id in case_ids:
            raise ValueError(f"Ragas result contains duplicate case ID {case_id}")
        case_ids.add(case_id)

    config = ragas_result.get("config", {})
    if not isinstance(config.get("source_result_sha256"), str):
        raise ValueError("Ragas result has no source-result fingerprint")
    return ragas_result


def reference_score(case: dict[str, Any], metric_name: str) -> float:
    metric = case["judgment"][metric_name]
    if metric_name == "context_precision":
        return float(metric["average_precision"])
    return float(metric["score"])


def summarize_numeric(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    references = [comparison["reference"] for comparison in comparisons]
    judge_scores = [comparison["judge"] for comparison in comparisons]
    absolute_errors = [comparison["absolute_error"] for comparison in comparisons]
    return {
        "reference_mean": fmean(references),
        "judge_mean": fmean(judge_scores),
        "mean_signed_error": fmean(
            judge - reference
            for judge, reference in zip(judge_scores, references, strict=True)
        ),
        "mean_absolute_error": fmean(absolute_errors),
        "max_absolute_error": max(absolute_errors),
        "reference_range": [min(references), max(references)],
        "judge_range": [min(judge_scores), max(judge_scores)],
        "catastrophic_disagreements": sum(
            comparison["catastrophic"] for comparison in comparisons
        ),
    }


def pairwise_band_ordering(
    labeled_scores: list[tuple[str, float]],
) -> dict[str, Any]:
    band_rank = {"low": 0, "medium": 1, "high": 2}
    ordered = 0
    tied = 0
    reversed_count = 0

    for left_index, (left_band, left_score) in enumerate(labeled_scores):
        for right_band, right_score in labeled_scores[left_index + 1:]:
            left_rank = band_rank[left_band]
            right_rank = band_rank[right_band]
            if left_rank == right_rank:
                continue

            expected = 1 if left_rank > right_rank else -1
            if math.isclose(left_score, right_score, abs_tol=1e-12):
                tied += 1
            elif (left_score - right_score) * expected > 0:
                ordered += 1
            else:
                reversed_count += 1

    total = ordered + tied + reversed_count
    accuracy = None if total == 0 else (ordered + 0.5 * tied) / total
    return {
        "pairs": total,
        "ordered": ordered,
        "tied": tied,
        "reversed": reversed_count,
        "accuracy": accuracy,
    }


def summarize_answer_relevancy(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    by_band = {}
    for band in ("low", "medium", "high"):
        scores = [
            comparison["judge"]
            for comparison in comparisons
            if comparison["reference_band"] == band
        ]
        by_band[band] = {
            "count": len(scores),
            "judge_mean": fmean(scores) if scores else None,
            "judge_range": [min(scores), max(scores)] if scores else None,
        }

    return {
        "by_reference_band": by_band,
        "pairwise_band_ordering": pairwise_band_ordering([
            (comparison["reference_band"], comparison["judge"])
            for comparison in comparisons
        ]),
        "catastrophic_disagreements": sum(
            comparison["catastrophic"] for comparison in comparisons
        ),
    }


def compare_judge(
    calibration: dict[str, Any],
    ragas_result: dict[str, Any],
    *,
    calibration_path: Path,
    ragas_result_path: Path,
    catastrophic_error_threshold: float = 0.5,
    relevancy_low_max: float = 0.4,
    relevancy_high_min: float = 0.7,
) -> dict[str, Any]:
    if not 0.0 < catastrophic_error_threshold <= 1.0:
        raise ValueError("Catastrophic error threshold must be in (0, 1]")
    if not 0.0 <= relevancy_low_max < relevancy_high_min <= 1.0:
        raise ValueError("Answer-relevancy thresholds must satisfy 0 <= low < high <= 1")

    source_sha = calibration["source_result"]["sha256"]
    ragas_source_sha = ragas_result["config"]["source_result_sha256"]
    if source_sha != ragas_source_sha:
        raise ValueError(
            "Calibration set and Ragas result reference different source runs"
        )

    ragas_by_case_id = {
        result["case_id"]: result
        for result in ragas_result["results"]
    }
    missing_case_ids = [
        case["case_id"]
        for case in calibration["cases"]
        if case["case_id"] not in ragas_by_case_id
    ]
    if missing_case_ids:
        raise ValueError(
            "Ragas result is missing calibration cases: "
            + ", ".join(missing_case_ids)
        )

    case_comparisons = []
    numeric_comparisons = {metric_name: [] for metric_name in NUMERIC_METRICS}
    answer_comparisons = []
    catastrophic_disagreements = []

    for case in calibration["cases"]:
        case_id = case["case_id"]
        ragas_metrics = ragas_by_case_id[case_id].get("metrics", {})
        missing_metrics = set(REQUIRED_METRICS) - ragas_metrics.keys()
        if missing_metrics:
            raise ValueError(
                f"Ragas case {case_id} is missing metrics: {sorted(missing_metrics)}"
            )

        metrics = {}
        for metric_name in NUMERIC_METRICS:
            reference = reference_score(case, metric_name)
            judge = validate_score(
                ragas_metrics[metric_name].get("value"),
                f"Ragas {metric_name} score for {case_id}",
            )
            absolute_error = abs(judge - reference)
            comparison = {
                "reference": reference,
                "judge": judge,
                "signed_error": judge - reference,
                "absolute_error": absolute_error,
                "catastrophic": absolute_error >= catastrophic_error_threshold,
            }
            metrics[metric_name] = comparison
            numeric_comparisons[metric_name].append(comparison)
            if comparison["catastrophic"]:
                catastrophic_disagreements.append({
                    "case_id": case_id,
                    "metric": metric_name,
                    "reference": reference,
                    "judge": judge,
                    "severity": absolute_error,
                })

        relevancy = case["judgment"]["answer_relevancy"]
        reference_band = relevancy["band"]
        judge_relevancy = validate_score(
            ragas_metrics["answer_relevancy"].get("value"),
            f"Ragas answer_relevancy score for {case_id}",
        )
        relevancy_catastrophic = (
            reference_band == "high" and judge_relevancy <= relevancy_low_max
        ) or (
            reference_band == "low" and judge_relevancy >= relevancy_high_min
        )
        answer_comparison = {
            "reference_band": reference_band,
            "judge": judge_relevancy,
            "abstention_present": relevancy["abstention_present"],
            "abstention_appropriate": relevancy["abstention_appropriate"],
            "catastrophic": relevancy_catastrophic,
        }
        metrics["answer_relevancy"] = answer_comparison
        answer_comparisons.append(answer_comparison)
        if relevancy_catastrophic:
            severity = (
                1.0 - judge_relevancy
                if reference_band == "high"
                else judge_relevancy
            )
            catastrophic_disagreements.append({
                "case_id": case_id,
                "metric": "answer_relevancy",
                "reference": reference_band,
                "judge": judge_relevancy,
                "severity": severity,
            })

        case_comparisons.append({
            "case_id": case_id,
            "question": case["question"],
            "selection_reason": case["selection_reason"],
            "metrics": metrics,
        })

    catastrophic_disagreements.sort(
        key=lambda disagreement: disagreement["severity"],
        reverse=True,
    )
    catastrophic_case_ids = sorted({
        disagreement["case_id"]
        for disagreement in catastrophic_disagreements
    })
    numeric_summary = {
        metric_name: summarize_numeric(numeric_comparisons[metric_name])
        for metric_name in NUMERIC_METRICS
    }

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "calibration": {
            "path": str(calibration_path),
            "sha256": sha256_file(calibration_path),
            "name": calibration.get("name"),
            "labeling": calibration["labeling"],
            "source_result_sha256": source_sha,
        },
        "ragas_result": {
            "path": str(ragas_result_path),
            "sha256": sha256_file(ragas_result_path),
            "config": ragas_result["config"],
        },
        "thresholds": {
            "numeric_catastrophic_absolute_error": catastrophic_error_threshold,
            "answer_relevancy_high_case_max_score": relevancy_low_max,
            "answer_relevancy_low_case_min_score": relevancy_high_min,
        },
        "summary": {
            "compared_cases": len(case_comparisons),
            "numeric_metrics": numeric_summary,
            "answer_relevancy": summarize_answer_relevancy(answer_comparisons),
            "total_catastrophic_disagreements": len(catastrophic_disagreements),
            "cases_with_catastrophic_disagreement": len(catastrophic_case_ids),
            "catastrophic_case_ids": catastrophic_case_ids,
            "catastrophic_disagreements": catastrophic_disagreements,
        },
        "cases": case_comparisons,
    }


def default_output_path(ragas_result_path: Path) -> Path:
    stem = ragas_result_path.stem.removeprefix("ragas__")
    return ragas_result_path.with_name(f"judge_calibration__{stem}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a Ragas judge with approved calibration labels."
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help=f"Approved calibration set (default: {DEFAULT_CALIBRATION_PATH})",
    )
    parser.add_argument(
        "--ragas-result",
        type=Path,
        required=True,
        help="Completed Ragas sidecar to compare",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report (default: beside the Ragas sidecar)",
    )
    parser.add_argument(
        "--catastrophic-error-threshold",
        type=float,
        default=0.5,
        help="Absolute-error threshold for numeric catastrophic disagreements",
    )
    parser.add_argument(
        "--relevancy-low-max",
        type=float,
        default=0.4,
        help="A high-reference case at or below this score is catastrophic",
    )
    parser.add_argument(
        "--relevancy-high-min",
        type=float,
        default=0.7,
        help="A low-reference case at or above this score is catastrophic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    ragas_result = load_ragas_result(args.ragas_result)
    report = compare_judge(
        calibration,
        ragas_result,
        calibration_path=args.calibration,
        ragas_result_path=args.ragas_result,
        catastrophic_error_threshold=args.catastrophic_error_threshold,
        relevancy_low_max=args.relevancy_low_max,
        relevancy_high_min=args.relevancy_high_min,
    )
    output_path = args.output or default_output_path(args.ragas_result)
    write_report_file(output_path, report)
    summary = report["summary"]
    console_summary = {
        "compared_cases": summary["compared_cases"],
        "numeric_metrics": {
            metric_name: {
                "mean_absolute_error": metric["mean_absolute_error"],
                "catastrophic_disagreements": metric[
                    "catastrophic_disagreements"
                ],
            }
            for metric_name, metric in summary["numeric_metrics"].items()
        },
        "answer_relevancy_pairwise_accuracy": summary["answer_relevancy"][
            "pairwise_band_ordering"
        ]["accuracy"],
        "answer_relevancy_catastrophic_disagreements": summary[
            "answer_relevancy"
        ]["catastrophic_disagreements"],
        "total_catastrophic_disagreements": summary[
            "total_catastrophic_disagreements"
        ],
        "cases_with_catastrophic_disagreement": summary[
            "cases_with_catastrophic_disagreement"
        ],
    }
    print(json.dumps(console_summary, indent=2))
    print(f"Judge calibration report saved to: {output_path}")


if __name__ == "__main__":
    main()
