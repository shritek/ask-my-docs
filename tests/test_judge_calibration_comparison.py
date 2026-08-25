import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.compare_judge_calibration import (
    compare_judge,
    default_output_path,
    load_calibration,
    load_ragas_result,
)


def make_calibration_case(
    case_id: str,
    *,
    score: float,
    band: str,
) -> dict:
    return {
        "case_id": case_id,
        "question": f"Question for {case_id}",
        "selection_reason": f"Reason for {case_id}",
        "review_status": "approved",
        "judgment": {
            "faithfulness": {"score": score},
            "answer_relevancy": {
                "band": band,
                "abstention_present": band == "low",
                "abstention_appropriate": True if band == "low" else None,
            },
            "context_precision": {"average_precision": score},
            "context_recall": {"score": score},
        },
    }


def make_ragas_case(
    case_id: str,
    *,
    faithfulness: float,
    answer_relevancy: float,
    context_precision: float,
    context_recall: float,
) -> dict:
    return {
        "case_id": case_id,
        "metrics": {
            "faithfulness": {"value": faithfulness},
            "answer_relevancy": {"value": answer_relevancy},
            "context_precision": {"value": context_precision},
            "context_recall": {"value": context_recall},
        },
    }


class JudgeCalibrationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {
            "name": "test-calibration",
            "source_result": {"sha256": "source-hash"},
            "labeling": {
                "review_status": "approved_for_project_calibration",
                "quality_tier": "silver",
            },
            "cases": [
                make_calibration_case("high", score=1.0, band="high"),
                make_calibration_case("medium", score=0.5, band="medium"),
                make_calibration_case("low", score=0.0, band="low"),
            ],
        }
        self.ragas_result = {
            "status": "completed",
            "errors": [],
            "config": {
                "source_result_sha256": "source-hash",
                "evaluator_llm_model": "test-judge",
            },
            "results": [
                make_ragas_case(
                    "high",
                    faithfulness=0.0,
                    answer_relevancy=0.3,
                    context_precision=1.0,
                    context_recall=0.75,
                ),
                make_ragas_case(
                    "medium",
                    faithfulness=0.5,
                    answer_relevancy=0.6,
                    context_precision=0.0,
                    context_recall=0.5,
                ),
                make_ragas_case(
                    "low",
                    faithfulness=0.75,
                    answer_relevancy=0.8,
                    context_precision=1.0,
                    context_recall=0.0,
                ),
            ],
        }

    def test_compares_numeric_metrics_and_relevancy_ordering(self):
        with TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"
            ragas_path = Path(directory) / "ragas.json"
            calibration_path.write_text(json.dumps(self.calibration))
            ragas_path.write_text(json.dumps(self.ragas_result))

            report = compare_judge(
                self.calibration,
                self.ragas_result,
                calibration_path=calibration_path,
                ragas_result_path=ragas_path,
            )

        summary = report["summary"]
        self.assertEqual(summary["compared_cases"], 3)
        self.assertAlmostEqual(
            summary["numeric_metrics"]["faithfulness"]["mean_absolute_error"],
            (1.0 + 0.0 + 0.75) / 3,
        )
        self.assertEqual(
            summary["numeric_metrics"]["faithfulness"][
                "catastrophic_disagreements"
            ],
            2,
        )
        self.assertEqual(
            summary["numeric_metrics"]["context_precision"]["judge_range"],
            [0.0, 1.0],
        )
        ordering = summary["answer_relevancy"]["pairwise_band_ordering"]
        self.assertEqual(ordering["pairs"], 3)
        self.assertEqual(ordering["reversed"], 3)
        self.assertEqual(ordering["accuracy"], 0.0)
        self.assertEqual(
            summary["answer_relevancy"]["catastrophic_disagreements"],
            2,
        )
        self.assertEqual(summary["cases_with_catastrophic_disagreement"], 3)
        self.assertEqual(
            summary["catastrophic_case_ids"],
            ["high", "low", "medium"],
        )
        self.assertTrue(report["cases"][0]["metrics"]["faithfulness"]["catastrophic"])

    def test_rejects_source_result_fingerprint_mismatch(self):
        self.ragas_result["config"]["source_result_sha256"] = "different-source"
        with TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"
            ragas_path = Path(directory) / "ragas.json"
            calibration_path.write_text(json.dumps(self.calibration))
            ragas_path.write_text(json.dumps(self.ragas_result))

            with self.assertRaisesRegex(ValueError, "different source runs"):
                compare_judge(
                    self.calibration,
                    self.ragas_result,
                    calibration_path=calibration_path,
                    ragas_result_path=ragas_path,
                )

    def test_loaders_reject_unapproved_or_incomplete_inputs(self):
        with TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"
            ragas_path = Path(directory) / "ragas.json"

            self.calibration["cases"][0]["review_status"] = "needs_review"
            calibration_path.write_text(json.dumps(self.calibration))
            with self.assertRaisesRegex(ValueError, "is not approved"):
                load_calibration(calibration_path)

            self.ragas_result["status"] = "completed_with_errors"
            ragas_path.write_text(json.dumps(self.ragas_result))
            with self.assertRaisesRegex(ValueError, "completed Ragas result"):
                load_ragas_result(ragas_path)

    def test_default_output_sits_beside_ragas_result(self):
        path = Path("evaluation/results/ragas__basic__judge-test.json")
        self.assertEqual(
            default_output_path(path),
            Path("evaluation/results/judge_calibration__basic__judge-test.json"),
        )


if __name__ == "__main__":
    unittest.main()
