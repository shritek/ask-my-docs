import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.compare_judge_stability import (
    compare_judge_stability,
    default_output_path,
)


def make_result(scores_by_case: dict[str, dict[str, float]]) -> dict:
    return {
        "status": "completed",
        "errors": [],
        "config": {
            "source_result_sha256": "source-hash",
            "evaluator_llm_model": "test-judge",
            "evaluator_reasoning_effort": "none",
        },
        "results": [
            {
                "case_id": case_id,
                "metrics": {
                    metric_name: {"value": value}
                    for metric_name, value in scores.items()
                },
            }
            for case_id, scores in scores_by_case.items()
        ],
    }


class JudgeStabilityComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline = make_result({
            "case-a": {
                "faithfulness": 1.0,
                "answer_relevancy": 0.8,
                "context_precision": 0.5,
                "context_recall": 0.0,
            },
            "case-b": {
                "faithfulness": 0.5,
                "answer_relevancy": 0.4,
                "context_precision": 1.0,
                "context_recall": 0.75,
            },
        })
        self.repeat = make_result({
            "case-a": {
                "faithfulness": 1.0,
                "answer_relevancy": 0.7,
                "context_precision": 0.0,
                "context_recall": 0.0,
            },
            "case-b": {
                "faithfulness": 0.75,
                "answer_relevancy": 0.4,
                "context_precision": 1.0,
                "context_recall": 0.5,
            },
        })

    def compare(self) -> dict:
        with TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.json"
            repeat_path = Path(directory) / "repeat.json"
            baseline_path.write_text(json.dumps(self.baseline))
            repeat_path.write_text(json.dumps(self.repeat))
            return compare_judge_stability(
                self.baseline,
                self.repeat,
                baseline_path=baseline_path,
                repeat_path=repeat_path,
            )

    def test_compares_per_metric_repeatability(self):
        report = self.compare()

        summary = report["summary"]
        self.assertEqual(summary["compared_cases"], 2)
        self.assertEqual(summary["cases_with_material_change"], 2)
        self.assertEqual(
            summary["material_change_case_ids"],
            ["case-a", "case-b"],
        )
        self.assertEqual(
            summary["metrics"]["faithfulness"],
            {
                "mean_absolute_delta": 0.125,
                "max_absolute_delta": 0.25,
                "exact_matches": 1,
                "material_changes": 1,
            },
        )
        self.assertEqual(
            summary["metrics"]["context_precision"]["max_absolute_delta"],
            0.5,
        )
        self.assertTrue(
            report["cases"][0]["metrics"]["context_precision"][
                "material_change"
            ]
        )

    def test_rejects_configuration_or_case_mismatch(self):
        self.repeat["config"]["evaluator_llm_model"] = "another-judge"
        with self.assertRaisesRegex(ValueError, "different evaluator configurations"):
            self.compare()

        self.repeat = make_result({
            "different-case": {
                "faithfulness": 1.0,
                "answer_relevancy": 0.8,
                "context_precision": 0.5,
                "context_recall": 0.0,
            }
        })
        with self.assertRaisesRegex(ValueError, "different case IDs"):
            self.compare()

    def test_default_output_sits_beside_repeated_result(self):
        path = Path("evaluation/results/ragas__judge-test__repeat-2.json")
        self.assertEqual(
            default_output_path(path),
            Path("evaluation/results/judge_stability__judge-test__repeat-2.json"),
        )


if __name__ == "__main__":
    unittest.main()
