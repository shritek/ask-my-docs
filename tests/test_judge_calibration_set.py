import hashlib
import json
from pathlib import Path
import unittest


CALIBRATION_PATH = Path("evaluation/judge_calibration_set.json")


def average_precision(relevance: list[bool]) -> float:
    relevant_count = sum(relevance)
    if relevant_count == 0:
        return 0.0

    relevant_seen = 0
    precision_sum = 0.0
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            relevant_seen += 1
            precision_sum += relevant_seen / rank
    return precision_sum / relevant_count


class JudgeCalibrationSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = json.loads(CALIBRATION_PATH.read_text())
        source_path = Path(cls.calibration["source_result"]["path"])
        cls.source_bytes = source_path.read_bytes()
        cls.source_run = json.loads(cls.source_bytes)
        cls.source_by_case_id = {
            result["case_id"]: result
            for result in cls.source_run["results"]
        }

    def test_source_result_fingerprint_matches(self):
        self.assertEqual(
            hashlib.sha256(self.source_bytes).hexdigest(),
            self.calibration["source_result"]["sha256"],
        )

    def test_contains_thirteen_unique_cases_from_the_source_run(self):
        cases = self.calibration["cases"]
        case_ids = [case["case_id"] for case in cases]

        self.assertEqual(len(cases), 13)
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertTrue(set(case_ids) <= self.source_by_case_id.keys())
        for case in cases:
            self.assertEqual(
                case["question"],
                self.source_by_case_id[case["case_id"]]["rag_result"]["question"],
            )

    def test_reviewed_claim_counts_and_scores_are_consistent(self):
        for case in self.calibration["cases"]:
            with self.subTest(case_id=case["case_id"]):
                judgment = case["judgment"]
                context_count = len(
                    self.source_by_case_id[case["case_id"]]["rag_result"]["contexts"]
                )
                for metric_name in ("faithfulness", "context_recall"):
                    metric = judgment[metric_name]
                    supported = sum(
                        claim["supported"]
                        for claim in metric["claims"]
                    )
                    total = len(metric["claims"])
                    self.assertEqual(metric["supported_claims"], supported)
                    self.assertEqual(metric["total_claims"], total)
                    self.assertAlmostEqual(metric["score"], supported / total)
                    for claim in metric["claims"]:
                        self.assertEqual(
                            bool(claim["evidence_contexts"]),
                            claim["supported"],
                        )
                        self.assertTrue(all(
                            1 <= context_index <= context_count
                            for context_index in claim["evidence_contexts"]
                        ))

                relevancy = judgment["answer_relevancy"]
                self.assertIn(relevancy["band"], {"high", "medium", "low"})
                if relevancy["abstention_present"]:
                    self.assertIsInstance(relevancy["abstention_appropriate"], bool)
                else:
                    self.assertIsNone(relevancy["abstention_appropriate"])

    def test_context_labels_match_retrieval_depth_and_average_precision(self):
        for case in self.calibration["cases"]:
            with self.subTest(case_id=case["case_id"]):
                source_case = self.source_by_case_id[case["case_id"]]
                precision = case["judgment"]["context_precision"]
                relevance = precision["context_relevance"]

                self.assertEqual(
                    len(relevance),
                    len(source_case["rag_result"]["contexts"]),
                )
                self.assertEqual(len(precision["notes"]), len(relevance))
                self.assertAlmostEqual(
                    precision["average_precision"],
                    average_precision(relevance),
                )

    def test_ai_review_provenance_and_approval_are_explicit(self):
        self.assertEqual(
            self.calibration["labeling"]["label_source"],
            "codex_reviewed",
        )
        self.assertEqual(
            self.calibration["labeling"]["review_status"],
            "approved_for_project_calibration",
        )
        self.assertEqual(self.calibration["labeling"]["quality_tier"], "silver")
        self.assertFalse(self.calibration["labeling"]["human_verified"])
        self.assertTrue(all(
            case["review_status"] == "approved"
            for case in self.calibration["cases"]
        ))


if __name__ == "__main__":
    unittest.main()
