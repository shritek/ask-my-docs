from copy import deepcopy
import unittest

from evaluation.finalize_retrieval_qrels import finalize_qrels
from evaluation.ranked_retrieval import POOL_SCHEMA_VERSION, QRELS_SCHEMA_VERSION


class FinalizeRetrievalQrelsTests(unittest.TestCase):
    def setUp(self):
        self.pool = {
            "schema_version": POOL_SCHEMA_VERSION,
            "config": {
                "chunking_strategies": ["semantic"],
                "pool_depth": 2,
            },
            "cases": [{
                "case_id": "case-1",
                "rankings": {"semantic": ["a", "b"]},
                "candidates": [{"chunk_id": "a"}, {"chunk_id": "b"}],
            }],
        }
        self.proposal = {
            "schema_version": QRELS_SCHEMA_VERSION,
            "status": "generated_pending_review",
            "errors": [],
            "cases": [{
                "case_id": "case-1",
                "question": "question",
                "reference_claims": [
                    {"claim_id": "c1", "claim": "combined claim"},
                ],
                "judgments": [
                    {
                        "chunk_id": "a",
                        "relevant": True,
                        "supported_claim_ids": ["c1"],
                        "notes": "old",
                    },
                    {
                        "chunk_id": "b",
                        "relevant": False,
                        "supported_claim_ids": [],
                        "notes": "old",
                    },
                ],
            }],
        }
        self.review = {
            "labeling": {
                "label_source": "test",
                "quality_tier": "silver",
                "review_status": "approved_for_project_evaluation",
                "reviewed_by": "test suite",
                "reviewed_at": "2026-08-31",
                "human_verified": False,
                "method": "test fixture",
            },
            "audit": {"cases_corrected": 1},
            "case_overrides": [{
                "review_id": "split-claim",
                "case_id": "case-1",
                "rationale": "Make claims atomic.",
                "reference_claims": [
                    {"claim_id": "c1", "claim": "first claim"},
                    {"claim_id": "c2", "claim": "second claim"},
                ],
                "claim_id_remap": {"c1": ["c1"]},
                "judgment_updates": [{
                    "chunk_id": "b",
                    "supported_claim_ids": ["c2"],
                    "notes": "reviewed",
                }],
            }],
        }

    def test_finalizes_copy_and_applies_reviewed_override(self):
        original = deepcopy(self.proposal)

        qrels = finalize_qrels(self.pool, self.proposal, self.review)

        self.assertEqual(self.proposal, original)
        self.assertEqual(qrels["status"], "approved")
        self.assertNotIn("errors", qrels)
        case = qrels["cases"][0]
        self.assertEqual(len(case["reference_claims"]), 2)
        self.assertEqual(case["judgments"][1]["supported_claim_ids"], ["c2"])
        self.assertTrue(case["judgments"][1]["relevant"])
        self.assertEqual(case["review_history"][0]["review_id"], "split-claim")

    def test_rejects_incomplete_proposal(self):
        self.proposal["status"] = "generated_with_errors"

        with self.assertRaisesRegex(ValueError, "complete"):
            finalize_qrels(self.pool, self.proposal, self.review)


if __name__ == "__main__":
    unittest.main()
