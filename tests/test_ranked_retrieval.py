import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.documents import Document

from evaluation.generate_retrieval_pool import build_pool
from evaluation.initialize_retrieval_qrels import initialize_qrels
from evaluation.ranked_retrieval import (
    POOL_SCHEMA_VERSION,
    QRELS_SCHEMA_VERSION,
    claim_coverage_at_k,
    ndcg_at_k,
    pooled_recall_at_k,
    precision_at_k,
    reciprocal_rank_at_k,
    score_pool,
)
from helpers.chunk_ids import create_chunk_id


class RankingMetricTests(unittest.TestCase):
    def test_binary_ranking_metrics(self):
        relevance = [True, False, True, True]

        self.assertAlmostEqual(precision_at_k(relevance, 3), 2 / 3)
        self.assertAlmostEqual(pooled_recall_at_k(relevance, 3), 2 / 3)
        self.assertEqual(reciprocal_rank_at_k(relevance, 3), 1.0)
        expected_ndcg = (1 + (1 / math.log2(4))) / (
            1 + (1 / math.log2(3)) + (1 / math.log2(4))
        )
        self.assertAlmostEqual(ndcg_at_k(relevance, 3), expected_ndcg)

    def test_metrics_return_zero_when_no_result_is_relevant(self):
        relevance = [False, False]

        self.assertEqual(precision_at_k(relevance, 2), 0.0)
        self.assertEqual(pooled_recall_at_k(relevance, 2), 0.0)
        self.assertEqual(reciprocal_rank_at_k(relevance, 2), 0.0)
        self.assertEqual(ndcg_at_k(relevance, 2), 0.0)

    def test_claim_coverage_uses_unique_supported_claims(self):
        from evaluation.ranked_retrieval import ChunkJudgment

        judgments = {
            "a": ChunkJudgment(True, frozenset({"c1"})),
            "b": ChunkJudgment(True, frozenset({"c1", "c2"})),
        }

        self.assertEqual(
            claim_coverage_at_k(["a", "b"], judgments, {"c1", "c2"}, 1),
            0.5,
        )
        self.assertEqual(
            claim_coverage_at_k(["a", "b"], judgments, {"c1", "c2"}, 2),
            1.0,
        )


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def similarity_search(self, query, k):
        self.calls.append((query, k))
        return self.documents[:k]

    def similarity_search_by_vector(self, embedding, k):
        self.calls.append((embedding, k))
        return self.documents[:k]


class PoolGenerationTests(unittest.TestCase):
    def test_build_pool_preserves_rankings_and_deduplicates_candidates(self):
        shared_id = create_chunk_id("https://example.com/shared", "semantic", "shared")
        other_id = create_chunk_id("https://example.com/other", "recursive-500", "other")
        shared = Document(
            page_content="shared",
            metadata={"chunk_id": shared_id, "source_url": "https://example.com/shared"},
        )
        other = Document(
            page_content="other",
            metadata={"chunk_id": other_id, "source_url": "https://example.com/other"},
        )
        stores = {
            "semantic": FakeVectorStore([shared]),
            "recursive-500": FakeVectorStore([other, shared]),
        }
        test_case = {
            "question": "question",
            "answer": "answer",
            "source_url": "https://example.com/shared",
            "title": "Shared",
        }

        cases = build_pool([test_case], stores, pool_depth=2)

        self.assertEqual(cases[0]["rankings"], {
            "semantic": [shared_id],
            "recursive-500": [other_id, shared_id],
        })
        self.assertEqual(len(cases[0]["candidates"]), 2)
        self.assertEqual(stores["semantic"].calls, [("question", 2)])

    def test_build_pool_embeds_each_question_once_across_strategies(self):
        chunk_ids = [
            create_chunk_id(
                f"https://example.com/{strategy}",
                strategy,
                strategy,
            )
            for strategy in ("semantic", "recursive-500")
        ]
        stores = {
            strategy: FakeVectorStore([
                Document(
                    page_content=strategy,
                    metadata={"chunk_id": chunk_id, "strategy": strategy},
                )
            ])
            for strategy, chunk_id in zip(
                ("semantic", "recursive-500"),
                chunk_ids,
                strict=True,
            )
        }
        embedding_calls = []

        build_pool(
            [{
                "question": "question",
                "answer": "answer",
                "source_url": "https://example.com",
                "title": "Example",
            }],
            stores,
            pool_depth=1,
            embed_query=lambda question: embedding_calls.append(question) or [0.5],
        )

        self.assertEqual(embedding_calls, ["question"])
        self.assertEqual(stores["semantic"].calls, [([0.5], 1)])
        self.assertEqual(stores["recursive-500"].calls, [([0.5], 1)])

    def test_qrels_template_is_unlabeled_and_fingerprints_pool(self):
        pool = {
            "schema_version": POOL_SCHEMA_VERSION,
            "cases": [{
                "case_id": "case-1",
                "question": "question",
                "candidates": [{"chunk_id": "chunk-1"}],
            }],
        }

        with TemporaryDirectory() as directory:
            pool_path = Path(directory) / "pool.json"
            pool_path.write_text(json.dumps(pool), encoding="utf-8")
            qrels = initialize_qrels(pool_path, pool)

        self.assertEqual(qrels["schema_version"], QRELS_SCHEMA_VERSION)
        self.assertEqual(qrels["labeling"]["review_status"], "not_started")
        self.assertIsNone(qrels["cases"][0]["judgments"][0]["relevant"])
        self.assertEqual(len(qrels["source_pool"]["sha256"]), 64)


class RankedRetrievalScoringTests(unittest.TestCase):
    def setUp(self):
        self.pool = {
            "schema_version": POOL_SCHEMA_VERSION,
            "config": {
                "chunking_strategies": ["semantic", "recursive-500"],
                "pool_depth": 4,
            },
            "cases": [{
                "case_id": "case-1",
                "rankings": {
                    "semantic": ["a", "b", "c", "d"],
                    "recursive-500": ["e", "f", "g", "h"],
                },
                "candidates": [
                    {"chunk_id": chunk_id}
                    for chunk_id in ("a", "b", "c", "d", "e", "f", "g", "h")
                ],
            }],
        }
        self.qrels = {
            "schema_version": QRELS_SCHEMA_VERSION,
            "labeling": {
                "label_source": "test",
                "quality_tier": "test",
                "review_status": "approved_for_project_evaluation",
                "reviewed_by": "test suite",
                "reviewed_at": "2026-08-27",
                "human_verified": False,
                "method": "fixed test fixture",
            },
            "cases": [{
                "case_id": "case-1",
                "reference_claims": [
                    {"claim_id": "c1", "claim": "first"},
                    {"claim_id": "c2", "claim": "second"},
                ],
                "judgments": [
                    {"chunk_id": "a", "relevant": True, "supported_claim_ids": ["c1"]},
                    {"chunk_id": "b", "relevant": False, "supported_claim_ids": []},
                    {"chunk_id": "c", "relevant": True, "supported_claim_ids": ["c2"]},
                    {"chunk_id": "d", "relevant": True, "supported_claim_ids": ["c1"]},
                    {"chunk_id": "e", "relevant": False, "supported_claim_ids": []},
                    {"chunk_id": "f", "relevant": True, "supported_claim_ids": ["c2"]},
                    {"chunk_id": "g", "relevant": False, "supported_claim_ids": []},
                    {"chunk_id": "h", "relevant": False, "supported_claim_ids": []},
                ],
            }],
        }

    def test_scores_and_aggregates_each_strategy(self):
        result = score_pool(self.pool, self.qrels, [1, 3])

        semantic = result["cases"][0]["strategies"]["semantic"]["3"]
        self.assertTrue(semantic["hit_at_k"])
        self.assertAlmostEqual(semantic["precision_at_k"], 2 / 3)
        self.assertAlmostEqual(semantic["pooled_recall_at_k"], 2 / 3)
        self.assertEqual(semantic["reciprocal_rank_at_k"], 1.0)
        self.assertEqual(semantic["claim_coverage_at_k"], 1.0)
        self.assertEqual(
            result["summary"]["semantic"]["3"]["precision_at_k"],
            semantic["precision_at_k"],
        )

        recursive = result["cases"][0]["strategies"]["recursive-500"]["3"]
        self.assertEqual(recursive["reciprocal_rank_at_k"], 0.5)
        self.assertEqual(recursive["claim_coverage_at_k"], 0.5)

    def test_rejects_incomplete_qrels(self):
        self.qrels["cases"][0]["judgments"].pop()

        with self.assertRaisesRegex(ValueError, "missing 1 candidate judgments"):
            score_pool(self.pool, self.qrels, [3])

    def test_rejects_relevance_without_claim_support(self):
        self.qrels["cases"][0]["judgments"][0]["supported_claim_ids"] = []

        with self.assertRaisesRegex(ValueError, "supports at least one"):
            score_pool(self.pool, self.qrels, [3])

    def test_rejects_cutoff_above_pool_depth(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            score_pool(self.pool, self.qrels, [5])

    def test_rejects_unapproved_qrels(self):
        self.qrels["labeling"]["review_status"] = "in_review"

        with self.assertRaisesRegex(ValueError, "approved_for_project_evaluation"):
            score_pool(self.pool, self.qrels, [3])

    def test_rejects_pool_candidates_that_do_not_match_rankings(self):
        self.pool["cases"][0]["candidates"] = [{"chunk_id": "a"}]

        with self.assertRaisesRegex(ValueError, "do not match rankings"):
            score_pool(self.pool, self.qrels, [3])


if __name__ == "__main__":
    unittest.main()
