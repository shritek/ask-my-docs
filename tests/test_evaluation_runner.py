import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from basic_rag.basic_rag import RAGResult
from evaluation.run_basic_evaluation import (
    EvaluationConfig,
    compute_source_metrics,
    evaluate_test_set,
)


class FakePipeline:
    def __init__(self, results):
        self.results = iter(results)
        self.questions = []

    def query(self, question):
        self.questions.append(question)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def make_rag_result(question, source_urls):
    chunk_ids = [f"chunk_{index:064x}" for index in range(1, len(source_urls) + 1)]
    return RAGResult(
        question=question,
        answer=f"answer for {question}",
        contexts=[f"context {index}" for index in range(len(source_urls))],
        retrieved_chunk_ids=chunk_ids,
        sources=[
            {"chunk_id": chunk_id, "source_url": source_url}
            for chunk_id, source_url in zip(chunk_ids, source_urls)
        ],
        retrieval_latency_ms=10.0,
        generation_latency_ms=90.0,
        total_latency_ms=100.0,
    )


class SourceMetricTests(unittest.TestCase):
    def test_source_metrics_report_rank_and_reciprocal_rank(self):
        metrics = compute_source_metrics(
            "https://example.com/expected/",
            [
                {"source_url": "https://example.com/other/"},
                {"source_url": "https://example.com/expected"},
            ],
        )

        self.assertEqual(metrics["source_hit_at_k"], True)
        self.assertEqual(metrics["expected_source_rank"], 2)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)

    def test_source_metrics_report_miss(self):
        metrics = compute_source_metrics(
            "https://example.com/expected",
            [{"source_url": "https://example.com/other"}],
        )

        self.assertEqual(metrics, {
            "source_hit_at_k": False,
            "expected_source_rank": None,
            "reciprocal_rank": 0.0,
        })


class EvaluationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            {
                "question": "first question",
                "answer": "first reference",
                "source_url": "https://example.com/first/",
                "title": "First",
            },
            {
                "question": "second question",
                "answer": "second reference",
                "source_url": "https://example.com/second/",
                "title": "Second",
            },
        ]
        self.config = EvaluationConfig(
            variant="basic",
            embedding_model="nomic",
            chunking_strategy="semantic",
            llm_model="llama3.1:8b",
            top_k=3,
            test_set_path="evaluation/test_set.json",
            test_set_sha256="test-set-hash",
            corpus_path="data/chunked_corpus_semantic.json",
            corpus_sha256="corpus-hash",
        )

    def test_run_persists_results_summary_and_resumes(self):
        pipeline = FakePipeline([
            make_rag_result(
                "first question",
                ["https://example.com/other", "https://example.com/first"],
            ),
            make_rag_result(
                "second question",
                ["https://example.com/other"],
            ),
        ])

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "run.json"
            run_data = evaluate_test_set(
                pipeline,
                self.test_cases,
                self.config,
                output_path,
            )

            self.assertEqual(run_data["status"], "completed")
            self.assertEqual(run_data["summary"]["completed_questions"], 2)
            self.assertEqual(run_data["summary"]["source_hit_rate"], 0.5)
            self.assertEqual(run_data["summary"]["mean_reciprocal_rank"], 0.25)
            self.assertEqual(run_data["summary"]["mean_total_latency_ms"], 100.0)
            self.assertEqual(pipeline.questions, ["first question", "second question"])
            self.assertFalse(output_path.with_suffix(".json.tmp").exists())

            persisted = json.loads(output_path.read_text())
            self.assertEqual(persisted["summary"], run_data["summary"])
            self.assertEqual(len(persisted["results"]), 2)

            resumed_pipeline = FakePipeline([])
            resumed = evaluate_test_set(
                resumed_pipeline,
                self.test_cases,
                self.config,
                output_path,
            )
            self.assertEqual(resumed_pipeline.questions, [])
            self.assertEqual(len(resumed["results"]), 2)

    def test_resume_rejects_configuration_mismatch(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "run.json"
            evaluate_test_set(
                FakePipeline([make_rag_result("first question", [])]),
                self.test_cases,
                self.config,
                output_path,
                limit=1,
            )
            changed_config = EvaluationConfig(
                **{**self.config.to_dict(), "top_k": 5}
            )

            with self.assertRaisesRegex(ValueError, "configuration"):
                evaluate_test_set(
                    FakePipeline([]),
                    self.test_cases,
                    changed_config,
                    output_path,
                )

    def test_failed_case_is_persisted_and_retried_on_resume(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "run.json"
            failed = evaluate_test_set(
                FakePipeline([RuntimeError("temporary failure")]),
                self.test_cases,
                self.config,
                output_path,
                limit=1,
            )

            self.assertEqual(failed["status"], "completed_with_errors")
            self.assertEqual(len(failed["errors"]), 1)
            self.assertEqual(failed["results"], [])

            resumed = evaluate_test_set(
                FakePipeline([
                    make_rag_result(
                        "first question",
                        ["https://example.com/first"],
                    )
                ]),
                self.test_cases,
                self.config,
                output_path,
                limit=1,
            )

            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["errors"], [])
            self.assertEqual(len(resumed["results"]), 1)


if __name__ == "__main__":
    unittest.main()
