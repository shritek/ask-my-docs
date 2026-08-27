from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from evaluation.run_ragas_evaluation import (
    DEFAULT_EVALUATOR_LLM_MODEL,
    DEFAULT_EVALUATOR_REASONING_EFFORT,
    RagasEvaluationConfig,
    RagasScorerSuite,
    default_output_path,
    evaluator_llm_options,
    evaluate_ragas,
    load_source_run,
    select_source_cases,
)


TEST_METRICS = ("faithfulness", "context_recall")


def make_source_case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "reference": {"answer": f"reference for {case_id}"},
        "rag_result": {
            "question": f"question for {case_id}",
            "answer": f"answer for {case_id}",
            "contexts": [f"context for {case_id}"],
        },
    }


class FakeScorerSuite:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def score(self, metric_name, source_case):
        key = (source_case["case_id"], metric_name)
        self.calls.append(key)
        response = self.responses.get(key, {"value": 0.75})
        if isinstance(response, Exception):
            raise response
        return dict(response)


class RecordingMetric:
    def __init__(self):
        self.calls = []

    def score(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=0.5, reason=None)


class RagasEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.source_cases = [make_source_case("case-1"), make_source_case("case-2")]
        self.config = RagasEvaluationConfig(
            source_result_path="evaluation/results/source.json",
            source_result_sha256="source-hash",
            evaluator_llm_model="llama3.1:8b",
            evaluator_embedding_model="nomic-embed-text",
            evaluator_max_tokens=4096,
            ollama_base_url="http://localhost:11434/v1",
            answer_relevancy_strictness=3,
            metric_names=TEST_METRICS,
            ragas_version="0.4.3",
            case_limit=None,
        )

    def test_scorer_maps_each_metric_to_the_required_ragas_inputs(self):
        suite = RagasScorerSuite.__new__(RagasScorerSuite)
        suite.scorers = {
            metric_name: RecordingMetric()
            for metric_name in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            )
        }
        source_case = self.source_cases[0]

        for metric_name in suite.scorers:
            suite.score(metric_name, source_case)

        rag_result = source_case["rag_result"]
        common_inputs = {
            "user_input": rag_result["question"],
            "retrieved_contexts": rag_result["contexts"],
        }
        self.assertEqual(
            suite.scorers["faithfulness"].calls,
            [{**common_inputs, "response": rag_result["answer"]}],
        )
        self.assertEqual(
            suite.scorers["answer_relevancy"].calls,
            [{
                "user_input": rag_result["question"],
                "response": rag_result["answer"],
            }],
        )
        expected_reference_inputs = {
            **common_inputs,
            "reference": source_case["reference"]["answer"],
        }
        self.assertEqual(
            suite.scorers["context_precision"].calls,
            [expected_reference_inputs],
        )
        self.assertEqual(
            suite.scorers["context_recall"].calls,
            [expected_reference_inputs],
        )

    def test_scores_persist_summary_and_resume_at_metric_granularity(self):
        scorer = FakeScorerSuite({
            ("case-1", "faithfulness"): {"value": 1.0, "reason": "supported"},
            ("case-1", "context_recall"): {"value": 0.5},
            ("case-2", "faithfulness"): {"value": 0.25},
            ("case-2", "context_recall"): {"value": 0.75},
        })

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "ragas.json"
            run_data = evaluate_ragas(
                scorer,
                self.source_cases,
                self.config,
                output_path,
            )

            self.assertEqual(run_data["status"], "completed")
            self.assertEqual(run_data["summary"]["completed_cases"], 2)
            self.assertEqual(
                run_data["summary"]["metrics"]["faithfulness"]["mean"],
                0.625,
            )
            self.assertEqual(
                run_data["summary"]["metrics"]["context_recall"]["mean"],
                0.625,
            )
            self.assertEqual(len(scorer.calls), 4)
            self.assertFalse(output_path.with_suffix(".json.tmp").exists())

            persisted = json.loads(output_path.read_text())
            self.assertEqual(persisted["summary"], run_data["summary"])
            self.assertEqual(
                persisted["results"][0]["metrics"]["faithfulness"]["reason"],
                "supported",
            )

            resumed_scorer = FakeScorerSuite()
            resumed = evaluate_ragas(
                resumed_scorer,
                self.source_cases,
                self.config,
                output_path,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed_scorer.calls, [])

    def test_failed_metric_is_recorded_and_only_that_metric_is_retried(self):
        limited_config = RagasEvaluationConfig(
            **{**self.config.to_dict(), "metric_names": TEST_METRICS, "case_limit": 1}
        )
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "ragas.json"
            failed_scorer = FakeScorerSuite({
                ("case-1", "faithfulness"): RuntimeError("judge unavailable"),
            })
            failed = evaluate_ragas(
                failed_scorer,
                self.source_cases,
                limited_config,
                output_path,
                limit=1,
            )

            self.assertEqual(failed["status"], "completed_with_errors")
            self.assertEqual(len(failed["errors"]), 1)
            self.assertNotIn(
                "faithfulness",
                failed["results"][0]["metrics"],
            )
            self.assertIn("context_recall", failed["results"][0]["metrics"])

            resumed_scorer = FakeScorerSuite({
                ("case-1", "faithfulness"): {"value": 1.0},
            })
            resumed = evaluate_ragas(
                resumed_scorer,
                self.source_cases,
                limited_config,
                output_path,
                limit=1,
            )

            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["errors"], [])
            self.assertEqual(
                resumed_scorer.calls,
                [("case-1", "faithfulness")],
            )

    def test_resume_rejects_configuration_mismatch(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "ragas.json"
            evaluate_ragas(
                FakeScorerSuite(),
                self.source_cases,
                self.config,
                output_path,
            )
            changed_config = RagasEvaluationConfig(
                **{
                    **self.config.to_dict(),
                    "metric_names": TEST_METRICS,
                    "evaluator_llm_model": "llama3.2:3b",
                }
            )

            with self.assertRaisesRegex(ValueError, "configuration"):
                evaluate_ragas(
                    FakeScorerSuite(),
                    self.source_cases,
                    changed_config,
                    output_path,
                )

    def test_explicit_case_selection_preserves_requested_order(self):
        selected_ids = ("case-2", "case-1")
        selected_config = replace(self.config, case_ids=selected_ids)
        scorer = FakeScorerSuite()

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "ragas.json"
            run_data = evaluate_ragas(
                scorer,
                self.source_cases,
                selected_config,
                output_path,
                case_ids=selected_ids,
            )

        self.assertEqual(
            [result["case_id"] for result in run_data["results"]],
            ["case-2", "case-1"],
        )
        self.assertEqual(run_data["summary"]["total_cases"], 2)
        self.assertEqual(scorer.calls[0], ("case-2", "faithfulness"))
        self.assertEqual(
            selected_config.to_dict()["case_ids"],
            ["case-2", "case-1"],
        )

    def test_explicit_case_selection_rejects_missing_ids(self):
        with self.assertRaisesRegex(ValueError, "missing selected case IDs"):
            select_source_cases(
                self.source_cases,
                case_ids=("missing-case",),
            )

    def test_load_source_run_requires_completed_valid_results(self):
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(json.dumps({
                "status": "running",
                "errors": [],
                "results": self.source_cases,
            }))
            with self.assertRaisesRegex(ValueError, "completed"):
                load_source_run(source_path)

            source_path.write_text(json.dumps({
                "status": "completed",
                "errors": [],
                "results": self.source_cases,
            }))
            self.assertEqual(load_source_run(source_path)["results"], self.source_cases)

    def test_default_output_path_contains_source_and_evaluator_identity(self):
        output_path = default_output_path(
            Path("evaluation/results/basic__semantic.json"),
            "llama3.1:8b",
            "nomic-embed-text",
            4096,
        )

        self.assertEqual(
            output_path,
            Path(
                "evaluation/results/"
                "ragas__basic__semantic__judge-llama3.1-8b__"
                "emb-nomic-embed-text__max-tokens-4096.json"
            ),
        )

    def test_default_output_path_records_calibration_selection(self):
        output_path = default_output_path(
            Path("evaluation/results/basic__semantic.json"),
            "gemma4:31b-mlx",
            "nomic-embed-text",
            4096,
            "abcdef1234567890",
            13,
        )

        self.assertEqual(
            output_path,
            Path(
                "evaluation/results/"
                "ragas__basic__semantic__judge-gemma4-31b-mlx__"
                "emb-nomic-embed-text__max-tokens-4096__"
                "cases-13-abcdef12.json"
            ),
        )

    def test_reasoning_effort_is_explicit_model_configuration(self):
        self.assertEqual(DEFAULT_EVALUATOR_LLM_MODEL, "gemma4:31b-mlx")
        self.assertEqual(DEFAULT_EVALUATOR_REASONING_EFFORT, "none")
        self.assertEqual(
            evaluator_llm_options(4096, "none"),
            {"max_tokens": 4096, "reasoning_effort": "none"},
        )
        self.assertEqual(
            evaluator_llm_options(4096, None),
            {"max_tokens": 4096},
        )

        config = replace(self.config, evaluator_reasoning_effort="none")
        self.assertEqual(config.to_dict()["evaluator_reasoning_effort"], "none")

        output_path = default_output_path(
            Path("evaluation/results/basic.json"),
            "qwen3.6:27b-mlx",
            "nomic-embed-text",
            4096,
            evaluator_reasoning_effort="none",
        )
        self.assertIn("reasoning-none", output_path.name)


if __name__ == "__main__":
    unittest.main()
