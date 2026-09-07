import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.propose_retrieval_qrels import (
    LABELING_RUBRIC_VERSION,
    PROPOSAL_PROTOCOL_VERSION,
    ProposalConfig,
    candidate_prompt_records,
    parse_json_object,
    propose_qrels,
    validate_proposal,
)
from evaluation.ranked_retrieval import POOL_SCHEMA_VERSION


def make_pool_case(case_id="case-1"):
    return {
        "case_id": case_id,
        "question": "How does the feature work?",
        "reference": {"answer": "It starts once. It stops once."},
        "rankings": {
            "first": ["chunk-a"],
            "second": ["chunk-b"],
        },
        "candidates": [
            {
                "chunk_id": "chunk-a",
                "text": "The feature starts once.",
                "source": {"strategy": "first", "source_url": "secret-a"},
            },
            {
                "chunk_id": "chunk-b",
                "text": "Unrelated material.",
                "source": {"strategy": "second", "source_url": "secret-b"},
            },
        ],
    }


def valid_proposal():
    return {
        "reference_claims": [
            {"claim_id": "c1", "claim": "The feature starts once."},
            {"claim_id": "c2", "claim": "The feature stops once."},
        ],
        "judgments": [
            {
                "chunk_id": "chunk-a",
                "relevant": True,
                "supported_claim_ids": ["c1"],
                "notes": "Direct support.",
            },
            {
                "chunk_id": "chunk-b",
                "relevant": False,
                "supported_claim_ids": [],
                "notes": "No support.",
            },
        ],
    }


def make_config():
    return ProposalConfig(
        source_pool_path="pool.json",
        source_pool_sha256="pool-sha",
        evaluator_model="judge",
        evaluator_reasoning_effort="none",
        evaluator_max_tokens=1024,
        ollama_base_url="http://localhost:11434/v1",
        labeling_rubric_version=LABELING_RUBRIC_VERSION,
        proposal_protocol_version=PROPOSAL_PROTOCOL_VERSION,
    )


class ProposalValidationTests(unittest.TestCase):
    def test_parses_plain_and_markdown_fenced_json(self):
        self.assertEqual(parse_json_object('{"ok": true}'), {"ok": True})
        self.assertEqual(
            parse_json_object('```json\n{"ok": true}\n```'),
            {"ok": True},
        )

    def test_prompt_candidates_hide_strategy_rank_and_source(self):
        pool_case = make_pool_case()

        first, first_aliases = candidate_prompt_records(pool_case)
        second, second_aliases = candidate_prompt_records(pool_case)

        self.assertEqual(first, second)
        self.assertEqual(first_aliases, second_aliases)
        self.assertEqual(set(first_aliases.values()), {"chunk-a", "chunk-b"})
        self.assertEqual(
            {record["candidate_id"] for record in first},
            set(first_aliases),
        )
        self.assertNotIn("strategy", json.dumps(first))
        self.assertNotIn("source_url", json.dumps(first))
        self.assertNotIn("chunk-a", json.dumps(first))

    def test_alias_protocol_maps_short_ids_back_to_stable_chunk_ids(self):
        pool_case = make_pool_case()
        _, aliases = candidate_prompt_records(pool_case)
        alias_by_chunk = {chunk_id: alias for alias, chunk_id in aliases.items()}
        proposal = valid_proposal()
        for judgment in proposal["judgments"]:
            judgment["candidate_id"] = alias_by_chunk[judgment.pop("chunk_id")]

        result = validate_proposal(
            pool_case,
            proposal,
            candidate_aliases=aliases,
            protocol_version=PROPOSAL_PROTOCOL_VERSION,
        )

        self.assertEqual(
            [judgment["chunk_id"] for judgment in result["judgments"]],
            ["chunk-a", "chunk-b"],
        )
        self.assertEqual(
            result["proposal_metadata"]["protocol_version"],
            PROPOSAL_PROTOCOL_VERSION,
        )

    def test_validates_and_restores_candidate_artifact_order(self):
        proposal = valid_proposal()
        proposal["judgments"].reverse()

        result = validate_proposal(make_pool_case(), proposal)

        self.assertEqual(
            [judgment["chunk_id"] for judgment in result["judgments"]],
            ["chunk-a", "chunk-b"],
        )
        self.assertEqual(result["reference_claims"][1]["claim_id"], "c2")

    def test_rejects_missing_candidate_judgment(self):
        proposal = valid_proposal()
        proposal["judgments"].pop()

        with self.assertRaisesRegex(ValueError, "missing 1"):
            validate_proposal(make_pool_case(), proposal)

    def test_rejects_relevance_without_supported_claim(self):
        proposal = valid_proposal()
        proposal["judgments"][0]["supported_claim_ids"] = []

        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate_proposal(make_pool_case(), proposal)


class FakeProposer:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.case_ids = []

    def propose(self, pool_case):
        self.case_ids.append(pool_case["case_id"])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        result = validate_proposal(pool_case, outcome)
        return result


class ProposalRunnerTests(unittest.TestCase):
    def setUp(self):
        second_case = make_pool_case("case-2")
        self.pool = {
            "schema_version": POOL_SCHEMA_VERSION,
            "cases": [make_pool_case(), second_case],
        }

    def test_persists_progress_and_resumes_only_failed_cases(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "qrels.json"
            first_proposer = FakeProposer([
                valid_proposal(),
                RuntimeError("temporary failure"),
            ])
            first = propose_qrels(
                self.pool,
                first_proposer,
                make_config(),
                output,
            )

            self.assertEqual(first["status"], "generated_with_errors")
            self.assertEqual(len(first["cases"]), 1)
            self.assertEqual(len(first["errors"]), 1)
            self.assertEqual(
                first["labeling"]["review_status"],
                "generated_pending_review",
            )

            resumed_proposer = FakeProposer([valid_proposal()])
            resumed = propose_qrels(
                self.pool,
                resumed_proposer,
                make_config(),
                output,
            )

            self.assertEqual(resumed["status"], "generated_pending_review")
            self.assertEqual(resumed_proposer.case_ids, ["case-2"])
            self.assertEqual(len(resumed["cases"]), 2)
            self.assertEqual(resumed["errors"], [])

    def test_resume_rejects_configuration_mismatch(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "qrels.json"
            propose_qrels(
                self.pool,
                FakeProposer([valid_proposal()]),
                make_config(),
                output,
                limit=1,
            )
            changed = ProposalConfig(
                **{**make_config().to_dict(), "evaluator_model": "other-judge"}
            )

            with self.assertRaisesRegex(ValueError, "configuration"):
                propose_qrels(
                    self.pool,
                    FakeProposer([]),
                    changed,
                    output,
                )

    def test_resume_migrates_valid_protocol_one_cases(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "qrels.json"
            original = propose_qrels(
                self.pool,
                FakeProposer([valid_proposal()]),
                make_config(),
                output,
                limit=1,
            )
            original["generation_config"].pop("proposal_protocol_version")
            original["cases"][0].pop("proposal_metadata")
            original.pop("proposal_protocols")
            output.write_text(json.dumps(original), encoding="utf-8")

            resumed = propose_qrels(
                self.pool,
                FakeProposer([]),
                make_config(),
                output,
                limit=1,
            )

            self.assertEqual(
                resumed["cases"][0]["proposal_metadata"]["protocol_version"],
                1,
            )
            self.assertEqual(
                resumed["generation_config"]["proposal_protocol_version"],
                PROPOSAL_PROTOCOL_VERSION,
            )
            self.assertIn("2", resumed["proposal_protocols"])


if __name__ == "__main__":
    unittest.main()
