from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from basic_rag.basic_rag import answer_query
from evaluation.run_basic_evaluation import EvaluationConfig, default_output_path, initialize_run, write_run_file
from helpers.chunk_ids import create_chunk_id
from hybrid_retrieval.hybrid_rag import (
    HybridConfig, HybridRAGPipeline, HybridRetriever, LexicalRetriever, fuse_rankings, tokenize,
)


def document(text, source="https://example.com/docs"):
    return Document(page_content=text, metadata={
        "chunk_id": create_chunk_id(source, "recursive-1000", text),
        "strategy": "recursive-1000", "source_url": source,
    })


class LexicalTests(unittest.TestCase):
    def test_preprocessing_matches_code_identifiers_and_punctuation(self):
        self.assertEqual(tokenize('`@app.on_event("STARTUP")`'), ["app", "on_event", "startup"])
        docs = [document("on_event handles startup"), document("other unrelated text"), document("different content")]
        lexical = LexicalRetriever(docs)
        self.assertEqual(lexical.retrieve("ON_EVENT()", 3), [docs[0]])

    def test_unmatched_query_does_not_return_arbitrary_zero_score_documents(self):
        lexical = LexicalRetriever([document("alpha"), document("beta")])
        self.assertEqual(lexical.retrieve("unknown", 3), [])
        self.assertEqual(lexical.retrieve("!!!", 3), [])

    def test_tied_scores_are_independent_of_corpus_order(self):
        docs = [document("shared", "https://example.com/a"), document("shared", "https://example.com/b")]
        self.assertEqual(
            LexicalRetriever(docs).retrieve("shared", 2),
            LexicalRetriever(list(reversed(docs))).retrieve("shared", 2),
        )

    def test_empty_corpus_has_clear_error(self):
        for docs in ([], [document("!!!")]):
            with self.assertRaisesRegex(ValueError, "searchable tokens"):
                LexicalRetriever(docs)


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c = [document(t) for t in ("alpha", "beta", "gamma")]

    def test_shared_evidence_gets_both_rank_votes_and_final_k_is_enforced(self):
        result = fuse_rankings([self.a, self.b], [self.c, self.b], HybridConfig(), 1)
        self.assertEqual(result, [self.b])

    def test_duplicates_in_one_arm_do_not_receive_extra_votes(self):
        self.assertEqual(
            fuse_rankings([self.a, self.a, self.b], [self.b], HybridConfig(), 2),
            fuse_rankings([self.a, self.b], [self.b], HybridConfig(), 2),
        )

    def test_zero_weight_arm_contributes_no_documents(self):
        self.assertEqual(fuse_rankings([self.a], [self.b], HybridConfig(vector_weight=1), 3), [self.a])
        self.assertEqual(fuse_rankings([self.a], [self.b], HybridConfig(vector_weight=0), 3), [self.b])

    def test_unequal_weights_can_outweigh_agreement_at_lower_ranks(self):
        # a: .99/61 exceeds b: .99/63 + .01/61, despite b appearing in both arms.
        self.assertEqual(
            fuse_rankings([self.a, self.c, self.b], [self.b], HybridConfig(vector_weight=0.99), 1),
            [self.a],
        )

    def test_ties_use_stable_chunk_ids(self):
        expected = sorted([self.a, self.b], key=lambda d: d.metadata["chunk_id"])
        self.assertEqual(fuse_rankings([self.a], [self.b], HybridConfig(), 2), expected)

    def test_invalid_configuration_rejected(self):
        for options in ({"candidate_k": 0}, {"rrf_c": 0}, {"vector_weight": float("nan")}, {"vector_weight": 1.1}):
            with self.assertRaises(ValueError):
                HybridConfig(**options)

    def test_conflicting_chunk_identity_rejected(self):
        changed = self.a.model_copy(update={"page_content": "changed"})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            fuse_rankings([self.a], [changed], HybridConfig(), 3)


class HybridPipelineTests(unittest.TestCase):
    def test_shared_generation_path_receives_fused_context_once(self):
        a, b = document("alpha"), document("beta")
        store = Mock()
        store.similarity_search.return_value = [a, b]
        llm = Mock()
        llm.invoke.return_value = AIMessage(content="A supported answer")
        retriever = HybridRetriever(store, [a, b], HybridConfig(candidate_k=10))
        result = answer_query(llm, retriever, "beta", top_k=1)
        self.assertEqual(result.contexts, ["beta"])
        self.assertEqual(result.sources, [b.metadata])
        store.similarity_search.assert_called_once_with("beta", k=10)
        llm.invoke.assert_called_once()
        self.assertNotIn("rrf", llm.invoke.call_args.args[0].to_string())

    def test_stale_vector_content_and_unknown_ids_fail_before_generation(self):
        corpus_doc = document("alpha")
        for returned in (document("unknown"), corpus_doc.model_copy(update={"page_content": "changed"})):
            store = Mock()
            store.similarity_search.return_value = [returned]
            llm = Mock()
            with self.assertRaisesRegex(ValueError, "rebuild"):
                answer_query(llm, HybridRetriever(store, [corpus_doc], HybridConfig()), "alpha")
            llm.invoke.assert_not_called()

    def test_no_lexical_matches_preserves_vector_ranking(self):
        docs = [document("alpha"), document("beta")]
        store = Mock()
        store.similarity_search.return_value = docs
        retriever = HybridRetriever(store, docs, HybridConfig())
        self.assertEqual(retriever.similarity_search("unknown", k=2), docs)

    @patch("hybrid_retrieval.hybrid_rag.get_embedder")
    def test_invalid_top_k_rejected_before_model_initialization(self, embedder):
        with self.assertRaisesRegex(ValueError, "candidate_k"):
            HybridRAGPipeline(top_k=11)
        embedder.assert_not_called()

    def test_duplicate_corpus_ids_rejected(self):
        doc = document("alpha")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            HybridRetriever(Mock(), [doc, doc], HybridConfig())


class HybridEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.basic = EvaluationConfig(
            "basic", "nomic", "recursive-1000", "llama3.1:8b", 3,
            "evaluation/test_set.json", "test-sha", "corpus.json", "corpus-sha",
        )

    def test_basic_output_and_resume_config_remain_compatible(self):
        self.assertNotIn("retrieval_config", self.basic.to_dict())
        self.assertEqual(default_output_path(self.basic).name, "basic__recursive-1000__nomic__llama3.1-8b__k3.json")

    def test_hybrid_settings_change_filename_and_block_incompatible_resume(self):
        config = replace(self.basic, variant="hybrid", retrieval_config=HybridConfig().to_dict())
        changed = replace(config, retrieval_config=HybridConfig(vector_weight=0.7).to_dict())
        self.assertNotEqual(default_output_path(config), default_output_path(changed))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            write_run_file(output, initialize_run(output, config, resume=True))
            with self.assertRaisesRegex(ValueError, "configuration"):
                initialize_run(output, changed, resume=True)


if __name__ == "__main__":
    unittest.main()
