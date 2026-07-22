import json
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from basic_rag.basic_rag import answer_query, get_vector_store_identity
from config.settings import EmbeddingModel, LLMModel
from helpers.chunk_ids import create_chunk_id
from helpers.llm_factory import get_llm_model


class VectorStoreIdentityTests(unittest.TestCase):
    def test_identity_contains_chunking_strategy_and_embedding_model(self):
        collection_name, persist_dir = get_vector_store_identity(
            "recursive-500",
            EmbeddingModel.NOMIC,
        )

        self.assertEqual(
            collection_name,
            "ask_my_docs_recursive-500__nomic",
        )
        self.assertEqual(
            persist_dir,
            "data/vector_stores/recursive-500__nomic",
        )

    def test_different_embedders_use_different_indices(self):
        nomic_identity = get_vector_store_identity(
            "semantic",
            EmbeddingModel.NOMIC,
        )
        openai_identity = get_vector_store_identity(
            "semantic",
            EmbeddingModel.OPENAI,
        )

        self.assertNotEqual(nomic_identity, openai_identity)


class DeterministicRagTests(unittest.TestCase):
    @patch("helpers.llm_factory.init_chat_model")
    def test_llm_uses_zero_temperature(self, init_chat_model):
        get_llm_model(LLMModel.LLAMA3_8B)

        init_chat_model.assert_called_once_with(
            "llama3.1:8b",
            model_provider="ollama",
            temperature=0,
        )

    def test_retrieves_and_generates_exactly_once(self):
        chunk_id = create_chunk_id(
            "https://example.com/fastapi",
            "semantic",
            "FastAPI uses type hints for validation.",
        )

        class FakeVectorStore:
            def __init__(self):
                self.calls = []

            def similarity_search(self, query, k):
                self.calls.append((query, k))
                return [
                    Document(
                        page_content="FastAPI uses type hints for validation.",
                        metadata={
                            "chunk_id": chunk_id,
                            "source_url": "https://example.com/fastapi",
                        },
                    )
                ]

        class FakeLlm:
            def __init__(self):
                self.calls = []

            def invoke(self, messages):
                self.calls.append(messages)
                return AIMessage(content="A grounded answer")

        vector_store = FakeVectorStore()
        llm = FakeLlm()

        with patch(
            "basic_rag.basic_rag.perf_counter",
            side_effect=[10.0, 10.025, 20.0, 20.075],
        ):
            result = answer_query(
                llm,
                vector_store,
                "How does validation work?",
                top_k=5,
            )

        self.assertEqual(result.question, "How does validation work?")
        self.assertEqual(result.answer, "A grounded answer")
        self.assertEqual(
            result.contexts,
            ["FastAPI uses type hints for validation."],
        )
        self.assertEqual(
            result.retrieved_chunk_ids,
            [chunk_id],
        )
        self.assertEqual(
            result.sources,
            [{
                "chunk_id": chunk_id,
                "source_url": "https://example.com/fastapi",
            }],
        )
        self.assertAlmostEqual(result.retrieval_latency_ms, 25.0)
        self.assertAlmostEqual(result.generation_latency_ms, 75.0)
        self.assertAlmostEqual(result.total_latency_ms, 100.0)
        self.assertEqual(vector_store.calls, [("How does validation work?", 5)])
        self.assertEqual(len(llm.calls), 1)

        prompt_text = llm.calls[0].to_string()
        self.assertIn("How does validation work?", prompt_text)
        self.assertIn("FastAPI uses type hints for validation.", prompt_text)
        self.assertIn("https://example.com/fastapi", prompt_text)

        serialized = json.dumps(result.to_dict())
        self.assertIn('"answer": "A grounded answer"', serialized)
        self.assertIn(f'"retrieved_chunk_ids": ["{chunk_id}"]', serialized)

    def test_rejects_retrieved_documents_without_stable_ids(self):
        class FakeVectorStore:
            def similarity_search(self, query, k):
                return [Document(page_content="content", metadata={})]

        with self.assertRaisesRegex(ValueError, "missing chunk IDs"):
            answer_query(
                llm=object(),
                vector_store=FakeVectorStore(),
                query="question",
            )


if __name__ == "__main__":
    unittest.main()
