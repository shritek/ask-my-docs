import unittest

from basic_rag.basic_rag import get_vector_store_identity
from config.settings import EmbeddingModel


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


if __name__ == "__main__":
    unittest.main()
