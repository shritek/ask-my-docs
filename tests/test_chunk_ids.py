import unittest
from unittest.mock import patch

from basic_rag.basic_rag import build_vector_store, get_document_by_chunk_id
from data.chunker import chunk_documents
from helpers.chunk_ids import create_chunk_id, is_stable_chunk_id


class ChunkIdTests(unittest.TestCase):
    def test_chunk_id_is_stable(self):
        first = create_chunk_id("https://example.com/docs", "semantic", "content")
        second = create_chunk_id("https://example.com/docs", "semantic", "content")

        self.assertEqual(first, second)
        self.assertTrue(is_stable_chunk_id(first))

    def test_chunk_id_changes_when_identity_changes(self):
        baseline = create_chunk_id("https://example.com/docs", "semantic", "content")

        self.assertNotEqual(
            baseline,
            create_chunk_id("https://example.com/other", "semantic", "content"),
        )
        self.assertNotEqual(
            baseline,
            create_chunk_id("https://example.com/docs", "recursive-500", "content"),
        )
        self.assertNotEqual(
            baseline,
            create_chunk_id("https://example.com/docs", "semantic", "updated content"),
        )

    def test_chunker_includes_id_in_record_and_metadata(self):
        class FakeSplitter:
            def split_text(self, text):
                return [text]

        chunks = chunk_documents(
            [{
                "content": "A sufficiently long chunk for the corpus.",
                "source_url": "https://example.com/docs",
                "title": "Docs",
            }],
            FakeSplitter(),
            "recursive-500",
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], chunks[0]["metadata"]["chunk_id"])
        self.assertTrue(is_stable_chunk_id(chunks[0]["chunk_id"]))

    @patch("basic_rag.basic_rag.Chroma")
    def test_vector_store_uses_chunk_id_as_chroma_id(self, chroma):
        chunk_id = create_chunk_id(
            "https://example.com/docs",
            "recursive-500",
            "content",
        )
        store = chroma.return_value

        build_vector_store(
            [{
                "chunk_id": chunk_id,
                "text": "content",
                "metadata": {"source_url": "https://example.com/docs"},
            }],
            embedder=object(),
            collection_name="test_collection",
            persist_dir="test_path",
        )

        store.add_texts.assert_called_once_with(
            texts=["content"],
            embeddings=None,
            metadatas=[{
                "source_url": "https://example.com/docs",
                "chunk_id": chunk_id,
            }],
            ids=[chunk_id],
        )

    def test_vector_store_rejects_legacy_ids(self):
        with self.assertRaisesRegex(ValueError, "Regenerate"):
            build_vector_store(
                [{
                    "chunk_id": "id_42",
                    "text": "content",
                    "metadata": {},
                }],
                embedder=object(),
                collection_name="test_collection",
                persist_dir="test_path",
            )

    def test_document_can_be_retrieved_by_chunk_id(self):
        class FakeVectorStore:
            def __init__(self, documents):
                self.documents = documents
                self.calls = []

            def get_by_ids(self, ids):
                self.calls.append(ids)
                return self.documents

        chunk_id = create_chunk_id(
            "https://example.com/docs",
            "recursive-500",
            "content",
        )
        expected = object()
        vector_store = FakeVectorStore([expected])

        document = get_document_by_chunk_id(vector_store, chunk_id)

        self.assertIs(document, expected)
        self.assertEqual(vector_store.calls, [[chunk_id]])


if __name__ == "__main__":
    unittest.main()
