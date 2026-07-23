import argparse
from dataclasses import asdict, dataclass
import os
import json
import logging
from time import perf_counter
from typing import Any
from config.settings import EmbeddingModel, DEFAULT_EMBEDDING, DEFAULT_LLM, CHUNKED_CORPUS_PATH, CHUNKING_STRATEGY_MAPPING, LLMModel
from helpers.embedding_factory import get_embedder
from helpers.chunk_ids import is_stable_chunk_id
from helpers.llm_factory import get_llm_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs from httpx and ollama
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)

DATA_DIR = "data"
RETRIEVAL_TOP_K = 3

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You answer questions using only the supplied documentation context. "
        "If the context does not contain enough information, say you don't know. "
        "Do not use outside knowledge.",
    ),
    (
        "human",
        "Question:\n{question}\n\nDocumentation context:\n{context}",
    ),
])


@dataclass(frozen=True)
class RAGResult:
    """Evaluation-ready output from one RAG query."""

    question: str
    answer: str
    contexts: list[str]
    retrieved_chunk_ids: list[str]
    sources: list[dict[str, Any]]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the result."""
        return asdict(self)


def get_vector_store_identity(
    chunking_strategy: str,
    embedding_model: EmbeddingModel,
) -> tuple[str, str]:
    """Return an index name and path unique to a chunking/embedder pair."""
    index_key = f"{chunking_strategy}__{embedding_model.value}"
    collection_name = f"ask_my_docs_{index_key}"
    persist_dir = os.path.join(DATA_DIR, "vector_stores", index_key)
    return collection_name, persist_dir


def load_chunks(corpus_path: str) -> list[dict]:
    """Loads chunked corpus from disk."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"❌ Chunked corpus not found at {corpus_path}. Run chunker.py first.")
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def build_vector_store(chunks: list[dict], embedder, collection_name: str, persist_dir: str) -> Chroma:
    """Embeds chunks and stores them in Chroma using batching to prevent Ollama crashes."""
    invalid_ids = [
        chunk.get("chunk_id")
        for chunk in chunks
        if not is_stable_chunk_id(chunk.get("chunk_id"))
    ]
    if invalid_ids:
        raise ValueError(
            "Chunk corpus contains legacy or missing chunk IDs. "
            "Regenerate it with `python -m data.chunker` before building the vector index."
        )

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {**c["metadata"], "chunk_id": c["chunk_id"]}
        for c in chunks
    ]

    # Initialize an empty Chroma client with the embedder
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_dir
    )

    # Process in batches to avoid overloading Ollama (EOF 400 errors)
    batch_size = 50
    total = len(texts)
    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        vector_store.add_texts(
            texts=batch_texts,
            embeddings=None,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )
        logger.info(f"Indexed batch {i // batch_size + 1}/{(total // batch_size) + 1} ({i + len(batch_texts)}/{total})")

    logger.info(f"✅ Vector store built: {total} chunks indexed in {collection_name}")
    return vector_store


def retrieve_documents(vector_store, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Document]:
    """Retrieve a fixed number of documents exactly once for a query."""
    return vector_store.similarity_search(query, k=top_k)


def get_document_by_chunk_id(vector_store, chunk_id: str) -> Document | None:
    """Retrieve the exact stored chunk associated with a logical chunk ID."""
    if not is_stable_chunk_id(chunk_id):
        raise ValueError(f"Invalid stable chunk ID: {chunk_id}")

    documents = vector_store.get_by_ids([chunk_id])
    return documents[0] if documents else None


def format_documents(documents: list[Document]) -> str:
    """Serialize retrieved documents for the synthesis prompt."""
    return "\n\n".join(
        f"Source: {document.metadata}\nContent: {document.page_content}"
        for document in documents
    )


def answer_query(llm, vector_store, query: str, top_k: int = RETRIEVAL_TOP_K) -> RAGResult:
    """Run deterministic retrieval and generation with evaluation metadata."""
    retrieval_started = perf_counter()
    documents = retrieve_documents(vector_store, query, top_k)
    retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000

    retrieved_chunk_ids = [
        document.metadata.get("chunk_id")
        for document in documents
    ]
    if not all(is_stable_chunk_id(chunk_id) for chunk_id in retrieved_chunk_ids):
        raise ValueError(
            "Retrieved documents contain legacy or missing chunk IDs. "
            "Regenerate the chunk corpus and vector index."
        )

    messages = RAG_PROMPT.invoke({
        "question": query,
        "context": format_documents(documents),
    })

    generation_started = perf_counter()
    response = llm.invoke(messages)
    generation_latency_ms = (perf_counter() - generation_started) * 1000

    return RAGResult(
        question=query,
        answer=response.content,
        contexts=[document.page_content for document in documents],
        retrieved_chunk_ids=retrieved_chunk_ids,
        sources=[dict(document.metadata) for document in documents],
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        total_latency_ms=retrieval_latency_ms + generation_latency_ms,
    )


def run(
    embedding_model: EmbeddingModel,
    chunking_strategy: str,
    llm_model: LLMModel,
    query: str,
) -> RAGResult:
    """Run one query through the configured Basic RAG pipeline."""
    embedder = get_embedder(embedding_model)
    llm = get_llm_model(llm_model)
    corpus_path = CHUNKED_CORPUS_PATH(chunking_strategy)
    chunks = load_chunks(corpus_path)

    collection_name, persist_dir = get_vector_store_identity(
        chunking_strategy,
        embedding_model,
    )

    # Use the vector store if it exists, otherwise build it
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info(f"Loading existing vector store from {persist_dir}...")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=persist_dir
        )
    else:
        vector_store = build_vector_store(chunks, embedder, collection_name, persist_dir)

    return answer_query(llm, vector_store, query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic RAG pipeline")
    parser.add_argument(
        "query",
        help="The question you want to ask the RAG system"
    )
    parser.add_argument(
        "--chunking_strategy",
        choices=list(CHUNKING_STRATEGY_MAPPING.keys()),
        required=True,
        help="Chunking strategy to use for retrieval"
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default=DEFAULT_EMBEDDING.value,
        choices=[e.value for e in EmbeddingModel]
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=DEFAULT_LLM.value,
        choices=[e.value for e in LLMModel]
    )
    args = parser.parse_args()

    result = run(
        embedding_model=EmbeddingModel(args.embedder),
        chunking_strategy=args.chunking_strategy,
        llm_model=LLMModel(args.llm),
        query=args.query
    )

    logger.info(f"\n🔍 Query: {args.query.strip()}")
    print("=" * 60)
    print(result.answer)
    print("\n" + "=" * 60)
