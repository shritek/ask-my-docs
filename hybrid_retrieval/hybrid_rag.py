"""BM25 + vector retrieval with a fixed, auditable rank-fusion baseline."""

import argparse
from dataclasses import asdict, dataclass
import math
import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from basic_rag.basic_rag import (
    RETRIEVAL_TOP_K, RAGResult, answer_query, load_chunks,
    load_or_build_vector_store,
)
from config.settings import (
    CHUNKED_CORPUS_PATH, CHUNKING_STRATEGY_MAPPING,
    DEFAULT_EMBEDDING, DEFAULT_LLM, EmbeddingModel, LLMModel,
)
from helpers.chunk_ids import create_chunk_id, is_stable_chunk_id
from helpers.embedding_factory import get_embedder
from helpers.llm_factory import get_llm_model


@dataclass(frozen=True)
class HybridConfig:
    candidate_k: int = 10
    rrf_c: int = 60
    vector_weight: float = 0.5
    # Changes to preprocessing or fixed BM25 parameters require a new version.
    algorithm_version: str = "bm25-okapi-rrf-v1"

    def __post_init__(self):
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        if self.rrf_c < 1:
            raise ValueError("rrf_c must be at least 1")
        if not math.isfinite(self.vector_weight) or not 0 <= self.vector_weight <= 1:
            raise ValueError("vector_weight must be finite and between 0 and 1")
        if self.algorithm_version != "bm25-okapi-rrf-v1":
            raise ValueError("Unsupported hybrid algorithm version")

    def to_dict(self) -> dict:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    """Case-fold words and Python identifiers; punctuation separates tokens."""
    return re.findall(r"\w+", text.casefold())


class LexicalRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = sorted(documents, key=lambda doc: doc.metadata["chunk_id"])
        tokenized = [tokenize(doc.page_content) for doc in self.documents]
        self.term_sets = [set(tokens) for tokens in tokenized]
        if not tokenized or not any(tokenized):
            raise ValueError("BM25 requires a corpus containing searchable tokens")
        self.index = BM25Okapi(tokenized, k1=1.5, b=0.75, epsilon=0.25)

    def retrieve(self, query: str, k: int) -> list[Document]:
        if k < 1:
            raise ValueError("k must be at least 1")
        tokens = tokenize(query)
        terms = set(tokens)
        # Do not let an all-zero, no-match list contribute arbitrary RRF votes.
        matched = [i for i, doc_terms in enumerate(self.term_sets) if terms & doc_terms]
        scores = self.index.get_scores(tokens) if matched else []
        ranked = sorted(
            matched,
            key=lambda i: (-float(scores[i]), self.documents[i].metadata["chunk_id"]),
        )
        return [self.documents[i] for i in ranked[:k]]


def fuse_rankings(
    vector: list[Document], lexical: list[Document], config: HybridConfig, k: int,
) -> list[Document]:
    """Weighted RRF: sum(weight / (c + rank)), deduplicated by chunk ID."""
    if k < 1:
        raise ValueError("k must be at least 1")
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking, weight in ((vector, config.vector_weight), (lexical, 1 - config.vector_weight)):
        if weight == 0:
            continue
        seen = set()
        unique_rank = 0
        for doc in ranking:
            chunk_id = doc.metadata.get("chunk_id")
            if not is_stable_chunk_id(chunk_id):
                raise ValueError("Fusion requires stable chunk IDs")
            if chunk_id in documents and documents[chunk_id] != doc:
                raise ValueError(f"Conflicting content or metadata for {chunk_id}")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique_rank += 1
            documents[chunk_id] = doc
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (config.rrf_c + unique_rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [documents[chunk_id] for chunk_id in ordered[:k]]


class HybridRetriever:
    """Adapter for the existing answer_query retrieval contract."""

    def __init__(self, vector_store, documents: list[Document], config: HybridConfig):
        self.vector_store = vector_store
        self.config = config
        self.by_id: dict[str, Document] = {}
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if not is_stable_chunk_id(chunk_id):
                raise ValueError("Hybrid corpus requires stable chunk IDs")
            if chunk_id in self.by_id:
                raise ValueError(f"Duplicate corpus chunk ID: {chunk_id}")
            self.by_id[chunk_id] = doc
        self.lexical = LexicalRetriever(documents)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        if not 1 <= k <= self.config.candidate_k:
            raise ValueError("top_k must be between 1 and candidate_k")
        vector = []
        if self.config.vector_weight > 0:
            for doc in self.vector_store.similarity_search(query, k=self.config.candidate_k):
                chunk_id = doc.metadata.get("chunk_id")
                canonical = self.by_id.get(chunk_id)
                if (canonical is None or canonical.page_content != doc.page_content
                        or canonical.metadata != doc.metadata):
                    raise ValueError("Vector index does not match the hybrid corpus; rebuild the index")
                # Preserve the exact same metadata in the generation prompt for both arms.
                vector.append(canonical)
        lexical = (
            self.lexical.retrieve(query, self.config.candidate_k)
            if self.config.vector_weight < 1 else []
        )
        return fuse_rankings(vector, lexical, self.config, k)


class HybridRAGPipeline:
    def __init__(
        self, embedding_model: EmbeddingModel = DEFAULT_EMBEDDING,
        chunking_strategy: str = "recursive-1000", llm_model: LLMModel = DEFAULT_LLM,
        top_k: int = RETRIEVAL_TOP_K, retrieval_config: HybridConfig | None = None,
    ):
        self.config = retrieval_config or HybridConfig()
        if not 1 <= top_k <= self.config.candidate_k:
            raise ValueError("top_k must be between 1 and candidate_k")
        if chunking_strategy not in CHUNKING_STRATEGY_MAPPING:
            raise ValueError("Unknown chunking strategy")
        self.top_k = top_k
        chunks = load_chunks(CHUNKED_CORPUS_PATH(chunking_strategy))
        documents = []
        for chunk in chunks:
            metadata = {**chunk["metadata"], "chunk_id": chunk["chunk_id"]}
            expected_id = create_chunk_id(metadata.get("source_url", ""), chunking_strategy, chunk["text"])
            if metadata.get("strategy") != chunking_strategy or chunk["chunk_id"] != expected_id:
                raise ValueError("Chunk identity does not match the configured corpus")
            documents.append(Document(page_content=chunk["text"], metadata=metadata))
        self.embedder = get_embedder(embedding_model)
        self.llm = get_llm_model(llm_model)
        self.vector_store = load_or_build_vector_store(embedding_model, chunking_strategy, self.embedder)
        self.retriever = HybridRetriever(self.vector_store, documents, self.config)

    def query(self, question: str) -> RAGResult:
        return answer_query(self.llm, self.retriever, question, self.top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 + vector Hybrid RAG")
    parser.add_argument("query")
    parser.add_argument("--chunking-strategy", choices=list(CHUNKING_STRATEGY_MAPPING), default="recursive-1000")
    parser.add_argument("--embedder", choices=[e.value for e in EmbeddingModel], default=DEFAULT_EMBEDDING.value)
    parser.add_argument("--llm", choices=[m.value for m in LLMModel], default=DEFAULT_LLM.value)
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--rrf-c", type=int, default=60)
    parser.add_argument("--vector-weight", type=float, default=0.5)
    args = parser.parse_args()
    pipeline = HybridRAGPipeline(
        EmbeddingModel(args.embedder), args.chunking_strategy, LLMModel(args.llm), args.top_k,
        HybridConfig(args.candidate_k, args.rrf_c, args.vector_weight),
    )
    result = pipeline.query(args.query)
    print(result.answer)
    print("\nRetrieved chunk IDs:", ", ".join(result.retrieved_chunk_ids))


if __name__ == "__main__":
    main()
