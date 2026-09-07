from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Callable

from basic_rag.basic_rag import load_or_build_vector_store, retrieve_documents
from config.settings import (
    CHUNKED_CORPUS_PATH,
    CHUNKING_STRATEGY_MAPPING,
    DEFAULT_EMBEDDING,
    EmbeddingModel,
)
from evaluation.ranked_retrieval import (
    POOL_SCHEMA_VERSION,
    atomic_write_json,
    sha256_file,
)
from evaluation.run_basic_evaluation import create_case_id, load_test_set
from helpers.chunk_ids import is_stable_chunk_id
from helpers.embedding_factory import get_embedder


DEFAULT_TEST_SET = Path("evaluation/test_set.json")
DEFAULT_OUTPUT_DIR = Path("evaluation/retrieval_pools")

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_pool(
    test_cases: list[dict[str, Any]],
    vector_stores: dict[str, Any],
    pool_depth: int,
    embed_query: Callable[[str], list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve and deduplicate candidate chunks for every test case."""
    if pool_depth < 1:
        raise ValueError("pool_depth must be at least 1")

    pooled_cases: list[dict[str, Any]] = []
    for index, test_case in enumerate(test_cases, start=1):
        logger.info(
            "[%s/%s] Retrieving: %s",
            index,
            len(test_cases),
            test_case["question"],
        )
        rankings: dict[str, list[str]] = {}
        candidates: dict[str, dict[str, Any]] = {}
        query_embedding = (
            embed_query(test_case["question"])
            if embed_query is not None
            else None
        )
        for strategy, vector_store in vector_stores.items():
            documents = (
                vector_store.similarity_search_by_vector(
                    query_embedding,
                    k=pool_depth,
                )
                if query_embedding is not None
                else retrieve_documents(
                    vector_store,
                    test_case["question"],
                    pool_depth,
                )
            )
            ranking: list[str] = []
            for document in documents:
                chunk_id = document.metadata.get("chunk_id")
                if not is_stable_chunk_id(chunk_id):
                    raise ValueError(
                        f"{strategy} returned a legacy or missing chunk ID"
                    )
                if chunk_id in ranking:
                    raise ValueError(
                        f"{strategy} returned duplicate chunk {chunk_id}"
                    )
                ranking.append(chunk_id)
                candidates.setdefault(chunk_id, {
                    "chunk_id": chunk_id,
                    "text": document.page_content,
                    "source": dict(document.metadata),
                })
            rankings[strategy] = ranking

        pooled_cases.append({
            "case_id": create_case_id(test_case),
            "question": test_case["question"],
            "reference": {
                "answer": test_case["answer"],
                "source_url": test_case["source_url"],
                "title": test_case["title"],
            },
            "rankings": rankings,
            "candidates": list(candidates.values()),
        })
    return pooled_cases


def default_output_path(embedding_model: str, pool_depth: int) -> Path:
    filename = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "-",
        f"all-strategies__{embedding_model}__pool-k{pool_depth}",
    )
    return DEFAULT_OUTPUT_DIR / f"{filename}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a pooled retrieval artifact for chunk-level labeling"
    )
    parser.add_argument(
        "--chunking-strategy",
        action="append",
        dest="chunking_strategies",
        choices=list(CHUNKING_STRATEGY_MAPPING.keys()),
        help="Strategy to include; repeat the option. Defaults to all strategies.",
    )
    parser.add_argument(
        "--embedder",
        choices=[model.value for model in EmbeddingModel],
        default=DEFAULT_EMBEDDING.value,
    )
    parser.add_argument("--pool-depth", type=int, default=10)
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.pool_depth < 1:
        parser.error("--pool-depth must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    strategies = args.chunking_strategies or list(CHUNKING_STRATEGY_MAPPING.keys())
    strategies = list(dict.fromkeys(strategies))
    embedding_model = EmbeddingModel(args.embedder)
    test_cases = load_test_set(args.test_set)
    if args.limit is not None:
        test_cases = test_cases[:args.limit]

    embedder = get_embedder(embedding_model)
    vector_stores = {
        strategy: load_or_build_vector_store(
            embedding_model,
            strategy,
            embedder,
        )
        for strategy in strategies
    }
    corpus_inputs = {
        strategy: {
            "path": str(Path(CHUNKED_CORPUS_PATH(strategy))),
            "sha256": sha256_file(Path(CHUNKED_CORPUS_PATH(strategy))),
        }
        for strategy in strategies
    }
    pool = {
        "schema_version": POOL_SCHEMA_VERSION,
        "created_at": utc_now(),
        "config": {
            "embedding_model": embedding_model.value,
            "chunking_strategies": strategies,
            "pool_depth": args.pool_depth,
            "test_set_path": str(args.test_set),
            "test_set_sha256": sha256_file(args.test_set),
            "corpora": corpus_inputs,
        },
        "cases": build_pool(
            test_cases,
            vector_stores,
            args.pool_depth,
            embed_query=embedder.embed_query,
        ),
    }
    output_path = args.output or default_output_path(
        embedding_model.value,
        args.pool_depth,
    )
    atomic_write_json(output_path, pool)
    print(f"Retrieval pool saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
