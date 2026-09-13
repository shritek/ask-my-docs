import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
import re
from statistics import fmean
from typing import Any

from basic_rag.basic_rag import BasicRAGPipeline, RAGResult, RETRIEVAL_TOP_K
from config.settings import (
    CHUNKED_CORPUS_PATH,
    CHUNKING_STRATEGY_MAPPING,
    DEFAULT_EMBEDDING,
    DEFAULT_LLM,
    EmbeddingModel,
    LLMModel,
)


DEFAULT_TEST_SET = Path("evaluation/test_set.json")
DEFAULT_RESULTS_DIR = Path("evaluation/results")
RESULT_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration that must remain constant within one evaluation run."""

    variant: str
    embedding_model: str
    chunking_strategy: str
    llm_model: str
    top_k: int
    test_set_path: str
    test_set_sha256: str
    corpus_path: str
    corpus_sha256: str
    retrieval_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.retrieval_config is None:
            data.pop("retrieval_config")
        return data


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def create_case_id(test_case: dict[str, Any]) -> str:
    """Return a stable ID for a question/reference/source combination."""
    identity = "\n".join((
        test_case["question"],
        test_case["answer"],
        test_case["source_url"],
    ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a content fingerprint for an evaluation input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_source_url(url: str) -> str:
    """Normalize URL formatting used by deterministic source metrics."""
    return url.strip().rstrip("/")


def compute_source_metrics(
    expected_source_url: str,
    retrieved_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute page-level hit and reciprocal-rank metrics."""
    expected = normalize_source_url(expected_source_url)
    source_rank = None

    for rank, source in enumerate(retrieved_sources, start=1):
        retrieved_url = source.get("source_url")
        if isinstance(retrieved_url, str) and normalize_source_url(retrieved_url) == expected:
            source_rank = rank
            break

    return {
        "source_hit_at_k": source_rank is not None,
        "expected_source_rank": source_rank,
        "reciprocal_rank": 0.0 if source_rank is None else 1.0 / source_rank,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate deterministic quality and latency metrics."""
    if not results:
        return {
            "completed_questions": 0,
            "source_hit_rate": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_retrieval_latency_ms": 0.0,
            "mean_generation_latency_ms": 0.0,
            "mean_total_latency_ms": 0.0,
        }

    return {
        "completed_questions": len(results),
        "source_hit_rate": fmean(
            float(result["metrics"]["source_hit_at_k"])
            for result in results
        ),
        "mean_reciprocal_rank": fmean(
            result["metrics"]["reciprocal_rank"]
            for result in results
        ),
        "mean_retrieval_latency_ms": fmean(
            result["rag_result"]["retrieval_latency_ms"]
            for result in results
        ),
        "mean_generation_latency_ms": fmean(
            result["rag_result"]["generation_latency_ms"]
            for result in results
        ),
        "mean_total_latency_ms": fmean(
            result["rag_result"]["total_latency_ms"]
            for result in results
        ),
    }


def load_test_set(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        test_cases = json.load(file)

    required_fields = {"question", "answer", "source_url", "title"}
    for index, test_case in enumerate(test_cases):
        missing = required_fields - test_case.keys()
        if missing:
            raise ValueError(
                f"Test case {index} is missing required fields: {sorted(missing)}"
            )

    return test_cases


def write_run_file(path: Path, run_data: dict[str, Any]) -> None:
    """Atomically persist progress so interrupted runs can resume safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(run_data, file, indent=2, ensure_ascii=False)
    temporary_path.replace(path)


def initialize_run(
    output_path: Path,
    config: EvaluationConfig,
    resume: bool,
) -> dict[str, Any]:
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as file:
            run_data = json.load(file)

        if run_data.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("Existing result file uses a different schema version")
        if run_data.get("config") != config.to_dict():
            raise ValueError(
                "Existing result configuration does not match this evaluation run"
            )
        return run_data

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "config": config.to_dict(),
        "started_at": utc_now(),
        "updated_at": None,
        "completed_at": None,
        "status": "running",
        "results": [],
        "errors": [],
        "summary": summarize_results([]),
    }


def evaluate_test_set(
    pipeline: BasicRAGPipeline,
    test_cases: list[dict[str, Any]],
    config: EvaluationConfig,
    output_path: Path,
    resume: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate test cases and persist progress after every question."""
    selected_cases = test_cases if limit is None else test_cases[:limit]
    run_data = initialize_run(output_path, config, resume)
    completed_case_ids = {
        result["case_id"]
        for result in run_data["results"]
    }

    for index, test_case in enumerate(selected_cases, start=1):
        case_id = create_case_id(test_case)
        if case_id in completed_case_ids:
            logger.info(f"[{index}/{len(selected_cases)}] Already completed; skipping")
            continue

        run_data["errors"] = [
            error
            for error in run_data["errors"]
            if error["case_id"] != case_id
        ]

        try:
            logger.info(
                f"[{index}/{len(selected_cases)}] Evaluating: "
                f"{test_case['question']}"
            )
            rag_result: RAGResult = pipeline.query(test_case["question"])
            result = {
                "case_id": case_id,
                "reference": {
                    "answer": test_case["answer"],
                    "source_url": test_case["source_url"],
                    "title": test_case["title"],
                },
                "rag_result": rag_result.to_dict(),
                "metrics": compute_source_metrics(
                    test_case["source_url"],
                    rag_result.sources,
                ),
            }
            run_data["results"].append(result)
            completed_case_ids.add(case_id)
        except Exception as error:
            logger.error(
                f"[{index}/{len(selected_cases)}] Failed with "
                f"{type(error).__name__}: {error}"
            )
            run_data["errors"].append({
                "case_id": case_id,
                "question": test_case["question"],
                "error_type": type(error).__name__,
                "message": str(error),
            })

        run_data["updated_at"] = utc_now()
        run_data["summary"] = summarize_results(run_data["results"])
        write_run_file(output_path, run_data)

    selected_case_ids = {create_case_id(test_case) for test_case in selected_cases}
    completed_selected_ids = completed_case_ids & selected_case_ids
    run_data["status"] = (
        "completed"
        if len(completed_selected_ids) == len(selected_case_ids)
        else "completed_with_errors"
    )
    run_data["updated_at"] = utc_now()
    run_data["completed_at"] = run_data["updated_at"]
    run_data["summary"] = summarize_results(run_data["results"])
    write_run_file(output_path, run_data)
    return run_data


def default_output_path(config: EvaluationConfig) -> Path:
    filename = "__".join((
        config.variant,
        config.chunking_strategy,
        config.embedding_model,
        config.llm_model,
        f"k{config.top_k}",
    ))
    safe_filename = re.sub(r"[^a-zA-Z0-9_.-]+", "-", filename)
    if config.retrieval_config is not None:
        fingerprint = hashlib.sha256(
            json.dumps(config.retrieval_config, sort_keys=True).encode()
        ).hexdigest()[:12]
        safe_filename += f"__retrieval-{fingerprint}"
    return DEFAULT_RESULTS_DIR / f"{safe_filename}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Basic or Hybrid RAG")
    parser.add_argument("--variant", choices=["basic", "hybrid"], default="basic")
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--rrf-c", type=int)
    parser.add_argument("--vector-weight", type=float)
    parser.add_argument(
        "--chunking-strategy",
        choices=list(CHUNKING_STRATEGY_MAPPING.keys()),
        required=True,
    )
    parser.add_argument(
        "--embedder",
        choices=[model.value for model in EmbeddingModel],
        default=DEFAULT_EMBEDDING.value,
    )
    parser.add_argument(
        "--llm",
        choices=[model.value for model in LLMModel],
        default=DEFAULT_LLM.value,
    )
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    hybrid_config = None
    if args.variant == "hybrid":
        from hybrid_retrieval.hybrid_rag import HybridConfig, HybridRAGPipeline
        overrides = {
            name: getattr(args, name)
            for name in ("candidate_k", "rrf_c", "vector_weight")
            if getattr(args, name) is not None
        }
        try:
            hybrid_config = HybridConfig(**overrides)
        except ValueError as error:
            parser.error(str(error))
        if args.top_k > hybrid_config.candidate_k:
            parser.error("--top-k cannot exceed --candidate-k")
    elif any(value is not None for value in (args.candidate_k, args.rrf_c, args.vector_weight)):
        parser.error("Hybrid retrieval options require --variant hybrid")

    test_set_path = args.test_set
    corpus_path = Path(CHUNKED_CORPUS_PATH(args.chunking_strategy))
    config = EvaluationConfig(
        variant=args.variant,
        embedding_model=args.embedder,
        chunking_strategy=args.chunking_strategy,
        llm_model=args.llm,
        top_k=args.top_k,
        test_set_path=str(test_set_path),
        test_set_sha256=sha256_file(test_set_path),
        corpus_path=str(corpus_path),
        corpus_sha256=sha256_file(corpus_path),
        retrieval_config=hybrid_config.to_dict() if hybrid_config else None,
    )
    output_path = args.output or default_output_path(config)

    # Reject incompatible resume files before loading models or touching indices.
    initialize_run(output_path, config, resume=not args.no_resume)
    pipeline_class = HybridRAGPipeline if hybrid_config else BasicRAGPipeline
    extra = {"retrieval_config": hybrid_config} if hybrid_config else {}
    pipeline = pipeline_class(
        embedding_model=EmbeddingModel(args.embedder),
        chunking_strategy=args.chunking_strategy,
        llm_model=LLMModel(args.llm),
        top_k=args.top_k,
        **extra,
    )
    test_cases = load_test_set(test_set_path)
    run_data = evaluate_test_set(
        pipeline=pipeline,
        test_cases=test_cases,
        config=config,
        output_path=output_path,
        resume=not args.no_resume,
        limit=args.limit,
    )

    print(json.dumps(run_data["summary"], indent=2))
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
