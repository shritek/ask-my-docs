import argparse
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import re
from statistics import fmean
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI

from config.settings import DEFAULT_LLM
from evaluation.compare_judge_calibration import load_calibration
from evaluation.ragas_compat import load_ragas_components
from evaluation.run_basic_evaluation import sha256_file, utc_now, write_run_file


DEFAULT_RESULTS_DIR = Path("evaluation/results")
DEFAULT_EVALUATOR_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_EVALUATOR_MAX_TOKENS = 4096
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
RESULT_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagasEvaluationConfig:
    """Inputs that must remain constant while a Ragas sidecar is resumed."""

    source_result_path: str
    source_result_sha256: str
    evaluator_llm_model: str
    evaluator_embedding_model: str
    evaluator_max_tokens: int
    ollama_base_url: str
    answer_relevancy_strictness: int
    metric_names: tuple[str, ...]
    ragas_version: str
    case_limit: int | None
    case_ids: tuple[str, ...] | None = None
    case_selection_path: str | None = None
    case_selection_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metric_names"] = list(self.metric_names)
        if self.case_ids is None:
            data.pop("case_ids")
            data.pop("case_selection_path")
            data.pop("case_selection_sha256")
        else:
            data["case_ids"] = list(self.case_ids)
        return data


class RagasScorerSuite:
    """Own the four initialized Ragas scorers for one evaluator configuration."""

    def __init__(
        self,
        evaluator_llm_model: str,
        evaluator_embedding_model: str,
        evaluator_max_tokens: int,
        ollama_base_url: str,
        answer_relevancy_strictness: int,
    ):
        components = load_ragas_components()
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=ollama_base_url,
        )
        evaluator_llm = components["llm_factory"](
            evaluator_llm_model,
            provider="openai",
            client=client,
            max_tokens=evaluator_max_tokens,
        )
        evaluator_embeddings = components["embedding_factory"](
            "openai",
            model=evaluator_embedding_model,
            client=client,
        )
        metric_classes = components["metric_classes"]
        self.ragas_version = components["version"]
        self.scorers = {
            "faithfulness": metric_classes["faithfulness"](llm=evaluator_llm),
            "answer_relevancy": metric_classes["answer_relevancy"](
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                strictness=answer_relevancy_strictness,
            ),
            "context_precision": metric_classes["context_precision"](
                llm=evaluator_llm,
            ),
            "context_recall": metric_classes["context_recall"](
                llm=evaluator_llm,
            ),
        }

    def score(self, metric_name: str, source_case: dict[str, Any]) -> dict[str, Any]:
        """Score one metric and retain Ragas reasoning when it is available."""
        rag_result = source_case["rag_result"]
        common_inputs = {
            "user_input": rag_result["question"],
            "retrieved_contexts": rag_result["contexts"],
        }
        if metric_name == "faithfulness":
            metric_inputs = {
                **common_inputs,
                "response": rag_result["answer"],
            }
        elif metric_name == "answer_relevancy":
            metric_inputs = {
                "user_input": rag_result["question"],
                "response": rag_result["answer"],
            }
        elif metric_name in {"context_precision", "context_recall"}:
            metric_inputs = {
                **common_inputs,
                "reference": source_case["reference"]["answer"],
            }
        else:
            raise ValueError(f"Unsupported Ragas metric: {metric_name}")

        result = self.scorers[metric_name].score(**metric_inputs)
        value = float(result.value)
        if not math.isfinite(value):
            raise ValueError(f"Ragas returned a non-finite {metric_name} score")
        score = {"value": value}
        reason = getattr(result, "reason", None)
        if reason is not None:
            score["reason"] = reason
        return score


def load_source_run(path: Path) -> dict[str, Any]:
    """Load a completed deterministic run that can be judged independently."""
    with path.open(encoding="utf-8") as file:
        source_run = json.load(file)

    if source_run.get("status") != "completed":
        raise ValueError("Ragas evaluation requires a completed source result")
    if source_run.get("errors"):
        raise ValueError("Ragas evaluation requires a source result without errors")
    if not source_run.get("results"):
        raise ValueError("Ragas evaluation requires at least one source result")

    required_case_fields = {"case_id", "reference", "rag_result"}
    required_rag_fields = {"question", "answer", "contexts"}
    case_ids = set()
    for index, source_case in enumerate(source_run["results"]):
        missing_case_fields = required_case_fields - source_case.keys()
        if missing_case_fields:
            raise ValueError(
                f"Source result {index} is missing fields: "
                f"{sorted(missing_case_fields)}"
            )
        missing_rag_fields = required_rag_fields - source_case["rag_result"].keys()
        if missing_rag_fields:
            raise ValueError(
                f"Source RAG result {index} is missing fields: "
                f"{sorted(missing_rag_fields)}"
            )
        if "answer" not in source_case["reference"]:
            raise ValueError(f"Source reference {index} is missing answer")
        if source_case["case_id"] in case_ids:
            raise ValueError(f"Source result contains duplicate case ID at {index}")
        case_ids.add(source_case["case_id"])

    return source_run


def select_source_cases(
    source_cases: list[dict[str, Any]],
    *,
    limit: int | None = None,
    case_ids: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Select either a prefix or an explicit ordered set of source cases."""
    if limit is not None and case_ids is not None:
        raise ValueError("Cannot combine a case limit with explicit case IDs")
    if case_ids is None:
        return source_cases if limit is None else source_cases[:limit]
    if not case_ids:
        raise ValueError("Explicit case selection must not be empty")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Explicit case selection contains duplicate case IDs")

    source_by_case_id = {
        source_case["case_id"]: source_case
        for source_case in source_cases
    }
    missing_case_ids = [
        case_id for case_id in case_ids if case_id not in source_by_case_id
    ]
    if missing_case_ids:
        raise ValueError(
            "Source result is missing selected case IDs: "
            + ", ".join(missing_case_ids)
        )
    return [source_by_case_id[case_id] for case_id in case_ids]


def summarize_ragas_results(
    results: list[dict[str, Any]],
    total_cases: int,
    metric_names: tuple[str, ...],
) -> dict[str, Any]:
    metric_summaries = {}
    for metric_name in metric_names:
        scores = [
            result["metrics"][metric_name]["value"]
            for result in results
            if metric_name in result["metrics"]
        ]
        metric_summaries[metric_name] = {
            "completed": len(scores),
            "mean": fmean(scores) if scores else None,
        }

    completed_cases = sum(
        all(metric_name in result["metrics"] for metric_name in metric_names)
        for result in results
    )
    metric_latencies = [
        metric["latency_ms"]
        for result in results
        for metric in result["metrics"].values()
    ]
    return {
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "metrics": metric_summaries,
        "mean_metric_latency_ms": (
            fmean(metric_latencies) if metric_latencies else None
        ),
    }


def initialize_ragas_run(
    output_path: Path,
    config: RagasEvaluationConfig,
    resume: bool,
    total_cases: int,
) -> dict[str, Any]:
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as file:
            run_data = json.load(file)

        if run_data.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("Existing Ragas result uses a different schema version")
        if run_data.get("config") != config.to_dict():
            raise ValueError(
                "Existing Ragas result configuration does not match this run"
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
        "summary": summarize_ragas_results(
            [],
            total_cases,
            config.metric_names,
        ),
    }


def evaluate_ragas(
    scorer_suite,
    source_cases: list[dict[str, Any]],
    config: RagasEvaluationConfig,
    output_path: Path,
    resume: bool = True,
    limit: int | None = None,
    case_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Add resumable Ragas scores without modifying the source RAG run."""
    if config.case_limit != limit:
        raise ValueError("Ragas configuration case_limit does not match limit")
    if config.case_ids != case_ids:
        raise ValueError("Ragas configuration case_ids do not match selection")

    selected_cases = select_source_cases(
        source_cases,
        limit=limit,
        case_ids=case_ids,
    )
    run_data = initialize_ragas_run(
        output_path,
        config,
        resume,
        len(selected_cases),
    )
    results_by_case_id = {
        result["case_id"]: result
        for result in run_data["results"]
    }

    for case_index, source_case in enumerate(selected_cases, start=1):
        case_id = source_case["case_id"]
        result = results_by_case_id.get(case_id)
        if result is None:
            result = {"case_id": case_id, "metrics": {}}
            run_data["results"].append(result)
            results_by_case_id[case_id] = result

        for metric_name in config.metric_names:
            if metric_name in result["metrics"]:
                logger.info(
                    f"[{case_index}/{len(selected_cases)}] "
                    f"{metric_name} already completed; skipping"
                )
                continue

            run_data["errors"] = [
                error
                for error in run_data["errors"]
                if not (
                    error["case_id"] == case_id
                    and error["metric"] == metric_name
                )
            ]
            started_at = perf_counter()
            try:
                logger.info(
                    f"[{case_index}/{len(selected_cases)}] Scoring {metric_name}"
                )
                metric_result = scorer_suite.score(metric_name, source_case)
                metric_result["latency_ms"] = (
                    perf_counter() - started_at
                ) * 1000
                result["metrics"][metric_name] = metric_result
            except Exception as error:
                logger.error(
                    f"[{case_index}/{len(selected_cases)}] {metric_name} failed "
                    f"with {type(error).__name__}: {error}"
                )
                run_data["errors"].append({
                    "case_id": case_id,
                    "metric": metric_name,
                    "error_type": type(error).__name__,
                    "message": str(error),
                })

            run_data["updated_at"] = utc_now()
            run_data["summary"] = summarize_ragas_results(
                run_data["results"],
                len(selected_cases),
                config.metric_names,
            )
            write_run_file(output_path, run_data)

    run_data["status"] = (
        "completed"
        if run_data["summary"]["completed_cases"] == len(selected_cases)
        else "completed_with_errors"
    )
    run_data["updated_at"] = utc_now()
    run_data["completed_at"] = run_data["updated_at"]
    run_data["summary"] = summarize_ragas_results(
        run_data["results"],
        len(selected_cases),
        config.metric_names,
    )
    write_run_file(output_path, run_data)
    return run_data


def default_output_path(
    source_result_path: Path,
    evaluator_llm_model: str,
    evaluator_embedding_model: str,
    evaluator_max_tokens: int,
    case_selection_sha256: str | None = None,
    case_count: int | None = None,
) -> Path:
    filename_parts = [
        "ragas",
        source_result_path.stem,
        f"judge-{evaluator_llm_model}",
        f"emb-{evaluator_embedding_model}",
        f"max-tokens-{evaluator_max_tokens}",
    ]
    if case_selection_sha256 is not None:
        if case_count is None:
            raise ValueError("Selected-case output requires a case count")
        filename_parts.append(
            f"cases-{case_count}-{case_selection_sha256[:8]}"
        )
    filename = "__".join(filename_parts)
    safe_filename = re.sub(r"[^a-zA-Z0-9_.-]+", "-", filename)
    return DEFAULT_RESULTS_DIR / f"{safe_filename}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add Ragas scores to a completed Basic RAG result"
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument(
        "--evaluator-llm",
        default=DEFAULT_LLM.value,
        help="Local evaluator model name exposed by Ollama",
    )
    parser.add_argument(
        "--evaluator-embedding-model",
        default=DEFAULT_EVALUATOR_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--evaluator-max-tokens",
        type=int,
        default=DEFAULT_EVALUATOR_MAX_TOKENS,
    )
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--answer-relevancy-strictness", type=int, default=3)
    parser.add_argument("--output", type=Path)
    case_selection = parser.add_mutually_exclusive_group()
    case_selection.add_argument("--limit", type=int)
    case_selection.add_argument(
        "--calibration-set",
        type=Path,
        help="Approved calibration set whose exact case IDs should be scored",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.answer_relevancy_strictness < 1:
        parser.error("--answer-relevancy-strictness must be at least 1")
    args.evaluator_llm = args.evaluator_llm.strip()
    if not args.evaluator_llm:
        parser.error("--evaluator-llm must not be empty")
    if args.evaluator_max_tokens < 1:
        parser.error("--evaluator-max-tokens must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    source_run = load_source_run(args.source_result)
    calibration = (
        load_calibration(args.calibration_set)
        if args.calibration_set is not None
        else None
    )
    case_ids = (
        tuple(case["case_id"] for case in calibration["cases"])
        if calibration is not None
        else None
    )
    case_selection_sha256 = (
        sha256_file(args.calibration_set)
        if args.calibration_set is not None
        else None
    )
    components = load_ragas_components()
    config = RagasEvaluationConfig(
        source_result_path=str(args.source_result),
        source_result_sha256=sha256_file(args.source_result),
        evaluator_llm_model=args.evaluator_llm,
        evaluator_embedding_model=args.evaluator_embedding_model,
        evaluator_max_tokens=args.evaluator_max_tokens,
        ollama_base_url=args.ollama_base_url,
        answer_relevancy_strictness=args.answer_relevancy_strictness,
        metric_names=METRIC_NAMES,
        ragas_version=components["version"],
        case_limit=args.limit,
        case_ids=case_ids,
        case_selection_path=(
            str(args.calibration_set)
            if args.calibration_set is not None
            else None
        ),
        case_selection_sha256=case_selection_sha256,
    )
    output_path = args.output or default_output_path(
        args.source_result,
        args.evaluator_llm,
        args.evaluator_embedding_model,
        args.evaluator_max_tokens,
        case_selection_sha256,
        len(case_ids) if case_ids is not None else None,
    )
    scorer_suite = RagasScorerSuite(
        evaluator_llm_model=args.evaluator_llm,
        evaluator_embedding_model=args.evaluator_embedding_model,
        evaluator_max_tokens=args.evaluator_max_tokens,
        ollama_base_url=args.ollama_base_url,
        answer_relevancy_strictness=args.answer_relevancy_strictness,
    )
    run_data = evaluate_ragas(
        scorer_suite=scorer_suite,
        source_cases=source_run["results"],
        config=config,
        output_path=output_path,
        resume=not args.no_resume,
        limit=args.limit,
        case_ids=case_ids,
    )

    print(json.dumps(run_data["summary"], indent=2))
    print(f"Ragas results saved to: {output_path}")


if __name__ == "__main__":
    main()
