from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


POOL_SCHEMA_VERSION = 1
QRELS_SCHEMA_VERSION = 1
SCORE_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Return a SHA-256 fingerprint for an immutable evaluation artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically so an interrupted command cannot corrupt output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(path)


@dataclass(frozen=True)
class ChunkJudgment:
    relevant: bool
    supported_claim_ids: frozenset[str]


def precision_at_k(relevance: list[bool], k: int) -> float:
    """Return relevant results divided by k."""
    _validate_k(k)
    return sum(relevance[:k]) / k


def pooled_recall_at_k(relevance: list[bool], k: int) -> float:
    """Return recall against relevant chunks found in the labeled pool."""
    _validate_k(k)
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant


def reciprocal_rank_at_k(relevance: list[bool], k: int) -> float:
    """Return the reciprocal rank of the first relevant result up to k."""
    _validate_k(k)
    for rank, is_relevant in enumerate(relevance[:k], start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: list[bool], k: int) -> float:
    """Return binary normalized discounted cumulative gain up to k."""
    _validate_k(k)
    retrieved = relevance[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(retrieved, start=1)
        if is_relevant
    )
    ideal_relevant = min(sum(relevance), k)
    if ideal_relevant == 0:
        return 0.0
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )
    return dcg / ideal_dcg


def claim_coverage_at_k(
    chunk_ids: list[str],
    judgments: dict[str, ChunkJudgment],
    reference_claim_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of required reference claims covered up to k."""
    _validate_k(k)
    if not reference_claim_ids:
        return 0.0

    supported: set[str] = set()
    for chunk_id in chunk_ids[:k]:
        supported.update(judgments[chunk_id].supported_claim_ids)
    return len(supported & reference_claim_ids) / len(reference_claim_ids)


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def _parse_qrels_case(
    qrels_case: dict[str, Any],
    candidate_ids: set[str],
) -> tuple[dict[str, ChunkJudgment], set[str]]:
    reference_claims = qrels_case.get("reference_claims", [])
    claim_ids = [claim.get("claim_id") for claim in reference_claims]
    if not claim_ids or any(not isinstance(claim_id, str) for claim_id in claim_ids):
        raise ValueError(
            f"Qrels case {qrels_case.get('case_id')} must define reference claims"
        )
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError(
            f"Qrels case {qrels_case.get('case_id')} has duplicate claim IDs"
        )
    reference_claim_ids = set(claim_ids)

    judgments: dict[str, ChunkJudgment] = {}
    for raw_judgment in qrels_case.get("judgments", []):
        chunk_id = raw_judgment.get("chunk_id")
        if chunk_id in judgments:
            raise ValueError(
                f"Qrels case {qrels_case.get('case_id')} has duplicate judgment "
                f"for {chunk_id}"
            )
        relevant = raw_judgment.get("relevant")
        if not isinstance(relevant, bool):
            raise ValueError(
                f"Judgment for {chunk_id} must set relevant to true or false"
            )
        supported_claim_ids = raw_judgment.get("supported_claim_ids", [])
        if not isinstance(supported_claim_ids, list) or any(
            not isinstance(claim_id, str) for claim_id in supported_claim_ids
        ):
            raise ValueError(
                f"Judgment for {chunk_id} has invalid supported_claim_ids"
            )
        unknown_claim_ids = set(supported_claim_ids) - reference_claim_ids
        if unknown_claim_ids:
            raise ValueError(
                f"Judgment for {chunk_id} references unknown claims: "
                f"{sorted(unknown_claim_ids)}"
            )
        if relevant != bool(supported_claim_ids):
            raise ValueError(
                f"Judgment for {chunk_id} must be relevant exactly when it "
                "supports at least one reference claim"
            )
        judgments[chunk_id] = ChunkJudgment(
            relevant=relevant,
            supported_claim_ids=frozenset(supported_claim_ids),
        )

    missing = candidate_ids - judgments.keys()
    extra = judgments.keys() - candidate_ids
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} candidate judgments")
        if extra:
            details.append(f"contains {len(extra)} unknown chunks")
        raise ValueError(
            f"Qrels case {qrels_case.get('case_id')} " + " and ".join(details)
        )

    return judgments, reference_claim_ids


def _validate_pool_case(
    pool_case: dict[str, Any],
    strategies: list[str],
) -> set[str]:
    rankings = pool_case.get("rankings", {})
    if list(rankings) != strategies:
        raise ValueError(
            f"Pool case {pool_case.get('case_id')} must contain each configured "
            "strategy in configuration order"
        )

    ranked_ids: set[str] = set()
    for strategy, chunk_ids in rankings.items():
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                f"Pool case {pool_case.get('case_id')} has duplicate chunks in "
                f"the {strategy} ranking"
            )
        ranked_ids.update(chunk_ids)

    candidates = pool_case.get("candidates", [])
    candidate_ids = [candidate.get("chunk_id") for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(
            f"Pool case {pool_case.get('case_id')} has duplicate candidates"
        )
    if set(candidate_ids) != ranked_ids:
        raise ValueError(
            f"Pool case {pool_case.get('case_id')} candidates do not match rankings"
        )
    return ranked_ids


def _validate_labeling_provenance(qrels: dict[str, Any]) -> None:
    labeling = qrels.get("labeling", {})
    required_text_fields = (
        "label_source",
        "quality_tier",
        "reviewed_by",
        "reviewed_at",
        "method",
    )
    missing = [
        field
        for field in required_text_fields
        if not isinstance(labeling.get(field), str) or not labeling[field].strip()
    ]
    if missing:
        raise ValueError(
            f"Qrels labeling provenance is incomplete: {sorted(missing)}"
        )
    if labeling.get("review_status") != "approved_for_project_evaluation":
        raise ValueError(
            "Qrels must be approved_for_project_evaluation before scoring"
        )
    if not isinstance(labeling.get("human_verified"), bool):
        raise ValueError("Qrels labeling must explicitly set human_verified")


def score_case(
    pool_case: dict[str, Any],
    qrels_case: dict[str, Any],
    cutoffs: list[int],
) -> dict[str, Any]:
    """Score every strategy ranking for one case against frozen qrels."""
    rankings = pool_case["rankings"]
    candidate_ids = {
        chunk_id for ranking in rankings.values() for chunk_id in ranking
    }
    judgments, reference_claim_ids = _parse_qrels_case(
        qrels_case,
        candidate_ids,
    )

    strategy_scores: dict[str, Any] = {}
    for strategy, chunk_ids in rankings.items():
        relevance = [judgments[chunk_id].relevant for chunk_id in chunk_ids]
        cutoff_scores: dict[str, Any] = {}
        for k in cutoffs:
            cutoff_scores[str(k)] = {
                "hit_at_k": any(relevance[:k]),
                "precision_at_k": precision_at_k(relevance, k),
                "pooled_recall_at_k": pooled_recall_at_k(relevance, k),
                "reciprocal_rank_at_k": reciprocal_rank_at_k(relevance, k),
                "ndcg_at_k": ndcg_at_k(relevance, k),
                "claim_coverage_at_k": claim_coverage_at_k(
                    chunk_ids,
                    judgments,
                    reference_claim_ids,
                    k,
                ),
            }
        strategy_scores[strategy] = cutoff_scores

    return {
        "case_id": pool_case["case_id"],
        "strategies": strategy_scores,
    }


def summarize_case_scores(
    cases: list[dict[str, Any]],
    strategies: list[str],
    cutoffs: list[int],
) -> dict[str, Any]:
    """Aggregate per-case scores by strategy and cutoff."""
    summary: dict[str, Any] = {}
    metric_names = (
        "hit_at_k",
        "precision_at_k",
        "pooled_recall_at_k",
        "reciprocal_rank_at_k",
        "ndcg_at_k",
        "claim_coverage_at_k",
    )
    for strategy in strategies:
        summary[strategy] = {}
        for k in cutoffs:
            scores = [case["strategies"][strategy][str(k)] for case in cases]
            summary[strategy][str(k)] = {
                metric_name: fmean(float(score[metric_name]) for score in scores)
                for metric_name in metric_names
            }
    return summary


def score_pool(
    pool: dict[str, Any],
    qrels: dict[str, Any],
    cutoffs: list[int],
) -> dict[str, Any]:
    """Score a complete candidate pool against its frozen qrels."""
    if pool.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError("Unsupported candidate-pool schema version")
    if qrels.get("schema_version") != QRELS_SCHEMA_VERSION:
        raise ValueError("Unsupported qrels schema version")
    _validate_labeling_provenance(qrels)

    normalized_cutoffs = sorted(set(cutoffs))
    if not normalized_cutoffs:
        raise ValueError("At least one cutoff is required")
    for k in normalized_cutoffs:
        _validate_k(k)

    pool_cases = pool.get("cases", [])
    qrels_cases = qrels.get("cases", [])
    pool_by_id = {case["case_id"]: case for case in pool_cases}
    qrels_by_id = {case["case_id"]: case for case in qrels_cases}
    if len(pool_by_id) != len(pool_cases) or len(qrels_by_id) != len(qrels_cases):
        raise ValueError("Pool and qrels case IDs must be unique")
    if pool_by_id.keys() != qrels_by_id.keys():
        raise ValueError("Pool and qrels must contain the same case IDs")

    strategies = pool["config"]["chunking_strategies"]
    pool_depth = pool["config"]["pool_depth"]
    if max(normalized_cutoffs) > pool_depth:
        raise ValueError("A cutoff cannot exceed the candidate-pool depth")

    for pool_case in pool_cases:
        _validate_pool_case(pool_case, strategies)

    cases = [
        score_case(
            pool_case,
            qrels_by_id[pool_case["case_id"]],
            normalized_cutoffs,
        )
        for pool_case in pool_cases
    ]
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "config": {
            "chunking_strategies": strategies,
            "pool_depth": pool_depth,
            "cutoffs": normalized_cutoffs,
        },
        "cases": cases,
        "summary": summarize_case_scores(cases, strategies, normalized_cutoffs),
    }
