from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any

from openai import OpenAI

from evaluation.ranked_retrieval import (
    POOL_SCHEMA_VERSION,
    QRELS_SCHEMA_VERSION,
    atomic_write_json,
    sha256_file,
)


DEFAULT_MODEL = "gemma4:31b-mlx"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
LABELING_RUBRIC_VERSION = 1
PROPOSAL_PROTOCOL_VERSION = 2

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ProposalConfig:
    source_pool_path: str
    source_pool_sha256: str
    evaluator_model: str
    evaluator_reasoning_effort: str
    evaluator_max_tokens: int
    ollama_base_url: str
    labeling_rubric_version: int
    proposal_protocol_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SYSTEM_PROMPT = """You create evidence-based retrieval relevance judgments.

Given one question, its reference answer, and an unordered collection of
candidate documentation chunks:

1. Split the reference answer into atomic factual claims required to answer the
   question. Give them IDs c1, c2, ... in reference-answer order.
2. For every candidate chunk, list each claim ID that the chunk directly
   supports. Evidence may be paraphrased, but it must be present in the chunk.
3. Set relevant=true exactly when supported_claim_ids is non-empty.

Judge only the supplied text. Do not infer relevance from a chunk ID, likely
source page, or outside knowledge. Do not reward general topical similarity.
Do not omit any candidate or alter its short candidate ID.

Return only a JSON object with this shape:
{
  "reference_claims": [
    {"claim_id": "c1", "claim": "atomic required claim"}
  ],
  "judgments": [
    {
      "candidate_id": "exact supplied short candidate ID",
      "relevant": true,
      "supported_claim_ids": ["c1"],
      "notes": "brief evidence explanation"
    }
  ]
}
"""


def _candidate_sort_key(case_id: str, chunk_id: str) -> str:
    """Hide original retrieval order using a stable case-specific ordering."""
    return hashlib.sha256(f"{case_id}\n{chunk_id}".encode()).hexdigest()


def candidate_prompt_records(
    pool_case: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Return aliased evidence without strategy, rank, URL, or stable chunk ID."""
    candidates = sorted(
        pool_case["candidates"],
        key=lambda candidate: _candidate_sort_key(
            pool_case["case_id"],
            candidate["chunk_id"],
        ),
    )
    records: list[dict[str, str]] = []
    aliases: dict[str, str] = {}
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = f"d{index}"
        records.append({
            "candidate_id": candidate_id,
            "text": candidate["text"],
        })
        aliases[candidate_id] = candidate["chunk_id"]
    return records, aliases


def build_user_prompt(pool_case: dict[str, Any]) -> str:
    records, _ = candidate_prompt_records(pool_case)
    payload = {
        "question": pool_case["question"],
        "reference_answer": pool_case["reference"]["answer"],
        "candidates": records,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse JSON even when a compatible endpoint adds Markdown fences."""
    stripped = content.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline == -1 or not stripped.endswith("```"):
            raise ValueError("Label proposer returned an incomplete JSON fence")
        stripped = stripped[first_newline + 1:-3].strip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Label proposer response must be a JSON object")
    return parsed


def validate_proposal(
    pool_case: dict[str, Any],
    proposal: dict[str, Any],
    candidate_aliases: dict[str, str] | None = None,
    protocol_version: int = 1,
) -> dict[str, Any]:
    """Validate and normalize one model-generated qrels case."""
    raw_claims = proposal.get("reference_claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ValueError("Proposal must contain at least one reference claim")

    claim_ids: list[str] = []
    claims: list[dict[str, str]] = []
    for index, raw_claim in enumerate(raw_claims, start=1):
        expected_id = f"c{index}"
        if raw_claim.get("claim_id") != expected_id:
            raise ValueError("Reference claim IDs must be sequential c1, c2, ...")
        claim = raw_claim.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"Reference claim {expected_id} has no text")
        claim_ids.append(expected_id)
        claims.append({"claim_id": expected_id, "claim": claim.strip()})

    raw_judgments = proposal.get("judgments")
    if not isinstance(raw_judgments, list):
        raise ValueError("Proposal judgments must be a list")
    expected_chunk_ids = {
        candidate["chunk_id"] for candidate in pool_case["candidates"]
    }
    judgments_by_id: dict[str, dict[str, Any]] = {}
    for raw_judgment in raw_judgments:
        if candidate_aliases is None:
            chunk_id = raw_judgment.get("chunk_id")
        else:
            candidate_id = raw_judgment.get("candidate_id")
            if candidate_id not in candidate_aliases:
                raise ValueError(
                    f"Proposal contains unknown candidate alias {candidate_id}"
                )
            chunk_id = candidate_aliases[candidate_id]
        if chunk_id in judgments_by_id:
            raise ValueError(f"Proposal contains duplicate judgment for {chunk_id}")
        if chunk_id not in expected_chunk_ids:
            raise ValueError(f"Proposal contains unknown chunk {chunk_id}")
        relevant = raw_judgment.get("relevant")
        supported_claim_ids = raw_judgment.get("supported_claim_ids")
        if not isinstance(relevant, bool):
            raise ValueError(f"Judgment for {chunk_id} has invalid relevance")
        if not isinstance(supported_claim_ids, list) or any(
            claim_id not in claim_ids for claim_id in supported_claim_ids
        ):
            raise ValueError(f"Judgment for {chunk_id} has invalid claim IDs")
        if len(supported_claim_ids) != len(set(supported_claim_ids)):
            raise ValueError(f"Judgment for {chunk_id} repeats a claim ID")
        if relevant != bool(supported_claim_ids):
            raise ValueError(
                f"Judgment for {chunk_id} relevance conflicts with claim support"
            )
        notes = raw_judgment.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError(f"Judgment for {chunk_id} has invalid notes")
        judgments_by_id[chunk_id] = {
            "chunk_id": chunk_id,
            "relevant": relevant,
            "supported_claim_ids": supported_claim_ids,
            "notes": notes.strip(),
        }

    missing = expected_chunk_ids - judgments_by_id.keys()
    if missing:
        raise ValueError(f"Proposal is missing {len(missing)} candidate judgments")

    return {
        "case_id": pool_case["case_id"],
        "question": pool_case["question"],
        "proposal_metadata": {"protocol_version": protocol_version},
        "reference_claims": claims,
        "judgments": [
            judgments_by_id[candidate["chunk_id"]]
            for candidate in pool_case["candidates"]
        ],
    }


class QrelsProposer:
    def __init__(
        self,
        model: str,
        reasoning_effort: str,
        max_tokens: int,
        base_url: str,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key="ollama", base_url=base_url)

    def propose(self, pool_case: dict[str, Any]) -> dict[str, Any]:
        _, candidate_aliases = candidate_prompt_records(pool_case)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(pool_case)},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Label proposer returned an empty response")
        return validate_proposal(
            pool_case,
            parse_json_object(content),
            candidate_aliases=candidate_aliases,
            protocol_version=PROPOSAL_PROTOCOL_VERSION,
        )


def initialize_run(config: ProposalConfig) -> dict[str, Any]:
    return {
        "schema_version": QRELS_SCHEMA_VERSION,
        "status": "generating",
        "created_at": utc_now(),
        "updated_at": None,
        "source_pool": {
            "path": config.source_pool_path,
            "sha256": config.source_pool_sha256,
        },
        "generation_config": config.to_dict(),
        "proposal_protocols": {
            "1": "The model copied full stable chunk IDs into its response.",
            "2": (
                "The model used short opaque aliases that were deterministically "
                "mapped back to stable chunk IDs."
            ),
        },
        "labeling": {
            "label_source": "model_generated",
            "quality_tier": "unreviewed",
            "review_status": "generated_pending_review",
            "reviewed_by": None,
            "reviewed_at": None,
            "human_verified": False,
            "method": None,
            "usage": (
                "Model-generated proposals must be reviewed and explicitly "
                "approved before the deterministic scorer will consume them."
            ),
        },
        "rubric": {
            "version": LABELING_RUBRIC_VERSION,
            "reference_claims": (
                "Atomic factual claims required by the reference answer."
            ),
            "relevant": (
                "True exactly when a chunk directly supports at least one "
                "required reference claim."
            ),
            "supported_claim_ids": (
                "Every required reference claim directly supported by the chunk."
            ),
        },
        "cases": [],
        "errors": [],
    }


def load_or_initialize_run(
    output_path: Path,
    config: ProposalConfig,
    resume: bool,
) -> dict[str, Any]:
    if not resume or not output_path.exists():
        return initialize_run(config)
    with output_path.open(encoding="utf-8") as file:
        run = json.load(file)
    if run.get("schema_version") != QRELS_SCHEMA_VERSION:
        raise ValueError("Existing proposal uses a different schema version")
    expected_config = config.to_dict()
    existing_config = run.get("generation_config", {})
    legacy_config = dict(expected_config)
    legacy_config.pop("proposal_protocol_version")
    if existing_config not in (expected_config, legacy_config):
        raise ValueError("Existing proposal configuration does not match")
    if existing_config == legacy_config:
        run["generation_config"] = expected_config
        for case in run.get("cases", []):
            case.setdefault("proposal_metadata", {"protocol_version": 1})
    run.setdefault("proposal_protocols", {
        "1": "The model copied full stable chunk IDs into its response.",
        "2": (
            "The model used short opaque aliases that were deterministically "
            "mapped back to stable chunk IDs."
        ),
    })
    return run


def propose_qrels(
    pool: dict[str, Any],
    proposer: Any,
    config: ProposalConfig,
    output_path: Path,
    resume: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    if pool.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError("Unsupported candidate-pool schema version")
    selected_cases = pool["cases"] if limit is None else pool["cases"][:limit]
    run = load_or_initialize_run(output_path, config, resume)
    completed_ids = {case["case_id"] for case in run["cases"]}

    for index, pool_case in enumerate(selected_cases, start=1):
        case_id = pool_case["case_id"]
        if case_id in completed_ids:
            logger.info("[%s/%s] Already proposed; skipping", index, len(selected_cases))
            continue
        run["errors"] = [
            error for error in run["errors"] if error["case_id"] != case_id
        ]
        try:
            logger.info(
                "[%s/%s] Proposing labels: %s",
                index,
                len(selected_cases),
                pool_case["question"],
            )
            run["cases"].append(proposer.propose(pool_case))
            completed_ids.add(case_id)
        except Exception as error:
            logger.error(
                "[%s/%s] Failed with %s: %s",
                index,
                len(selected_cases),
                type(error).__name__,
                error,
            )
            run["errors"].append({
                "case_id": case_id,
                "error_type": type(error).__name__,
                "message": str(error),
            })
        run["updated_at"] = utc_now()
        atomic_write_json(output_path, run)

    selected_ids = {case["case_id"] for case in selected_cases}
    run["status"] = (
        "generated_pending_review"
        if selected_ids <= completed_ids
        else "generated_with_errors"
    )
    run["updated_at"] = utc_now()
    atomic_write_json(output_path, run)
    return run


def default_output_path(pool_path: Path, model: str) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model)
    return pool_path.with_name(f"{pool_path.stem}__proposed-{safe_model}.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose chunk-level qrels with a local model for later review"
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    with args.pool.open(encoding="utf-8") as file:
        pool = json.load(file)
    config = ProposalConfig(
        source_pool_path=str(args.pool),
        source_pool_sha256=sha256_file(args.pool),
        evaluator_model=args.model,
        evaluator_reasoning_effort=args.reasoning_effort,
        evaluator_max_tokens=args.max_tokens,
        ollama_base_url=args.ollama_base_url,
        labeling_rubric_version=LABELING_RUBRIC_VERSION,
        proposal_protocol_version=PROPOSAL_PROTOCOL_VERSION,
    )
    output_path = args.output or default_output_path(args.pool, args.model)
    proposer = QrelsProposer(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        base_url=args.ollama_base_url,
    )
    run = propose_qrels(
        pool,
        proposer,
        config,
        output_path,
        resume=not args.no_resume,
        limit=args.limit,
    )
    print(json.dumps({
        "status": run["status"],
        "proposed_cases": len(run["cases"]),
        "errors": len(run["errors"]),
    }, indent=2))
    print(f"Qrels proposals saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
