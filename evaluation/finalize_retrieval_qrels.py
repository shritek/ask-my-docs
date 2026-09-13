from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from evaluation.ranked_retrieval import (
    atomic_write_json,
    score_pool,
    sha256_file,
)


def apply_case_override(
    qrels_case: dict[str, Any],
    override: dict[str, Any],
) -> None:
    """Apply one reviewed claim decomposition and its judgment corrections."""
    replacement_claims = override["reference_claims"]
    replacement_claim_ids = {
        claim["claim_id"] for claim in replacement_claims
    }
    claim_id_remap = override["claim_id_remap"]

    for judgment in qrels_case["judgments"]:
        remapped: list[str] = []
        for old_claim_id in judgment["supported_claim_ids"]:
            if old_claim_id not in claim_id_remap:
                raise ValueError(
                    f"Override for {qrels_case['case_id']} does not map "
                    f"existing claim {old_claim_id}"
                )
            for new_claim_id in claim_id_remap[old_claim_id]:
                if new_claim_id not in remapped:
                    remapped.append(new_claim_id)
        judgment["supported_claim_ids"] = remapped
        judgment["relevant"] = bool(remapped)

    judgments_by_id = {
        judgment["chunk_id"]: judgment
        for judgment in qrels_case["judgments"]
    }
    for update in override.get("judgment_updates", []):
        chunk_id = update["chunk_id"]
        if chunk_id not in judgments_by_id:
            raise ValueError(
                f"Override for {qrels_case['case_id']} references unknown "
                f"chunk {chunk_id}"
            )
        supported_claim_ids = update["supported_claim_ids"]
        unknown_claim_ids = set(supported_claim_ids) - replacement_claim_ids
        if unknown_claim_ids:
            raise ValueError(
                f"Override for {chunk_id} references unknown claims: "
                f"{sorted(unknown_claim_ids)}"
            )
        judgment = judgments_by_id[chunk_id]
        judgment["supported_claim_ids"] = supported_claim_ids
        judgment["relevant"] = bool(supported_claim_ids)
        judgment["notes"] = update["notes"]

    qrels_case["reference_claims"] = replacement_claims
    qrels_case.setdefault("review_history", []).append({
        "review_id": override["review_id"],
        "rationale": override["rationale"],
    })


def finalize_qrels(
    pool: dict[str, Any],
    proposal: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Create approved qrels from a complete proposal and review manifest."""
    if proposal.get("status") != "generated_pending_review":
        raise ValueError("Qrels proposal must be complete and pending review")
    if proposal.get("errors"):
        raise ValueError("Qrels proposal must not contain generation errors")

    qrels = deepcopy(proposal)
    qrels_by_id = {case["case_id"]: case for case in qrels["cases"]}
    for override in review.get("case_overrides", []):
        case_id = override["case_id"]
        if case_id not in qrels_by_id:
            raise ValueError(f"Review references unknown case {case_id}")
        apply_case_override(qrels_by_id[case_id], override)

    qrels["status"] = "approved"
    qrels["finalized_at"] = review["labeling"]["reviewed_at"]
    qrels["labeling"] = review["labeling"]
    qrels["review_summary"] = review["audit"]
    qrels.pop("errors", None)

    # The scorer performs the complete qrels/pool/provenance validation. The
    # result is intentionally discarded; official metrics are generated later.
    score_pool(pool, qrels, [1])
    return qrels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize reviewed retrieval qrels without changing proposals"
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.pool.open(encoding="utf-8") as file:
        pool = json.load(file)
    with args.proposal.open(encoding="utf-8") as file:
        proposal = json.load(file)
    with args.review.open(encoding="utf-8") as file:
        review = json.load(file)

    expected_proposal_sha = review.get("source_proposal", {}).get("sha256")
    if sha256_file(args.proposal) != expected_proposal_sha:
        raise ValueError("Review does not reference this exact proposal artifact")
    expected_pool_sha = proposal.get("source_pool", {}).get("sha256")
    if sha256_file(args.pool) != expected_pool_sha:
        raise ValueError("Proposal does not reference this exact candidate pool")

    qrels = finalize_qrels(pool, proposal, review)
    qrels["source_proposal"] = {
        "path": str(args.proposal),
        "sha256": expected_proposal_sha,
    }
    qrels["review_manifest"] = {
        "path": str(args.review),
        "sha256": sha256_file(args.review),
    }
    atomic_write_json(args.output, qrels)
    print(f"Approved retrieval qrels saved to: {args.output}")


if __name__ == "__main__":
    main()
