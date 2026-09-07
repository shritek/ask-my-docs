from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from evaluation.ranked_retrieval import (
    POOL_SCHEMA_VERSION,
    QRELS_SCHEMA_VERSION,
    atomic_write_json,
    sha256_file,
)


def initialize_qrels(pool_path: Path, pool: dict) -> dict:
    """Create an intentionally incomplete judgment template from a pool."""
    if pool.get("schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError("Unsupported candidate-pool schema version")

    return {
        "schema_version": QRELS_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_pool": {
            "path": str(pool_path),
            "sha256": sha256_file(pool_path),
        },
        "labeling": {
            "label_source": None,
            "quality_tier": None,
            "review_status": "not_started",
            "reviewed_by": None,
            "reviewed_at": None,
            "human_verified": None,
            "method": None,
            "usage": (
                "Complete and freeze these labels before calculating ranked "
                "retrieval metrics. Do not treat pooled recall as exhaustive "
                "corpus recall."
            ),
        },
        "rubric": {
            "reference_claims": (
                "Split the reference answer into atomic claims required to "
                "answer the question."
            ),
            "relevant": (
                "A chunk is relevant exactly when it provides evidence for at "
                "least one required reference claim."
            ),
            "supported_claim_ids": (
                "List every reference claim for which the chunk supplies "
                "evidence. Leave empty when relevant is false."
            ),
        },
        "cases": [
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "reference_claims": [],
                "judgments": [
                    {
                        "chunk_id": candidate["chunk_id"],
                        "relevant": None,
                        "supported_claim_ids": [],
                        "notes": "",
                    }
                    for candidate in case["candidates"]
                ],
            }
            for case in pool["cases"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a chunk-level retrieval-qrels template"
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.pool.open(encoding="utf-8") as file:
        pool = json.load(file)
    qrels = initialize_qrels(args.pool, pool)
    atomic_write_json(args.output, qrels)
    print(f"Qrels template saved to: {args.output}")


if __name__ == "__main__":
    main()
