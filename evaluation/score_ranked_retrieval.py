from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from evaluation.ranked_retrieval import atomic_write_json, score_pool, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate deterministic ranking metrics from frozen qrels"
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, action="append", dest="cutoffs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.pool.open(encoding="utf-8") as file:
        pool = json.load(file)
    with args.qrels.open(encoding="utf-8") as file:
        qrels = json.load(file)

    expected_pool_sha = qrels.get("source_pool", {}).get("sha256")
    actual_pool_sha = sha256_file(args.pool)
    if expected_pool_sha != actual_pool_sha:
        raise ValueError(
            "Qrels do not reference this exact candidate-pool artifact"
        )

    result = score_pool(pool, qrels, args.cutoffs or [1, 3, 5, 10])
    result.update({
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "pool_path": str(args.pool),
            "pool_sha256": actual_pool_sha,
            "qrels_path": str(args.qrels),
            "qrels_sha256": sha256_file(args.qrels),
        },
    })
    atomic_write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))
    print(f"Ranked retrieval scores saved to: {args.output}")


if __name__ == "__main__":
    main()
