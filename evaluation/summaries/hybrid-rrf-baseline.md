# Hybrid RRF baseline: first generation run

Recorded 2026-09-10. This is the first untuned hybrid configuration, not a
selected architecture winner. The full 50-case generation run completed without
errors and passes the existing Ragas source-file validator. No hybrid Ragas
scores have been generated yet.

## Fixed configuration

Both runs use recursive-1000, nomic embeddings, llama3.1:8b, the same 50-question
test set, final K=3, and the existing generation prompt. Test-set and corpus
fingerprints match. Hybrid uses 10 candidates per arm, equal weights, RRF c=60,
and the versioned BM25/tokenization settings in D026.

| Metric | Basic vector | Hybrid RRF |
| --- | ---: | ---: |
| Source Hit@3 | 0.940 | 0.940 |
| Source MRR@3 | 0.807 | 0.830 |
| Mean retrieval latency (ms) | 21.50 | 28.98 |
| Mean generation latency (ms) | 3695.14 | 4020.85 |
| Mean total latency (ms) | 3716.64 | 4049.82 |

Hybrid improves source reciprocal rank on 9 questions, regresses on 6, and ties
on 35. It gains one expected-page hit and loses one. These are page-level
measurements, not chunk relevance or answer correctness. Latencies are from
separate local runs without repeated timing trials, and exclude initialization;
the observed difference cannot all be attributed to fusion overhead.

## Two cases worth reviewing

- `a678325c7a8d`: request parameter mapping and `Body(embed=True)`. Hybrid finds
  the expected page at rank 3, which Basic missed. Inspecting the generated
  answers shows a partial recovery: hybrid discusses embedding the body, but
  its first part explains response serialization instead of request parameter
  mapping. A gained source hit does not establish a correct answer.
- `508a42f0beed`: application-wide dependencies. Basic finds the expected page
  at rank 3 and answers using `FastAPI(dependencies=[...])`. Hybrid misses that
  page and abstains. Fusion can displace useful vector evidence as well as add
  useful lexical evidence.

This is a targeted inspection of the two changed hit/miss cases, not a full
claim-level audit. The next quality check is frozen-Gemma Ragas on the immutable
hybrid run, followed by inspection of important answer-quality differences.
Keep the Basic baseline until that comparison supports a change. Do not tune
weights based only on the small source-MRR gain.

## Artifacts

- Basic: `evaluation/results/basic__recursive-1000__nomic__llama3.1-8b__k3.json`
  SHA-256: `01e50a2f3a894cab85ed9a8f1b8cb81e8d1a3c807d86721604cca43558cd2901`
- Hybrid: `evaluation/results/hybrid__recursive-1000__nomic__llama3.1-8b__k3__retrieval-1102b24c35e9.json`
  SHA-256: `ceaf3bff60ed2f8816a55499288dfd8505f6ab2bf2d83f09aaa81ac56c683be6`

The old chunking qrels omit some new hybrid candidates. Do not treat missing
labels as irrelevant or reuse the old report as a hybrid evaluation.
