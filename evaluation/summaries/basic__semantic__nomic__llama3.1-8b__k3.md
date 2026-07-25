# Basic RAG semantic baseline

## Configuration

| Setting | Value |
| --- | --- |
| Variant | `basic` |
| Chunking | `semantic` |
| Embedder | `nomic` |
| Generator | `llama3.1:8b` |
| Retrieval depth | `k=3` |
| Questions | 50 |
| Test-set SHA-256 | `fd2f7ef19c2e99a09f323842374a281118857e83cf38eaa355c73e543c81e993` |
| Corpus SHA-256 | `c0111f1644480fe2103448b77fdcf02d60d0482b1491b8900d977fd69616246d` |

## Deterministic results

| Metric | Result |
| --- | ---: |
| Completed questions | 50/50 |
| Source hit rate at 3 | 0.84 |
| Mean reciprocal rank | 0.7667 |
| Expected source at rank 1 | 35 |
| Expected source at rank 2 | 6 |
| Expected source at rank 3 | 1 |
| Expected source absent | 8 |
| Mean retrieval latency | 26.72 ms |
| Mean generation latency | 4,314.66 ms |
| Mean total latency | 4,341.39 ms |

Latency is environment-dependent and was measured using the local Ollama
runtime. It is useful as a baseline on the same machine, not as a portable
performance claim.

## Audit of the eight expected-source misses

| Case | Audit classification | Learning |
| --- | --- | --- |
| `fb18f9b4` — APISpec and NestJS | Genuine retrieval miss; correct abstention | Exact framework names are a strong candidate for lexical or hybrid retrieval. |
| `00592e5d` — lifespan | Genuine, partial, and redundant retrieval | Similar chunks repeated the deprecation fact but omitted startup/shutdown behavior and sub-application scope; retrieval diversity matters. |
| `2063af3d` — `status_code` | Relevant alternate source retrieved | Exact expected-page matching understated retrieval quality; acceptable alternate sources should be represented in stronger ground truth. |
| `3fc3f007` — process replication | Genuine retrieval miss; correct abstention | Broad topical similarity did not recover the required `--workers` and Kubernetes contrast. |
| `a678325c` — request body mapping | Genuine wrong-topic retrieval | The answer was grounded in retrieved text but irrelevant to `Body(embed=True)`; faithfulness alone does not imply correctness. |
| `9c2c3aed` — Swagger defaults | Relevant alternate source retrieved | The response was mostly correct but omitted exact implementation details; page-level hit rate produced another false negative. |
| `cbc42bd9` — self-hosted docs | Genuine retrieval miss | The model abstained, then added generic filler; answer discipline should stop after a justified abstention. |
| `508a42f0` — global dependencies | Genuine retrieval miss; correct abstention | A title, link, or release-note mention is not sufficient evidence for an answer. |

Six cases were genuine retrieval failures and two retrieved useful evidence from
pages other than the single expected URL. This is why source hit rate and MRR
must be interpreted alongside answer/context metrics and manual audits.

## Next use

This is an initial baseline, not the winner of the chunking experiment. Add
Ragas metrics, run the same configuration for `recursive-500` and
`recursive-1000`, and then compare all three strategies under identical inputs.
