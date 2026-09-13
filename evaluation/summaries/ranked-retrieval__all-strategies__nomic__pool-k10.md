# Ranked retrieval comparison: chunking strategies

## Configuration

- Test set: 50 fixed FastAPI documentation questions
- Embedder: `nomic-embed-text`
- Candidate depth: 10 per strategy
- Evaluation cutoffs: 1, 3, 5, 10
- Strategies: `recursive-500`, `recursive-1000`, `semantic`
- Relevance labels: Gemma-generated, Codex-reviewed silver qrels
- Human verification: false

The qrels contain 1,500 chunk judgments and atomic reference-claim support.
They are linked to the exact candidate pool and raw proposal artifacts by
SHA-256. Pooled Recall@K is strategy-local recall within the labeled top-10
pool, not exhaustive corpus recall.

Review was targeted: it compared 39 existing calibration labels, inspected the
six disagreements and selected outliers, and corrected one case. It did not
independently verify all 1,500 judgments. These scores are provisional evidence;
errors in claim decomposition or support labels can affect the comparison.

## Results at the application retrieval depth (K=3)

| Strategy | Hit@3 | Precision@3 | Pooled Recall@3 | MRR@3 | nDCG@3 | Claim Coverage@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `recursive-500` | 0.800 | 0.440 | 0.545 | 0.693 | 0.596 | 0.446 |
| `recursive-1000` | **0.880** | **0.520** | **0.609** | 0.770 | **0.683** | **0.665** |
| `semantic` | 0.840 | 0.447 | 0.592 | **0.770** | 0.653 | 0.621 |

Under these labels at K=3, `recursive-1000` retrieves relevant evidence more often, returns a
higher proportion of relevant chunks, orders them better, and covers more of
the reference-answer claims. Semantic matches its mean reciprocal rank across
all questions, including misses scored as zero; this does not imply identical
first-hit ranks among successful retrievals.

## Depth sensitivity

| K | Strategy | Hit@K | Precision@K | Pooled Recall@K | nDCG@K | Claim Coverage@K |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `recursive-500` | 0.600 | 0.600 | 0.281 | 0.600 | 0.305 |
| 1 | `recursive-1000` | 0.680 | 0.680 | 0.305 | 0.680 | 0.391 |
| 1 | `semantic` | **0.700** | **0.700** | **0.367** | **0.700** | **0.487** |
| 5 | `recursive-500` | 0.880 | 0.360 | 0.687 | 0.646 | 0.525 |
| 5 | `recursive-1000` | 0.900 | **0.400** | 0.721 | 0.693 | **0.724** |
| 5 | `semantic` | **0.920** | 0.368 | **0.755** | **0.715** | 0.714 |
| 10 | `recursive-500` | **0.960** | 0.264 | **0.960** | 0.758 | 0.657 |
| 10 | `recursive-1000` | **0.960** | **0.284** | **0.960** | 0.795 | 0.783 |
| 10 | `semantic` | 0.940 | 0.246 | 0.940 | **0.795** | **0.796** |

Semantic is strongest at K=1 and becomes competitive again at deeper cutoffs.
`recursive-1000` is strongest at the current K=3 and remains balanced at K=5.
This shows why the retrieval depth must be held constant when choosing a
chunking strategy.

## Interpretation

This report does not lock the Experiment 0 winner. It measures retrieval only;
the final decision must combine these deterministic results with Basic RAG
answers, the frozen Gemma Ragas scores, latency, and per-case failure analysis
for all three strategies. The current evidence makes `recursive-1000` the
retrieval-layer leader at K=3 and the strategy to beat in that comparison.
