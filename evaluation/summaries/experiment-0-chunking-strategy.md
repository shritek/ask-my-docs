# Experiment 0: chunking strategy

## Configuration

- Test set: 50 fixed FastAPI documentation questions
- Architecture: Basic RAG
- Embedding model: `nomic-embed-text`
- Generation model: `llama3.1:8b`
- Retrieval depth: K=3
- Ragas judge: `gemma4:31b-mlx`, reasoning disabled, 4,096 max tokens
- Strategies: `recursive-500`, `recursive-1000`, `semantic`

All three Basic RAG runs and all three 50-case Ragas sidecars completed with no
errors. The deterministic retrieval report is in
`evaluation/summaries/ranked-retrieval__all-strategies__nomic__pool-k10.md`.

## Results

| Strategy | Source hit rate | Source MRR | Mean total latency (s) | Faithfulness | Answer relevancy | Context precision | Context recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `recursive-500` | 0.880 | 0.800 | 3.434 | 0.707 | 0.655 | 0.523 | 0.491 |
| `recursive-1000` | **0.940** | **0.807** | **3.717** | **0.834** | 0.769 | **0.633** | **0.710** |
| `semantic` | 0.840 | 0.767 | 4.341 | 0.768 | **0.787** | 0.598 | 0.642 |

## Decision

Use `recursive-1000` as the provisional chunking strategy for the next
retrieval-architecture experiment. It provides the strongest evidence coverage
and answer faithfulness, the highest source hit rate, and the lowest latency of
the two leading strategies. Semantic's 0.018 answer-relevancy advantage does
not outweigh its weaker retrieval metrics and 0.624 second higher mean latency
in this workload.

This decision holds chunking constant so the next experiment can compare
retrieval architectures. It is a provisional experiment baseline, not a claim
that recursive-1000 is optimal for every corpus or query distribution.

## Limitations and follow-up

Ragas scores use the frozen Gemma judge. Chunk-level retrieval scores use
AI-generated, Codex-reviewed silver qrels with targeted review of disagreements
and outliers; all 1,500 judgments were not independently human-verified. The
50 questions are synthetic documentation questions, so production traffic may
change the ranking. Future evaluation should add varied real-user queries and a
human-verified holdout before treating the choice as final.

Per-case raw results remain in `evaluation/results/` for failure analysis and
reproduction. Later changes to the prompt, judge, embedding model, retrieval
depth, or test set require a new controlled comparison.
