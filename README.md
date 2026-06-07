# Production RAG Application — Ask My Docs

A domain-specific retrieval-augmented generation (RAG) system built with progressively advanced retrieval techniques, systematic evaluation, and CI-gated quality gates. This project demonstrates production-grade RAG patterns used in enterprise AI systems.

> **Status:** 🚧 In Progress

---

## Roadmap

### Phase 0 · Data Ingestion ✅
- [x] Scrape FastAPI docs via `SitemapLoader`
- [x] Clean HTML and save raw pages to `data/corpus_snapshot.json`

### Phase 1 · Chunking Pipeline
- [x] Implement `chunker.py` with `--strategy` flag
- [x] Generate all three chunked corpora (`recursive-500`, `recursive-1000`, `semantic`)

### Phase 2 · Basic RAG
- [ ] Build basic vector retrieval pipeline using `nomic-embed-text` and Chroma

### Phase 3 · Evaluation Setup
- [ ] Build fixed evaluation test set (25 Q&A pairs from FastAPI docs corpus)
- [ ] Implement Ragas evaluation pipeline (`ragas_eval.py`)
- [ ] Verify scores run end-to-end against basic RAG

### Phase 4 · Experiment 0 — Chunking Strategy
- [ ] Evaluate all three chunking strategies using basic RAG + `nomic-embed-text`
- [ ] Compare Ragas scores across strategies
- [ ] Lock down winning chunking strategy for all subsequent phases

### Phase 5 · Hybrid Retrieval
- [ ] Add BM25 retriever alongside vector search
- [ ] Combine via LangChain `EnsembleRetriever`
- [ ] Evaluate against fixed test set and compare to Phase 2

### Phase 6 · Cross-Encoder Reranking
- [ ] Add Cohere Rerank or local `MiniLM-L6-v2`
- [ ] Evaluate against fixed test set and compare to Phase 5

### Phase 7 · Experiments 1 & 2 — 3×3 Evaluation Grid
- [ ] Experiment 1: compare retrieval architectures (fixed embedder, fixed chunking)
- [ ] Experiment 2: compare embedding models (fixed architecture, fixed chunking)
- [ ] Document results with before/after metrics

### Phase 8 · CI Pipeline
- [ ] Add GitHub Actions workflow
- [ ] Gate PRs on Ragas score thresholds
- [ ] Prevent silent retrieval regressions

---

## Project Structure

```
ask-my-docs/
├── README.md
├── pyproject.toml                        # Project metadata and dependencies
├── uv.lock                               # Auto-generated lockfile — committed to git
├── .env.example
│
├── config/
│   └── settings.py                       # Central config — model choices, paths, thresholds
│
├── helpers/
│   ├── embedding_factory.py              # Returns embedder based on config/param
│   ├── llm_factory.py                    # Returns LLM based on config/param
│   └── vector_store_factory.py           # Returns vector store based on config/param
│
├── data/
│   ├── loaders/
│   │   └── sitemap_loader.py             # Scrapes FastAPI docs, saves raw pages
│   ├── chunker.py                        # Reads raw corpus, chunks, saves chunked corpus
│   ├── corpus_snapshot.json              # Raw pages — generated once, not committed to git
│   ├── chunked_corpus_recursive500.json  # Chunking strategy A — not committed to git
│   ├── chunked_corpus_recursive1000.json # Chunking strategy B — not committed to git
│   └── chunked_corpus_semantic.json      # Chunking strategy C — not committed to git
│
├── evaluation/
│   ├── test_set.json                     # Fixed — never modified after creation
│   ├── ragas_eval.py                     # Ragas evaluation pipeline
│   └── results/
│       ├── exp0_chunking.json            # Experiment 0 — chunking strategy comparison
│       ├── exp1_architecture.json        # Experiment 1 — retrieval architecture comparison
│       └── exp2_embedder.json            # Experiment 2 — embedding model comparison
│
├── basic_rag/
│   └── basic_rag.py
├── hybrid_retrieval/
│   └── hybrid_rag.py
├── reranking/
│   └── reranked_rag.py
└── ci_pipeline/
    └── .github/
        └── workflows/
            └── rag_eval.yml
```

---

## Design Decisions

**Pluggable dependencies via factory pattern**

Embedding models, LLMs, and vector stores are injected as parameters rather than hardcoded into each RAG variant. This keeps retrieval architecture code decoupled from infrastructure choices, and makes running controlled experiments a single CLI flag rather than a code change.

```bash
# Same architecture, three embedders — no code changes between runs
uv run 04_reranking/reranked_rag.py --embedder nomic
uv run 04_reranking/reranked_rag.py --embedder qwen3
uv run 04_reranking/reranked_rag.py --embedder openai
```

**Corpus snapshot generated once, chunking kept separate**

Scraping and chunking are intentionally separate steps. The raw corpus is scraped once and saved to disk. Chunking runs independently against the raw snapshot — meaning chunking strategy can be changed without re-scraping. Each strategy produces its own snapshot file so all three chunked corpora exist on disk simultaneously.

```
sitemap_loader.py → corpus_snapshot.json → chunker.py --strategy X → chunked_corpus_X.json → RAG variants
```

**Chunking strategy locked down before the 3×3 evaluation grid**

Chunking is evaluated first (Experiment 0) once basic RAG and the evaluation pipeline are in place. The winning strategy is then held constant across Experiments 1 and 2, ensuring the 3×3 grid isolates only the intended variables — retrieval architecture and embedding model.

```
Experiment 0 · Chunking strategy      → lock down winner
       ↓
Experiment 1 · Retrieval architecture → fixed chunking (winner from Exp 0)
       ↓
Experiment 2 · Embedding model        → fixed chunking + fixed architecture (best from Exp 1)
```

---

## Experiment Design

> **Principle:** Hold everything constant except the one variable under test. Each experiment's winner feeds into the next as a fixed constant — forming a clean dependency chain.

### Experiment 0 — Chunking strategy
Fixed architecture: basic RAG · Fixed embedder: `nomic-embed-text` · Fixed test set · Fixed LLM
*Run in Phase 4 — winning strategy locked for all subsequent experiments.*

| Strategy | Chunk Size | Overlap | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|---|---|
| Recursive Character | 500 | 50 | — | — | — |
| Recursive Character | 1000 | 100 | — | — | — |
| Semantic Chunking | auto | auto | — | — | — |

### Experiment 1 — Retrieval architecture
Fixed embedder: `nomic-embed-text` · Fixed chunking: winner from Exp 0 · Fixed test set · Fixed LLM

| Architecture | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| Basic RAG | — | — | — |
| Hybrid Retrieval | — | — | — |
| Reranking | — | — | — |

### Experiment 2 — Embedding model
Fixed architecture: best from Exp 1 · Fixed chunking: winner from Exp 0 · Fixed test set · Fixed LLM

| Embedder | Size | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|---|
| `nomic-embed-text` | 274 MB | — | — | — |
| `Qwen3-Embedding-8B` | ~8 GB | — | — | — |
| `text-embedding-3-small` | API | — | — | — |

*Results populated after evaluation runs. See `evaluation/results/` for raw scores.*

---

## Tech Stack (planned)

| Component | Choice | Reason |
|---|---|---|
| Framework | LangChain | Industry standard, modular retriever APIs |
| LLM | `llama3.1:8b` via Ollama | Local, free, no API dependency |
| Embedders | `nomic-embed-text`, `Qwen3-Embedding-8B`, `text-embedding-3-small` | Experiment variable — injected via factory |
| Vector store | Chroma → Qdrant | Chroma for prototyping; Qdrant for native hybrid search support |
| Reranker | Cohere / `MiniLM-L6-v2` | API option vs free local option |
| Evaluation | Ragas | Reference-free RAG metrics, CI-compatible |
| Data source | FastAPI docs via `SitemapLoader` | Technical domain, verifiable ground truth |

---

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for environment and package management
- [Ollama](https://ollama.ai) installed and running locally
- OpenAI API key (optional — for `text-embedding-3-small` in Experiment 2)
- Cohere API key (optional — for Cohere Rerank in Phase 6)

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/ask-my-docs.git
cd ask-my-docs

# Install dependencies (creates venv and installs in one step)
uv sync

# Pull local models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Copy and fill in API keys
cp .env.example .env
```

### Generate corpus (run once, in order)

```bash
# Step 1 — scrape raw pages
uv run data/loaders/sitemap_loader.py

# Step 2 — chunk into segments (run per strategy)
uv run data/chunker.py --strategy recursive-500    # default baseline
uv run data/chunker.py --strategy recursive-1000
uv run data/chunker.py --strategy semantic

# All output files are gitignored — regenerate locally before running experiments
```

### Run a variant

```bash
# Basic RAG with default settings
uv run 02_basic_rag/basic_rag.py

# Specify embedder and chunking strategy
uv run 02_basic_rag/basic_rag.py --embedder nomic --strategy recursive-500
uv run 02_basic_rag/basic_rag.py --embedder nomic --strategy recursive-1000
uv run 02_basic_rag/basic_rag.py --embedder nomic --strategy semantic

# Run evaluation
uv run evaluation/ragas_eval.py --variant basic --embedder nomic --strategy recursive-500
uv run evaluation/ragas_eval.py --variant reranking --embedder qwen3
```

---

## CI Thresholds (planned)

```
faithfulness:       >= 0.70
answer_relevancy:   >= 0.75
context_precision:  >= 0.65
```

PRs that cause scores to drop below these thresholds will fail automatically.

---

## References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/rerank-2)
- [Ragas Documentation](https://docs.ragas.io)
- [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/hybrid-queries/)
