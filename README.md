# Production RAG Application — Ask My Docs

A domain-specific retrieval-augmented generation (RAG) system built with progressively advanced retrieval techniques, systematic evaluation, and CI-gated quality gates. This project demonstrates production-grade RAG patterns used in enterprise AI systems.

> **Status:** 🚧 In Progress

---

## Roadmap

### Phase 0 · Data Pipeline (corpus snapshot)
- [ ] Scrape FastAPI docs via `SitemapLoader`
- [ ] Chunk documents with `RecursiveCharacterTextSplitter`
- [ ] Save snapshot to `data/corpus_snapshot.json` (generated once, reused by all experiments)
- [ ] Add PDF and HuggingFace dataset loaders as alternative sources

### Phase 1 · Basic RAG (baseline)
- [ ] Build basic vector retrieval pipeline
- [ ] Establish fixed evaluation test set (25 Q&A pairs from FastAPI docs corpus)
- [ ] Run baseline Ragas scores

### Phase 2 · Hybrid Retrieval
- [ ] Add BM25 retriever alongside vector search
- [ ] Combine via LangChain `EnsembleRetriever`
- [ ] Evaluate against fixed test set and compare to Phase 1

### Phase 3 · Cross-Encoder Reranking
- [ ] Add Cohere Rerank or local `MiniLM-L6-v2`
- [ ] Evaluate against fixed test set and compare to Phase 2

### Phase 4 · Evaluation Framework
- [ ] Formalize Ragas evaluation pipeline
- [ ] Experiment 1: compare retrieval architectures (fixed embedder)
- [ ] Experiment 2: compare embedding models (fixed architecture)
- [ ] Document results with before/after metrics

### Phase 5 · CI Pipeline
- [ ] Add GitHub Actions workflow
- [ ] Gate PRs on Ragas score thresholds
- [ ] Prevent silent retrieval regressions

---

## Project Structure

```
ask-my-docs/
├── README.md
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py                 # Central config — model choices, paths, thresholds
│
├── helpers/
│   ├── embedding_factory.py        # Returns embedder based on config/param
│   ├── llm_factory.py              # Returns LLM based on config/param
│   └── vector_store_factory.py     # Returns vector store based on config/param
│
├── data/
│   ├── loaders/
│   │   ├── sitemap_loader.py       # Web source — FastAPI docs (primary)
│   │   ├── pdf_loader.py           # Offline — PDF files
│   │   └── huggingface_loader.py   # Offline — HuggingFace datasets
│   └── corpus_snapshot.json        # Generated once, not committed to git
│
├── evaluation/
│   ├── test_set.json               # Fixed — never modified after creation
│   └── results/
│       ├── basic_rag.json
│       ├── hybrid_rag.json
│       └── reranked_rag.json
│
├── 01_basic_rag/
│   └── basic_rag.py
├── 02_hybrid_retrieval/
│   └── hybrid_rag.py
├── 03_reranking/
│   └── reranked_rag.py
├── 04_evaluation/
│   └── ragas_eval.py
└── 05_ci_pipeline/
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
python 03_reranking/reranked_rag.py --embedder nomic
python 03_reranking/reranked_rag.py --embedder qwen3
python 03_reranking/reranked_rag.py --embedder openai
```

**Corpus snapshot generated once**

The corpus is scraped, chunked, and saved to disk before any experiment runs. Every RAG variant and every embedder reads from the identical snapshot — eliminating variability from web scraping, chunking randomness, or document ordering between runs.

---

## Experiment Design

> **Principle:** Hold everything constant except the one variable under test. The corpus snapshot and fixed test set ensure all comparisons are fair and reproducible.

### Experiment 1 — Retrieval architecture
Fixed embedder: `nomic-embed-text` · Fixed corpus · Fixed test set · Fixed LLM

| Architecture | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| Basic RAG | — | — | — |
| Hybrid Retrieval | — | — | — |
| Reranking | — | — | — |

### Experiment 2 — Embedding model
Fixed architecture: reranking (best from Experiment 1) · Fixed corpus · Fixed test set · Fixed LLM

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
- [Ollama](https://ollama.ai) installed and running locally
- OpenAI API key (optional — for `text-embedding-3-small` in Experiment 2)
- Cohere API key (optional — for Cohere Rerank in Phase 3)

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/ask-my-docs.git
cd ask-my-docs

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Pull local models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Copy and fill in API keys
cp .env.example .env
```

### Generate corpus snapshot (run once)

```bash
python data/loaders/sitemap_loader.py
# Saves chunked documents to data/corpus_snapshot.json
# This file is not committed to git — regenerate if needed
```

### Run a variant

```bash
# Default embedder (nomic-embed-text)
python 01_basic_rag/basic_rag.py

# Specify embedder explicitly
python 01_basic_rag/basic_rag.py --embedder nomic
python 01_basic_rag/basic_rag.py --embedder qwen3
python 01_basic_rag/basic_rag.py --embedder openai

# Run evaluation against fixed test set
python 04_evaluation/ragas_eval.py --variant reranking --embedder qwen3
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
