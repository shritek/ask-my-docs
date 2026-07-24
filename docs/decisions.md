# Architecture and Technical Decision Log

This log records decisions that materially affect the architecture,
experiments, reproducibility, operations, or maintainability of Ask My Docs.
The initial entries are retrospective summaries of work completed before the
log was introduced on 2026-07-23.

Decision statuses:

- **Accepted**: current project direction.
- **Provisional**: current baseline that still requires evaluation.
- **Planned**: agreed direction that has not been implemented.
- **Superseded**: retained for history but replaced by a later decision.

## Template for future decisions

Copy this structure to the bottom of the log and assign the next decision ID:

```markdown
## DNNN — Short decision title

- **Status:** Accepted | Provisional | Planned | Superseded by DNNN
- **Recorded:** YYYY-MM-DD

### Context

What problem, constraint, or uncertainty required a decision?

### Decision

What approach was selected?

### Consequences

- What becomes easier or harder?
- What tradeoffs or operational requirements follow?

### Follow-up

What remains unresolved or needs evaluation?
```

---

## D001 — Use controlled, sequential RAG experiments

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

Chunking, retrieval architecture, embedding models, prompts, and LLMs can all
change RAG quality. Changing multiple variables together makes improvements
difficult to attribute.

### Decision

Evaluate one major variable at a time. Select a chunking strategy first, then
hold it constant while comparing retrieval architectures, and finally compare
embedding models using the selected chunking and retrieval approach.

### Consequences

- Each experiment has a clear independent variable.
- Winners become fixed inputs to later experiments.
- The project prioritizes evaluation infrastructure before hybrid retrieval or
  reranking.
- Some locally optimal parameters may need revisiting after the architecture
  changes.

### Follow-up

Implement the evaluation pipeline and record raw configurations with every
result so later experiments remain reproducible.

---

## D002 — Snapshot the source corpus separately from chunking

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

Re-scraping FastAPI documentation for every chunking experiment would mix
source-content changes with chunking changes.

### Decision

Scrape FastAPI documentation once into `data/corpus_snapshot.json`. Generate
each chunked corpus independently from that snapshot.

### Consequences

- Chunking strategies operate on the same source content.
- Experiments do not depend on the live website after snapshot creation.
- Generated corpus and index artifacts remain gitignored and must be rebuilt
  locally.
- A new source snapshot represents a new dataset version and should not be
  silently mixed with old results.

---

## D003 — Keep model and storage choices behind factories

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

Embedding models, LLMs, and vector stores are experiment variables. Hardcoding
them inside each RAG implementation would duplicate code and make controlled
comparisons error-prone.

### Decision

Select embedders, LLMs, and vector stores through enums and factory functions.
Pass those choices into RAG variants rather than editing implementation code
between experiment runs.

### Consequences

- Experiment configuration is explicit.
- Retrieval architecture stays decoupled from provider selection.
- Factory branches must be covered by tests and kept compatible with provider
  library APIs.

---

## D004 — Use Chroma for the prototype and keep Qdrant as a later option

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

The first phases need a simple local vector store. Native hybrid retrieval may
later benefit from a more production-oriented engine.

### Decision

Use persisted local Chroma indexes for the Basic RAG and early experiments.
Keep Qdrant as a planned option when hybrid retrieval requirements justify the
additional operational complexity.

### Consequences

- Current development remains local and inexpensive.
- Qdrant integration is not considered complete merely because a factory
  branch exists; it requires an explicit client and collection design before
  use.

---

## D005 — Isolate vector indexes by chunking strategy and embedding model

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

An index originally keyed only by chunking strategy could be reopened with a
different embedding model. That would compare query and document vectors from
different embedding spaces or fail because of dimension mismatch.

### Decision

Include both chunking strategy and embedding model in the Chroma collection
name and persistence path, for example `semantic__nomic`.

### Consequences

- Each chunking/embedder pair gets an independent index.
- Switching embedders cannot silently reuse incompatible document vectors.
- The first use of a new combination must build a new index.

---

## D006 — Make Basic RAG a deterministic retrieve-then-generate baseline

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

The original LangGraph tool-calling loop allowed the LLM to decide whether and
how often to retrieve. That introduced an uncontrolled variable into chunking
and embedding experiments.

### Decision

Basic RAG always performs exactly one vector retrieval followed by exactly one
LLM synthesis call using a fixed grounded prompt. Configure Ollama chat models
with temperature zero.

### Consequences

- Retrieval behavior is consistent across baseline experiments.
- Retrieval and generation can be tested independently.
- Temperature zero reduces sampling variance but does not guarantee
  bit-for-bit identical local inference.
- Agentic retrieval can still be evaluated later as a separate architecture;
  it is not part of the Basic RAG baseline.

---

## D007 — Keep top-k at three as a baseline, not an optimum

- **Status:** Provisional
- **Recorded:** 2026-07-23 (retrospective)

### Context

Basic RAG needs a fixed retrieval depth for controlled experiments. The project
has not yet measured the best number of chunks to pass to generation.

### Decision

Retrieve the three most similar chunks for the initial baseline. Keep `top_k`
configurable internally, but defer a formal retrieval-depth experiment until
the evaluation pipeline and retrieval architecture are established.

### Consequences

- All initial runs receive the same number of chunks.
- Different chunking strategies can still produce different total context
  lengths even with the same `k`.
- A later experiment may compare recall, precision, answer quality, token
  volume, and latency across several `k` values.

---

## D008 — Current recursive chunk sizes are character-based

- **Status:** Provisional
- **Recorded:** 2026-07-23 (retrospective)

### Context

The recursive splitters use `length_function=len`. Therefore the configured
sizes 500 and 1000 represent characters, not tokens. Common industry examples
using 256, 512, or 1024 often refer to tokens and are not directly comparable.

### Decision

Retain the existing character-based strategies for the current baseline and
make the unit explicit. Do not mix a token-based migration into unrelated
chunk-ID or structured-result changes.

### Consequences

- Roughly, 500 characters may be about 100–150 English tokens and 1000
  characters about 200–300 tokens.
- Token-aware chunking remains a possible future experiment or migration.
- Any change to chunk sizing or units requires regenerating corpora and
  indexes and creates a new experimental configuration.

---

## D009 — Use stable, content-addressed chunk IDs

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

Sequential IDs such as `id_42` changed when corpus ordering or earlier chunk
counts changed. The initial indexing path also discarded logical IDs and let
Chroma generate unrelated record IDs.

### Decision

Generate a SHA-256 ID from source URL, chunking strategy, and chunk text. Use
the `chunk_<digest>` value in the chunk record, metadata, and Chroma record ID.
Record the scheme as `sha256-v1` in chunk-corpus metadata.

### Consequences

- Identical chunk identity produces the same ID independent of corpus order.
- Source, strategy, or content changes produce a different ID.
- Retrieval traces can identify and fetch exact stored chunks.
- Legacy or missing IDs are rejected before indexing.
- Changing the ID scheme requires regenerating chunk corpora and indexes.

---

## D010 — Rebuild disposable indexes instead of versioning this migration

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

Adding logical chunk IDs changed stored index records. Existing local indexes
would otherwise be reopened without the new metadata. Index path versioning was
considered but could be mistaken for an embedding-model version.

### Decision

Delete and rebuild the gitignored local vector stores from regenerated chunk
corpora. Keep the established `{chunking_strategy}__{embedding_model}` names.

### Consequences

- The current learning project avoids unnecessary migration infrastructure.
- This approach assumes generated indexes are disposable and reproducible.
- A production deployment should use an index manifest or explicit schema
  migration when old and new indexes must coexist or support rollback.

---

## D011 — Return evaluation-ready structured RAG results

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

Returning or printing only an answer discards the evidence needed to determine
whether a failure came from retrieval or generation.

### Decision

Return an immutable `RAGResult` containing the question, answer, retrieved
contexts, stable chunk IDs, source metadata, retrieval latency, generation
latency, and their sum. Provide `to_dict()` for JSON serialization. Keep CLI
printing outside the programmatic pipeline.

### Consequences

- Ragas and deterministic evaluators can consume the same pipeline directly.
- `contexts`, `retrieved_chunk_ids`, and `sources` are positionally aligned.
- Missing or legacy retrieved chunk IDs fail visibly.
- Source URLs provide human-readable provenance and page-level evaluation;
  chunk IDs provide exact machine-level identity.
- Current total latency is retrieval plus generation and excludes minor prompt
  formatting and orchestration time.

---

## D012 — Use module execution for repository CLIs

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

Running nested files directly can omit the repository root from Python's module
search path and cause imports such as `config` or `helpers` to fail.

### Decision

Document commands using module execution, for example:

```bash
uv run python -m basic_rag.basic_rag
uv run python -m data.chunker
```

### Consequences

- Commands work without setting `PYTHONPATH=.`.
- Repository-root imports resolve consistently from the project root.
- A packaged console entry point may replace these commands later.

---

## D013 — Use a fixed synthetic evaluation set as the starting point

- **Status:** Provisional
- **Recorded:** 2026-07-23 (retrospective)

### Context

Controlled comparisons require the same questions and reference answers across
all configurations.

### Decision

Keep the generated 50-question FastAPI test set fixed while building the first
evaluation pipeline. Retain source URL and title with each pair.

### Consequences

- Every experiment can run against the same questions.
- The set is synthetic rather than fully human-authored gold data.
- Manual review and supporting reference passages or chunk IDs remain follow-up
  work before treating it as a high-confidence benchmark.

---

## D014 — Preserve source URLs alongside chunk IDs

- **Status:** Accepted
- **Recorded:** 2026-07-23 (retrospective)

### Context

Chunk IDs identify exact machine records but do not by themselves show a human
where the information originated.

### Decision

Keep complete source metadata, including `source_url`, in retrieved documents
and structured results even though the chunk ID is also present.

### Consequences

- Applications can display citations and users can verify claims.
- Evaluation can compare expected and retrieved pages without another lookup.
- Chunk IDs and URLs intentionally overlap in purpose: exact record identity
  versus human-readable provenance.

---

## D015 — Track semantic-chunking reproducibility as a known concern

- **Status:** Provisional
- **Recorded:** 2026-07-23 (retrospective)

### Context

Regenerating the same semantic strategy produced 1,952 chunks instead of the
previous 1,926, while both recursive strategy counts remained unchanged.
Semantic chunking also currently depends on the deprecated
`langchain-experimental` package.

### Decision

Treat generated semantic corpora as versioned experimental artifacts whose
metadata and counts must be recorded with results. Do not assume semantic
chunking is reproducible merely because the source snapshot and visible
parameters are unchanged.

### Consequences

- Experiment reports must include corpus generation metadata and chunk count.
- Dependency and model versions may need to be captured or pinned more
  explicitly.
- Replacing the deprecated semantic chunker is future work and should be its
  own evaluated change.
