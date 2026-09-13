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

---

## D016 — Reuse one initialized pipeline and persist deterministic evaluation first

- **Status:** Accepted
- **Recorded:** 2026-07-23

### Context

Evaluating 50 questions by calling the original `run()` function would recreate
the embedder and LLM clients, reopen Chroma, and inspect corpus state for every
question. Starting directly with LLM-judged metrics would also make it harder
to verify whether the evaluation runner and retrieval behavior were correct.
Long local runs need to survive interruption without starting over.

### Decision

Initialize one `BasicRAGPipeline` per evaluation configuration and reuse it for
all questions. Persist a JSON run file atomically after every question and
resume only when the saved configuration matches. Fingerprint the test set and
chunk corpus with SHA-256 so regenerated inputs cannot be silently mixed into
the same run.

Calculate deterministic page-level metrics before adding Ragas:

- source hit at `k`;
- expected source rank;
- reciprocal rank and aggregate mean reciprocal rank;
- retrieval, generation, and combined latency.

### Consequences

- Expensive clients and the vector store are initialized once per run.
- An interrupted run loses at most the in-flight question.
- Failed cases are recorded and retried when the run resumes.
- Configuration or dataset changes require a new output file or an explicit
  non-resume run.
- Source metrics are cheap and reproducible, but page-level relevance does not
  prove that the exact supporting passage was retrieved.
- Ragas remains a separate next layer for LLM-judged answer and context quality.

### Follow-up

Add Ragas metrics to the persisted raw results without replacing the
deterministic metrics. Manually audit the synthetic references and add
supporting passages or expected chunk IDs where stronger retrieval ground
truth is needed.

---

## D017 — Commit completed official runs together with curated summaries

- **Status:** Accepted
- **Recorded:** 2026-07-24

### Context

The resumable evaluator stores every generated answer, retrieved context, chunk
ID, source, and per-case metric. The first 50-case raw result is about 372 KB,
which is modest at the planned experiment scale. These details let readers
verify a summary and reproduce a manual audit. Because generated chunk corpora
are not committed, the raw result is also the most accessible evidence of what
the retriever supplied during a run.

Temporary and interrupted runs do not provide the same lasting value and would
create source-control noise if committed indiscriminately.

### Decision

- Commit completed raw runs that serve as official baselines or planned
  experiment results under `evaluation/results/`.
- Commit a concise interpretation of each meaningful run or comparison under
  `evaluation/summaries/`.
- Do not commit smoke tests, interrupted checkpoints, debugging runs, or
  duplicate executions. Write those outside the repository, such as under
  `/tmp`.
- Treat committed raw results as immutable evidence. A changed configuration,
  model, benchmark, or corpus produces a distinctly named result rather than
  overwriting an existing official run.
- Keep configuration and input fingerprints in raw results and summaries so
  the evidence remains traceable.

### Consequences

- Readers can inspect both the conclusions and the evidence behind them.
- Manual audit findings remain independently reviewable at the individual-case
  level.
- The repository will grow with official experiments, but the current file size
  and planned experiment grid make that cost acceptable.
- Contributors must distinguish intentional experiment evidence from temporary
  evaluator output before staging files.

---

## D018 — Evaluate immutable RAG runs with resumable Ragas sidecars

- **Status:** Accepted
- **Recorded:** 2026-07-29

### Context

Ragas evaluation is substantially more expensive and less deterministic than
page-level source metrics because it makes several judge-model calls per case.
Re-running retrieval and answer generation while adding judge scores would mix
two sources of variation and waste the completed Basic RAG run. Updating that
committed run in place would also conflict with D017, which treats official raw
runs as immutable evidence.

Ragas 0.4 replaces its legacy dataset metrics with a collections API that can
score individual cases. Its scores depend on the evaluator LLM, evaluator
embeddings, prompts, and library version, so those are part of the experiment
configuration rather than universal ground truth.

### Decision

- Score the four planned metrics: faithfulness, answer relevancy, context
  precision with the reference answer, and context recall with the reference
  answer.
- Read answers and retrieved contexts from a completed deterministic result;
  do not rerun the RAG pipeline.
- Write a separate Ragas sidecar containing the source-result SHA-256, evaluator
  configuration, per-case scores, optional judge reasoning, metric latency, and
  aggregate means.
- Persist atomically after every metric. On resume, retry only failed or missing
  case/metric pairs and reject configuration mismatches.
- Use Ragas 0.4.3 collections metrics through Ollama's OpenAI-compatible API.
  Default to `llama3.1:8b` as the local judge and `nomic-embed-text` for answer
  relevancy, while keeping both explicit in the result configuration.
- Set and record a 4,096-token evaluator output limit. The Ragas default of
  1,024 truncated structured context-recall results on real benchmark cases;
  changing this limit requires a new sidecar rather than mixing evaluator
  configurations on resume.
- Disable Ragas telemetry for this local evaluation workflow.
- Isolate the Ragas 0.4.3 compatibility shim for its import of the removed
  `langchain_community.chat_models.vertexai` module. The sentinel type is safe
  here because this project uses Ollama, not Vertex AI. Remove the shim when an
  upgraded Ragas release fixes the upstream import.

### Consequences

- The same generated answers can be rescored by a different judge without
  changing or duplicating the source run.
- Interruptions lose at most the in-flight metric call, which matters because a
  full run requires several hundred local model operations.
- Comparisons are valid only when the evaluator configuration is held constant.
- A local 8B judge is accessible and reproducible enough for the first learning
  baseline, but it may misunderstand rubric prompts or favor its own style.
  Scores require manual spot checks and should later be calibrated against a
  stronger judge or human labels.
- Ragas metrics complement deterministic source rank and manual audit; they do
  not replace either.

---

## D019 — Calibrate LLM judges against a provenance-tracked challenge set

- **Status:** Accepted
- **Recorded:** 2026-08-21

### Context

The first complete Ragas run was operationally successful but produced results
that contradicted direct inspection. Context precision was effectively 1.0 for
all 50 cases, including known irrelevant retrievals, while clearly grounded
answers sometimes received zero faithfulness. A metric implementation does not
make its LLM judge reliable automatically.

Judge calibration requires examples that distinguish concepts that aggregate
scores can hide. In particular, an answer may be faithful to retrieved text but
irrelevant to the question, and retrieved contexts may rank one useful chunk
first while still omitting most required reference claims.

### Decision

- Maintain a small judge-calibration challenge set separate from RAG tuning and
  release holdout datasets.
- Select deliberately varied cases rather than a representative random sample:
  clear positives, partial evidence, redundant evidence, alternate sources,
  wrong-topic grounded answers, retrieval misses, and appropriate or
  inappropriate abstentions.
- Label atomic generated-answer claims for faithfulness, answer-relevancy bands,
  per-context relevance for average precision, and atomic reference claims for
  context recall.
- Allow AI review against the immutable source evidence without seeing candidate
  judge scores. Record the reviewer, acceptance, review method, and whether human
  verification actually occurred.
- For this straightforward documentation corpus, use Codex-reviewed labels
  accepted by the project owner as `silver` project reference labels. Do not
  describe them as human-authored gold labels or an external benchmark.
- Compare candidate judges with the approved labels using agreement, absolute
  error, catastrophic disagreements, and repeated-run stability. Freeze the
  chosen judge configuration before comparing RAG architectures.

### Consequences

- Ragas scores cannot be used for chunking or retrieval decisions until judge
  calibration is complete.
- Label review is focused on 13 diagnostic cases instead of all 50 cases.
- AI-reviewed silver labels make calibration practical for this learning
  project, but comparison against them measures agreement with the trusted
  reviewer rather than independent human judgment.
- The challenge set tests judge correctness, not expected production query
  distribution, so it must not be reported as an end-user quality estimate.
- The calibration set may evolve when a new metric, judge family, or failure
  mode is introduced, but changes require renewed review and versioning. Human
  verification can promote the label quality tier later without obscuring the
  original provenance.

---

## D020 — Reject `llama3.1:8b` as the frozen Ragas judge

- **Status:** Accepted
- **Recorded:** 2026-08-24

### Context

The complete Ragas sidecar used `llama3.1:8b` as its evaluator. Comparing its
scores with the 13 approved calibration cases produced 18 catastrophic
metric-level disagreements across 9 cases. Numeric disagreements are called
catastrophic when absolute error is at least 0.5. For answer relevancy, a
reviewed high-relevancy answer scoring at most 0.4 or a low-relevancy answer
scoring at least 0.7 is catastrophic.

Context precision was effectively 1.0 for all 13 cases even though the approved
labels range from 0.0 to 1.0. Context recall had mean absolute error 0.5362 and
gave several genuine retrieval misses scores between 0.75 and 1.0.
Faithfulness had mean absolute error 0.4595 and assigned a fully supported
FieldInfo answer a score of 0.0. Answer relevancy separated most high and low
cases, but scored one grounded wrong-topic answer 0.9369 despite its low label.

### Decision

- Do not freeze `llama3.1:8b` or use its Ragas scores to select chunking,
  retrieval, or generation architectures.
- Keep the complete Ragas sidecar and deterministic calibration report as
  diagnostic evidence; an operationally complete evaluation is not the same as
  a trustworthy evaluation.
- Compare future candidate judges on the same 13 cases before paying to score
  all 50 cases. Reuse the immutable Basic RAG answers and contexts rather than
  rerunning the RAG pipeline.
- Link every comparison to both input files by SHA-256 and reject source-run
  mismatches. Report numeric bias, mean and maximum absolute error, score range,
  catastrophic disagreements, and answer-relevancy pairwise band ordering.
- Treat the catastrophic thresholds as triage signals, not acceptance criteria.
  Judge selection still requires inspecting the failure types and repeated-run
  stability.

### Consequences

- The first full Ragas run remains useful for learning and debugging, but its
  aggregate means are not RAG quality baselines.
- A stronger judge must be evaluated next. Only the 13 calibration cases need
  scoring during candidate selection, reducing local runtime substantially.
- Answer relevancy appears more promising than the other three metrics, but the
  wrong-topic failure prevents using it independently without further
  calibration.
- No chunking experiment should begin until a candidate judge passes calibration
  and its complete evaluator configuration is frozen.

---

## D021 — Score candidate judges on exact calibration case IDs

- **Status:** Accepted
- **Recorded:** 2026-08-24

### Context

Judge selection should use the 13 deliberately chosen calibration cases before
running another 50-case evaluation. The existing Ragas runner's `--limit`
option selects a prefix of the Basic RAG results, but the calibration cases are
distributed throughout that run. `--limit 13` would therefore evaluate the
wrong sample. The runner also restricted evaluator names to application LLM
enum values, preventing locally available experimental judges such as
`gemma4:31b-mlx` and `qwen3.6:27b-mlx`.

### Decision

- Accept any non-empty evaluator model name and pass it to the local
  Ollama-compatible endpoint. Application generator enums do not define the
  judge-experiment search space.
- Add `--calibration-set` as mutually exclusive with `--limit`. Load only an
  approved calibration set and select its exact case IDs in calibration order.
- Reject missing or duplicate selected IDs rather than silently shrinking or
  changing the sample.
- Record the ordered IDs, calibration path, and calibration SHA-256 in the
  resumable Ragas configuration.
- Add the selected case count and short calibration fingerprint to the default
  filename so calibration sidecars cannot collide with full-run sidecars.
- Omit selection-only configuration fields for ordinary full or prefix runs,
  preserving compatibility with existing sidecars.

### Consequences

- Candidate judges can be compared using 13 × 4 metric calls instead of 50 × 4
  calls, while reusing the same generated answers and contexts.
- Calibration changes produce a new filename and configuration mismatch instead
  of accidentally resuming an older sample.
- Model-name validation occurs at the local inference endpoint; a typo fails at
  scoring time and remains resumable after correction only when written to a
  correctly configured sidecar.
- A candidate that passes this challenge set still requires repeated-run
  stability checks before its evaluator configuration is frozen.

---

## D022 — Make evaluator reasoning effort explicit

- **Status:** Accepted
- **Recorded:** 2026-08-25

### Context

The first `gemma4:31b-mlx` calibration case ran with the Ollama server's default
thinking behavior. Four metric calls took about 23 minutes, or 347 seconds per
metric on average. Repeating that case with reasoning effort set to `none`
produced the same four scores while reducing average metric latency to about 49
seconds. Leaving thinking behavior implicit also means a server or model default
can change what appears to be the same evaluator configuration.

### Decision

- Add an optional `--evaluator-reasoning-effort` setting with the values
  supported by Ollama's OpenAI-compatible endpoint: `none`, `low`, `medium`,
  `high`, and `max`.
- Pass an explicitly selected value through the Ragas LLM adapter and record it
  in both the sidecar configuration and generated filename.
- Use `reasoning_effort=none` for the Gemma and Qwen calibration experiments.
  This is a calibrated judge setting, not a general claim that reasoning should
  always be disabled.
- Continue omitting the field when no value is supplied so existing Ragas
  sidecars retain their original serialized configuration. Expose this behavior
  as `--evaluator-reasoning-effort server-default` after `none` becomes the CLI
  default, allowing legacy sidecars to remain resumable.

### Consequences

- Candidate comparisons no longer accidentally mix different thinking modes.
- The observed one-case Gemma speedup makes independent repeated runs feasible
  on local hardware, but the repeated full calibration set must still establish
  score stability.
- A future evaluator model or Ragas prompt change requires recalibration even if
  `reasoning_effort=none` remains fixed.

---

## D023 — Freeze Gemma as a qualified local Ragas judge

- **Status:** Accepted
- **Recorded:** 2026-08-25

### Context

The approved 13-case challenge set was scored with `gemma4:31b-mlx` and
`qwen3.6:27b-mlx` under the same Ragas configuration. Gemma had lower
faithfulness MAE (0.1809 versus 0.2314), lower context-precision MAE (0.1603
versus 0.2756), and better answer-relevancy band ordering (0.92 versus 0.89).
Both had context-recall MAE 0.0517. Gemma produced 5 catastrophic metric
disagreements across 5 cases; Qwen produced 7 across 6 cases. Qwen was about
20% faster but had one transient structured-output schema failure that required
resuming the failed metric.

An independent second Gemma pass reproduced all 13 context-precision and
context-recall scores exactly. Twelve of 13 answer-relevancy scores were exact;
the other changed by 0.0176. Twelve of 13 faithfulness scores were exact; case
`4767e7a5` changed from 0.3333 to 0.0. Faithfulness therefore had mean absolute
repeat drift 0.0256 and one material change under the 0.25 diagnostic threshold.

Manual failure analysis also showed that the remaining flags are not
interchangeable. They include a clear lifespan context-precision error,
debatable partial-context boundaries, difficulty scoring an abstention's
absence claim, and a wrong-topic answer that all three tested judges rated as
answer-relevant. The last failure is a Ragas answer-relevancy design limitation,
not evidence that Gemma alone is defective.

### Decision

- Freeze `gemma4:31b-mlx` as the project's local Ragas judge with Ragas 0.4.3,
  `reasoning_effort=none`, `nomic-embed-text` evaluator embeddings, a 4,096-token
  output limit, and answer-relevancy strictness 3.
- Make that judge and reasoning setting the Ragas runner defaults. Keep every
  setting serialized in result sidecars so an explicit CLI override forms a
  distinct evaluator configuration.
- Treat Gemma as the best calibrated candidate available for controlled
  architecture comparisons, not as ground truth and not as a production
  quality gate.
- Interpret context recall as the strongest calibrated metric. Use
  faithfulness and context precision with their documented failure cases.
  Never use answer relevancy alone to detect wrong-topic answers.
- Retain deterministic source metrics, raw evidence, and manual audit alongside
  Ragas scores. Do not relabel the silver challenge cases after observing
  candidate outputs merely to improve agreement.
- Measure repeated-run stability with a deterministic report that rejects
  configuration or case-set mismatches and flags score deltas of at least 0.25.

### Consequences

- Phase 3 judge calibration is complete, so full Basic RAG scoring and chunking
  experiments can proceed with a fixed evaluator configuration.
- Full 50-case Ragas evaluation will be slower than Qwen by roughly 20% on the
  measured local hardware, a deliberate trade for better reference agreement
  and cleaner structured-output behavior.
- Aggregate score changes must be traced back to per-case results before making
  architecture decisions, especially for answer relevancy and partial-context
  cases.
- Judge, Ragas, prompt, embedding, reasoning, strictness, or output-limit
  changes create a new calibration regime and require a fresh comparison before
  results are placed on the same chart.

---

## D024 — Evaluate retrieval with pooled chunk judgments

- **Status:** Accepted
- **Recorded:** 2026-08-27

### Context

The deterministic Basic RAG evaluator knows one expected source URL per
question. Source Hit@K and page-level MRR are useful diagnostics, but a page hit
does not prove that the retrieved chunk contains supporting evidence. The
existing data also cannot produce chunk-level Precision@K, Recall@K, nDCG@K,
or context-sufficiency measurements without relevance judgments.

Chunk IDs are strategy-specific because chunking changes the text boundaries.
Consequently, treating relevant chunks from all strategies as one global recall
denominator would unfairly penalize every strategy for chunks it cannot
retrieve from another strategy's index.

### Decision

- Retrieve the top 10 chunks for every fixed test question from each chunking
  strategy and store the rankings, chunk text, source metadata, input
  fingerprints, and configuration in an immutable candidate-pool artifact.
- Keep chunk-level judgments in a separate qrels artifact linked to the exact
  pool by SHA-256. Split each reference answer into atomic required claims.
- Permit the calibrated Gemma model to generate resumable label proposals, but
  hide retrieval strategy, rank, source URL, and stable chunk ID from its prompt
  and present candidates through short opaque aliases in a stable shuffled
  order. Map aliases back to exact stable IDs only after validation.
- Treat model output as `generated_pending_review`. Require explicit labeling
  provenance and `approved_for_project_evaluation` status before scoring; a
  successful structured model response is not automatically a frozen qrels
  artifact.
- Define a chunk as relevant exactly when it supports at least one required
  reference claim. Record the supported claim IDs so retrieval sufficiency can
  be measured separately from relevance.
- Reject incomplete judgments, unknown claims or chunks, mismatched pool
  fingerprints, and cutoffs deeper than the labeled pool.
- Calculate Hit@K, Precision@K, reciprocal rank@K, binary nDCG@K, claim
  coverage@K, and strategy-local pooled Recall@K at fixed cutoffs.
- Define pooled Recall@K using all relevant chunks in that strategy's labeled
  top-10 ranking as the denominator. Do not report it as exhaustive corpus
  recall.

### Consequences

- Ranking changes can be scored repeatedly without new LLM calls once qrels
  are frozen.
- Precision and nDCG measure context relevance and ordering; claim coverage
  measures whether the retrieved set contains the required answer evidence.
- The candidate pool bounds what can be labeled relevant, so pooled recall may
  overestimate true corpus recall when a strategy misses relevant chunks below
  depth 10.
- Binary relevance avoids subjective relevance grades. Claim coverage retains
  more information about partial versus complete evidence.
- Creating the qrels is a one-time labeling cost and its provenance and review
  status must be recorded. AI-assisted silver labels must not be presented as
  human-authored gold labels.
- Model proposals reduce repetitive labeling work but introduce model bias.
  Hiding system identity reduces direct preference bias; it does not replace
  evidence review or make the labels independent of the proposing model.
- The proposal task uses a distinct prompt and an 8,192-token output limit;
  the model's Ragas calibration does not independently validate these labels.
  Approval follows a targeted audit, not an exhaustive verification of all
  1,500 judgments, so the comparison remains provisional.

### Follow-up

Run Basic RAG and frozen-judge Ragas evaluation for all three strategies before
selecting the Experiment 0 chunking winner. The deterministic retrieval report
currently favors `recursive-1000` at the application's K=3, but that is only
one layer of the final decision.

### Outcome

- Generated 1,500 candidates: 50 questions × 10 chunks × 3 strategies.
- Protocol v1 produced 41 valid cases and nine deterministic long-ID
  transcription failures. Protocol v2 replaced model-facing chunk IDs with
  `d1`–`d30` aliases and recovered all nine; each case records its protocol.
- Preserved the complete raw proposals and applied review through a separate
  fingerprinted manifest and deterministic finalizer.
- Compared 39 semantic top-3 judgments with the existing calibration set. The
  raw proposal agreed on 33. All six differences were inspected; one
  non-atomic lifespan claim was split and corrected, raising agreement to 36.
  The three remaining label differences follow the calibration set's stricter
  claim-level evidence rather than its broader context-precision labels.
- Approved the result as AI-generated, Codex-reviewed silver project labels
  with `human_verified: false`.

---

## D025 — Use recursive-1000 as the Experiment 0 chunking baseline

- **Status:** Provisional
- **Recorded:** 2026-09-09

### Context

Experiment 0 compares the three chunking strategies while holding the Basic RAG
architecture, `nomic-embed-text`, `llama3.1:8b`, K=3, test set, and frozen Gemma
judge constant. The comparison needs one strategy fixed before retrieval
architecture experiments begin.

### Decision

Use `recursive-1000` as the provisional baseline for subsequent experiments.
It leads at K=3 on source hit rate (0.940), source MRR (0.807), chunk claim
coverage (0.665), and the main Ragas metrics: faithfulness (0.834), context
precision (0.633), and context recall (0.710). Its mean total latency is 3.717 s.

Semantic remains a meaningful alternative: it has the highest answer relevancy
(0.787), but its retrieval metrics are lower and mean total latency is 4.341 s.
Recursive-500 trails both on answer quality despite a comparable source hit rate.

### Consequences

- The next retrieval-architecture experiment can vary hybrid search or reranking
  while holding chunking at `recursive-1000`.
- The baseline favors answer faithfulness and evidence coverage over the small
  semantic answer-relevancy advantage.
- Results are specific to this 50-question synthetic FastAPI workload and the
  frozen local judge. They do not establish a universal chunk-size rule.
- The baseline remains provisional because qrels are silver labels with targeted
  review and the test questions do not represent the full variation of user
  traffic.

### Follow-up

Run the retrieval-architecture comparison using `recursive-1000`, then revisit
the choice if real-user queries, a human-verified holdout, or a changed corpus
materially alters the evidence.

### Evidence

See `evaluation/summaries/experiment-0-chunking-strategy.md` and the three Basic
RAG plus three frozen-Gemma Ragas result sidecars under `evaluation/results/`.

---

## D026 — Establish a BM25 and vector rank-fusion baseline

- **Status:** Accepted
- **Recorded:** 2026-09-10

### Context

The next controlled experiment varies retrieval architecture. The vector-only
baseline can miss exact technical identifiers. We need a lexical retrieval arm
without changing chunking, generation, or the test set while PR #14 is reviewed.

### Decision

- Build BM25Okapi (`rank-bm25==0.2.2`, k1=1.5, b=0.75, epsilon=0.25) once in
  memory from the same recursive-1000 corpus used by the existing Chroma index.
  Case-fold and tokenize with Unicode word matching, preserving underscores.
  Do not stem or remove stopwords in this initial baseline.
- Retrieve up to 10 candidates per arm, use equal-weight reciprocal-rank fusion
  with c=60, and pass the best 3 unique chunks to the existing generation path.
  These are initial settings, not measured optima. Scores are not normalized or
  mixed directly. Deduplicate and break score ties by stable chunk ID.
- Exclude chunks with no lexical token overlap from the BM25 ranking, even if
  the library returns a full list of zero-scoring documents. Preserve genuine
  matches with zero or negative scores, which Okapi can produce in small corpora.
- Implement a small explicit fusion function instead of the roadmap's proposed
  `EnsembleRetriever`. This makes ID validation, deterministic ties, duplicate
  handling, zero-weight behavior, and corpus/index mismatch errors explicit
  without adding a direct dependency on the legacy LangChain retriever package.
- Reuse the Basic RAG prompt, result schema, generation model, evaluator, and
  checkpoint machinery. Add an opt-in hybrid variant and serialize retrieval
  settings in its configuration; include their hash in default output filenames.
  Preserve existing Basic RAG configuration serialization and filenames.
- Validate corpus content-addressed IDs at startup and each vector candidate
  against the canonical corpus before generation. Do not insert fusion scores
  into source metadata: doing so would also change the generation prompt.

### Consequences

- Hybrid retrieval requires no vector-index migration or additional service.
  BM25 startup time and memory scale with corpus size; query timing excludes
  startup, as in the existing Basic RAG runner.
- Unknown lexical queries fall back to vector evidence. The default still
  requires both retrieval components to be available; errors are not silently
  converted into a different architecture.
- A small custom fusion function carries a maintenance cost covered by tests
  for ties, duplicates, weights, candidate limits, and identity conflicts.
- Old qrels cannot automatically score new hybrid candidates. Preserve the old
  pool and labels and judge additional candidates for future chunk-level comparisons.

### Follow-up

Compare the first hybrid generation run with the recursive-1000 Basic baseline,
then use the frozen Gemma judge before claiming a quality improvement. Do not
tune weights against page hit rate alone. Extend the candidate judgments when
needed, then proceed to a separately evaluated reranking step. The missing full
semantic Ragas raw artifact remains an Experiment 0 reproducibility limitation;
this branch does not reconstruct those lost per-case scores.
