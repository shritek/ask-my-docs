# Repository Working Agreement

## Branch and pull-request workflow

- Create a feature branch for code, test, dependency, configuration, or other
  implementation changes.
- Small documentation-only changes may be committed directly to `main`.
- Pull-request descriptions should explain the problem, decision, tradeoffs,
  migration or operational impact, and validation performed.
## Decision log

- Update `docs/decisions.md` whenever work introduces or changes an
  architectural or meaningful technical decision.
- Make the decision-log update in the same branch or change as the decision.
- Record context, the selected approach, rationale, consequences, and any
  unresolved follow-up work.
- Do not silently rewrite past decisions. Add a new entry and mark an older
  entry superseded when direction changes.
- Skip routine implementation details that do not affect design, operations,
  experiments, reproducibility, or maintainability.

## Verification

- Run the relevant unit tests and `git diff --check` before committing.
- For RAG pipeline changes, perform a local Ollama/Chroma smoke test when the
  change affects live retrieval, indexing, prompting, or generation behavior.
