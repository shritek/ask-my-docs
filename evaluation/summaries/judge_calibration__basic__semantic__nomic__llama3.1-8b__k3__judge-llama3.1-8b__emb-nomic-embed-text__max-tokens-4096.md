# Judge calibration — `llama3.1:8b`

## Configuration

| Setting | Value |
| --- | --- |
| Reference set | `basic_semantic_judge_calibration_v1` |
| Reference cases | 13 |
| Label tier | Codex-reviewed `silver`; project-owner accepted; not human-verified |
| Judge | `llama3.1:8b` |
| Evaluator embedding | `nomic-embed-text` |
| Ragas | `0.4.3` |
| Maximum evaluator output | 4,096 tokens |
| Answer-relevancy strictness | 3 |

The report compares scores from the completed Ragas sidecar with the approved
labels. It does not rerun retrieval, generation, or judging. Both inputs point
to Basic RAG source run SHA-256
`72e93b90bef51a2566887c958152ef7ca611bb804c5c3f1fcd417d1e2176439f`.

## Numeric agreement

| Metric | Reference mean | Judge mean | Signed bias | MAE | Maximum error | Catastrophic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Faithfulness | 0.8718 | 0.4636 | -0.4082 | 0.4595 | 1.0000 | 5 |
| Context precision | 0.5833 | 1.0000 | +0.4167 | 0.4167 | 1.0000 | 5 |
| Context recall | 0.4235 | 0.8773 | +0.4538 | 0.5362 | 1.0000 | 7 |

A numeric disagreement is classified as catastrophic when its absolute error is
at least 0.5. There were 18 catastrophic metric-level disagreements affecting 9
of the 13 cases.

## Answer relevancy

| Reference band | Cases | Mean judge score | Judge range |
| --- | ---: | ---: | ---: |
| Low | 7 | 0.1338 | 0.0000–0.9369 |
| Medium | 2 | 0.4597 | 0.0000–0.9194 |
| High | 4 | 0.9355 | 0.8697–0.9810 |

The judge ordered 39 of 50 cross-band pairs correctly, tied 6, and reversed 5;
counting ties as half-correct gives 0.84 pairwise ordering accuracy. One result
was catastrophic: case `a678325c` answered a different question but received
0.9369 despite its reviewed `low` label.

## Interpretation

- Context precision has no discriminatory power in this run: every calibration
  case received effectively the same score of 1.0, including five cases whose
  approved precision is 0.0.
- Context recall is strongly inflated on retrieval misses. Three zero-recall
  cases received 1.0, and no calibration case received less than 0.5.
- Faithfulness substantially under-scores multiple grounded answers. The clear
  FieldInfo case `ff873844` received 0.0 against an approved score of 1.0.
- Answer relevancy separates most clear positives and abstentions, but the
  wrong-topic failure shows that its aggregate performance hides a serious
  error mode.

## Decision

Reject `llama3.1:8b` as the frozen Ragas judge for architecture decisions. Keep
this run as diagnostic evidence, score a stronger candidate on only these 13
cases, and compare it with the same labels before running another full 50-case
judge evaluation.
