# Local Ragas judge selection

## Frozen configuration

| Setting | Value |
| --- | --- |
| Judge | `gemma4:31b-mlx` |
| Reasoning effort | `none` |
| Evaluator embedding | `nomic-embed-text` |
| Ragas | `0.4.3` |
| Maximum evaluator output | 4,096 tokens |
| Answer-relevancy strictness | 3 |
| Calibration set | 13 Codex-reviewed `silver` cases; project-owner accepted; not human-verified |

## Candidate agreement

| Metric | Gemma | Qwen | Better |
| --- | ---: | ---: | --- |
| Faithfulness MAE | 0.1809 | 0.2314 | Gemma |
| Context-precision MAE | 0.1603 | 0.2756 | Gemma |
| Context-recall MAE | 0.0517 | 0.0517 | Tie |
| Answer-relevancy pairwise ordering | 0.92 | 0.89 | Gemma |
| Catastrophic metric disagreements | 5 | 7 | Gemma |
| Cases with a catastrophic disagreement | 5 | 6 | Gemma |
| Mean metric latency | 38.45 s | 30.65 s | Qwen |

Gemma was selected because its agreement advantage spans multiple metrics.
Qwen was about 20% faster on the measured local hardware but produced one
transient incorrect structured-output schema; resuming the single failed metric
completed its result.

## Gemma repeatability

| Metric | Mean absolute drift | Maximum drift | Exact matches | Material changes |
| --- | ---: | ---: | ---: | ---: |
| Faithfulness | 0.0256 | 0.3333 | 12/13 | 1 |
| Context precision | 0.0000 | 0.0000 | 13/13 | 0 |
| Context recall | 0.0000 | 0.0000 | 13/13 | 0 |
| Answer relevancy | 0.0014 | 0.0176 | 12/13 | 0 |

A material change is an absolute delta of at least 0.25. The only material
repeat change was faithfulness for case `4767e7a5`, from 0.3333 to 0.0. This
case already has ambiguous partial-evidence boundaries: its approved
faithfulness is 0.6667 while its approved context precision is 0.5833 and
context recall is 0.0.

## Known limitations

- Context recall had the strongest calibration agreement and perfect repeated
  scores in this challenge set.
- Context precision still made a clear error on lifespan case `00592e5d` and is
  sensitive to whether partially useful evidence counts as relevant.
- Faithfulness struggled with an abstention framed as an absence claim and had
  one material repeated-run change.
- Answer relevancy gave wrong-topic case `a678325c` approximately 0.99. The same
  failure appeared with Llama and Qwen, indicating that this metric cannot be
  used alone to detect semantically adjacent wrong-topic answers.
- The labels are project calibration references, not human-authored gold labels
  or a production-quality benchmark.

## Decision

Freeze Gemma for controlled RAG architecture comparisons, accepting its local
runtime cost. Use Ragas as one evidence layer rather than a quality oracle:
retain deterministic retrieval metrics, raw per-case evidence, and manual audit,
and never tune solely to an aggregate Ragas mean.

## Reproducibility artifacts

- [Gemma calibration sidecar](../results/ragas__basic__semantic__nomic__llama3.1-8b__k3__judge-gemma4-31b-mlx__emb-nomic-embed-text__max-tokens-4096__reasoning-none__cases-13-ff02dc99.json)
- [Gemma calibration comparison](../results/judge_calibration__basic__semantic__nomic__llama3.1-8b__k3__judge-gemma4-31b-mlx__emb-nomic-embed-text__max-tokens-4096__reasoning-none__cases-13-ff02dc99.json)
- [Gemma independent repeat](../results/ragas__basic__semantic__nomic__llama3.1-8b__k3__judge-gemma4-31b-mlx__emb-nomic-embed-text__max-tokens-4096__reasoning-none__cases-13-ff02dc99__repeat-2.json)
- [Gemma repeat calibration comparison](../results/judge_calibration__basic__semantic__nomic__llama3.1-8b__k3__judge-gemma4-31b-mlx__emb-nomic-embed-text__max-tokens-4096__reasoning-none__cases-13-ff02dc99__repeat-2.json)
- [Gemma stability comparison](../results/judge_stability__basic__semantic__nomic__llama3.1-8b__k3__judge-gemma4-31b-mlx__emb-nomic-embed-text__max-tokens-4096__reasoning-none__cases-13-ff02dc99__repeat-2.json)
- [Qwen calibration sidecar](../results/ragas__basic__semantic__nomic__llama3.1-8b__k3__judge-qwen3.6-27b-mlx__emb-nomic-embed-text__max-tokens-4096__reasoning-none__cases-13-ff02dc99.json)
- [Qwen calibration comparison](../results/judge_calibration__basic__semantic__nomic__llama3.1-8b__k3__judge-qwen3.6-27b-mlx__emb-nomic-embed-text__max-tokens-4096__reasoning-none__cases-13-ff02dc99.json)
