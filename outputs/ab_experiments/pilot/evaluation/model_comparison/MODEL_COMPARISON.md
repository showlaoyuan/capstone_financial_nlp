# DeepSeek V4 Pro vs Solar Pro 3 — Pilot Comparison

Selected prompts: DeepSeek **B**, Solar **A**.

## Overall results

| Metric | DeepSeek | Solar | Difference (DeepSeek − Solar) | Winner |
|---|---:|---:|---:|---|
| answer_accuracy | 71.38% | 62.62% | +8.75 pp | DeepSeek V4 Pro |
| strict_accuracy | 65.88% | 58.63% | +7.25 pp | DeepSeek V4 Pro |
| format_compliance | 91.75% | 87.62% | +4.13 pp | DeepSeek V4 Pro |

## Answer accuracy by dataset

| Dataset | DeepSeek | Solar | Difference (DeepSeek − Solar) | Winner |
|---|---:|---:|---:|---|
| fpb | 65.50% | 72.50% | -7.00 pp | Solar Pro 3 |
| fiqa_sa | 70.50% | 83.00% | -12.50 pp | Solar Pro 3 |
| finqa | 61.00% | 20.00% | +41.00 pp | DeepSeek V4 Pro |
| financial_mmlu_ko | 88.50% | 75.00% | +13.50 pp | DeepSeek V4 Pro |

## Efficiency

| Metric | DeepSeek | Solar | Solar reduction |
|---|---:|---:|---:|
| Average latency | 970.21 ms | 605.10 ms | 37.63% |
| Median latency | 916.47 ms | 406.07 ms | 55.69% |
| Estimated cost | $0.111444 | $0.057363 | 48.53% |

## Interpretation

DeepSeek V4 Pro is stronger overall, especially on FinQA numeric reasoning and financial_mmlu_ko. Solar Pro 3 is stronger on the two sentiment tasks and is faster and cheaper in this pilot. This supports a task-dependent conclusion rather than a universal claim that one model is always better.

Limitation: The fixed pilot contains 200 records per dataset. The result is specific to this sample, model versions, prompts, decoding settings, API conditions, and the evaluation rules; it is not a universal claim about Chinese or Korean language models.
