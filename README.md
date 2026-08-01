# Financial NLP Data and Evaluation

This repository comes from our team Financial NLP capstone project. My role in the team was **Data & Evaluation Contributor**, and this version documents the part I was responsible for: preparing the data, running the experiments, and organizing the results.

I returned to this work after the original capstone so that the data pipeline and evaluation process would be easier to follow and reproduce. I reorganized the files, reran the relevant steps, checked the generated data, and kept the experiment records together. This is not a new standalone project or a claim over the whole team project; it is a cleaned-up and completed version of my own contribution.

## What I worked on

I downloaded and inspected the four datasets, converted their different source formats into a consistent evaluation format, validated the converted records, and recorded the conversion statistics. I then built smoke, pilot, and full evaluation sets, prepared fixed request manifests, tested prompt configurations, ran the API experiments, parsed the responses, and summarized accuracy, response-format, latency, and cost results.

The repository keeps the scripts, pilot samples, request manifests, model outputs, and evaluation summaries needed to trace this process. Large raw and processed datasets are excluded from GitHub, but they can be regenerated with the included scripts. API keys are read from environment variables and are not stored in the repository.

## Data and experiment setup

| Dataset | Task | Full records |
|---|---|---:|
| FPB | Financial sentiment classification | 4,846 |
| FiQA-SA | Financial sentiment classification | 1,173 |
| FinQA | Financial numerical reasoning | 8,281 |
| financial-mmlu-ko | Korean financial multiple-choice QA | 455 |
| **Total** |  | **14,755** |

I processed and validated all 14,755 records. For the API comparison, I used a fixed pilot set of 200 records from each dataset, or 800 records in total. I also used 20 records per dataset as a smoke test before the pilot run. The same pilot records and deterministic request manifests were used for both models.

The two models compared were:

- DeepSeek V4 Pro with Prompt B
- Solar Pro 3 with Prompt A

Answer Accuracy checks whether the parsed answer matches the reference. Strict Accuracy additionally requires the full response to follow the requested answer-only format. Format Compliance checks whether the response contains only the requested label, choice number, or numerical answer. FinQA answers were scored with displayed-precision tolerance and percent/decimal equivalence. I also recorded average latency, median latency, and estimated API cost.

## Results

### Overall pilot results

| Metric | DeepSeek V4 Pro | Solar Pro 3 | Difference |
|---|---:|---:|---:|
| Answer Accuracy | 71.38% | 62.62% | DeepSeek +8.75 pp |
| Strict Accuracy | 65.88% | 58.63% | DeepSeek +7.25 pp |
| Format Compliance | 91.75% | 87.62% | DeepSeek +4.13 pp |

### Answer accuracy by dataset

| Dataset | DeepSeek V4 Pro | Solar Pro 3 | Higher result |
|---|---:|---:|---|
| FPB | 65.50% | 72.50% | Solar Pro 3 |
| FiQA-SA | 70.50% | 83.00% | Solar Pro 3 |
| FinQA | 61.00% | 20.00% | DeepSeek V4 Pro |
| financial-mmlu-ko | 88.50% | 75.00% | DeepSeek V4 Pro |

### Efficiency

| Metric | DeepSeek V4 Pro | Solar Pro 3 | Solar reduction |
|---|---:|---:|---:|
| Average latency | 970.21 ms | 605.10 ms | 37.63% |
| Median latency | 916.47 ms | 406.07 ms | 55.69% |
| Estimated cost | $0.111444 | $0.057363 | 48.53% |

The results were mixed by task. Solar Pro 3 performed better on FPB and FiQA-SA, the two sentiment datasets. DeepSeek V4 Pro performed much better on FinQA numerical reasoning and also scored higher on the Korean financial multiple-choice dataset. Across the full pilot set, DeepSeek V4 Pro had higher answer accuracy, strict accuracy, and format compliance, while Solar Pro 3 was faster and cheaper.

## Limitations

The model comparison is based on the fixed 800-record pilot set rather than all 14,755 processed records. The four datasets cover different tasks, so the combined score should not be read as one uniform benchmark. The results also depend on the selected samples, prompts, model versions, decoding settings, and API conditions used for these runs. Network conditions and provider load can affect latency, and the reported cost is an estimate based on the pricing used at the time of the experiment.

Because of these limits, the results describe this experiment rather than proving that either model is generally better. More detail about my role is in [`MY_CONTRIBUTIONS.md`](MY_CONTRIBUTIONS.md), and the experiment limitations are documented in [`LIMITATIONS.md`](LIMITATIONS.md).
