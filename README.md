# Financial NLP Dataset Rebuild and LLM Evaluation

## 1. Project Overview

This project rebuilds and extends the data and evaluation component of a previous team-based Financial NLP capstone project.

The original capstone project involved financial dataset processing, model evaluation, and experiment execution. This independent rebuild focuses on the part for which I was primarily responsible: data preparation, unified format conversion, data validation, evaluation set construction, API-based experiment execution, and result organization.

The rebuilt project improves reproducibility and extends the original work by comparing one Chinese large language model and one Korean large language model:

- DeepSeek V4 Pro
- Solar Pro 3

The purpose is not to claim that one model is universally better. Instead, the project examines how the two models perform across different financial NLP tasks under a controlled pilot evaluation.

## 2. Project Objectives

The main objectives are:

1. Download and inspect four financial NLP datasets.
2. Convert heterogeneous raw datasets into consistent evaluation formats.
3. Validate data quality and record conversion statistics.
4. Construct smoke, pilot, and full evaluation sets.
5. Test alternative prompt designs.
6. Evaluate DeepSeek V4 Pro and Solar Pro 3 using the same dataset samples.
7. Compare answer accuracy, format compliance, latency, and estimated API cost.
8. Preserve scripts, manifests, raw responses, and evaluation summaries for reproducibility.

## 3. Datasets

Four datasets are included in the rebuilt data pipeline.

| Dataset | Primary task | Full records |
|---|---|---:|
| FPB | Financial sentiment classification | 4,846 |
| FiQA-SA | Financial sentiment classification | 1,173 |
| FinQA | Financial numerical reasoning | 8,281 |
| financial-mmlu-ko | Korean financial multiple-choice QA | 455 |
| Total | — | 14,755 |

The full evaluation pool contains 14,755 processed records.

For the API-based pilot comparison, 200 records were selected from each dataset, resulting in 800 evaluation records in total.

## 4. Evaluation Design

The project uses three evaluation scales:

- Smoke evaluation: 20 records per dataset, used to verify the execution pipeline.
- Pilot evaluation: 200 records per dataset, used for the formal model comparison.
- Full evaluation set: all 14,755 processed records, preserved for future evaluation and reproducibility.

The full dataset was processed and validated, but the API model comparison was conducted on the fixed 800-record pilot set to control time and API cost.

Each model was evaluated using its selected prompt configuration:

- DeepSeek V4 Pro: Prompt B
- Solar Pro 3: Prompt A

The same fixed pilot records and deterministic request manifests were used to improve comparability.

## 5. Evaluation Metrics

Three primary correctness metrics are reported:

- Answer Accuracy: whether the parsed answer matches the reference answer.
- Strict Accuracy: whether the answer is correct and the complete response follows the required answer-only format.
- Format Compliance: whether the response contains only the requested label, choice number, or numerical answer.

For FinQA, numerical answers are evaluated using displayed-precision tolerance and percent/decimal equivalence.

Efficiency is evaluated using:

- Average latency
- Median latency
- Estimated API cost

## 6. Pilot Comparison Results

### Overall Results

| Metric | DeepSeek V4 Pro | Solar Pro 3 | Difference |
|---|---:|---:|---:|
| Answer Accuracy | 71.38% | 62.62% | DeepSeek +8.75 pp |
| Strict Accuracy | 65.88% | 58.63% | DeepSeek +7.25 pp |
| Format Compliance | 91.75% | 87.62% | DeepSeek +4.13 pp |

DeepSeek V4 Pro achieved the higher overall accuracy, strict accuracy, and format compliance on the fixed pilot set.

### Answer Accuracy by Dataset

| Dataset | DeepSeek V4 Pro | Solar Pro 3 | Higher result |
|---|---:|---:|---|
| FPB | 65.50% | 72.50% | Solar Pro 3 |
| FiQA-SA | 70.50% | 83.00% | Solar Pro 3 |
| FinQA | 61.00% | 20.00% | DeepSeek V4 Pro |
| financial-mmlu-ko | 88.50% | 75.00% | DeepSeek V4 Pro |

The results indicate task-dependent strengths:

- Solar Pro 3 performed better on the two financial sentiment tasks.
- DeepSeek V4 Pro performed substantially better on FinQA numerical reasoning.
- DeepSeek V4 Pro also achieved higher accuracy on the Korean financial multiple-choice dataset.

These results support a task-dependent interpretation rather than a universal claim that either model is always superior.

### Efficiency Results

| Metric | DeepSeek V4 Pro | Solar Pro 3 | Solar reduction |
|---|---:|---:|---:|
| Average latency | 970.21 ms | 605.10 ms | 37.63% |
| Median latency | 916.47 ms | 406.07 ms | 55.69% |
| Estimated cost | $0.111444 | $0.057363 | 48.53% |

Solar Pro 3 was faster and cheaper in this pilot experiment, while DeepSeek V4 Pro achieved higher overall accuracy.

## 7. Main Findings

The experiment produced three main findings:

1. DeepSeek V4 Pro had higher overall correctness and format compliance.
2. Solar Pro 3 performed better on FPB and FiQA-SA sentiment classification.
3. Solar Pro 3 provided lower latency and lower estimated cost.

Therefore, model selection should depend on the task and deployment requirements:

- DeepSeek V4 Pro may be more suitable when numerical reasoning and overall correctness are prioritized.
- Solar Pro 3 may be more suitable when sentiment classification, response speed, and API cost are prioritized.

## 8. Project Structure

```text
capstone_financial_nlp_rebuild/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── evaluation/
│   │   ├── smoke/
│   │   ├── pilot/
│   │   └── full/
│   └── metadata/
├── scripts/
├── outputs/
│   └── ab_experiments/
├── results/
│   ├── inspection/
│   ├── conversion/
│   ├── validation/
│   └── evaluation/
├── docs/
├── README.md
├── MY_CONTRIBUTIONS.md
├── LIMITATIONS.md
├── requirements.txt
└── .gitignore
```
## 9. Reproducibility

The local project workflow generates and preserves:

- Dataset inspection results
- Unified processed datasets
- Data validation summaries
- Smoke, pilot, and full evaluation files
- Request manifests
- Raw API responses
- Parsed model answers
- Prompt comparison results
- Final model comparison tables and summaries

To keep the GitHub repository lightweight, the large raw datasets, processed datasets, and full evaluation files are excluded from version control. They can be regenerated locally using the included scripts. The repository retains the smoke and pilot evaluation sets, experiment outputs, summaries, and documentation.

API keys must be provided through environment variables and must not be committed to GitHub.

Before reproducing an API experiment, users should review the model provider's current model names, prices, rate limits, and API conditions.

## 10. Limitations

The main limitations include:

- The formal API comparison uses a pilot sample of 800 records rather than all 14,755 records.
- Results are specific to the selected samples, prompts, model versions, decoding settings, and API conditions.
- The four datasets represent different tasks and cannot be treated as a single homogeneous benchmark.
- API latency can be influenced by network conditions and provider load.
- Estimated costs may change when providers update their pricing.
- The comparison does not prove that either model is universally superior.
- Statistical significance testing has not yet been included.

More detailed limitations are provided in `LIMITATIONS.md`.

## 11. Relationship to the Original Capstone Project

This repository is an independent reconstruction and extension of the data and experiment-execution component of the original team capstone project.

It does not claim ownership of the entire original team project. The work documented here focuses on my own area of responsibility and adds a new reproducible comparison between Chinese and Korean large language models.

See `MY_CONTRIBUTIONS.md` for a detailed description of my individual work.

## 12. Future Work

Possible future improvements include:

- Increasing the API evaluation sample size
- Adding repeated runs to measure result stability
- Applying statistical significance testing
- Performing detailed error analysis
- Comparing additional Chinese and Korean models
- Evaluating multilingual prompt robustness
- Adding automated visual reports