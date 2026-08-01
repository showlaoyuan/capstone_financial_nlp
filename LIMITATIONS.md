# Project Limitations

## 1. Scope of the API Evaluation

The complete processed evaluation pool contains 14,755 records. However, the formal API-based comparison was conducted on a fixed pilot set of 800 records:

- 200 FPB records
- 200 FiQA-SA records
- 200 FinQA records
- 200 financial-mmlu-ko records

The full dataset was processed, validated, and preserved, but it was not completely evaluated through the paid model APIs because of time and cost constraints.

Therefore, the pilot results should not be interpreted as exact estimates of performance on every record in the full dataset.

## 2. Dataset Differences

The four datasets represent different financial NLP tasks:

- FPB: financial sentiment classification
- FiQA-SA: financial sentiment classification
- FinQA: financial numerical reasoning
- financial-mmlu-ko: Korean financial multiple-choice question answering

Because the task types, languages, answer formats, and difficulty levels differ, their scores are not directly interchangeable.

The combined overall accuracy provides a summary of this pilot experiment, but it should not be treated as the performance of a single homogeneous benchmark.

## 3. Sampling Limitations

The formal comparison used a fixed sample of 200 records from each dataset.

This balanced design gives every dataset the same weight in the overall pilot result, even though the original datasets have different sizes.

For example, FinQA contains substantially more full records than financial-mmlu-ko, but both contribute 200 records to the pilot comparison.

This improves task-level comparability but means that the overall pilot score does not reflect the original dataset-size distribution.

## 4. Model and API Version Dependence

The results are specific to the model versions and API services available at the time of execution.

Model providers may later change:

- Model implementations
- Model names
- Default parameters
- Tokenization
- Safety behavior
- Rate limits
- Pricing
- Server infrastructure

As a result, rerunning the same requests in the future may not produce exactly the same outputs, latency, or estimated cost.

## 5. Prompt Dependence

Model performance is influenced by prompt wording and answer-format instructions.

A controlled prompt A/B experiment was conducted, and the selected configurations were:

- DeepSeek V4 Pro: Prompt B
- Solar Pro 3: Prompt A

Using the strongest observed prompt for each model reflects a practical model-specific configuration. However, it also means that the two models were not evaluated with identical prompt wording in the final comparison.

The results therefore represent the performance of each selected model-and-prompt configuration, rather than the isolated effect of the model alone.

## 6. Output Parsing and Rule-Based Scoring

The evaluation pipeline automatically parses model responses and compares them with reference answers.

This process may be affected by:

- Unexpected explanations
- Additional punctuation
- Alternative label expressions
- Numerical formatting
- Percentage and decimal representation
- Rounding differences
- Invalid or incomplete responses

Rules were added to handle the expected answer formats. For FinQA, displayed-precision tolerance and percent/decimal equivalence were considered.

Nevertheless, automated rules cannot perfectly capture every semantically equivalent answer.

## 7. Reference-Answer Limitations

The evaluation assumes that the reference answers in the processed datasets are correct and sufficiently unambiguous.

Possible source-dataset issues include:

- Annotation errors
- Ambiguous sentiment
- Multiple reasonable interpretations
- Incomplete contextual information
- Differences in financial terminology
- Numerical-answer formatting inconsistencies

This project validates data structure and answer format, but it does not manually re-annotate every source record.

## 8. Language and Cultural Coverage

The comparison is described as a Chinese–Korean model comparison because DeepSeek V4 Pro and Solar Pro 3 were developed by Chinese and Korean organizations respectively.

However, the benchmark itself is not evenly balanced between Chinese and Korean language data:

- FPB, FiQA-SA, and FinQA primarily contain English content.
- financial-mmlu-ko contains Korean financial questions.
- A Chinese-language financial dataset was not included.

Therefore, the experiment compares models from China and Korea, but it is not a complete multilingual comparison of Chinese and Korean financial language understanding.

No conclusion should be drawn about the general superiority of one country's AI systems.

## 9. Latency Measurement Limitations

API response time can be influenced by factors outside the model itself, including:

- Network conditions
- Geographic routing
- Provider server load
- Temporary rate limiting
- Request scheduling
- API infrastructure differences

The measured latency reflects the actual experiment environment, but it is not a controlled hardware benchmark.

Although Solar Pro 3 was faster in this pilot run, the exact latency difference may change under different conditions.

## 10. Cost Estimation Limitations

API cost was estimated using token usage and the applicable pricing assumptions recorded during the experiment.

Actual cost may differ because of:

- Provider pricing updates
- Currency conversion
- Discounts or free credits
- Cached-token pricing
- Billing-unit differences
- Rounding rules
- Unreported provider-side processing

The reported cost should therefore be interpreted as an experiment-specific estimate rather than a permanent price comparison.

## 11. Statistical Limitations

The current comparison reports descriptive metrics, including:

- Answer accuracy
- Strict accuracy
- Format compliance
- Dataset-level accuracy
- Average and median latency
- Estimated cost

The project does not currently include:

- Confidence intervals
- Statistical significance tests
- Bootstrap analysis
- Multiple independent sampling rounds
- Repeated API runs for each record

Consequently, observed differences should be interpreted cautiously, especially when the score gap is small.

## 12. Reproducibility Limitations

The repository preserves processed data, fixed evaluation records, request manifests, responses, scripts, and result summaries.

However, exact reproduction may still depend on:

- Access to the same APIs
- Availability of the same model versions
- Correct environment variables
- Compatible Python dependencies
- Provider-side model updates
- Current API conditions

API keys are intentionally excluded for security reasons. Users reproducing the experiment must provide their own valid credentials and may incur API costs.

## 13. Relationship to the Original Capstone Project

This repository reconstructs and extends the data and experiment-execution area of an earlier team capstone project.

It does not reproduce or claim ownership of every component developed by the original team.

The repository focuses on the work associated with:

- Dataset preparation
- Unified conversion
- Data validation
- Evaluation-set construction
- API experiment execution
- Result organization
- Documentation

The Chinese–Korean model comparison is an independent extension added during the reconstruction.

## 14. Appropriate Interpretation

The results support the following limited conclusions:

- DeepSeek V4 Pro achieved higher overall correctness on this fixed pilot set.
- Solar Pro 3 performed better on the two included sentiment datasets.
- DeepSeek V4 Pro performed better on the included numerical-reasoning and Korean multiple-choice datasets.
- Solar Pro 3 produced lower measured latency and lower estimated cost in this experiment.
- Model strengths differed by task.

The results do not prove that:

- One model is universally better.
- One country's AI technology is generally superior.
- The same ranking will appear on other datasets.
- The same results will appear with different prompts or model versions.
- Pilot performance will exactly match full-dataset performance.

These limitations should be considered whenever the project results are presented, discussed, or reused.