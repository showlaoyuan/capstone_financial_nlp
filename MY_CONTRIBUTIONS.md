# My Contributions

## 1. Scope of My Work

This repository documents my individual work on the data preparation and experiment-execution component of a Financial NLP capstone project.

In the original team project, my main responsibilities were related to:

- Financial dataset collection
- Dataset structure inspection
- Unified-format conversion
- Data quality checking
- Evaluation dataset preparation
- A/B experiment execution support
- Experiment output organization

This rebuilt repository focuses specifically on these responsibilities. It does not represent the work of the entire original team.

## 2. Dataset Collection and Inspection

I collected and organized four financial NLP datasets:

- Financial PhraseBank (FPB)
- FiQA Sentiment Analysis (FiQA-SA)
- FinQA
- financial-mmlu-ko

Because these datasets use different file structures, fields, labels, and task formats, I inspected each dataset separately before conversion.

The inspection stage included:

- Checking file formats and dataset splits
- Counting records
- Identifying available fields
- Examining label distributions
- Detecting missing or unusual values
- Saving inspection summaries for later verification

## 3. Unified Data Conversion

I converted the four heterogeneous datasets into consistent task-specific formats.

The conversion process included:

- Mapping original dataset fields to standardized fields
- Normalizing text values and labels
- Preserving source dataset and split information
- Creating stable record identifiers
- Handling sentiment classification, numerical reasoning, and multiple-choice tasks separately
- Saving conversion summaries and processed datasets

The goal was not to force every dataset into exactly the same semantic structure. Instead, common metadata was standardized while task-specific information was preserved.

## 4. Data Validation

After conversion, I validated the processed data to check whether:

- Record counts matched the expected totals
- Required fields were present
- Record identifiers were unique
- Labels and answer formats were valid
- Missing values were handled correctly
- Dataset and split metadata were preserved
- Converted records remained traceable to their source data

Validation reports were saved so that the data-processing results could be reviewed without rerunning every script.

## 5. Evaluation Set Construction

I prepared three evaluation scales:

- Smoke set: 20 records per dataset
- Pilot set: 200 records per dataset
- Full set: all 14,755 processed records

The smoke set was used to test whether the API execution and evaluation pipeline worked correctly.

The fixed pilot set was used for the formal model comparison. Using the same records for both models reduced sampling differences and improved comparability.

The full evaluation set was preserved for future experiments, although it was not completely evaluated through paid APIs because of cost and time constraints.

## 6. Prompt Experimentation

I conducted a controlled prompt A/B experiment to compare alternative answer instructions.

The experiment examined:

- Answer correctness
- Strict answer correctness
- Output-format compliance
- Differences between datasets
- Whether a prompt produced unnecessary explanatory text

Based on the prompt experiment results, the selected configurations were:

- DeepSeek V4 Pro: Prompt B
- Solar Pro 3: Prompt A

These selected prompts were then used in the formal pilot comparison.

## 7. Chinese–Korean Model Comparison

I extended the original data and evaluation work by comparing:

- DeepSeek V4 Pro, representing a Chinese-developed model
- Solar Pro 3, representing a Korean-developed model

Both models were evaluated on the same fixed pilot set of 800 records.

The comparison covered:

- Answer accuracy
- Strict accuracy
- Format compliance
- Dataset-level performance
- Average and median response latency
- Estimated API cost

The purpose was to examine task-dependent differences between the two models, not to claim that either country or model is universally superior.

## 8. Experiment Execution and Result Organization

I executed the API evaluation workflow and preserved the main experiment artifacts, including:

- Request manifests
- Raw API responses
- Parsed answers
- Per-record evaluation results
- Prompt comparison results
- Model-level summaries
- Dataset-level comparison tables
- Latency and estimated-cost statistics
- Markdown, CSV, and JSON reports

This organization makes it possible to inspect both the final metrics and the intermediate experiment records.

## 9. Documentation

I prepared documentation describing:

- Dataset structures
- Conversion and validation logic
- Evaluation-set construction
- Prompt experiment design
- Model comparison methodology
- Main results
- Reproducibility conditions
- Project limitations

The documentation separates verified results from interpretation and records the limits of the experiment.

## 10. Tools and Technical Skills Applied

The main tools and technical concepts applied in this work include:

- Python
- pandas
- JSON and JSONL processing
- CSV data processing
- pathlib-based file management
- Dataset schema mapping
- Data validation
- Stratified or controlled sampling
- API request execution
- Environment-variable management
- Response parsing
- Rule-based evaluation
- Experiment logging
- Markdown documentation
- Git and GitHub project organization

## 11. Relationship to the Original Team Project

This repository is an independent reconstruction and extension of the area for which I was responsible in the original capstone project.

It does not claim that I developed every component of the original team system.

My contribution is specifically centered on:

- Data
- Evaluation-set preparation
- Experiment execution
- Result verification
- Reproducible documentation

The rebuilt version improves this part by processing the complete dataset collection, preserving intermediate artifacts, and adding a reproducible Chinese–Korean model comparison.

## 12. What I Learned

Through this reconstruction, I developed a clearer understanding of:

- Why datasets must be inspected before conversion
- How different NLP tasks require different schemas
- How to preserve traceability during data processing
- Why fixed samples are important for fair comparison
- How prompts affect both correctness and output format
- Why accuracy, latency, and cost should be evaluated together
- How API outputs can be parsed and scored automatically
- How experimental limitations affect the interpretation of results
- How to organize a project so that other people can understand and reproduce it

This project also helped me move from simply obtaining experimental outputs to understanding the complete data and evaluation workflow.