# Dataset Inspection Report

## 1. Inspection scope

This report inspects FPB, FiQA-SA, FinQA, and financial-mmlu-ko.

- Raw files were read only.
- No raw rows were deleted or modified.
- Dataset conversion was not performed.

## 2. Dataset summary

Total rows across all splits: **14,755**

| dataset | split | row_count | column_count | columns |
| --- | --- | --- | --- | --- |
| fpb | full | 4846 | 2 | sentence \| label |
| fiqa_sa | train | 822 | 6 | _id \| sentence \| target \| aspect \| score \| type |
| fiqa_sa | valid | 117 | 6 | _id \| sentence \| target \| aspect \| score \| type |
| fiqa_sa | test | 234 | 6 | _id \| sentence \| target \| aspect \| score \| type |
| finqa | train | 6251 | 11 | pre_text \| post_text \| filename \| table_ori \| table \| qa \| id \| table_retrieved \| text_retrieved \| table_retrieved_all \| text_retrieved_all |
| finqa | dev | 883 | 11 | pre_text \| post_text \| filename \| table_ori \| table \| qa \| id \| table_retrieved \| text_retrieved \| table_retrieved_all \| text_retrieved_all |
| finqa | test | 1147 | 11 | pre_text \| post_text \| filename \| table_ori \| table \| qa \| id \| table_retrieved \| text_retrieved \| table_retrieved_all \| text_retrieved_all |
| financial_mmlu_ko | test | 455 | 2 | conversation_id \| conversations |

## 3. Missing values

Total detected top-level missing values: **0**

No null values or blank strings were detected in the inspected top-level columns.

## 4. Duplicate checks

| dataset | split | check_type | column | duplicate_row_count | duplicate_group_count |
| --- | --- | --- | --- | --- | --- |
| financial_mmlu_ko | test | column | conversation_id | 208 | 104 |
| fiqa_sa | test | column | _id | 8 | 4 |
| fiqa_sa | test | column | sentence | 8 | 4 |
| fiqa_sa | train | column | _id | 100 | 48 |
| fiqa_sa | train | column | sentence | 100 | 48 |
| fiqa_sa | valid | column | _id | 11 | 5 |
| fiqa_sa | valid | column | sentence | 11 | 5 |
| fpb | full | column | sentence | 16 | 8 |
| fpb | full | full_row | ALL_COLUMNS | 12 | 6 |

Duplicate findings remain unchanged in raw data.

## 5. Label and answer distributions

| dataset | split | distribution_name | value | count | percentage |
| --- | --- | --- | --- | --- | --- |
| financial_mmlu_ko | test | answer | 1 | 88 | 19.34 |
| financial_mmlu_ko | test | answer | 2 | 126 | 27.69 |
| financial_mmlu_ko | test | answer | 3 | 117 | 25.71 |
| financial_mmlu_ko | test | answer | 4 | 116 | 25.49 |
| financial_mmlu_ko | test | answer | 5 | 8 | 1.76 |
| finqa | dev | answer_type | missing | 12 | 1.36 |
| finqa | dev | answer_type | numeric | 851 | 96.38 |
| finqa | dev | answer_type | text | 10 | 1.13 |
| finqa | dev | answer_type | yes_no | 10 | 1.13 |
| finqa | test | answer_type | missing | 14 | 1.22 |
| finqa | test | answer_type | numeric | 1100 | 95.9 |
| finqa | test | answer_type | text | 11 | 0.96 |
| finqa | test | answer_type | yes_no | 22 | 1.92 |
| finqa | train | answer_type | missing | 48 | 0.77 |
| finqa | train | answer_type | numeric | 5968 | 95.47 |
| finqa | train | answer_type | text | 115 | 1.84 |
| finqa | train | answer_type | yes_no | 120 | 1.92 |
| fiqa_sa | test | inspection_sentiment_label | negative | 88 | 37.61 |
| fiqa_sa | test | inspection_sentiment_label | neutral | 2 | 0.85 |
| fiqa_sa | test | inspection_sentiment_label | positive | 144 | 61.54 |
| fiqa_sa | train | inspection_sentiment_label | negative | 264 | 32.12 |
| fiqa_sa | train | inspection_sentiment_label | neutral | 12 | 1.46 |
| fiqa_sa | train | inspection_sentiment_label | positive | 546 | 66.42 |
| fiqa_sa | valid | inspection_sentiment_label | negative | 47 | 40.17 |
| fiqa_sa | valid | inspection_sentiment_label | positive | 70 | 59.83 |
| fpb | full | label | negative | 604 | 12.46 |
| fpb | full | label | neutral | 2879 | 59.41 |
| fpb | full | label | positive | 1363 | 28.13 |

## 6. Day 2 conclusion

- Four raw datasets were loaded successfully.
- The inspected total is 14,755 rows.
- Fields, types, splits, missing values, duplicates, and task-aware distributions were checked.
- Raw data remains unchanged.
- Next: dataset conversion and stable canonical ID generation.
