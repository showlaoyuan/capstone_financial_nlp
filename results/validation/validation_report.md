# Day 4 - Processed Dataset Validation Report

- **Overall status:** PASS WITH WARNINGS
- **Errors:** 0
- **Warnings:** 7
- **Issue details:** `results\validation\validation_errors.csv`

## Summary

| Dataset | Rows | Splits | Errors | Warnings | Fresh conversion | Status |
|---|---:|---|---:|---:|---|---|
| fpb | 4,846 | full: 4,846 | 0 | 2 | YES | PASS WITH WARNINGS |
| fiqa_sa | 1,173 | train: 822, valid: 117, test: 234 | 0 | 2 | YES | PASS WITH WARNINGS |
| finqa | 8,281 | train: 6,251, valid: 883, test: 1,147 | 0 | 1 | YES | PASS WITH WARNINGS |
| financial_mmlu_ko | 455 | test: 455 | 0 | 2 | YES | PASS WITH WARNINGS |

## Checks

- UTF-8 and JSONL format.
- Total rows and split rows.
- Required fields and fixed metadata.
- Canonical ID uniqueness and sequence.
- Sentiment labels and FiQA score mapping.
- FinQA context, table, answer, and program.
- Korean choices and answer range.
- Duplicate content and reused source IDs.
- Exact comparison with a fresh conversion from frozen raw data.

## Dataset details

### fpb

- Duplicate content rows/groups: 16/8
- Duplicate canonical ID rows: 0
- Labels: neutral: 2,879, negative: 604, positive: 1,363

### fiqa_sa

- Duplicate content rows/groups: 119/57
- Duplicate canonical ID rows: 0
- Labels: negative: 399, positive: 760, neutral: 14
- Reused source ID rows: 119

### finqa

- Duplicate content rows/groups: 311/148
- Duplicate canonical ID rows: 0
- Reused source ID rows: 0

### financial_mmlu_ko

- Duplicate content rows/groups: 2/1
- Duplicate canonical ID rows: 0
- Reused source ID rows: 208
- Choice counts: 5: 29, 4: 426
- Missing optional subject/category: 455/455
- The frozen source only provides `conversation_id` and `conversations`; blank subject/category is not an error.

## Errors and warnings

| Severity | Dataset | Check | Record ID | Message |
|---|---|---|---|---|
| WARNING | fpb | duplicate_text | - | Duplicate text retained. 16 rows in 8 groups. |
| WARNING | fpb | label_conflict | - | 2 duplicate texts have conflicting labels. |
| WARNING | fiqa_sa | duplicate_text | - | Repeated sentences retained because annotations may differ. 119 rows in 57 groups. |
| WARNING | fiqa_sa | duplicate_source_id | - | Original _id reused: 119 rows in 57 groups. Canonical IDs remain unique. |
| WARNING | finqa | duplicate_question | - | 311 rows in 148 repeated-question groups. |
| WARNING | financial_mmlu_ko | duplicate_question | - | 2 rows in 1 repeated-question groups. |
| WARNING | financial_mmlu_ko | duplicate_source_id | - | conversation_id reused: 208 rows in 104 groups. Canonical IDs remain unique. |

## Interpretation

- ERROR: fix it before Day 5.
- WARNING: record and explain it; do not automatically delete data.
- PASS WITH WARNINGS is acceptable when the error count is zero.

**Next step:** Day 4 is complete; build the full evaluation files.
