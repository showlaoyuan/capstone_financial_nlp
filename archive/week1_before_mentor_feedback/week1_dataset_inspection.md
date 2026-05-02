# Week 1 Dataset Inspection Notes

This document preserves verified Week 1 inspection notes from the early prototype stage before mentor feedback.

## Financial PhraseBank (FPB)

- Shape: `(4846, 2)`
- Columns:
  - `sentence`
  - `label`
- Data types:
  - `sentence`: string
  - `label`: int64
- Label mapping:
  - `0 = negative`
  - `1 = neutral`
  - `2 = positive`
- Task type: financial sentiment classification

## FiQA

- Dataset structure:
  - train: 822
  - test: 234
  - valid: 117
- Train shape: `(822, 6)`
- Columns:
  - `_id`
  - `sentence`
  - `target`
  - `aspect`
  - `score`
  - `type`
- Data types:
  - `_id`: string
  - `sentence`: string
  - `target`: string
  - `aspect`: string
  - `score`: float64
  - `type`: string
- Original label type: sentiment score
- Converted sentiment labels:
  - `0 = negative`
  - `1 = neutral`
  - `2 = positive`
- Train label distribution after conversion:
  - label 2: 532
  - label 0: 250
  - label 1: 40
- Important note: The exact score-to-label conversion rule must be verified from the preprocessing script before final submission.
