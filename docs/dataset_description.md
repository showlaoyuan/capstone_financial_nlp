# Dataset Description

This document describes the current intended 4-dataset project structure.
Only verified information is written as confirmed values.
If a current-version value has not been rechecked in the latest local workflow, it is marked as `TODO: verify after PyCharm rerun`.

## FPB

- Full name: Financial PhraseBank
- Task type: financial sentiment classification
- Current project status: included in the revised 4-dataset workflow
- Current shape: `TODO: verify after PyCharm rerun`
- Current columns: `TODO: verify after PyCharm rerun`
- Current label mapping: `TODO: verify after PyCharm rerun`

Verified Week 1 inspection note:

- Week 1 shape: `(4846, 2)`
- Week 1 columns: `sentence`, `label`
- Week 1 data types:
  - `sentence`: string
  - `label`: int64
- Week 1 label mapping:
  - `0 = negative`
  - `1 = neutral`
  - `2 = positive`

## FiQA-SA

- Task type: financial sentiment analysis / sentiment classification after conversion
- Current project status: included in the revised 4-dataset workflow
- Current dataset structure: `TODO: verify after PyCharm rerun`
- Current shape: `TODO: verify after PyCharm rerun`
- Current columns: `TODO: verify after PyCharm rerun`
- Current score-to-label conversion rule: `TODO: verify after PyCharm rerun`

Verified Week 1 inspection note:

- Week 1 dataset structure:
  - train: 822
  - test: 234
  - valid: 117
- Week 1 train shape: `(822, 6)`
- Week 1 columns:
  - `_id`
  - `sentence`
  - `target`
  - `aspect`
  - `score`
  - `type`
- Week 1 data types:
  - `_id`: string
  - `sentence`: string
  - `target`: string
  - `aspect`: string
  - `score`: float64
  - `type`: string
- Week 1 converted sentiment labels:
  - `0 = negative`
  - `1 = neutral`
  - `2 = positive`
- Week 1 train label distribution after conversion:
  - label 2: 532
  - label 0: 250
  - label 1: 40

## FinQA

- Task type: financial question answering / financial reasoning
- Important note: FinQA is not a sentiment classification dataset.
- Current project status: included in the revised 4-dataset workflow
- Current shape: `TODO: verify after PyCharm rerun`
- Current columns / fields: `TODO: verify after PyCharm rerun`
- Current answer format: `TODO: verify after PyCharm rerun`

## financial-mmlu-ko

- Intended usage: Korean financial knowledge evaluation / multiple-choice QA if applicable
- Current project status: intended for the revised 4-dataset workflow
- Current shape: `TODO: verify after PyCharm rerun`
- Current columns / fields: `TODO: verify after PyCharm rerun`
- Current answer format: `TODO: verify after PyCharm rerun`
