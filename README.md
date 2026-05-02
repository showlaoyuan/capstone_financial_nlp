# capstone_financial_nlp

## Project Overview

This repository contains a Financial NLP capstone project prepared for a graduation project and graduate school AI/NLP portfolio.

The project began as a small-scale Week 1 prototype before mentor feedback and was later revised into a more rigorous 4-dataset Financial NLP evaluation workflow.

## Current Repository State

- Old Week 1 prototype files and scripts are preserved in `archive/week1_before_mentor_feedback/`.
- The original prototype scripts were archived to avoid confusion with the revised workflow.
- The current revised scripts will be added later after the project is rerun in PyCharm and the outputs are verified.
- Revised data and results will be added later after the PyCharm rerun and verification process.

## Revised Project Direction

The revised direction is a 4-dataset Financial NLP evaluation workflow using around 5000 samples.

The current intended focus includes:

- dataset validation
- experiment input checking
- relabeling
- cross-checking
- experiment logs
- cost and latency tracking
- result organization

## Dataset List

- FPB
- FiQA-SA
- FinQA
- financial-mmlu-ko

See [docs/dataset_description.md](docs/dataset_description.md) for the current dataset overview and verification notes.

## My Contribution

Role: `Data & Evaluation Contributor`

Main responsibilities:

- FPB / FiQA / FinQA / financial-mmlu-ko dataset organization
- unified format conversion
- column and label validation
- experiment input file checking
- manual relabeling of error samples
- cross-checking with teammate
- experiment log collection
- token/cost/latency tracking
- result table and graph preparation
- final submission material preparation

See [docs/my_contribution.md](docs/my_contribution.md) for more detail.

## Project Structure

```text
capstone_financial_nlp/
|-- AGENTS.md
|-- app/
|   \-- api/
|-- archive/
|   \-- week1_before_mentor_feedback/
|       |-- README.md
|       |-- week1_dataset_inspection.md
|       |-- app/
|       |   \-- api/
|       \-- data/
|           \-- processed/
|-- data/
|   |-- metadata/
|   |-- processed/
|   |-- raw/
|   \-- unified/
|-- docs/
|   |-- dataset_description.md
|   |-- my_contribution.md
|   \-- project_revision_history.md
|-- LICENSE
\-- README.md
```

## Version Note

Archived Week 1 prototype files are stored in `archive/week1_before_mentor_feedback/`.
These archived prototype files are kept for traceability and are not final experimental results.

The current revised scripts, revised 5000-sample workflow outputs, and final verified results will be added later after the PyCharm rerun and verification process.
