from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FULL_DIR = PROJECT_ROOT / "data" / "evaluation" / "full"
VALIDATION_CSV = (
    PROJECT_ROOT / "results" / "validation" / "validation_errors.csv"
)
SUMMARY_CSV = (
    PROJECT_ROOT / "results" / "evaluation" / "full_eval_summary.csv"
)
FORMAT_DOC = PROJECT_ROOT / "docs" / "full_evaluation_format.md"

CONFIG = {
    "fpb": (
        PROCESSED_DIR / "fpb" / "fpb_processed.jsonl",
        FULL_DIR / "fpb_full_eval.jsonl",
        4846,
    ),
    "fiqa_sa": (
        PROCESSED_DIR / "fiqa_sa" / "fiqa_sa_processed.jsonl",
        FULL_DIR / "fiqa_sa_full_eval.jsonl",
        1173,
    ),
    "finqa": (
        PROCESSED_DIR / "finqa" / "finqa_processed.jsonl",
        FULL_DIR / "finqa_full_eval.jsonl",
        8281,
    ),
    "financial_mmlu_ko": (
        PROCESSED_DIR
        / "financial_mmlu_ko"
        / "financial_mmlu_ko_processed.jsonl",
        FULL_DIR / "financial_mmlu_ko_full_eval.jsonl",
        455,
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{path}: blank line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}: invalid JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(
                    f"{path}: line {line_number} is not a JSON object"
                )
            records.append(record)
    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_day4_pass() -> None:
    """没有第4天结果，或者仍存在 ERROR，就不允许进入第5天。"""
    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"Day 4 result not found: {VALIDATION_CSV}"
        )

    with VALIDATION_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        issues = list(csv.DictReader(file))

    errors = [
        row
        for row in issues
        if str(row.get("severity", "")).upper() == "ERROR"
    ]
    if errors:
        raise RuntimeError(
            f"Day 4 still has {len(errors)} ERROR item(s)."
        )


def join_text(value) -> str:
    if not isinstance(value, list):
        return ""
    return "\n".join(
        str(item).strip()
        for item in value
        if item is not None and str(item).strip()
    )


def table_to_markdown(table) -> str:
    if not isinstance(table, list) or not table:
        return "(no table)"

    rows = [row if isinstance(row, list) else [row] for row in table]
    width = max(len(row) for row in rows)

    def clean(value) -> str:
        return (
            "" if value is None else str(value)
        ).replace("|", r"\|").replace("\n", " ").strip()

    rows = [
        [clean(row[index]) if index < len(row) else ""
         for index in range(width)]
        for row in rows
    ]

    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def common_record(record, prompt, reference, extra_metadata=None):
    metadata = {
        "source_id": record.get("source_id"),
        "source": record.get("source"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "task_type": record["task_type"],
        "language": record["language"],
        "split": record["split"],
        "prompt": prompt,
        "reference": reference,
        "metadata": metadata,
    }


def build_fpb(record):
    prompt = (
        "You are evaluating financial sentiment classification.\n"
        "Classify the sentiment of the financial text.\n"
        "Return only one label: negative, neutral, or positive.\n\n"
        f"Text:\n{record['text']}"
    )
    return common_record(record, prompt, {"label": record["label"]})


def build_fiqa_sa(record):
    prompt = (
        "You are evaluating financial sentiment analysis.\n"
        "Classify the sentiment of the financial text.\n"
        "Return only one label: negative, neutral, or positive.\n\n"
        f"Text:\n{record['text']}"
    )
    return common_record(
        record,
        prompt,
        {
            "label": record["label"],
            "score": record.get("score"),
        },
        {
            "aspect": record.get("aspect"),
            "target": record.get("target"),
            "type": record.get("type"),
        },
    )


def build_finqa(record):
    context = record.get("context", {})
    before = join_text(context.get("pre_text", [])) or "(none)"
    after = join_text(context.get("post_text", [])) or "(none)"
    table = table_to_markdown(record.get("table", []))

    prompt = (
        "You are evaluating financial numerical reasoning.\n"
        "Use the report context and table to answer the question.\n"
        "Return only the final answer.\n\n"
        f"Context before table:\n{before}\n\n"
        f"Table:\n{table}\n\n"
        f"Context after table:\n{after}\n\n"
        f"Question:\n{record['question']}"
    )
    return common_record(
        record,
        prompt,
        {
            "answer": record["answer"],
            "program": record.get("program"),
        },
    )


def build_financial_mmlu_ko(record):
    choices = record.get("choices", [])
    choice_text = "\n".join(
        f"{number}. {choice}"
        for number, choice in enumerate(choices, start=1)
    )
    prompt = (
        "다음 금융 객관식 문제에 답하세요.\n"
        "정답의 보기 번호만 출력하세요.\n\n"
        f"문제:\n{record['question']}\n\n"
        f"보기:\n{choice_text}"
    )

    answer = str(record.get("answer", "")).strip()
    answer_text = None
    if answer.isdigit() and 1 <= int(answer) <= len(choices):
        answer_text = choices[int(answer) - 1]

    return common_record(
        record,
        prompt,
        {"answer": answer, "answer_text": answer_text},
        {
            "subject": record.get("subject"),
            "category": record.get("category"),
        },
    )


BUILDERS = {
    "fpb": build_fpb,
    "fiqa_sa": build_fiqa_sa,
    "finqa": build_finqa,
    "financial_mmlu_ko": build_financial_mmlu_ko,
}


def validate_records(dataset, records, expected_rows):
    if len(records) != expected_rows:
        raise ValueError(
            f"{dataset}: expected {expected_rows:,}, "
            f"found {len(records):,}"
        )

    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}: duplicate ID found")

    for number, record in enumerate(records, start=1):
        if not str(record.get("prompt", "")).strip():
            raise ValueError(f"{dataset}: blank prompt at row {number}")
        if not isinstance(record.get("reference"), dict):
            raise TypeError(
                f"{dataset}: reference is not a dictionary at row {number}"
            )
        if not record["reference"]:
            raise ValueError(
                f"{dataset}: blank reference at row {number}"
            )


def save_summary(rows):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset", "rows", "split_counts", "output_file", "sha256"
    ]
    with SUMMARY_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_format_doc():
    FORMAT_DOC.parent.mkdir(parents=True, exist_ok=True)
    FORMAT_DOC.write_text(
        """# Full Evaluation File Format

`data/evaluation/full/` contains model-evaluation inputs generated from
validated processed data.

Common fields:

- `id`: canonical ID
- `dataset`, `task_type`, `language`, `split`: task metadata
- `prompt`: model input
- `reference`: gold answer used for scoring
- `metadata`: source traceability and optional task information

Run again with:

```powershell
python scripts\\4_build_full_eval_files.py
```
""",
        encoding="utf-8",
    )


def main():
    print("=" * 70)
    print("Day 5 - Build Full Evaluation Files")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")

    print("\n[CHECK] Day 4 validation")
    require_day4_pass()
    print("[PASS] errors=0")

    summary = []
    total_rows = 0

    for dataset, (input_path, output_path, expected_rows) in CONFIG.items():
        print(f"\n[START] {dataset}")
        processed = load_jsonl(input_path)
        evaluation = [BUILDERS[dataset](record) for record in processed]

        validate_records(dataset, evaluation, expected_rows)
        save_jsonl(evaluation, output_path)

        split_counts = Counter(record["split"] for record in evaluation)
        summary.append({
            "dataset": dataset,
            "rows": len(evaluation),
            "split_counts": ", ".join(
                f"{key}={value}"
                for key, value in sorted(split_counts.items())
            ),
            "output_file": str(output_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(output_path),
        })
        total_rows += len(evaluation)
        print(
            f"[SAVED] {output_path.relative_to(PROJECT_ROOT)} "
            f"({len(evaluation):,} rows)"
        )

    save_summary(summary)
    save_format_doc()

    expected_total = sum(config[2] for config in CONFIG.values())
    if total_rows != expected_total:
        raise ValueError(
            f"Expected {expected_total:,} total rows, "
            f"found {total_rows:,}"
        )

    print(f"\n[SAVED] {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"[SAVED] {FORMAT_DOC.relative_to(PROJECT_ROOT)}")
    print("=" * 70)
    print(
        f"Day 5 COMPLETE: 4 files, {total_rows:,} total rows."
    )
    print("Next step: Day 6 pilot evaluation sets.")
    print("=" * 70)


if __name__ == "__main__":
    main()