from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULT_DIR = PROJECT_ROOT / "results" / "validation"
REPORT_PATH = RESULT_DIR / "validation_report.md"
ERRORS_PATH = RESULT_DIR / "validation_errors.csv"

CONFIG = {
    "fpb": {
        "file": PROCESSED_DIR / "fpb" / "fpb_processed.jsonl",
        "rows": 4846,
        "splits": {"full": 4846},
        "task": "financial_sentiment_classification",
        "language": "en",
        "source": "takala/financial_phrasebank:sentences_50agree",
        "fields": ["id", "source_id", "dataset", "task_type", "language",
                   "split", "source", "text", "label"],
    },
    "fiqa_sa": {
        "file": PROCESSED_DIR / "fiqa_sa" / "fiqa_sa_processed.jsonl",
        "rows": 1173,
        "splits": {"train": 822, "valid": 117, "test": 234},
        "task": "financial_sentiment_analysis",
        "language": "en",
        "source": "TheFinAI/fiqa-sentiment-classification",
        "fields": ["id", "source_id", "dataset", "task_type", "language",
                   "split", "source", "text", "score", "label", "aspect",
                   "target", "type"],
    },
    "finqa": {
        "file": PROCESSED_DIR / "finqa" / "finqa_processed.jsonl",
        "rows": 8281,
        "splits": {"train": 6251, "valid": 883, "test": 1147},
        "task": "financial_numerical_reasoning",
        "language": "en",
        "source": "czyssrs/FinQA",
        "fields": ["id", "source_id", "dataset", "task_type", "language",
                   "split", "source", "question", "context", "table",
                   "answer", "program"],
    },
    "financial_mmlu_ko": {
        "file": PROCESSED_DIR / "financial_mmlu_ko"
        / "financial_mmlu_ko_processed.jsonl",
        "rows": 455,
        "splits": {"test": 455},
        "task": "korean_financial_multiple_choice",
        "language": "ko",
        "source": "allganize/financial-mmlu-ko",
        "fields": ["id", "source_id", "dataset", "task_type", "language",
                   "split", "source", "question", "choices", "answer",
                   "subject", "category"],
    },
}

LABELS = {"negative", "neutral", "positive"}
ISSUE_COLUMNS = ["dataset", "severity", "check", "record_id", "line_number",
                 "field", "message", "observed_value"]
issues = []
stats = {name: {} for name in CONFIG}


# ============================================================
# 通用工具
# ============================================================

def short(value, limit=200):
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(value)
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    return text if len(text) <= limit else text[:limit - 3] + "..."


def add_issue(dataset, severity, check, message, record_id="", line_number="",
              field="", observed_value=""):
    issues.append({
        "dataset": dataset,
        "severity": severity,
        "check": check,
        "record_id": "" if record_id is None else str(record_id),
        "line_number": line_number,
        "field": field,
        "message": message,
        "observed_value": short(observed_value),
    })


def blank(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return not value
    return False


def duplicate_info(values):
    counts = Counter(values)
    groups = sum(count > 1 for count in counts.values())
    rows = sum(count for count in counts.values() if count > 1)
    return rows, groups


def load_jsonl(dataset, path):
    records = []
    if not path.exists():
        add_issue(dataset, "ERROR", "file", f"File not found: {path}")
        return records

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    add_issue(dataset, "ERROR", "jsonl", "Blank line.",
                              line_number=line_number)
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    add_issue(dataset, "ERROR", "jsonl",
                              f"Invalid JSON: {error.msg}",
                              line_number=line_number, observed_value=line)
                    continue
                if not isinstance(record, dict):
                    add_issue(dataset, "ERROR", "jsonl",
                              "Each line must be one JSON object.",
                              line_number=line_number, observed_value=record)
                    continue
                record["_line"] = line_number
                records.append(record)
    except UnicodeDecodeError as error:
        add_issue(dataset, "ERROR", "utf8",
                  f"File is not valid UTF-8: {error}")
    return records


def public_record(record):
    return {key: value for key, value in record.items() if key != "_line"}


def counts_for(dataset):
    related = [item for item in issues
               if item["dataset"] in {dataset, "all"}]
    return (
        sum(item["severity"] == "ERROR" for item in related),
        sum(item["severity"] == "WARNING" for item in related),
    )


# ============================================================
# 公共字段、数量、ID、编码
# ============================================================

def validate_common(dataset, records):
    config = CONFIG[dataset]
    stats[dataset]["rows"] = len(records)

    if len(records) != config["rows"]:
        add_issue(dataset, "ERROR", "row_count",
                  f"Expected {config['rows']:,}, found {len(records):,}.",
                  observed_value=len(records))

    split_counts = Counter(record.get("split") for record in records)
    stats[dataset]["splits"] = dict(split_counts)

    for split, expected in config["splits"].items():
        actual = split_counts.get(split, 0)
        if actual != expected:
            add_issue(dataset, "ERROR", "split_count",
                      f"{split}: expected {expected:,}, found {actual:,}.",
                      field="split", observed_value=actual)

    unexpected = [split for split in split_counts
                  if split not in config["splits"]]
    if unexpected:
        add_issue(dataset, "ERROR", "split_value",
                  "Unexpected split values.", field="split",
                  observed_value=unexpected)

    ids = []
    by_split = defaultdict(list)

    for record in records:
        line = record["_line"]
        rid = record.get("id", "")

        missing = [field for field in config["fields"]
                   if field not in record]
        if missing:
            add_issue(dataset, "ERROR", "required_fields",
                      "Required fields are missing.", rid, line,
                      observed_value=missing)

        constants = {
            "dataset": dataset,
            "task_type": config["task"],
            "language": config["language"],
            "source": config["source"],
        }
        for field, expected in constants.items():
            if record.get(field) != expected:
                add_issue(dataset, "ERROR", "constant_field",
                          f"Expected {field}={expected!r}.", rid, line,
                          field, record.get(field))

        if blank(rid):
            add_issue(dataset, "ERROR", "canonical_id",
                      "Canonical ID is blank.", line_number=line, field="id")
        else:
            ids.append(str(rid))

        if blank(record.get("source_id")):
            add_issue(dataset, "WARNING", "source_id",
                      "source_id is blank.", rid, line, "source_id")

        serialized = json.dumps(public_record(record), ensure_ascii=False)
        if "\ufffd" in serialized or "\x00" in serialized:
            add_issue(dataset, "ERROR", "encoding_content",
                      "Replacement character or NUL found.", rid, line)

        by_split[str(record.get("split"))].append(record)

    duplicate_rows, duplicate_groups = duplicate_info(ids)
    stats[dataset]["duplicate_id_rows"] = duplicate_rows
    if duplicate_groups:
        add_issue(dataset, "ERROR", "canonical_id_unique",
                  f"{duplicate_rows} rows in {duplicate_groups} duplicate "
                  f"ID groups.", field="id")

    for split, grouped in by_split.items():
        if split not in config["splits"]:
            continue
        for number, record in enumerate(grouped, start=1):
            expected_id = f"{dataset}-{split}-{number:06d}"
            if record.get("id") != expected_id:
                add_issue(dataset, "ERROR", "canonical_id_sequence",
                          f"Expected {expected_id!r}.", record.get("id"),
                          record["_line"], "id", record.get("id"))


# ============================================================
# FPB 与 FiQA-SA
# ============================================================

def expected_fiqa_label(score):
    if isinstance(score, bool):
        return None
    try:
        number = float(score)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return "negative" if number < 0 else "positive" if number > 0 else "neutral"


def validate_sentiment(dataset, records):
    texts, source_ids = [], []
    labels = Counter()
    text_labels = defaultdict(set)

    for record in records:
        rid, line = record.get("id"), record["_line"]
        text, label = record.get("text"), record.get("label")

        if not isinstance(text, str) or not text.strip():
            add_issue(dataset, "ERROR", "text",
                      "text must be a non-blank string.",
                      rid, line, "text", text)
        else:
            normalized = text.strip()
            texts.append(normalized)
            text_labels[normalized].add(str(label))

        if label not in LABELS:
            add_issue(dataset, "ERROR", "label",
                      "Invalid sentiment label.",
                      rid, line, "label", label)
        else:
            labels[label] += 1

        if dataset == "fiqa_sa":
            expected = expected_fiqa_label(record.get("score"))
            if expected is None:
                add_issue(dataset, "ERROR", "score",
                          "score is not numeric.", rid, line, "score",
                          record.get("score"))
            elif label != expected:
                add_issue(dataset, "ERROR", "score_label_mapping",
                          f"score should map to {expected!r}.",
                          rid, line, "label",
                          {"score": record.get("score"), "label": label})

        if not blank(record.get("source_id")):
            source_ids.append(str(record.get("source_id")))

    rows, groups = duplicate_info(texts)
    stats[dataset]["duplicate_content_rows"] = rows
    stats[dataset]["duplicate_content_groups"] = groups
    stats[dataset]["labels"] = dict(labels)

    if groups:
        message = "Duplicate text retained."
        if dataset == "fiqa_sa":
            message = "Repeated sentences retained because annotations may differ."
        add_issue(dataset, "WARNING", "duplicate_text",
                  f"{message} {rows} rows in {groups} groups.", field="text")

    conflicts = [(text, sorted(values)) for text, values in text_labels.items()
                 if len(values) > 1]
    if dataset == "fpb" and conflicts:
        add_issue(dataset, "WARNING", "label_conflict",
                  f"{len(conflicts)} duplicate texts have conflicting labels.",
                  field="text,label", observed_value=conflicts[:5])

    if dataset == "fiqa_sa":
        source_rows, source_groups = duplicate_info(source_ids)
        stats[dataset]["duplicate_source_id_rows"] = source_rows
        if source_groups:
            add_issue(dataset, "WARNING", "duplicate_source_id",
                      f"Original _id reused: {source_rows} rows in "
                      f"{source_groups} groups. Canonical IDs remain unique.",
                      field="source_id")


# ============================================================
# FinQA
# ============================================================

def validate_finqa(records):
    questions, source_ids = [], []

    for record in records:
        rid, line = record.get("id"), record["_line"]
        question = record.get("question")
        context = record.get("context")
        table = record.get("table")
        program = record.get("program")

        if not isinstance(question, str) or not question.strip():
            add_issue("finqa", "ERROR", "question",
                      "question must be a non-blank string.",
                      rid, line, "question", question)
        else:
            questions.append(question.strip())

        if not isinstance(context, dict):
            add_issue("finqa", "ERROR", "context",
                      "context must be a dictionary.",
                      rid, line, "context", type(context).__name__)
        else:
            for key in ["pre_text", "post_text"]:
                if not isinstance(context.get(key), list):
                    add_issue("finqa", "ERROR", "context",
                              f"context.{key} must be a list.",
                              rid, line, f"context.{key}", context.get(key))

        valid_table = (
            isinstance(table, list)
            and bool(table)
            and all(isinstance(row, list) for row in table)
        )
        if not valid_table:
            add_issue("finqa", "ERROR", "table",
                      "table must be a non-empty list of rows.",
                      rid, line, "table", table)

        if blank(record.get("answer")):
            add_issue("finqa", "ERROR", "answer",
                      "answer is blank.", rid, line, "answer")

        if blank(program) or not isinstance(program, (list, str)):
            add_issue("finqa", "ERROR", "program",
                      "program must be a non-blank list or string.",
                      rid, line, "program", program)

        if not blank(record.get("source_id")):
            source_ids.append(str(record.get("source_id")))

    rows, groups = duplicate_info(questions)
    stats["finqa"]["duplicate_content_rows"] = rows
    stats["finqa"]["duplicate_content_groups"] = groups
    if groups:
        add_issue("finqa", "WARNING", "duplicate_question",
                  f"{rows} rows in {groups} repeated-question groups.",
                  field="question")

    source_rows, source_groups = duplicate_info(source_ids)
    stats["finqa"]["duplicate_source_id_rows"] = source_rows
    if source_groups:
        add_issue("finqa", "ERROR", "duplicate_source_id",
                  f"{source_rows} rows in {source_groups} duplicate "
                  f"source-ID groups.", field="source_id")


# ============================================================
# financial-mmlu-ko
# ============================================================

HANGUL = re.compile(r"[가-힣]")
INTEGER = re.compile(r"^[1-9]\d*$")


def validate_mmlu(records):
    questions, source_ids = [], []
    choice_counts = Counter()
    missing_subject = missing_category = 0

    for record in records:
        rid, line = record.get("id"), record["_line"]
        question = record.get("question")
        choices = record.get("choices")
        answer = record.get("answer")

        if not isinstance(question, str) or not question.strip():
            add_issue("financial_mmlu_ko", "ERROR", "question",
                      "question must be a non-blank string.",
                      rid, line, "question", question)
        else:
            questions.append(question.strip())

        if not isinstance(choices, list):
            add_issue("financial_mmlu_ko", "ERROR", "choices",
                      "choices must be a list.", rid, line, "choices",
                      type(choices).__name__)
            choices = []
        else:
            choice_counts[len(choices)] += 1
            if len(choices) < 2:
                add_issue("financial_mmlu_ko", "ERROR", "choices_count",
                          "At least two choices are required.",
                          rid, line, "choices", len(choices))
            elif len(choices) not in {4, 5}:
                add_issue("financial_mmlu_ko", "WARNING", "choices_count",
                          "The source normally has 4 or 5 choices.",
                          rid, line, "choices", len(choices))
            if any(not isinstance(choice, str) or not choice.strip()
                   for choice in choices):
                add_issue("financial_mmlu_ko", "ERROR", "choices_content",
                          "Every choice must be a non-blank string.",
                          rid, line, "choices", choices)

        answer_text = "" if answer is None else str(answer).strip()
        if not INTEGER.fullmatch(answer_text):
            add_issue("financial_mmlu_ko", "ERROR", "answer_format",
                      "answer must be an option number such as '1'.",
                      rid, line, "answer", answer)
        elif choices and int(answer_text) > len(choices):
            add_issue("financial_mmlu_ko", "ERROR", "answer_range",
                      "answer is outside the available choice range.",
                      rid, line, "answer",
                      {"answer": answer_text, "choice_count": len(choices)})

        korean_text = " ".join([str(question or "")]
                               + [str(choice) for choice in choices])
        if korean_text and not HANGUL.search(korean_text):
            add_issue("financial_mmlu_ko", "WARNING", "korean_text",
                      "No Hangul found in question or choices.",
                      rid, line, "question,choices")

        missing_subject += blank(record.get("subject"))
        missing_category += blank(record.get("category"))
        if not blank(record.get("source_id")):
            source_ids.append(str(record.get("source_id")))

    rows, groups = duplicate_info(questions)
    stats["financial_mmlu_ko"]["duplicate_content_rows"] = rows
    stats["financial_mmlu_ko"]["duplicate_content_groups"] = groups
    stats["financial_mmlu_ko"]["choice_counts"] = dict(choice_counts)
    stats["financial_mmlu_ko"]["missing_subject"] = missing_subject
    stats["financial_mmlu_ko"]["missing_category"] = missing_category

    if groups:
        add_issue("financial_mmlu_ko", "WARNING", "duplicate_question",
                  f"{rows} rows in {groups} repeated-question groups.",
                  field="question")

    source_rows, source_groups = duplicate_info(source_ids)
    stats["financial_mmlu_ko"]["duplicate_source_id_rows"] = source_rows
    if source_groups:
        add_issue("financial_mmlu_ko", "WARNING", "duplicate_source_id",
                  f"conversation_id reused: {source_rows} rows in "
                  f"{source_groups} groups. Canonical IDs remain unique.",
                  field="source_id")


# ============================================================
# raw → processed 精确一致性
# ============================================================

def find_converter():
    candidates = [
        PROJECT_ROOT / "scripts" / "2_convert_datasets.py",
        PROJECT_ROOT / "scripts" / "02_convert_datasets.py",
    ]
    return next((path for path in candidates if path.exists()), None)


def import_converter(path):
    spec = importlib.util.spec_from_file_location("day3_converter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_conversion(all_records):
    path = find_converter()
    if path is None:
        add_issue("all", "ERROR", "conversion_consistency",
                  "Cannot find 2_convert_datasets.py or 02_convert_datasets.py.")
        for dataset in CONFIG:
            stats[dataset]["conversion_match"] = False
        return

    try:
        module = import_converter(path)
    except Exception as error:
        add_issue("all", "ERROR", "conversion_consistency",
                  f"Cannot import conversion script: {error}")
        for dataset in CONFIG:
            stats[dataset]["conversion_match"] = False
        return

    functions = {
        "fpb": "convert_fpb",
        "fiqa_sa": "convert_fiqa_sa",
        "finqa": "convert_finqa",
        "financial_mmlu_ko": "convert_financial_mmlu_ko",
    }

    for dataset, name in functions.items():
        try:
            regenerated, raw_count = getattr(module, name)()
        except Exception as error:
            stats[dataset]["conversion_match"] = False
            add_issue(dataset, "ERROR", "conversion_consistency",
                      f"Fresh raw conversion failed: {error}")
            continue

        processed = [public_record(record) for record in all_records[dataset]]
        match = processed == regenerated
        stats[dataset]["conversion_match"] = match

        if raw_count != CONFIG[dataset]["rows"]:
            add_issue(dataset, "ERROR", "raw_count",
                      f"Converter read {raw_count:,}; expected "
                      f"{CONFIG[dataset]['rows']:,}.",
                      observed_value=raw_count)

        if not match:
            if len(processed) != len(regenerated):
                detail = {
                    "processed": len(processed),
                    "regenerated": len(regenerated),
                }
                message = "Processed and freshly converted row counts differ."
            else:
                different = [index for index, pair in enumerate(
                    zip(processed, regenerated), start=1
                ) if pair[0] != pair[1]]
                detail = different[:20]
                message = f"{len(different):,} records differ from fresh conversion."

            add_issue(dataset, "ERROR", "conversion_consistency",
                      message, observed_value=detail)


# ============================================================
# 保存 validation_report.md 与 validation_errors.csv
# ============================================================

def mapping_text(mapping):
    return ", ".join(f"{key}: {value:,}" for key, value in mapping.items()) or "-"


def save_outputs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with ERRORS_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ISSUE_COLUMNS)
        writer.writeheader()
        writer.writerows(issues)

    total_errors = sum(item["severity"] == "ERROR" for item in issues)
    total_warnings = sum(item["severity"] == "WARNING" for item in issues)
    overall = ("FAIL" if total_errors else
               "PASS WITH WARNINGS" if total_warnings else "PASS")

    lines = [
        "# Day 4 - Processed Dataset Validation Report",
        "",
        f"- **Overall status:** {overall}",
        f"- **Errors:** {total_errors}",
        f"- **Warnings:** {total_warnings}",
        f"- **Issue details:** `{ERRORS_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Summary",
        "",
        "| Dataset | Rows | Splits | Errors | Warnings | Fresh conversion | Status |",
        "|---|---:|---|---:|---:|---|---|",
    ]

    for dataset in CONFIG:
        errors, warnings = counts_for(dataset)
        status = "FAIL" if errors else "PASS WITH WARNINGS" if warnings else "PASS"
        match = "YES" if stats[dataset].get("conversion_match") else "NO"
        lines.append(
            f"| {dataset} | {stats[dataset].get('rows', 0):,} | "
            f"{mapping_text(stats[dataset].get('splits', {}))} | "
            f"{errors} | {warnings} | {match} | {status} |"
        )

    lines += [
        "",
        "## Checks",
        "",
        "- UTF-8 and JSONL format.",
        "- Total rows and split rows.",
        "- Required fields and fixed metadata.",
        "- Canonical ID uniqueness and sequence.",
        "- Sentiment labels and FiQA score mapping.",
        "- FinQA context, table, answer, and program.",
        "- Korean choices and answer range.",
        "- Duplicate content and reused source IDs.",
        "- Exact comparison with a fresh conversion from frozen raw data.",
        "",
        "## Dataset details",
        "",
    ]

    for dataset, current in stats.items():
        lines += [
            f"### {dataset}",
            "",
            "- Duplicate content rows/groups: "
            f"{current.get('duplicate_content_rows', 0):,}/"
            f"{current.get('duplicate_content_groups', 0):,}",
            "- Duplicate canonical ID rows: "
            f"{current.get('duplicate_id_rows', 0):,}",
        ]
        if "labels" in current:
            lines.append("- Labels: " + mapping_text(current["labels"]))
        if "duplicate_source_id_rows" in current:
            lines.append("- Reused source ID rows: "
                         f"{current['duplicate_source_id_rows']:,}")
        if dataset == "financial_mmlu_ko":
            lines += [
                "- Choice counts: " + mapping_text(current.get("choice_counts", {})),
                "- Missing optional subject/category: "
                f"{current.get('missing_subject', 0):,}/"
                f"{current.get('missing_category', 0):,}",
                "- The frozen source only provides `conversation_id` and "
                "`conversations`; blank subject/category is not an error.",
            ]
        lines.append("")

    lines += ["## Errors and warnings", ""]
    if not issues:
        lines.append("No errors or warnings.")
    else:
        lines += [
            "| Severity | Dataset | Check | Record ID | Message |",
            "|---|---|---|---|---|",
        ]
        for item in issues[:100]:
            message = item["message"].replace("|", "\\|")
            record_id = item["record_id"].replace("|", "\\|") or "-"
            lines.append(
                f"| {item['severity']} | {item['dataset']} | "
                f"{item['check']} | {record_id} | {message} |"
            )
        if len(issues) > 100:
            lines += ["", f"Only the first 100 of {len(issues)} issues are "
                      "shown. See `validation_errors.csv` for all details."]

    lines += [
        "",
        "## Interpretation",
        "",
        "- ERROR: fix it before Day 5.",
        "- WARNING: record and explain it; do not automatically delete data.",
        "- PASS WITH WARNINGS is acceptable when the error count is zero.",
        "",
        "**Next step:** " + (
            "fix errors and rerun Day 4."
            if total_errors
            else "Day 4 is complete; build the full evaluation files."
        ),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("Day 4 - Full Processed Dataset Validation")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")

    all_records = {}

    for dataset in CONFIG:
        print(f"\n[START] {dataset}")
        records = load_jsonl(dataset, CONFIG[dataset]["file"])
        all_records[dataset] = records

        validate_common(dataset, records)
        if dataset in {"fpb", "fiqa_sa"}:
            validate_sentiment(dataset, records)
        elif dataset == "finqa":
            validate_finqa(records)
        else:
            validate_mmlu(records)

        errors, warnings = counts_for(dataset)
        print(f"[CHECKED] {dataset}: {len(records):,} rows, "
              f"errors={errors}, warnings={warnings}")

    print("\n[START] raw -> processed exact consistency")
    validate_conversion(all_records)
    save_outputs()

    errors = sum(item["severity"] == "ERROR" for item in issues)
    warnings = sum(item["severity"] == "WARNING" for item in issues)

    print(f"\n[SAVED] {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[SAVED] {ERRORS_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 70)

    if errors:
        print(f"Validation FAILED: errors={errors}, warnings={warnings}")
        raise SystemExit(1)

    status = "PASSED WITH WARNINGS" if warnings else "PASSED"
    print(f"Validation {status}: errors=0, warnings={warnings}")
    print("Next step: Day 5 full evaluation files.")
    print("=" * 70)


if __name__ == "__main__":
    main()