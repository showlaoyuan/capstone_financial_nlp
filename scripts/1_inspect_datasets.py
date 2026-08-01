import json
from pathlib import Path

import pandas as pd


# ============================================================
# 1. 路径与配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "inspection"

DATASET_DIRS = {
    "fpb": RAW_DIR / "fpb",
    "fiqa_sa": RAW_DIR / "fiqa_sa",
    "finqa": RAW_DIR / "finqa",
    "financial_mmlu_ko": RAW_DIR / "financial_mmlu_ko",
}

DUPLICATE_COLUMNS = [
    "id", "_id", "conversation_id", "sentence", "question", "text"
]


# ============================================================
# 2. 读取数据
# ============================================================

def load_fpb():
    """读取 Financial PhraseBank 50Agree。"""
    path = (
        DATASET_DIRS["fpb"]
        / "extracted"
        / "FinancialPhraseBank-v1.0"
        / "Sentences_50Agree.txt"
    )
    if not path.exists():
        raise FileNotFoundError(f"Cannot find FPB file: {path}")

    records = []
    with path.open("r", encoding="latin-1") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            if "@" not in line:
                print(f"[WARNING] FPB line {line_number} was skipped.")
                continue

            sentence, label = line.rsplit("@", 1)
            records.append({
                "sentence": sentence.strip(),
                "label": label.strip(),
            })

    return {"full": pd.DataFrame(records)}


def load_fiqa_sa():
    """读取 FiQA-SA 的 train、valid、test。"""
    data_dir = DATASET_DIRS["fiqa_sa"] / "data"
    result = {}

    for split in ["train", "valid", "test"]:
        files = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"Cannot find FiQA-SA {split} file in: {data_dir}"
            )
        result[split] = pd.read_parquet(files[0])

    return result


def load_finqa():
    """读取 FinQA 的 train、dev、test。"""
    data_dir = DATASET_DIRS["finqa"] / "dataset"
    result = {}

    for split in ["train", "dev", "test"]:
        path = data_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Cannot find FinQA file: {path}")

        with path.open("r", encoding="utf-8") as file:
            result[split] = pd.DataFrame(json.load(file))

    return result


def load_financial_mmlu_ko():
    """读取 financial-mmlu-ko 的 test。"""
    data_dir = DATASET_DIRS["financial_mmlu_ko"] / "data"
    files = sorted(data_dir.glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Cannot find financial-mmlu-ko test file in: {data_dir}"
        )
    return {"test": pd.read_parquet(files[0])}


def load_all_datasets():
    """加载四个数据集。"""
    loaders = {
        "fpb": load_fpb,
        "fiqa_sa": load_fiqa_sa,
        "finqa": load_finqa,
        "financial_mmlu_ko": load_financial_mmlu_ko,
    }
    all_datasets = {}

    for name, loader in loaders.items():
        try:
            all_datasets[name] = loader()
            print(f"[LOADED] {name}")
        except Exception as error:
            raise RuntimeError(f"Failed to load {name}: {error}") from error

    return all_datasets


# ============================================================
# 3. 控制台检查
# ============================================================

def print_file_list():
    """输出四个数据集目录中的文件。"""
    print("\n" + "=" * 70)
    print("Dataset folders and files")
    print("=" * 70)

    for name, path in DATASET_DIRS.items():
        if not path.exists():
            print(f"\n- {name}: {path} [MISSING]")
            continue

        files = sorted(item for item in path.rglob("*") if item.is_file())
        print(f"\n- {name}: {path} [OK]")
        print(f"  File count: {len(files)}")
        for file_path in files:
            print(f"  - {file_path.relative_to(PROJECT_ROOT)}")


def print_structures(all_datasets):
    """输出各 split 的行数、字段和数据类型。"""
    print("\n" + "=" * 70)
    print("Dataset structure")
    print("=" * 70)

    for name, splits in all_datasets.items():
        for split, dataframe in splits.items():
            print(f"\nDataset: {name}")
            print(f"Split: {split}")
            print(f"Rows: {len(dataframe)}")
            print(f"Columns: {list(dataframe.columns)}")
            print("Data types:")
            for column, dtype in dataframe.dtypes.items():
                print(f"  - {column}: {dtype}")


# ============================================================
# 4. Summary 与空值
# ============================================================

def build_summary(all_datasets):
    """生成数据集规模与字段汇总。"""
    records = []

    for name, splits in all_datasets.items():
        for split, dataframe in splits.items():
            records.append({
                "dataset": name,
                "split": split,
                "row_count": len(dataframe),
                "column_count": len(dataframe.columns),
                "columns": " | ".join(map(str, dataframe.columns)),
                "memory_mb": round(
                    dataframe.memory_usage(deep=True).sum() / 1024 / 1024,
                    4,
                ),
            })

    return pd.DataFrame(records)


def build_missing_report(all_datasets):
    """统计顶层字段的 null 与空字符串。"""
    records = []

    for name, splits in all_datasets.items():
        for split, dataframe in splits.items():
            for column in dataframe.columns:
                series = dataframe[column]
                null_count = int(series.isna().sum())
                blank_count = int(series.map(
                    lambda value: isinstance(value, str) and not value.strip()
                ).sum())
                total = null_count + blank_count

                records.append({
                    "dataset": name,
                    "split": split,
                    "column": column,
                    "row_count": len(dataframe),
                    "null_count": null_count,
                    "blank_string_count": blank_count,
                    "total_missing": total,
                    "missing_rate_percent": round(
                        total / len(dataframe) * 100 if len(dataframe) else 0,
                        2,
                    ),
                })

    return pd.DataFrame(records)


# ============================================================
# 5. 重复值
# ============================================================

def normalize_value(value):
    """把复杂对象转换成可比较文本。"""
    if value is None:
        return "<NULL>"

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str
        )

    if not pd.api.types.is_scalar(value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str
        )

    if pd.isna(value):
        return "<NULL>"

    return str(value).strip()


def record_duplicate_details(
    records,
    dataframe,
    mask,
    dataset,
    split,
    check_type,
    column,
    values,
):
    """保存重复行的索引与内容预览。"""
    for position, duplicated in enumerate(mask.tolist()):
        if duplicated:
            records.append({
                "dataset": dataset,
                "split": split,
                "check_type": check_type,
                "column": column,
                "row_index": str(dataframe.index[position]),
                "value_preview": str(values.iloc[position])[:300],
            })


def build_duplicate_reports(all_datasets):
    """检查完整重复行及常见 ID、文本字段重复。"""
    summaries = []
    details = []

    for dataset, splits in all_datasets.items():
        for split, dataframe in splits.items():
            normalized = dataframe.copy()
            for column in normalized.columns:
                normalized[column] = normalized[column].map(normalize_value)

            signatures = normalized.apply(
                lambda row: " || ".join(
                    f"{column}={row[column]}" for column in normalized.columns
                ),
                axis=1,
            )
            full_mask = signatures.duplicated(keep=False)

            summaries.append({
                "dataset": dataset,
                "split": split,
                "check_type": "full_row",
                "column": "ALL_COLUMNS",
                "duplicate_row_count": int(full_mask.sum()),
                "duplicate_group_count": int(
                    signatures[full_mask].nunique()
                ),
            })
            record_duplicate_details(
                details,
                dataframe,
                full_mask,
                dataset,
                split,
                "full_row",
                "ALL_COLUMNS",
                signatures,
            )

            for column in DUPLICATE_COLUMNS:
                if column not in dataframe.columns:
                    continue

                values = dataframe[column].map(normalize_value)
                mask = (
                    ~values.isin(["", "<NULL>"])
                    & values.duplicated(keep=False)
                )

                summaries.append({
                    "dataset": dataset,
                    "split": split,
                    "check_type": "column",
                    "column": column,
                    "duplicate_row_count": int(mask.sum()),
                    "duplicate_group_count": int(values[mask].nunique()),
                })
                record_duplicate_details(
                    details,
                    dataframe,
                    mask,
                    dataset,
                    split,
                    "column",
                    column,
                    values,
                )

    summary = pd.DataFrame(summaries).sort_values(
        ["dataset", "split", "check_type", "column"]
    )
    detail = pd.DataFrame(
        details,
        columns=[
            "dataset",
            "split",
            "check_type",
            "column",
            "row_index",
            "value_preview",
        ],
    )
    return summary, detail


# ============================================================
# 6. 标签与答案分布
# ============================================================

def fiqa_label(score):
    """把 FiQA score 转成检查用情感标签。"""
    score = pd.to_numeric(score, errors="coerce")
    if pd.isna(score):
        return "missing"
    if score < 0:
        return "negative"
    if score > 0:
        return "positive"
    return "neutral"


def finqa_answer_type(qa):
    """把 FinQA 答案归为 numeric、yes_no、text 等类型。"""
    if not isinstance(qa, dict):
        return "unparsed"

    answer = qa.get("answer")
    if answer is None or not str(answer).strip():
        return "missing"

    text = str(answer).strip()
    if text.lower() in {"yes", "no"}:
        return "yes_no"

    cleaned = (
        text.replace(",", "").replace("$", "").replace("%", "").strip()
    )
    try:
        float(cleaned)
        return "numeric"
    except ValueError:
        return "text"


def mmlu_answer(conversations):
    """从 conversations 中提取 assistant/gpt 答案。"""
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()

    if isinstance(conversations, str):
        try:
            conversations = json.loads(conversations)
        except json.JSONDecodeError:
            return "unparsed"

    if not isinstance(conversations, (list, tuple)):
        return "unparsed"

    for turn in conversations:
        if not isinstance(turn, dict):
            continue

        role = str(turn.get("from", turn.get("role", ""))).lower()
        value = turn.get("value", turn.get("content"))

        if role in {"gpt", "assistant"}:
            if value is None or not str(value).strip():
                return "missing"
            return str(value).strip()

    return "unparsed"


def add_distribution(records, dataset, split, name, series):
    """统计一个 Series 中各值的数量和比例。"""
    series = series.map(
        lambda value: (
            "<MISSING>"
            if value is None or not str(value).strip()
            else str(value).strip()
        )
    )
    total = len(series)

    for value, count in series.value_counts(dropna=False).items():
        records.append({
            "dataset": dataset,
            "split": split,
            "distribution_name": name,
            "value": value,
            "count": int(count),
            "percentage": round(count / total * 100 if total else 0, 2),
        })


def build_distributions(all_datasets):
    """生成四个数据集对应的标签或答案分布。"""
    records = []

    add_distribution(
        records,
        "fpb",
        "full",
        "label",
        all_datasets["fpb"]["full"]["label"],
    )

    for split, dataframe in all_datasets["fiqa_sa"].items():
        add_distribution(
            records,
            "fiqa_sa",
            split,
            "inspection_sentiment_label",
            dataframe["score"].map(fiqa_label),
        )

    for split, dataframe in all_datasets["finqa"].items():
        add_distribution(
            records,
            "finqa",
            split,
            "answer_type",
            dataframe["qa"].map(finqa_answer_type),
        )

    mmlu = all_datasets["financial_mmlu_ko"]["test"]
    add_distribution(
        records,
        "financial_mmlu_ko",
        "test",
        "answer",
        mmlu["conversations"].map(mmlu_answer),
    )

    return pd.DataFrame(records).sort_values(
        ["dataset", "split", "distribution_name", "value"]
    ).reset_index(drop=True)


# ============================================================
# 7. 保存 CSV 与 Markdown 报告
# ============================================================

def save_csv(dataframe, filename):
    """保存 UTF-8 CSV。"""
    path = OUTPUT_DIR / filename
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {path.relative_to(PROJECT_ROOT)}")


def markdown_table(dataframe, columns):
    """将 DataFrame 转为简单 Markdown 表格。"""
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in dataframe.iterrows():
        values = [
            str(row[column]).replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def save_inspect_report(summary, missing, duplicate_summary, distributions):
    """生成 inspect_report.md。"""
    path = OUTPUT_DIR / "inspect_report.md"
    total_rows = int(summary["row_count"].sum())
    total_missing = int(missing["total_missing"].sum())
    duplicate_only = duplicate_summary[
        duplicate_summary["duplicate_row_count"] > 0
    ]

    lines = [
        "# Dataset Inspection Report",
        "",
        "## 1. Inspection scope",
        "",
        "This report inspects FPB, FiQA-SA, FinQA, and financial-mmlu-ko.",
        "",
        "- Raw files were read only.",
        "- No raw rows were deleted or modified.",
        "- Dataset conversion was not performed.",
        "",
        "## 2. Dataset summary",
        "",
        f"Total rows across all splits: **{total_rows:,}**",
        "",
        markdown_table(
            summary,
            ["dataset", "split", "row_count", "column_count", "columns"],
        ),
        "",
        "## 3. Missing values",
        "",
        f"Total detected top-level missing values: **{total_missing}**",
        "",
    ]

    if total_missing == 0:
        lines.append(
            "No null values or blank strings were detected "
            "in the inspected top-level columns."
        )
    else:
        lines.append(markdown_table(
            missing[missing["total_missing"] > 0],
            [
                "dataset",
                "split",
                "column",
                "total_missing",
                "missing_rate_percent",
            ],
        ))

    lines.extend(["", "## 4. Duplicate checks", ""])

    if duplicate_only.empty:
        lines.append("No duplicate rows or duplicate key values were detected.")
    else:
        lines.append(markdown_table(
            duplicate_only,
            [
                "dataset",
                "split",
                "check_type",
                "column",
                "duplicate_row_count",
                "duplicate_group_count",
            ],
        ))

    lines.extend([
        "",
        "Duplicate findings remain unchanged in raw data.",
        "",
        "## 5. Label and answer distributions",
        "",
        markdown_table(
            distributions,
            [
                "dataset",
                "split",
                "distribution_name",
                "value",
                "count",
                "percentage",
            ],
        ),
        "",
        "## 6. Day 2 conclusion",
        "",
        "- Four raw datasets were loaded successfully.",
        f"- The inspected total is {total_rows:,} rows.",
        "- Fields, types, splits, missing values, duplicates, and "
        "task-aware distributions were checked.",
        "- Raw data remains unchanged.",
        "- Next: dataset conversion and stable canonical ID generation.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {path.relative_to(PROJECT_ROOT)}")


# ============================================================
# 8. 主程序
# ============================================================

def main():
    """运行第 2 天完整检查流程。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Dataset Inspection")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    print_file_list()

    print("\n" + "=" * 70)
    print("Loading datasets")
    print("=" * 70)
    all_datasets = load_all_datasets()

    print_structures(all_datasets)

    print("\n" + "=" * 70)
    print("Saving inspection reports")
    print("=" * 70)

    summary = build_summary(all_datasets)
    missing = build_missing_report(all_datasets)
    duplicate_summary, duplicate_records = build_duplicate_reports(
        all_datasets
    )
    distributions = build_distributions(all_datasets)

    save_csv(summary, "dataset_summary.csv")
    save_csv(missing, "missing_values.csv")
    save_csv(duplicate_summary, "duplicate_summary.csv")
    save_csv(duplicate_records, "duplicate_records.csv")
    save_csv(distributions, "label_distribution.csv")
    save_inspect_report(
        summary,
        missing,
        duplicate_summary,
        distributions,
    )

    print("\n" + "=" * 70)
    print("Inspection completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()