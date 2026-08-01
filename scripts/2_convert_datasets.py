from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# 1. 路径与配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUMMARY_DIR = PROJECT_ROOT / "results" / "conversion"

OUTPUTS = {
    "fpb": PROCESSED_DIR / "fpb" / "fpb_processed.jsonl",
    "fiqa_sa": PROCESSED_DIR / "fiqa_sa" / "fiqa_sa_processed.jsonl",
    "finqa": PROCESSED_DIR / "finqa" / "finqa_processed.jsonl",
    "financial_mmlu_ko": (
        PROCESSED_DIR
        / "financial_mmlu_ko"
        / "financial_mmlu_ko_processed.jsonl"
    ),
}


# ============================================================
# 2. 通用工具
# ============================================================

def canonical_id(dataset: str, split: str, number: int) -> str:
    """同一份冻结 raw 重跑时，ID 保持一致。"""
    return f"{dataset}-{split}-{number:06d}"


def to_python(value):
    """把 Parquet 中的数组、NumPy 标量等转成可写入 JSON 的对象。"""
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(key): to_python(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_python(item) for item in value]

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return to_python(value.tolist())

    if pd.api.types.is_scalar(value):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        if hasattr(value, "item"):
            try:
                return value.item()
            except (AttributeError, ValueError):
                pass

        return value

    return str(value)


def clean_text(value):
    """转成去除首尾空白的字符串；空内容返回 None。"""
    value = to_python(value)
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def save_jsonl(records: list[dict], path: Path) -> None:
    """保存为 UTF-8 JSONL，一行一条记录。"""
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


def check_basic(dataset: str, raw_count: int, records: list[dict]) -> None:
    """第 3 天只检查数量与 canonical ID；完整字段验证留到第 4 天。"""
    ids = [record["id"] for record in records]

    if len(records) != raw_count:
        raise ValueError(
            f"{dataset}: raw={raw_count}, processed={len(records)}"
        )

    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}: duplicate canonical ID found.")


def fiqa_label(score):
    """按照 score 的正负转换为三分类情感标签。"""
    numeric = pd.to_numeric(score, errors="coerce")

    if pd.isna(numeric):
        return None
    if numeric < 0:
        return "negative"
    if numeric > 0:
        return "positive"
    return "neutral"


# ============================================================
# 3. FPB 转换
# ============================================================

def convert_fpb():
    path = (
        RAW_DIR
        / "fpb"
        / "extracted"
        / "FinancialPhraseBank-v1.0"
        / "Sentences_50Agree.txt"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    records = []

    with path.open("r", encoding="latin-1") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue
            if "@" not in line:
                raise ValueError(f"FPB line {line_no} has no '@'.")

            sentence, label = line.rsplit("@", 1)
            number = len(records) + 1

            records.append({
                "id": canonical_id("fpb", "full", number),
                "source_id": f"line-{line_no:06d}",
                "dataset": "fpb",
                "task_type": "financial_sentiment_classification",
                "language": "en",
                "split": "full",
                "source": "takala/financial_phrasebank:sentences_50agree",
                "text": sentence.strip(),
                "label": label.strip().lower(),
            })

    return records, len(records)


# ============================================================
# 4. FiQA-SA 转换
# ============================================================

def convert_fiqa_sa():
    data_dir = RAW_DIR / "fiqa_sa" / "data"
    records = []
    raw_count = 0

    for split in ["train", "valid", "test"]:
        files = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"FiQA-SA {split} file not found in {data_dir}"
            )

        dataframe = pd.read_parquet(files[0])
        raw_count += len(dataframe)

        for number, (_, row) in enumerate(
            dataframe.iterrows(),
            start=1,
        ):
            score = to_python(row.get("score"))

            records.append({
                "id": canonical_id("fiqa_sa", split, number),
                "source_id": clean_text(row.get("_id")),
                "dataset": "fiqa_sa",
                "task_type": "financial_sentiment_analysis",
                "language": "en",
                "split": split,
                "source": "TheFinAI/fiqa-sentiment-classification",
                "text": clean_text(row.get("sentence")),
                "score": score,
                "label": fiqa_label(score),
                "aspect": clean_text(row.get("aspect")),
                "target": clean_text(row.get("target")),
                "type": clean_text(row.get("type")),
            })

    return records, raw_count


# ============================================================
# 5. FinQA 转换
# ============================================================

def convert_finqa():
    data_dir = RAW_DIR / "finqa" / "dataset"
    records = []
    raw_count = 0

    split_map = {
        "train": "train",
        "dev": "valid",
        "test": "test",
    }

    for raw_split, split in split_map.items():
        path = data_dir / f"{raw_split}.json"
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            items = json.load(file)

        if not isinstance(items, list):
            raise TypeError(f"{path} must contain a JSON list.")

        raw_count += len(items)

        for number, item in enumerate(items, start=1):
            qa = item.get("qa", {})
            if not isinstance(qa, dict):
                qa = {}

            source_id = (
                clean_text(item.get("id"))
                or clean_text(item.get("filename"))
                or f"{raw_split}-row-{number:06d}"
            )

            program = qa.get(
                "program",
                qa.get(
                    "program_re",
                    item.get("program", item.get("program_re")),
                ),
            )

            finqa_answer = qa.get("exe_ans")

            if finqa_answer is None or (
                isinstance(finqa_answer, str)
                and not finqa_answer.strip()
            ):
                finqa_answer = qa.get(
                    "answer",
                    item.get("answer"),
                )

            records.append({
                "id": canonical_id("finqa", split, number),
                "source_id": source_id,
                "dataset": "finqa",
                "task_type": "financial_numerical_reasoning",
                "language": "en",
                "split": split,
                "source": "czyssrs/FinQA",
                "question": clean_text(
                    qa.get("question", item.get("question"))
                ),
                "context": {
                    "pre_text": to_python(item.get("pre_text", [])),
                    "post_text": to_python(item.get("post_text", [])),
                },
                "table": to_python(item.get("table", [])),
                "answer": to_python(finqa_answer),
                "program": to_python(program),
            })

    return records, raw_count


# ============================================================
# 6. financial-mmlu-ko 转换
# ============================================================

OPTION_RE = re.compile(
    r"^\s*(?:"
    r"[\(\[]?[1-9][\)\]\.\:\-]"
    r"|[\(\[]?[A-Ha-h][\)\]\.\:\-]"
    r"|[①②③④⑤⑥⑦⑧⑨]"
    r")\s*(.+?)\s*$"
)


def conversation_list(value):
    value = to_python(value)

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []

    if not isinstance(value, list):
        return []

    return [turn for turn in value if isinstance(turn, dict)]


def get_turn_text(conversations, roles):
    for turn in conversations:
        role = str(
            turn.get("from", turn.get("role", ""))
        ).strip().lower()

        if role in roles:
            return clean_text(
                turn.get("value", turn.get("content"))
            )

    return None


def split_question_choices(prompt):
    """从题目文本中尽量分离 question 与 choices。"""
    if not prompt:
        return None, []

    lines = [
        line.strip()
        for line in prompt.replace("\r\n", "\n").split("\n")
        if line.strip()
    ]

    question_lines = []
    choices = []
    started = False

    for line in lines:
        match = OPTION_RE.match(line)

        if match:
            started = True
            choices.append(match.group(1).strip())
        elif started and choices:
            choices[-1] = f"{choices[-1]} {line}".strip()
        else:
            question_lines.append(line)

    if not choices:
        return prompt.strip(), []

    question = "\n".join(question_lines).strip()
    return question or prompt.strip(), choices


def convert_financial_mmlu_ko():
    data_dir = RAW_DIR / "financial_mmlu_ko" / "data"
    files = sorted(data_dir.glob("test-*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"financial-mmlu-ko test file not found in {data_dir}"
        )

    dataframe = pd.read_parquet(files[0])
    records = []

    for number, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        conversations = conversation_list(row.get("conversations"))
        prompt = get_turn_text(conversations, {"human", "user"})
        answer = get_turn_text(conversations, {"gpt", "assistant"})
        question, choices = split_question_choices(prompt)

        records.append({
            "id": canonical_id(
                "financial_mmlu_ko",
                "test",
                number,
            ),
            "source_id": clean_text(row.get("conversation_id")),
            "dataset": "financial_mmlu_ko",
            "task_type": "korean_financial_multiple_choice",
            "language": "ko",
            "split": "test",
            "source": "allganize/financial-mmlu-ko",
            "question": question,
            "choices": choices,
            "answer": answer,
            "subject": clean_text(row.get("subject")),
            "category": clean_text(row.get("category")),
        })

    return records, len(dataframe)


# ============================================================
# 7. 主程序
# ============================================================

def main():
    converters = {
        "fpb": convert_fpb,
        "fiqa_sa": convert_fiqa_sa,
        "finqa": convert_finqa,
        "financial_mmlu_ko": convert_financial_mmlu_ko,
    }
    summary = []

    print("=" * 70)
    print("Day 3 - Dataset Conversion")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")

    for dataset, converter in converters.items():
        print(f"\n[START] {dataset}")

        try:
            records, raw_count = converter()
            check_basic(dataset, raw_count, records)
            save_jsonl(records, OUTPUTS[dataset])
        except Exception as error:
            raise RuntimeError(
                f"Failed to convert {dataset}: {error}"
            ) from error

        output_path = OUTPUTS[dataset]
        summary.append({
            "dataset": dataset,
            "raw_rows": raw_count,
            "processed_rows": len(records),
            "unique_ids": True,
            "output_file": str(
                output_path.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
        })

        print(
            f"[OK] {dataset}: {len(records):,} rows -> "
            f"{output_path.relative_to(PROJECT_ROOT)}"
        )

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / "conversion_summary.csv"
    pd.DataFrame(summary).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    total = sum(row["processed_rows"] for row in summary)

    print(f"\n[SAVED] {summary_path.relative_to(PROJECT_ROOT)}")
    print("=" * 70)
    print(f"Conversion completed: {total:,} rows.")
    print("Next step: Day 4 full validation.")
    print("=" * 70)


if __name__ == "__main__":
    main()