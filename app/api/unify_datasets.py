import pandas as pd
import json
import os

# 项目根目录
BASE_DIR = r"E:/capstone_financial_nlp"

# 输入输出目录
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "unified")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. FPB
# =========================
print("Processing FPB...")

fpb_path = os.path.join(RAW_DIR, "fpb", "fpb_raw.csv")
df_fpb = pd.read_csv(fpb_path)

# 统一列名
if "sentiment" in df_fpb.columns:
    df_fpb = df_fpb.rename(columns={"sentiment": "label"})
elif "label" in df_fpb.columns:
    pass
else:
    raise ValueError(f"FPB 中没有找到 sentiment 或 label 列，当前列名: {list(df_fpb.columns)}")

if "sentence" in df_fpb.columns:
    df_fpb = df_fpb.rename(columns={"sentence": "text"})
elif "text" in df_fpb.columns:
    pass
else:
    raise ValueError(f"FPB 中没有找到 sentence 或 text 列，当前列名: {list(df_fpb.columns)}")

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
df_fpb["label"] = df_fpb["label"].map(label_map)

df_fpb["id"] = ["fpb_" + str(i) for i in range(len(df_fpb))]
df_fpb["source_dataset"] = "FPB"

df_fpb_unified = df_fpb[["id", "text", "label", "source_dataset"]]
df_fpb_unified.to_csv(os.path.join(OUTPUT_DIR, "fpb_unified.csv"), index=False, encoding="utf-8-sig")

print("FPB done.")
print(df_fpb_unified.head(3))
print(df_fpb_unified["label"].value_counts())
print()

# =========================
# 2. FiQA
# =========================
print("Processing FiQA...")

fiqa_train_path = os.path.join(RAW_DIR, "fiqa", "fiqa_train.csv")
fiqa_test_path = os.path.join(RAW_DIR, "fiqa", "fiqa_test.csv")
fiqa_val_path = os.path.join(RAW_DIR, "fiqa", "fiqa_validation.csv")

df_fiqa_train = pd.read_csv(fiqa_train_path)
df_fiqa_test = pd.read_csv(fiqa_test_path)
df_fiqa_val = pd.read_csv(fiqa_val_path)

df_fiqa = pd.concat([df_fiqa_train, df_fiqa_test, df_fiqa_val], ignore_index=True)

def map_fiqa_label(score):
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

if "id" in df_fiqa.columns:
    df_fiqa["id"] = df_fiqa["id"].astype(str)
elif "_id" in df_fiqa.columns:
    df_fiqa["id"] = df_fiqa["_id"].astype(str)
else:
    df_fiqa["id"] = ["fiqa_" + str(i) for i in range(len(df_fiqa))]

if "text" in df_fiqa.columns:
    df_fiqa["text"] = df_fiqa["text"]
elif "sentence" in df_fiqa.columns:
    df_fiqa["text"] = df_fiqa["sentence"]
else:
    raise ValueError(f"FiQA 中没有找到 text 或 sentence 列，当前列名: {list(df_fiqa.columns)}")

if "sentiment_score" in df_fiqa.columns:
    df_fiqa["label"] = df_fiqa["sentiment_score"].apply(map_fiqa_label)
elif "score" in df_fiqa.columns:
    df_fiqa["label"] = df_fiqa["score"].apply(map_fiqa_label)
elif "label" in df_fiqa.columns:
    df_fiqa["label"] = df_fiqa["label"]
else:
    raise ValueError(f"FiQA 中没有找到 sentiment_score / score / label 列，当前列名: {list(df_fiqa.columns)}")

df_fiqa["source_dataset"] = "FiQA"

df_fiqa_unified = df_fiqa[["id", "text", "label", "source_dataset"]]
df_fiqa_unified.to_csv(os.path.join(OUTPUT_DIR, "fiqa_unified.csv"), index=False, encoding="utf-8-sig")

print("FiQA done.")
print(df_fiqa_unified.head(3))
print()

# =========================
# 3. FinQA
# =========================
print("Processing FinQA...")

finqa_path = os.path.join(RAW_DIR, "finqa", "train.json")

with open(finqa_path, "r", encoding="utf-8") as f:
    finqa_data = json.load(f)

finqa_rows = []

for i, item in enumerate(finqa_data):
    question = str(item.get("qa", {}).get("question", "")).strip()

    pre_text = item.get("pre_text", "")
    post_text = item.get("post_text", "")

    if isinstance(pre_text, list):
        pre_text = " ".join(pre_text)
    if isinstance(post_text, list):
        post_text = " ".join(post_text)

    context = (str(pre_text) + " " + str(post_text)).strip()
    answer = item.get("qa", {}).get("answer", None)

    if answer is not None:
        answer = str(answer).strip()

    text = f"Question: {question} Context: {context}"

    finqa_rows.append({
        "id": f"finqa_{i}",
        "text": text,
        "label": answer,
        "source_dataset": "FinQA"
    })

df_finqa_unified = pd.DataFrame(finqa_rows)
df_finqa_unified = df_finqa_unified.dropna(subset=["label"])
df_finqa_unified = df_finqa_unified[df_finqa_unified["label"].astype(str).str.strip() != ""]

df_finqa_unified.to_csv(
    os.path.join(OUTPUT_DIR, "finqa_unified.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("FinQA done.")
print(df_finqa_unified.head(3))
print("Remaining rows:", len(df_finqa_unified))
print("Label null count:", df_finqa_unified["label"].isna().sum())
print()

# =========================
# 4. Final Check
# =========================
print("===== Final Check =====")

for file_name in [
    "fpb_unified.csv",
    "fiqa_unified.csv",
    "finqa_unified.csv"
]:
    file_path = os.path.join(OUTPUT_DIR, file_name)
    df = pd.read_csv(file_path)
    print(file_name)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(2))
    print("-" * 50)