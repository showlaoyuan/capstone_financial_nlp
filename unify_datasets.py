import pandas as pd
import json
import os

# 创建输出目录
os.makedirs("data/unified", exist_ok=True)

# =========================
# 1. FPB -> unified
# =========================
print("Processing FPB...")

fpb_path = "data/raw/fpb_raw.csv"
df_fpb = pd.read_csv(fpb_path)

# 检查原始列
print("FPB columns:", df_fpb.columns.tolist())

# 情感标签统一映射
# 你之前的数据里 label 可能已经是数字，也可能是 sentiment 文本
if "sentiment" in df_fpb.columns:
    sentiment_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    df_fpb["label"] = df_fpb["sentiment"].map(sentiment_map)
elif "label" in df_fpb.columns:
    # 如果本来就是数字标签，就直接保留
    pass
else:
    raise ValueError("FPB 数据里没找到 sentiment 或 label 列")

# 统一列名
if "sentence" in df_fpb.columns:
    df_fpb["text"] = df_fpb["sentence"]
elif "text" in df_fpb.columns:
    pass
else:
    raise ValueError("FPB 数据里没找到 sentence 或 text 列")

df_fpb_unified = pd.DataFrame({
    "id": range(len(df_fpb)),
    "text": df_fpb["text"],
    "label": df_fpb["label"],
    "source_dataset": "FPB"
})

fpb_out = "data/unified/fpb_unified.csv"
df_fpb_unified.to_csv(fpb_out, index=False, encoding="utf-8-sig")
print(f"Saved: {fpb_out}")
print(df_fpb_unified.head(), "\n")


# =========================
# 2. FiQA -> unified
# =========================
print("Processing FiQA...")

# 注意：已修改为你新下载的训练集路径
fiqa_path = "data/raw/fiqa_train.csv"
df_fiqa = pd.read_csv(fiqa_path)

print("FiQA columns:", df_fiqa.columns.tolist())

# FiQA 一般有 score，可以把情感分数映射成三分类
# 规则可先这样：
# score < 0  -> negative(0)
# score = 0  -> neutral(1)
# score > 0  -> positive(2)

# 注意：已将 score 修改为 sentiment_score
if "sentiment_score" not in df_fiqa.columns:
    raise ValueError("FiQA 数据里没找到 sentiment_score 列")

def map_fiqa_score_to_label(score):
    if score < 0:
        return 0
    elif score == 0:
        return 1
    else:
        return 2

# 注意：已将 score 修改为 sentiment_score
df_fiqa["label"] = df_fiqa["sentiment_score"].apply(map_fiqa_score_to_label)

if "sentence" in df_fiqa.columns:
    df_fiqa["text"] = df_fiqa["sentence"]
else:
    raise ValueError("FiQA 数据里没找到 sentence 列")

# id 优先用原始 _id，没有的话重新编号
if "_id" in df_fiqa.columns:
    unified_id = df_fiqa["_id"]
else:
    unified_id = range(len(df_fiqa))

df_fiqa_unified = pd.DataFrame({
    "id": unified_id,
    "text": df_fiqa["text"],
    "label": df_fiqa["label"],
    "source_dataset": "FiQA"
})

fiqa_out = "data/unified/fiqa_unified.csv"
df_fiqa_unified.to_csv(fiqa_out, index=False, encoding="utf-8-sig")
print(f"Saved: {fiqa_out}")
print(df_fiqa_unified.head(), "\n")


# =========================
# 3. FinQA -> unified
# =========================
print("Processing FinQA...")

finqa_path = "data/raw/train.json"
with open(finqa_path, "r", encoding="utf-8") as f:
    finqa_data = json.load(f)

print("Number of FinQA samples:", len(finqa_data))

rows = []
for i, item in enumerate(finqa_data):
    question = str(item.get("question", "")).strip()
    context = str(item.get("pre_text", "")).strip()

    # pre_text 有时候是 list，需要拼接
    if isinstance(item.get("pre_text", ""), list):
        context = " ".join(item.get("pre_text", []))

    # 有些 FinQA 还会有 post_text
    post_text = item.get("post_text", "")
    if isinstance(post_text, list):
        post_text = " ".join(post_text)
    else:
        post_text = str(post_text).strip()

    full_context = (context + " " + post_text).strip()
    text = (question + " " + full_context).strip()

    answer = str(item.get("answer", "")).strip()

    rows.append({
        "id": i,
        "text": text,
        "label": answer,
        "source_dataset": "FinQA"
    })

df_finqa_unified = pd.DataFrame(rows)

finqa_out = "data/unified/finqa_unified.csv"
df_finqa_unified.to_csv(finqa_out, index=False, encoding="utf-8-sig")
print(f"Saved: {finqa_out}")
print(df_finqa_unified.head(), "\n")


# =========================
# 4. 简单验证
# =========================
print("===== Validation =====")
for name, df in [
    ("FPB", df_fpb_unified),
    ("FiQA", df_fiqa_unified),
    ("FinQA", df_finqa_unified)
]:
    print(f"\n{name}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3))
    print("Missing values:")
    print(df.isnull().sum())