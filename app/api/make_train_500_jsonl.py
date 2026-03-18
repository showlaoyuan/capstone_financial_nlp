import pandas as pd
import json

# 读取数据
fpb_path = "../../data/unified/fpb_train.csv"
fiqa_path = "../../data/unified/fiqa_train.csv"

df_fpb = pd.read_csv(fpb_path)
df_fiqa = pd.read_csv(fiqa_path)

# 统一列名
df_fpb = df_fpb[["text", "label"]].copy()
df_fiqa = df_fiqa[["text", "label"]].copy()

# 合并FPB和FiQA
df_all = pd.concat([df_fpb, df_fiqa], ignore_index=True)

# 选500条
df_500 = df_all.sample(n=500, random_state=42).reset_index(drop=True)

# 转成instruction格式
output_path = "../../data/processed/train_500.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df_500.iterrows():
        item = {
            "instruction": "Classify the sentiment of the following financial text.",
            "input": row["text"],
            "output": row["label"]
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved to {output_path}")
print(f"Total lines: {len(df_500)}")