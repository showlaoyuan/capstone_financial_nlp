import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# =========================
# 1. 项目根目录与路径
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

data_path = BASE_DIR / "data" / "raw" / "fpb" / "fpb_raw.csv"
model_path = BASE_DIR / "results" / "models" / "finbert_model" / "checkpoint-342"

print("Using data:", data_path)
print("Using model:", model_path)

if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

if not model_path.exists():
    raise FileNotFoundError(f"Model folder not found: {model_path}")

# =========================
# 2. 读取数据
# =========================
df = pd.read_csv(data_path)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())

# =========================
# 3. 划分测试集
#    和训练时保持一致思路
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

print("\nTest set shape:")
print(test_df.shape)

# =========================
# 4. 转为 Hugging Face Dataset
# =========================
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# =========================
# 5. 加载 tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    local_files_only=True
)

# =========================
# 6. 分词
# =========================
def tokenize_function(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test_dataset = test_dataset.map(tokenize_function, batched=True)

# =========================
# 7. 只保留模型需要的列
# =========================
keep_cols = ["input_ids", "attention_mask", "label"]
remove_cols = [col for col in test_dataset.column_names if col not in keep_cols]
test_dataset = test_dataset.remove_columns(remove_cols)

test_dataset.set_format("torch")

# =========================
# 8. 加载模型
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    str(model_path),
    local_files_only=True
)

# =========================
# 9. 创建 Trainer
# =========================
trainer = Trainer(model=model)

# =========================
# 10. 预测
# =========================
predictions = trainer.predict(test_dataset)

logits = predictions.predictions
labels = predictions.label_ids
preds = np.argmax(logits, axis=1)

# =========================
# 11. 输出评估结果
# =========================
print("\nAccuracy:")
print(accuracy_score(labels, preds))

print("\nClassification Report:")
print(classification_report(labels, preds, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, preds))