import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# =========================
# 路径设置
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# 训练数据路径
data_path = PROJECT_ROOT / "data" / "raw" / "fpb" / "fpb_raw.csv"

# 模型输出路径
model_output_dir = PROJECT_ROOT / "results" / "models" / "finbert_model"
model_output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 读取数据
# =========================
df = pd.read_csv(data_path)
print("Dataset size:", len(df))
print("Columns:", df.columns.tolist())

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# =========================
# 模型与分词器
# =========================
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 只保留 Trainer 需要的字段
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
)
test_dataset = test_dataset.remove_columns(
    [col for col in test_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
)

train_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# =========================
# 评估指标
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# =========================
# 训练参数
# =========================
training_args = TrainingArguments(
    output_dir=str(model_output_dir),
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# =========================
# 开始训练
# =========================
trainer.train()

results = trainer.evaluate()
print("\nEvaluation Results:")
print(results)