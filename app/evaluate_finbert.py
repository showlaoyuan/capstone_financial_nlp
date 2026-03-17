import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# 读取数据
df = pd.read_csv("data/fpb_raw.csv")

# 划分测试集（和训练时保持一致思路）
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# 转成 Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# 加载你训练好的模型
model_path = "../finbert_model/checkpoint-342"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 分词
def tokenize_function(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test_dataset = test_dataset.map(tokenize_function, batched=True)

# 只保留模型需要的列
test_dataset = test_dataset.remove_columns(
    [col for col in test_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
)

test_dataset.set_format("torch")

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 创建 Trainer
trainer = Trainer(model=model)

# 预测
predictions = trainer.predict(test_dataset)

logits = predictions.predictions
labels = predictions.label_ids
preds = np.argmax(logits, axis=1)

# 输出结果
print("Accuracy:")
print(accuracy_score(labels, preds))

print("\nClassification Report:")
print(classification_report(labels, preds, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, preds))