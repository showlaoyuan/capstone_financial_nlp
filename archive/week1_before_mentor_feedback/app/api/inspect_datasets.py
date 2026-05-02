from datasets import load_dataset
import pandas as pd
from pathlib import Path
import os

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# 保存目录
FIQA_DIR = BASE_DIR / "data" / "raw" / "fiqa"
os.makedirs(FIQA_DIR, exist_ok=True)

# 保存路径
output_path = FIQA_DIR / "fiqa_raw.csv"

print("Downloading FiQA...")
ds = load_dataset("SALT-NLP/FLUE-FiQA")

df = pd.DataFrame(ds["train"])
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Saved: {output_path}")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())