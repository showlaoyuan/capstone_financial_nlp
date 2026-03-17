from datasets import load_dataset
import pandas as pd

print("Downloading FiQA...")
ds = load_dataset("SALT-NLP/FLUE-FiQA")

df = pd.DataFrame(ds["train"])
df.to_csv("data/fiqa_raw.csv", index=False)

print("Saved: data/fiqa_raw.csv")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())