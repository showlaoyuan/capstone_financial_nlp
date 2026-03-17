from datasets import load_dataset
import pandas as pd
import json

print("===== FPB =====")
df_fpb = pd.read_csv("../data/raw/fpb_raw.csv")
print("Shape:")
print(df_fpb.shape)
print("\nHead:")
print(df_fpb.head(3))
print("\nDtypes:")
print(df_fpb.dtypes)

print("\n===== FiQA =====")
ds = load_dataset("TheFinAI/fiqa-sentiment-classification")

print("Dataset structure:")
print(ds)

df_fiqa = ds["train"].to_pandas()

print("\nShape:")
print(df_fiqa.shape)

print("\nHead:")
print(df_fiqa.head(3))

print("\nDtypes:")
print(df_fiqa.dtypes)

print("\nColumns:")
print(df_fiqa.columns)

def convert_score_to_label(score):
    if score <= -0.05:
        return 0
    elif score >= 0.05:
        return 2
    else:
        return 1

df_fiqa["label"] = df_fiqa["score"].apply(convert_score_to_label)

print("\nFiQA sentiment label counts:")
print(df_fiqa["label"].value_counts())

print("\nFiQA sentence / score / label preview:")
print(df_fiqa[["sentence", "score", "label"]].head(5))

print("\n===== FinQA =====")
with open("../data/raw/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Number of samples:")
print(len(data))

print("\nColumns:")
print(data[0].keys())