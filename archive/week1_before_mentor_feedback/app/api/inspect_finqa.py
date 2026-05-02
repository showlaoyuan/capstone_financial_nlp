import json
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# FinQA train.json 路径
finqa_path = BASE_DIR / "data" / "raw" / "finqa" / "train.json"

print("Using file:", finqa_path)

with open(finqa_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Number of samples:")
print(len(data))

print("\nType of first sample:")
print(type(data[0]))

print("\nKeys of first sample:")
print(data[0].keys())

print("\nFirst sample preview:")
print(data[0])