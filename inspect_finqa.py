import json

with open("data/raw/train.json", "r") as f:
    data = json.load(f)

print("Number of samples:", len(data))
print("Columns:", data[0].keys())