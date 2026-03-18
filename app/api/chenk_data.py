import json

finqa_path = r"/data/raw/train.json"

with open(finqa_path, "r", encoding="utf-8") as f:
    data = json.load(f)

empty_count = 0

for i, item in enumerate(data):
    answer = item.get("qa", {}).get("answer", None)

    if answer is None or str(answer).strip() == "":
        empty_count += 1
        print(f"Index: {i}")
        print(f"ID: {item.get('id')}")
        print(f"Question: {item.get('qa', {}).get('question')}")
        print(f"Answer: {answer}")
        print("-" * 50)

print("Total empty answers:", empty_count)