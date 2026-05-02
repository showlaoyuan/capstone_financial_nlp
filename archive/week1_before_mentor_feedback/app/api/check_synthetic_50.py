from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
input_path = PROJECT_ROOT / "data" / "processed" / "synthetic_50.jsonl"

if not input_path.exists():
    raise FileNotFoundError(f"未找到文件: {input_path}")

rows = []
valid_labels = {"positive", "negative", "neutral"}

with open(input_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"第 {line_num} 行 JSON 格式错误: {e}")

        # 检查字段
        keys = set(obj.keys())
        expected_keys = {"instruction", "input", "output"}
        if keys != expected_keys:
            raise ValueError(
                f"第 {line_num} 行字段不对。当前字段: {keys}，应为: {expected_keys}"
            )

        # 检查 instruction
        expected_instruction = (
            "Classify the sentiment of the following financial news sentence "
            "as positive, negative, or neutral."
        )
        if obj["instruction"] != expected_instruction:
            raise ValueError(f"第 {line_num} 行 instruction 不符合要求。")

        # 检查 input
        if not isinstance(obj["input"], str) or not obj["input"].strip():
            raise ValueError(f"第 {line_num} 行 input 为空或不是字符串。")

        # 检查 output
        if obj["output"] not in valid_labels:
            raise ValueError(f"第 {line_num} 行 output 不合法: {obj['output']}")

        rows.append(obj)

print(f"文件路径: {input_path}")
print(f"总条数: {len(rows)}")

if len(rows) != 50:
    raise ValueError(f"条数不是 50，而是 {len(rows)}")

print("\n前 3 条样本预览:")
for i, row in enumerate(rows[:3], start=1):
    print(f"\nSample {i}:")
    print(row)

print("\n检查通过。")