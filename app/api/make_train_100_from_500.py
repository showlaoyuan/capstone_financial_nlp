from pathlib import Path
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 路径
input_path = PROJECT_ROOT / "data" / "processed" / "train_500.jsonl"
output_path = PROJECT_ROOT / "data" / "processed" / "train_100.jsonl"

# 检查源文件是否存在
if not input_path.exists():
    raise FileNotFoundError(f"未找到源文件: {input_path}")

# 读取全部 jsonl
rows = []
with open(input_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            rows.append(obj)
        except json.JSONDecodeError as e:
            raise ValueError(f"第 {line_num} 行不是合法 JSON: {e}")

print(f"源文件路径: {input_path}")
print(f"源文件总条数: {len(rows)}")

# 严格检查是否为 500 条
if len(rows) != 500:
    raise ValueError(f"train_500.jsonl 条数不是 500，而是 {len(rows)}。请先确认源文件。")

# 取前 100 条
rows_100 = rows[:100]

# 写出 train_100.jsonl
with open(output_path, "w", encoding="utf-8") as f:
    for obj in rows_100:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"已生成: {output_path}")
print(f"写入条数: {len(rows_100)}")
print("Done.")