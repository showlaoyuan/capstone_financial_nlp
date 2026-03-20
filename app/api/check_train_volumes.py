from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

files = {
    "train_100.jsonl": PROJECT_ROOT / "data" / "processed" / "train_100.jsonl",
    "train_500.jsonl": PROJECT_ROOT / "data" / "processed" / "train_500.jsonl",
}

for name, path in files.items():
    print(f"\nChecking {name}...")
    if not path.exists():
        print(f"  不存在: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())

    print(f"  路径: {path}")
    print(f"  条数: {count}")