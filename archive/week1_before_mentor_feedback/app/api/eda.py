import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 以当前文件位置为基准，自动找到项目根目录
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# 数据文件路径
file_path = PROJECT_ROOT / "data" / "raw" / "fpb" / "fpb_raw.csv"

# 读取数据
df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())
print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel counts:")
print(df["label"].value_counts())

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

df["label_name"] = df["label"].map(label_map)

print("\nLabel name counts:")
print(df["label_name"].value_counts())

# 统计顺序固定
order = ["negative", "neutral", "positive"]
label_counts = df["label_name"].value_counts().reindex(order)

# 画图
plt.figure(figsize=(9, 6))
bars = plt.bar(label_counts.index, label_counts.values, edgecolor="black", linewidth=1.0)

# 标题和坐标轴
plt.title("Financial Sentiment Distribution", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Sentiment Category", fontsize=12, labelpad=10)
plt.ylabel("Number of Samples", fontsize=12, labelpad=10)

# 网格线
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.gca().set_axisbelow(True)

# 显示每个柱子的数值
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 30,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=11
    )

# 美化边距
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()

plt.show()