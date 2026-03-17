import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# 读取数据
df = pd.read_csv("data/raw/fpb_raw.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel counts:")
print(df["label"].value_counts())

# 数字标签 → 文字标签
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

df["label_name"] = df["label"].map(label_map)

print("\nLabel name counts:")
print(df["label_name"].value_counts())

# 统计数量
counts = df["label_name"].value_counts()

# 画图
plt.figure(figsize=(8,5))

colors = ["#4C72B0", "#55A868", "#C44E52"]

bars = plt.bar(counts.index, counts.values, color=colors)

plt.title("Financial Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Count", fontsize=12)

# 横着显示
plt.xticks(rotation=0)

# 在柱子上显示数字
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 10,
             int(height),
             ha='center')

plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()