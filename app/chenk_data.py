import pandas as pd

# 请把这里的路径换成你电脑上 fiqa_train.csv 的实际相对路径
# 比如可能是 "data/fiqa_train.csv"
file_path = "../data/raw/fiqa_train.csv"

# 读取数据
df = pd.read_csv(file_path)

# 打印所有的列名
print("这个文件包含的列有：", df.columns.tolist())

# 顺便打印前两行数据，看看具体内容长什么样
print("\n前两行数据是：\n", df.head(2))
