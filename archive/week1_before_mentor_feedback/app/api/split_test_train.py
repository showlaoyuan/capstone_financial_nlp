import pandas as pd

# ===== 1. FPB =====
df_fpb = pd.read_csv(r"E:\capstone_financial_nlp\data\unified\fpb_unified.csv")
test_fpb = df_fpb.sample(n=200, random_state=42)
train_fpb = df_fpb.drop(test_fpb.index)

test_fpb.to_csv(r"E:\capstone_financial_nlp\data\unified\fpb_test_200.csv", index=False)
train_fpb.to_csv(r"E:\capstone_financial_nlp\data\unified\fpb_train.csv", index=False)

# ===== 2. FiQA =====
df_fiqa = pd.read_csv(r"E:\capstone_financial_nlp\data\unified\fiqa_unified.csv")
test_fiqa = df_fiqa.sample(n=200, random_state=42)
train_fiqa = df_fiqa.drop(test_fiqa.index)

test_fiqa.to_csv(r"E:\capstone_financial_nlp\data\unified\fiqa_test_200.csv", index=False)
train_fiqa.to_csv(r"E:\capstone_financial_nlp\data\unified\fiqa_train.csv", index=False)

# ===== 3. FinQA =====
df_finqa = pd.read_csv(r"E:\capstone_financial_nlp\data\unified\finqa_unified.csv")
test_finqa = df_finqa.sample(n=200, random_state=42)
train_finqa = df_finqa.drop(test_finqa.index)

test_finqa.to_csv(r"E:\capstone_financial_nlp\data\unified\finqa_test_200.csv", index=False)
train_finqa.to_csv(r"E:\capstone_financial_nlp\data\unified\finqa_train.csv", index=False)

print("Done.")
print("FPB:", len(test_fpb), len(train_fpb))
print("FiQA:", len(test_fiqa), len(train_fiqa))
print("FinQA:", len(test_finqa), len(train_finqa))