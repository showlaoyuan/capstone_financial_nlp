import pandas as pd

input_path = "data/raw/Sentences_50Agree.txt"
output_path = "data/fpb_raw.csv"

rows = []

with open(input_path, "r", encoding="latin1") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        sentence, label = line.rsplit("@", 1)

        rows.append({
            "sentence": sentence.strip(),
            "label_text": label.strip().lower()
        })

df = pd.DataFrame(rows)

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2

}

df["label"] = df["label_text"].map(label_map)
df = df[["sentence", "label"]]

df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Saved:", output_path)
print("Shape:", df.shape)
print(df.head())