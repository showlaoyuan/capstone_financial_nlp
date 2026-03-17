from datasets import load_dataset
import pandas as pd
import os

def ensure_dirs():
    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../data/metadata", exist_ok=True)


def save_fpb():
    print("Converting local Financial PhraseBank txt...")

    txt_path = "../data/raw/Sentences_50Agree.txt"
    output_path = "../data/raw/fpb_raw.csv"

    with open(txt_path, "r", encoding="latin-1") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Financial PhraseBank 原始格式一般是：sentence@label
        if "@ " in line:
            sentence, label = line.rsplit("@ ", 1)
        elif "@" in line:
            sentence, label = line.rsplit("@", 1)
        else:
            continue

        label = label.strip().lower()

        label_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }

        if label in label_map:
            data.append([sentence.strip(), label_map[label]])

    df = pd.DataFrame(data, columns=["sentence", "label"])
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())
def save_fiqa():
    print("\n[2] Downloading FiQA...")
    ds = load_dataset("pauri32/fiqa-2018")

    for split in ds.keys():
        df = pd.DataFrame(ds[split])
        output_path = f"data/raw/fiqa_{split}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(f"  Shape ({split}):", df.shape)
        print(f"  Columns ({split}):", list(df.columns))


if __name__ == "__main__":
    ensure_dirs()

    save_fpb()
    save_fiqa()
    # save_finqa()

    print("\nAll dataset downloads completed.")