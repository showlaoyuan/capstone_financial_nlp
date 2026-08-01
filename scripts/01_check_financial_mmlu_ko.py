from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parquet_path = (
        project_root
        / "data"
        / "raw"
        / "financial_mmlu_ko"
        / "data"
        / "test-00000-of-00001.parquet"
    )

    if not parquet_path.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    print("=" * 60)
    print("financial-mmlu-ko raw data inspection")
    print("=" * 60)

    print(f"File: {parquet_path}")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    if "conversation_id" in df.columns:
        print(
            "\nDuplicate conversation_id:",
            df["conversation_id"].duplicated().sum(),
        )

    print("\nFirst record:")
    print(df.iloc[0].to_dict())

    if len(df) == 455:
        print("\nPASS: Row count is 455.")
    else:
        print(f"\nWARNING: Expected 455 rows, but found {len(df)}.")


if __name__ == "__main__":
    main()