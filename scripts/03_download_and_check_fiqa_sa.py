from __future__ import annotations

import csv
import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download


DATASET_NAME = "fiqa_sa"
REPO_ID = "TheFinAI/fiqa-sentiment-classification"

EXPECTED_SPLIT_ROWS = {
    "train": 822,
    "valid": 117,
    "test": 234,
}

EXPECTED_TOTAL_ROWS = 1173


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA256 checksum of one file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def upsert_manifest(
    manifest_path: Path,
    new_row: dict[str, str | int],
) -> None:
    """Add or replace one dataset row while preserving existing rows."""
    fieldnames = [
        "dataset_name",
        "source_repo",
        "revision",
        "download_date",
        "split",
        "expected_rows",
        "raw_path",
        "notes",
    ]

    existing_rows: list[dict[str, str]] = []

    if manifest_path.exists():
        with manifest_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            existing_rows = list(csv.DictReader(file))

    existing_rows = [
        row
        for row in existing_rows
        if row.get("dataset_name") != DATASET_NAME
    ]

    existing_rows.append(
        {
            key: str(new_row.get(key, ""))
            for key in fieldnames
        }
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def update_checksums(
    checksum_path: Path,
    project_root: Path,
    raw_dir: Path,
) -> int:
    """Update FiQA-SA checksum rows while preserving other datasets."""
    fieldnames = [
        "dataset_name",
        "file_path",
        "file_size_bytes",
        "sha256",
    ]

    existing_rows: list[dict[str, str]] = []

    if checksum_path.exists():
        with checksum_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            existing_rows = list(csv.DictReader(file))

    existing_rows = [
        row
        for row in existing_rows
        if row.get("dataset_name") != DATASET_NAME
    ]

    downloaded_files = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    ]

    for downloaded_file in downloaded_files:
        existing_rows.append(
            {
                "dataset_name": DATASET_NAME,
                "file_path": downloaded_file.relative_to(
                    project_root
                ).as_posix(),
                "file_size_bytes": str(
                    downloaded_file.stat().st_size
                ),
                "sha256": calculate_sha256(downloaded_file),
            }
        )

    with checksum_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    return len(downloaded_files)


def find_split_file(raw_dir: Path, split_name: str) -> Path:
    """Find the parquet file belonging to one split."""
    candidates = list(
        (raw_dir / "data").glob(f"{split_name}-*.parquet")
    )

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one parquet file for split "
            f"'{split_name}', found {len(candidates)}."
        )

    return candidates[0]


def inspect_split(
    split_name: str,
    parquet_path: Path,
) -> pd.DataFrame:
    """Read and inspect one FiQA-SA split."""
    df = pd.read_parquet(parquet_path)

    print()
    print("-" * 60)
    print(f"Split: {split_name}")
    print("-" * 60)

    print(f"File: {parquet_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nMissing values:")
    print(df.isnull().sum())

    if "_id" in df.columns:
        print(
            "Duplicate _id inside split:",
            df["_id"].duplicated().sum(),
        )

    if "type" in df.columns:
        print("\nType distribution:")
        print(df["type"].value_counts(dropna=False))

    if "score" in df.columns:
        print("\nScore summary:")
        print(df["score"].describe())

    print("\nFirst record:")
    print(df.iloc[0].to_dict())

    expected_rows = EXPECTED_SPLIT_ROWS[split_name]

    if len(df) == expected_rows:
        print(
            f"\nPASS: {split_name} row count "
            f"is {expected_rows}."
        )
    else:
        print(
            f"\nWARNING: Expected {expected_rows} rows, "
            f"but found {len(df)}."
        )

    return df


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw" / "fiqa_sa"
    metadata_dir = project_root / "data" / "metadata"

    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Reading FiQA-SA repository information...")

    api = HfApi()
    dataset_info = api.dataset_info(REPO_ID)
    revision = dataset_info.sha

    print(f"Repository: {REPO_ID}")
    print(f"Revision: {revision}")
    print("Downloading repository snapshot...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=raw_dir,
    )

    split_dataframes: dict[str, pd.DataFrame] = {}

    print()
    print("=" * 60)
    print("FiQA-SA raw data inspection")
    print("=" * 60)

    for split_name in ["train", "valid", "test"]:
        parquet_path = find_split_file(
            raw_dir=raw_dir,
            split_name=split_name,
        )

        split_dataframes[split_name] = inspect_split(
            split_name=split_name,
            parquet_path=parquet_path,
        )

    combined_df = pd.concat(
        [
            dataframe.assign(split=split_name)
            for split_name, dataframe
            in split_dataframes.items()
        ],
        ignore_index=True,
    )

    print()
    print("=" * 60)
    print("Combined FiQA-SA inspection")
    print("=" * 60)

    print(f"Total rows: {len(combined_df)}")

    if "_id" in combined_df.columns:
        print(
            "Duplicate _id across all splits:",
            combined_df["_id"].duplicated().sum(),
        )

    if "sentence" in combined_df.columns:
        print(
            "Duplicate sentences across all splits:",
            combined_df["sentence"].duplicated().sum(),
        )

    if len(combined_df) == EXPECTED_TOTAL_ROWS:
        print(
            f"PASS: Total row count is "
            f"{EXPECTED_TOTAL_ROWS}."
        )
    else:
        print(
            f"WARNING: Expected {EXPECTED_TOTAL_ROWS} "
            f"total rows, but found {len(combined_df)}."
        )

    manifest_path = metadata_dir / "dataset_manifest.csv"

    upsert_manifest(
        manifest_path,
        {
            "dataset_name": DATASET_NAME,
            "source_repo": REPO_ID,
            "revision": revision,
            "download_date": date.today().isoformat(),
            "split": "train/valid/test",
            "expected_rows": EXPECTED_TOTAL_ROWS,
            "raw_path": "data/raw/fiqa_sa",
            "notes": (
                "Official splits preserved: "
                "train=822, valid=117, test=234; "
                "raw files not modified."
            ),
        },
    )

    checksum_path = metadata_dir / "file_checksums.csv"

    checked_file_count = update_checksums(
        checksum_path=checksum_path,
        project_root=project_root,
        raw_dir=raw_dir,
    )

    print()
    print("FiQA-SA download and inspection completed.")
    print(f"Files recorded in checksums: {checked_file_count}")
    print(f"Manifest: {manifest_path}")
    print(f"Checksums: {checksum_path}")


if __name__ == "__main__":
    main()