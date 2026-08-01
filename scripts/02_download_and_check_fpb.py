from __future__ import annotations

import csv
import hashlib
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATASET_NAME = "fpb"
REPO_ID = "takala/financial_phrasebank"
CONFIG_NAME = "sentences_50agree"
EXPECTED_ROWS = 4846


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
    """Add or replace one dataset row without deleting existing rows."""
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
            reader = csv.DictReader(file)
            existing_rows = list(reader)

    existing_rows = [
        row
        for row in existing_rows
        if row.get("dataset_name") != DATASET_NAME
    ]

    existing_rows.append(
        {key: str(new_row.get(key, "")) for key in fieldnames}
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
    """Update FPB checksums while preserving other dataset records."""
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
            reader = csv.DictReader(file)
            existing_rows = list(reader)

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


def inspect_fpb(text_path: Path) -> None:
    """Inspect the sentences_50agree raw text file."""
    records: list[tuple[str, str]] = []
    malformed_lines: list[int] = []

    with text_path.open(
        "r",
        encoding="latin-1",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            if "@" not in line:
                malformed_lines.append(line_number)
                continue

            # Split from the right in case the sentence itself contains @.
            sentence, label = line.rsplit("@", 1)

            records.append(
                (
                    sentence.strip(),
                    label.strip().lower(),
                )
            )

    sentences = [sentence for sentence, _ in records]
    labels = [label for _, label in records]

    print()
    print("=" * 60)
    print("FPB raw data inspection")
    print("=" * 60)
    print(f"Selected config: {CONFIG_NAME}")
    print(f"Raw text file: {text_path}")
    print(f"Number of rows: {len(records)}")
    print(f"Malformed rows: {len(malformed_lines)}")
    print(
        "Duplicate sentences:",
        len(sentences) - len(set(sentences)),
    )
    print(f"Label distribution: {dict(Counter(labels))}")

    if records:
        print("\nFirst record:")
        print(
            {
                "sentence": records[0][0],
                "label": records[0][1],
            }
        )

    if len(records) == EXPECTED_ROWS:
        print(f"\nPASS: Row count is {EXPECTED_ROWS}.")
    else:
        print(
            f"\nWARNING: Expected {EXPECTED_ROWS} rows, "
            f"but found {len(records)}."
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw" / "fpb"
    snapshot_dir = raw_dir / "hf_snapshot"
    extracted_dir = raw_dir / "extracted"
    metadata_dir = project_root / "data" / "metadata"

    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Reading FPB dataset information...")

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
        local_dir=snapshot_dir,
    )

    zip_files = list(
        snapshot_dir.rglob("FinancialPhraseBank-v1.0.zip")
    )

    if not zip_files:
        raise FileNotFoundError(
            "FinancialPhraseBank-v1.0.zip was not found."
        )

    zip_path = zip_files[0]

    print(f"Source archive: {zip_path}")
    print("Extracting source archive...")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extracted_dir)

    target_files = [
        path
        for path in extracted_dir.rglob("*.txt")
        if path.name.lower() == "sentences_50agree.txt"
    ]

    if not target_files:
        raise FileNotFoundError(
            "Sentences_50Agree.txt was not found."
        )

    text_path = target_files[0]

    inspect_fpb(text_path)

    manifest_path = metadata_dir / "dataset_manifest.csv"

    upsert_manifest(
        manifest_path,
        {
            "dataset_name": DATASET_NAME,
            "source_repo": REPO_ID,
            "revision": revision,
            "download_date": date.today().isoformat(),
            "split": "unsplit",
            "expected_rows": EXPECTED_ROWS,
            "raw_path": "data/raw/fpb",
            "notes": (
                "Selected config: sentences_50agree; "
                "source ZIP preserved; no official train/valid/test split."
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
    print("FPB download and inspection completed.")
    print(f"Files recorded in checksums: {checked_file_count}")
    print(f"Manifest: {manifest_path}")
    print(f"Checksums: {checksum_path}")


if __name__ == "__main__":
    main()