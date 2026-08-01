from __future__ import annotations

import csv
import hashlib
from datetime import date
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATASET_NAME = "financial-mmlu-ko"
REPO_ID = "allganize/financial-mmlu-ko"
EXPECTED_ROWS = 455
SPLIT = "test"


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA256 checksum of one file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = (
        project_root
        / "data"
        / "raw"
        / "financial_mmlu_ko"
    )

    metadata_dir = project_root / "data" / "metadata"

    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Reading dataset information...")

    api = HfApi()
    dataset_info = api.dataset_info(REPO_ID)

    revision = dataset_info.sha

    print(f"Repository: {REPO_ID}")
    print(f"Revision: {revision}")
    print("Downloading raw dataset...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=raw_dir,
    )

    manifest_path = metadata_dir / "dataset_manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset_name",
                "source_repo",
                "revision",
                "download_date",
                "split",
                "expected_rows",
                "raw_path",
                "notes",
            ],
        )

        writer.writeheader()
        writer.writerow(
            {
                "dataset_name": DATASET_NAME,
                "source_repo": REPO_ID,
                "revision": revision,
                "download_date": date.today().isoformat(),
                "split": SPLIT,
                "expected_rows": EXPECTED_ROWS,
                "raw_path": "data/raw/financial_mmlu_ko",
                "notes": "Raw public dataset snapshot; original files not modified.",
            }
        )

    checksum_path = metadata_dir / "file_checksums.csv"

    downloaded_files = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    ]

    with checksum_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset_name",
                "file_path",
                "file_size_bytes",
                "sha256",
            ],
        )

        writer.writeheader()

        for downloaded_file in downloaded_files:
            writer.writerow(
                {
                    "dataset_name": DATASET_NAME,
                    "file_path": downloaded_file.relative_to(
                        project_root
                    ).as_posix(),
                    "file_size_bytes": downloaded_file.stat().st_size,
                    "sha256": calculate_sha256(downloaded_file),
                }
            )

    print()
    print("Download completed successfully.")
    print(f"Raw directory: {raw_dir}")
    print(f"Downloaded files: {len(downloaded_files)}")
    print(f"Manifest: {manifest_path}")
    print(f"Checksums: {checksum_path}")


if __name__ == "__main__":
    main()