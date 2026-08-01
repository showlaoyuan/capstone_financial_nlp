from __future__ import annotations

import csv
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import date
from pathlib import Path


DATASET_NAME = "finqa"
SOURCE_REPO = "czyssrs/FinQA"

GITHUB_API_URL = (
    "https://api.github.com/repos/czyssrs/FinQA/commits/main"
)

EXPECTED_SPLIT_ROWS = {
    "train": 6251,
    "dev": 883,
    "test": 1147,
}

EXPECTED_TOTAL_ROWS = 8281


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA256 checksum of one file."""
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_latest_commit_sha() -> str:
    """Resolve the current main-branch commit for reproducibility."""
    request = urllib.request.Request(
        GITHUB_API_URL,
        headers={
            "User-Agent": "capstone-financial-nlp-rebuild",
            "Accept": "application/vnd.github+json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        data = json.loads(response.read().decode("utf-8"))

    commit_sha = data.get("sha")

    if not commit_sha:
        raise RuntimeError(
            "Could not resolve the FinQA GitHub commit SHA."
        )

    return commit_sha


def download_archive(
    commit_sha: str,
    archive_path: Path,
) -> None:
    """Download the official repository archive at one exact commit."""
    if archive_path.exists():
        print(f"Archive already exists: {archive_path}")
        return

    archive_url = (
        f"https://codeload.github.com/"
        f"czyssrs/FinQA/zip/{commit_sha}"
    )

    request = urllib.request.Request(
        archive_url,
        headers={
            "User-Agent": "capstone-financial-nlp-rebuild",
        },
    )

    print("Downloading official FinQA archive...")

    with urllib.request.urlopen(
        request,
        timeout=180,
    ) as response:
        with archive_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)


def extract_dataset_files(
    archive_path: Path,
    dataset_dir: Path,
) -> None:
    """Extract only the three labeled official dataset splits."""
    required_files = [
        "train.json",
        "dev.json",
        "test.json",
    ]

    dataset_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive_names = archive.namelist()

        for filename in required_files:
            matching_files = [
                name
                for name in archive_names
                if name.endswith(f"/dataset/{filename}")
            ]

            if len(matching_files) != 1:
                raise FileNotFoundError(
                    f"Expected one {filename}, "
                    f"but found {len(matching_files)}."
                )

            source_name = matching_files[0]
            destination = dataset_dir / filename

            with archive.open(source_name) as source_file:
                with destination.open("wb") as output_file:
                    shutil.copyfileobj(
                        source_file,
                        output_file,
                    )


def inspect_split(
    split_name: str,
    json_path: Path,
) -> list[dict]:
    """Inspect one official FinQA split."""
    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise TypeError(
            f"{json_path} does not contain a JSON list."
        )

    source_ids: list[str] = []
    missing_id = 0
    missing_question = 0
    missing_answer = 0
    missing_program = 0
    malformed_qa = 0

    for record in records:
        source_id = record.get("id")

        if source_id:
            source_ids.append(str(source_id))
        else:
            missing_id += 1

        qa = record.get("qa")

        if not isinstance(qa, dict):
            malformed_qa += 1
            continue

        if not qa.get("question"):
            missing_question += 1

        if "answer" not in qa:
            missing_answer += 1

        if "program" not in qa:
            missing_program += 1

    duplicate_ids = (
        len(source_ids) - len(set(source_ids))
    )

    print()
    print("-" * 60)
    print(f"Split: {split_name}")
    print("-" * 60)
    print(f"File: {json_path}")
    print(f"Rows: {len(records)}")
    print(f"Duplicate IDs inside split: {duplicate_ids}")
    print(f"Missing ID: {missing_id}")
    print(f"Malformed qa field: {malformed_qa}")
    print(f"Missing question: {missing_question}")
    print(f"Missing answer field: {missing_answer}")
    print(f"Missing program field: {missing_program}")

    if records:
        first_record = records[0]
        first_qa = first_record.get("qa", {})

        print("\nFirst record summary:")
        print(
            {
                "id": first_record.get("id"),
                "question": first_qa.get("question"),
                "answer": first_qa.get("answer"),
                "program": first_qa.get("program"),
                "pre_text_count": len(
                    first_record.get("pre_text", [])
                ),
                "post_text_count": len(
                    first_record.get("post_text", [])
                ),
                "table_rows": len(
                    first_record.get("table", [])
                ),
            }
        )

    expected_rows = EXPECTED_SPLIT_ROWS[split_name]

    if len(records) == expected_rows:
        print(
            f"\nPASS: {split_name} row count "
            f"is {expected_rows}."
        )
    else:
        print(
            f"\nWARNING: Expected {expected_rows} rows, "
            f"but found {len(records)}."
        )

    return records


def upsert_manifest(
    manifest_path: Path,
    commit_sha: str,
) -> None:
    """Add FinQA while preserving previous dataset rows."""
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
            "dataset_name": DATASET_NAME,
            "source_repo": SOURCE_REPO,
            "revision": commit_sha,
            "download_date": date.today().isoformat(),
            "split": "train/dev/test",
            "expected_rows": str(EXPECTED_TOTAL_ROWS),
            "raw_path": "data/raw/finqa",
            "notes": (
                "Official labeled splits preserved: "
                "train=6251, dev=883, test=1147; "
                "private_test excluded because gold "
                "references are unavailable."
            ),
        }
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(existing_rows)


def update_checksums(
    checksum_path: Path,
    project_root: Path,
    raw_dir: Path,
) -> int:
    """Record FinQA checksums without deleting prior records."""
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
        if path.is_file()
        and ".cache" not in path.parts
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
                "sha256": calculate_sha256(
                    downloaded_file
                ),
            }
        )

    with checksum_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(existing_rows)

    return len(downloaded_files)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw" / "finqa"
    archive_dir = raw_dir / "source_archive"
    dataset_dir = raw_dir / "dataset"
    metadata_dir = project_root / "data" / "metadata"

    archive_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Resolving official FinQA repository revision...")

    commit_sha = get_latest_commit_sha()

    print(f"Repository: {SOURCE_REPO}")
    print(f"Commit SHA: {commit_sha}")

    archive_path = (
        archive_dir
        / f"FinQA-{commit_sha}.zip"
    )

    download_archive(
        commit_sha=commit_sha,
        archive_path=archive_path,
    )

    extract_dataset_files(
        archive_path=archive_path,
        dataset_dir=dataset_dir,
    )

    print()
    print("=" * 60)
    print("FinQA raw data inspection")
    print("=" * 60)

    all_ids: list[str] = []
    total_rows = 0

    for split_name in ["train", "dev", "test"]:
        records = inspect_split(
            split_name=split_name,
            json_path=dataset_dir / f"{split_name}.json",
        )

        total_rows += len(records)

        all_ids.extend(
            str(record.get("id"))
            for record in records
            if record.get("id")
        )

    print()
    print("=" * 60)
    print("Combined FinQA inspection")
    print("=" * 60)
    print(f"Total rows: {total_rows}")
    print(
        "Duplicate IDs across all splits:",
        len(all_ids) - len(set(all_ids)),
    )

    if total_rows == EXPECTED_TOTAL_ROWS:
        print(
            f"PASS: Total row count is "
            f"{EXPECTED_TOTAL_ROWS}."
        )
    else:
        print(
            f"WARNING: Expected {EXPECTED_TOTAL_ROWS} "
            f"rows, but found {total_rows}."
        )

    manifest_path = (
        metadata_dir / "dataset_manifest.csv"
    )

    upsert_manifest(
        manifest_path=manifest_path,
        commit_sha=commit_sha,
    )

    checksum_path = (
        metadata_dir / "file_checksums.csv"
    )

    checked_file_count = update_checksums(
        checksum_path=checksum_path,
        project_root=project_root,
        raw_dir=raw_dir,
    )

    print()
    print("FinQA download and inspection completed.")
    print(
        f"Files recorded in checksums: "
        f"{checked_file_count}"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Checksums: {checksum_path}")


if __name__ == "__main__":
    main()