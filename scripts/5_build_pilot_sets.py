from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_DIR = PROJECT_ROOT / "data" / "evaluation" / "full"
PILOT_DIR = PROJECT_ROOT / "data" / "evaluation" / "pilot"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
DOCS_DIR = PROJECT_ROOT / "docs"

PILOT_SIZE = 200
SMOKE_SIZE = 20
RANDOM_SEED = 42

CONFIG = {
    "fpb": {
        "input": FULL_DIR / "fpb_full_eval.jsonl",
        "pilot": PILOT_DIR / "fpb_pilot_eval.jsonl",
        "smoke": PILOT_DIR / "fpb_smoke20_eval.jsonl",
        "expected_full_rows": 4846,
    },
    "fiqa_sa": {
        "input": FULL_DIR / "fiqa_sa_full_eval.jsonl",
        "pilot": PILOT_DIR / "fiqa_sa_pilot_eval.jsonl",
        "smoke": PILOT_DIR / "fiqa_sa_smoke20_eval.jsonl",
        "expected_full_rows": 1173,
    },
    "finqa": {
        "input": FULL_DIR / "finqa_full_eval.jsonl",
        "pilot": PILOT_DIR / "finqa_pilot_eval.jsonl",
        "smoke": PILOT_DIR / "finqa_smoke20_eval.jsonl",
        "expected_full_rows": 8281,
    },
    "financial_mmlu_ko": {
        "input": FULL_DIR / "financial_mmlu_ko_full_eval.jsonl",
        "pilot": PILOT_DIR / "financial_mmlu_ko_pilot_eval.jsonl",
        "smoke": PILOT_DIR / "financial_mmlu_ko_smoke20_eval.jsonl",
        "expected_full_rows": 455,
    },
}

COMBINED_PILOT = PILOT_DIR / "combined_pilot_eval.jsonl"
COMBINED_SMOKE = PILOT_DIR / "combined_smoke20_eval.jsonl"
SUMMARY_CSV = RESULTS_DIR / "pilot_eval_summary.csv"
SELECTION_CSV = METADATA_DIR / "pilot_selection_manifest.csv"
FORMAT_DOC = DOCS_DIR / "pilot_evaluation_format.md"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Full evaluation file not found: {path}\n"
            "Run Day 5 first."
        )

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{path}: blank line {line_number}")

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}: invalid JSON at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"{path}: line {line_number} is not a JSON object"
                )

            record["_full_line_number"] = line_number
            records.append(record)

    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            clean_record = {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
            }
            file.write(
                json.dumps(
                    clean_record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def stratum_key(record: dict) -> str:
    """
    情感任务按 split + label 分层；
    其他任务按 split 分层。
    """
    dataset = str(record.get("dataset", ""))
    split = str(record.get("split", "unknown"))

    if dataset in {"fpb", "fiqa_sa"}:
        reference = record.get("reference", {})
        label = (
            reference.get("label", "unknown")
            if isinstance(reference, dict)
            else "unknown"
        )
        return f"split={split}|label={label}"

    return f"split={split}"


def allocate_quotas(
    group_sizes: dict[str, int],
    target_size: int,
) -> dict[str, int]:
    """
    按各层原始数量比例分配抽样名额。
    当目标数量足够时，每个非空层至少保留 1 条。
    """
    total_rows = sum(group_sizes.values())
    target_size = min(target_size, total_rows)

    if target_size <= 0:
        return {key: 0 for key in group_sizes}

    quotas = {key: 0 for key in group_sizes}
    nonempty_keys = [
        key for key, size in group_sizes.items()
        if size > 0
    ]

    if target_size >= len(nonempty_keys):
        for key in nonempty_keys:
            quotas[key] = 1

    assigned = sum(quotas.values())
    remaining = target_size - assigned

    while remaining > 0:
        capacities = {
            key: group_sizes[key] - quotas[key]
            for key in nonempty_keys
            if group_sizes[key] - quotas[key] > 0
        }

        if not capacities:
            break

        capacity_total = sum(capacities.values())
        raw_additions = {
            key: remaining * capacity / capacity_total
            for key, capacity in capacities.items()
        }

        floor_additions = {
            key: min(
                capacities[key],
                int(raw_additions[key]),
            )
            for key in capacities
        }

        floor_total = sum(floor_additions.values())

        if floor_total > 0:
            for key, addition in floor_additions.items():
                quotas[key] += addition
            remaining -= floor_total
            continue

        ranked_keys = sorted(
            capacities,
            key=lambda key: (
                raw_additions[key] - int(raw_additions[key]),
                capacities[key],
                key,
            ),
            reverse=True,
        )

        for key in ranked_keys:
            if remaining == 0:
                break
            if quotas[key] < group_sizes[key]:
                quotas[key] += 1
                remaining -= 1

    return quotas


def stratified_sample(
    records: list[dict],
    target_size: int,
    seed: int,
) -> list[dict]:
    groups = defaultdict(list)

    for record in records:
        groups[stratum_key(record)].append(record)

    group_sizes = {
        key: len(group)
        for key, group in groups.items()
    }
    quotas = allocate_quotas(group_sizes, target_size)

    selected = []

    for group_number, key in enumerate(sorted(groups), start=1):
        group = groups[key]
        quota = quotas[key]

        group_rng = random.Random(seed + group_number * 1009)
        chosen = group_rng.sample(group, quota)
        selected.extend(chosen)

    selected.sort(
        key=lambda record: (
            str(record.get("dataset", "")),
            str(record.get("split", "")),
            str(record.get("id", "")),
        )
    )

    return selected


def add_sample_id(records: list[dict]) -> list[dict]:
    output = []

    for record in records:
        copied = dict(record)
        canonical_id = str(copied.get("id", "")).strip()

        if not canonical_id:
            raise ValueError("A selected record has a blank id.")

        copied["sample_id"] = canonical_id
        output.append(copied)

    return output


def validate_full_records(
    dataset: str,
    records: list[dict],
    expected_rows: int,
) -> None:
    if len(records) != expected_rows:
        raise ValueError(
            f"{dataset}: expected {expected_rows:,} full rows, "
            f"found {len(records):,}"
        )

    ids = [str(record.get("id", "")).strip() for record in records]

    if any(not item for item in ids):
        raise ValueError(f"{dataset}: blank id found")

    if len(ids) != len(set(ids)):
        raise ValueError(f"{dataset}: duplicate id found")

    for line_number, record in enumerate(records, start=1):
        if not str(record.get("prompt", "")).strip():
            raise ValueError(
                f"{dataset}: blank prompt at loaded row {line_number}"
            )

        reference = record.get("reference")
        if not isinstance(reference, dict) or not reference:
            raise ValueError(
                f"{dataset}: invalid reference at loaded row "
                f"{line_number}"
            )


def validate_selection(
    dataset: str,
    full: list[dict],
    pilot: list[dict],
    smoke: list[dict],
) -> None:
    full_ids = {record["id"] for record in full}
    pilot_ids = [record["id"] for record in pilot]
    smoke_ids = [record["id"] for record in smoke]

    expected_pilot = min(PILOT_SIZE, len(full))
    expected_smoke = min(SMOKE_SIZE, expected_pilot)

    if len(pilot) != expected_pilot:
        raise ValueError(
            f"{dataset}: expected {expected_pilot} pilot rows, "
            f"found {len(pilot)}"
        )

    if len(smoke) != expected_smoke:
        raise ValueError(
            f"{dataset}: expected {expected_smoke} smoke rows, "
            f"found {len(smoke)}"
        )

    if len(pilot_ids) != len(set(pilot_ids)):
        raise ValueError(f"{dataset}: duplicate pilot id found")

    if len(smoke_ids) != len(set(smoke_ids)):
        raise ValueError(f"{dataset}: duplicate smoke id found")

    if not set(pilot_ids).issubset(full_ids):
        raise ValueError(
            f"{dataset}: pilot contains an id not found in full data"
        )

    if not set(smoke_ids).issubset(set(pilot_ids)):
        raise ValueError(
            f"{dataset}: smoke20 is not a subset of pilot"
        )

    for record in pilot + smoke:
        if record.get("sample_id") != record.get("id"):
            raise ValueError(
                f"{dataset}: sample_id and id are not aligned"
            )


def count_strata(records: list[dict]) -> str:
    counts = defaultdict(int)

    for record in records:
        counts[stratum_key(record)] += 1

    return "; ".join(
        f"{key}:{counts[key]}"
        for key in sorted(counts)
    )


def save_summary(rows: list[dict]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "dataset",
        "full_rows",
        "pilot_rows",
        "smoke20_rows",
        "pilot_strata",
        "random_seed",
        "pilot_file",
        "smoke20_file",
        "pilot_sha256",
        "smoke20_sha256",
    ]

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_selection_manifest(rows: list[dict]) -> None:
    SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "dataset",
        "sample_id",
        "split",
        "stratum",
        "full_line_number",
        "in_pilot",
        "in_smoke20",
    ]

    with SELECTION_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_format_doc() -> None:
    FORMAT_DOC.parent.mkdir(parents=True, exist_ok=True)

    FORMAT_DOC.write_text(
        """# Pilot Evaluation Files

Day 6 creates two deterministic subsets from each Day 5 full evaluation
file.

## Files

- `*_pilot_eval.jsonl`: 200 records per dataset for small A/B experiments.
- `*_smoke20_eval.jsonl`: 20 records per dataset for API and parsing tests.
- `combined_pilot_eval.jsonl`: all four pilot files combined.
- `combined_smoke20_eval.jsonl`: all four smoke-test files combined.

## Selection rules

- Random seed: 42.
- Sentiment datasets are stratified by `split + reference.label`.
- FinQA and financial-mmlu-ko are stratified by `split`.
- `smoke20` is always a subset of the corresponding pilot set.
- `sample_id` is copied from the stable canonical `id`.
- A and B runs must use the same `sample_id` list.

## Important

`reference` is retained for scoring, but it must never be inserted into
the model request prompt.

Run again with:

```powershell
python scripts\\5_build_pilot_sets.py
```
""",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("Day 6 - Build Pilot and Smoke-Test Evaluation Sets")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Pilot size per dataset: {PILOT_SIZE}")
    print(f"Smoke-test size per dataset: {SMOKE_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")

    all_pilot = []
    all_smoke = []
    summary_rows = []
    selection_rows = []

    for dataset_number, (dataset, config) in enumerate(
        CONFIG.items(),
        start=1,
    ):
        print(f"\n[START] {dataset}")

        full_records = load_jsonl(config["input"])
        validate_full_records(
            dataset,
            full_records,
            config["expected_full_rows"],
        )

        pilot_records = stratified_sample(
            full_records,
            PILOT_SIZE,
            RANDOM_SEED + dataset_number * 10000,
        )
        pilot_records = add_sample_id(pilot_records)

        smoke_records = stratified_sample(
            pilot_records,
            SMOKE_SIZE,
            RANDOM_SEED + dataset_number * 20000,
        )
        smoke_records = add_sample_id(smoke_records)

        validate_selection(
            dataset,
            full_records,
            pilot_records,
            smoke_records,
        )

        save_jsonl(pilot_records, config["pilot"])
        save_jsonl(smoke_records, config["smoke"])

        smoke_ids = {
            record["sample_id"]
            for record in smoke_records
        }

        for record in pilot_records:
            selection_rows.append({
                "dataset": dataset,
                "sample_id": record["sample_id"],
                "split": record.get("split"),
                "stratum": stratum_key(record),
                "full_line_number": record.get("_full_line_number"),
                "in_pilot": True,
                "in_smoke20": record["sample_id"] in smoke_ids,
            })

        summary_rows.append({
            "dataset": dataset,
            "full_rows": len(full_records),
            "pilot_rows": len(pilot_records),
            "smoke20_rows": len(smoke_records),
            "pilot_strata": count_strata(pilot_records),
            "random_seed": RANDOM_SEED,
            "pilot_file": str(
                config["pilot"].relative_to(PROJECT_ROOT)
            ),
            "smoke20_file": str(
                config["smoke"].relative_to(PROJECT_ROOT)
            ),
            "pilot_sha256": file_sha256(config["pilot"]),
            "smoke20_sha256": file_sha256(config["smoke"]),
        })

        all_pilot.extend(pilot_records)
        all_smoke.extend(smoke_records)

        print(
            f"[SAVED] {config['pilot'].relative_to(PROJECT_ROOT)} "
            f"({len(pilot_records)} rows)"
        )
        print(
            f"[SAVED] {config['smoke'].relative_to(PROJECT_ROOT)} "
            f"({len(smoke_records)} rows)"
        )

    save_jsonl(all_pilot, COMBINED_PILOT)
    save_jsonl(all_smoke, COMBINED_SMOKE)
    save_summary(summary_rows)
    save_selection_manifest(selection_rows)
    save_format_doc()

    expected_pilot_total = sum(
        min(PILOT_SIZE, config["expected_full_rows"])
        for config in CONFIG.values()
    )
    expected_smoke_total = sum(
        min(SMOKE_SIZE, PILOT_SIZE, config["expected_full_rows"])
        for config in CONFIG.values()
    )

    if len(all_pilot) != expected_pilot_total:
        raise ValueError(
            f"Combined pilot expected {expected_pilot_total} rows, "
            f"found {len(all_pilot)}"
        )

    if len(all_smoke) != expected_smoke_total:
        raise ValueError(
            f"Combined smoke expected {expected_smoke_total} rows, "
            f"found {len(all_smoke)}"
        )

    print(f"\n[SAVED] {COMBINED_PILOT.relative_to(PROJECT_ROOT)}")
    print(f"[SAVED] {COMBINED_SMOKE.relative_to(PROJECT_ROOT)}")
    print(f"[SAVED] {SUMMARY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"[SAVED] {SELECTION_CSV.relative_to(PROJECT_ROOT)}")
    print(f"[SAVED] {FORMAT_DOC.relative_to(PROJECT_ROOT)}")
    print("=" * 72)
    print(
        f"Pilot sets complete: {len(all_pilot)} pilot rows and "
        f"{len(all_smoke)} smoke-test rows."
    )
    print("No API was called and no paid usage occurred.")
    print("Next: create the A/B run-output template.")
    print("=" * 72)


if __name__ == "__main__":
    main()