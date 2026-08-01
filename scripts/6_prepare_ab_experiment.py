from __future__ import annotations

import csv
import hashlib
import json

from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PILOT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "pilot"
    / "combined_pilot_eval.jsonl"
)
SMOKE_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "pilot"
    / "combined_smoke20_eval.jsonl"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ab_experiments"
CONFIG_DIR = OUTPUT_ROOT / "config"
SMOKE_DIR = OUTPUT_ROOT / "smoke20"
PILOT_DIR = OUTPUT_ROOT / "pilot"
DOCS_DIR = PROJECT_ROOT / "docs"

PROMPT_CATALOG = CONFIG_DIR / "prompt_catalog.json"
EXPERIMENT_CONFIG = CONFIG_DIR / "experiment_config.json"
RUN_METADATA_TEMPLATE = CONFIG_DIR / "run_metadata_template.json"
API_LOG_TEMPLATE = CONFIG_DIR / "api_call_log_template.csv"
README_FILE = DOCS_DIR / "ab_experiment_guide.md"

EXPECTED_COUNTS = {
    "pilot": 800,
    "smoke20": 80,
}

VARIANTS = {
    "A": {
        "name": "direct_baseline",
        "system_prompt": (
            "You are a financial NLP evaluation assistant. "
            "Follow the task exactly. Return only the final answer "
            "in the format requested by the user prompt."
        ),
        "description": (
            "简洁基线提示词：直接完成任务，只返回最终答案。"
        ),
    },
    "B": {
        "name": "constraint_enhanced",
        "system_prompt": (
            "You are a careful financial NLP evaluation assistant. "
            "Read every instruction and all supplied evidence carefully. "
            "Reason privately, follow the requested output format exactly, "
            "and return only the final answer without explanations."
        ),
        "description": (
            "增强约束提示词：强调仔细阅读、内部推理和严格输出格式。"
        ),
    },
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run scripts\\5_build_pilot_sets.py first."
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(
                    f"{path}: blank line at {line_number}"
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}: invalid JSON at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"{path}: line {line_number} is not an object"
                )

            records.append(record)

    return records


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_records(
    records: list[dict],
    expected_count: int,
    scope: str,
) -> None:
    if len(records) != expected_count:
        raise ValueError(
            f"{scope}: expected {expected_count} rows, "
            f"found {len(records)}"
        )

    required_fields = {
        "id",
        "sample_id",
        "dataset",
        "split",
        "prompt",
        "reference",
    }

    sample_ids = []

    for row_number, record in enumerate(records, start=1):
        missing = required_fields - set(record)

        if missing:
            raise ValueError(
                f"{scope}: row {row_number} missing fields: "
                f"{sorted(missing)}"
            )

        sample_id = str(record["sample_id"]).strip()
        canonical_id = str(record["id"]).strip()

        if not sample_id:
            raise ValueError(
                f"{scope}: blank sample_id at row {row_number}"
            )

        if sample_id != canonical_id:
            raise ValueError(
                f"{scope}: sample_id != id at row {row_number}"
            )

        if not str(record["prompt"]).strip():
            raise ValueError(
                f"{scope}: blank prompt at row {row_number}"
            )

        if not isinstance(record["reference"], dict):
            raise TypeError(
                f"{scope}: reference must be an object "
                f"at row {row_number}"
            )

        sample_ids.append(sample_id)

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{scope}: duplicate sample_id found")


def create_request_manifest(
    records: list[dict],
    scope: str,
    variant: str,
) -> list[dict]:
    variant_config = VARIANTS[variant]
    output = []

    for request_number, record in enumerate(records, start=1):
        output.append({
            "request_number": request_number,
            "scope": scope,
            "variant": variant,
            "prompt_name": variant_config["name"],
            "sample_id": record["sample_id"],
            "dataset": record["dataset"],
            "split": record["split"],
            "system_prompt": variant_config["system_prompt"],
            "user_prompt": record["prompt"],
            "status": "pending",
        })

    return output


def create_prediction_template(
    manifest: list[dict],
) -> list[dict]:
    return [
        {
            "request_number": row["request_number"],
            "scope": row["scope"],
            "variant": row["variant"],
            "prompt_name": row["prompt_name"],
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "split": row["split"],
            "model": None,
            "prediction": None,
            "status": "pending",
            "error_type": None,
            "error_message": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "latency_ms": None,
            "estimated_cost_usd": None,
            "created_at_utc": None,
        }
        for row in manifest
    ]


def create_variant_files(
    records: list[dict],
    scope: str,
    scope_dir: Path,
    variant: str,
) -> dict:
    variant_dir = scope_dir / f"prompt_{variant.lower()}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    manifest = create_request_manifest(
        records,
        scope,
        variant,
    )
    predictions = create_prediction_template(manifest)

    manifest_path = variant_dir / "request_manifest.jsonl"
    predictions_path = variant_dir / "predictions.jsonl"
    run_metadata_path = variant_dir / "run_metadata.json"
    api_log_path = variant_dir / "api_call_log.csv"

    save_jsonl(manifest_path, manifest)
    save_jsonl(predictions_path, predictions)

    metadata = {
        "experiment_name": "financial_nlp_prompt_ab",
        "scope": scope,
        "variant": variant,
        "prompt_name": VARIANTS[variant]["name"],
        "status": "prepared_not_run",
        "model": None,
        "api_provider": None,
        "temperature": 0,
        "seed": 42,
        "record_count": len(records),
        "successful_calls": 0,
        "failed_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "started_at_utc": None,
        "finished_at_utc": None,
        "source_file": str(
            (
                SMOKE_FILE if scope == "smoke20" else PILOT_FILE
            ).relative_to(PROJECT_ROOT)
        ),
        "request_manifest": str(
            manifest_path.relative_to(PROJECT_ROOT)
        ),
        "predictions_file": str(
            predictions_path.relative_to(PROJECT_ROOT)
        ),
        "api_log_file": str(
            api_log_path.relative_to(PROJECT_ROOT)
        ),
        "source_sha256": sha256(
            SMOKE_FILE if scope == "smoke20" else PILOT_FILE
        ),
        "prepared_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "notes": (
            "The reference answer is intentionally excluded from "
            "request_manifest.jsonl to prevent label leakage."
        ),
    }
    save_json(run_metadata_path, metadata)
    save_api_log(api_log_path)

    return {
        "scope": scope,
        "variant": variant,
        "rows": len(manifest),
        "manifest_path": manifest_path,
        "predictions_path": predictions_path,
        "metadata_path": run_metadata_path,
        "api_log_path": api_log_path,
    }


def save_api_log(path: Path) -> None:
    columns = [
        "request_number",
        "scope",
        "variant",
        "sample_id",
        "dataset",
        "attempt",
        "status",
        "http_status",
        "error_type",
        "error_message",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_ms",
        "estimated_cost_usd",
        "requested_at_utc",
        "completed_at_utc",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()


def save_global_templates() -> None:
    save_json(
        PROMPT_CATALOG,
        {
            "experiment_name": "financial_nlp_prompt_ab",
            "variants": VARIANTS,
            "important_rule": (
                "A and B must use the same sample_id list, model, "
                "temperature, and decoding settings. Only the prompt "
                "variant may differ."
            ),
        },
    )

    save_json(
        EXPERIMENT_CONFIG,
        {
            "experiment_name": "financial_nlp_prompt_ab",
            "default_scope": "smoke20",
            "available_scopes": {
                "smoke20": {
                    "records": 80,
                    "purpose": "API connection and parsing test",
                },
                "pilot": {
                    "records": 800,
                    "purpose": "small formal A/B experiment",
                },
            },
            "variants": ["A", "B"],
            "recommended_first_run": {
                "scope": "smoke20",
                "variant": "A",
                "records": 80,
            },
            "generation_settings": {
                "temperature": 0,
                "seed": 42,
                "max_output_tokens": 256,
            },
            "retry_settings": {
                "max_attempts": 3,
                "initial_wait_seconds": 2,
            },
            "privacy_rule": (
                "Do not send reference answers to the model."
            ),
        },
    )

    save_json(
        RUN_METADATA_TEMPLATE,
        {
            "experiment_name": None,
            "scope": None,
            "variant": None,
            "prompt_name": None,
            "status": "prepared_not_run",
            "api_provider": None,
            "model": None,
            "temperature": 0,
            "seed": 42,
            "record_count": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "started_at_utc": None,
            "finished_at_utc": None,
            "notes": None,
        },
    )

    save_api_log(API_LOG_TEMPLATE)


def verify_ab_alignment(
    scope_dir: Path,
    expected_count: int,
) -> None:
    paths = {
        variant: (
            scope_dir
            / f"prompt_{variant.lower()}"
            / "request_manifest.jsonl"
        )
        for variant in VARIANTS
    }

    manifests = {
        variant: load_jsonl(path)
        for variant, path in paths.items()
    }

    for variant, rows in manifests.items():
        if len(rows) != expected_count:
            raise ValueError(
                f"{scope_dir.name} variant {variant}: "
                f"expected {expected_count}, found {len(rows)}"
            )

    ids_a = [row["sample_id"] for row in manifests["A"]]
    ids_b = [row["sample_id"] for row in manifests["B"]]

    if ids_a != ids_b:
        raise ValueError(
            f"{scope_dir.name}: A/B sample IDs are not aligned"
        )


def save_readme() -> None:
    README_FILE.parent.mkdir(parents=True, exist_ok=True)

    README_FILE.write_text(
        """# A/B Experiment Guide

## Purpose

This directory stores reproducible Prompt A/B experiment inputs,
predictions, API logs, and run metadata.

## Prompt variants

- Prompt A: direct baseline.
- Prompt B: stronger reading and output-format constraints.

The model, temperature, seed, sample IDs, and decoding parameters must
remain identical. Only the prompt variant may change.

## Scopes

- `smoke20`: 80 records total. Run this first to test API access,
  response parsing, retry behavior, and logging.
- `pilot`: 800 records total. Run only after smoke20 succeeds.

## Important files

Each scope and prompt variant contains:

- `request_manifest.jsonl`: model request input. It does not include
  reference answers.
- `predictions.jsonl`: empty prediction result template.
- `api_call_log.csv`: one row per API attempt.
- `run_metadata.json`: settings, token use, cost, and run status.

## Safety rule

The `reference` field is used only after prediction for evaluation.
Never include it in an API request.

## Current state

Running `scripts/6_prepare_ab_experiment.py` only creates files and
directories. It does not call an API and costs nothing.
""",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("Day 6 - Prepare A/B Experiment Structure")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")

    pilot_records = load_jsonl(PILOT_FILE)
    smoke_records = load_jsonl(SMOKE_FILE)

    validate_records(
        pilot_records,
        EXPECTED_COUNTS["pilot"],
        "pilot",
    )
    validate_records(
        smoke_records,
        EXPECTED_COUNTS["smoke20"],
        "smoke20",
    )

    save_global_templates()
    save_readme()

    results = []

    for scope, records, directory in [
        ("smoke20", smoke_records, SMOKE_DIR),
        ("pilot", pilot_records, PILOT_DIR),
    ]:
        print(f"\n[START] {scope}")

        for variant in VARIANTS:
            result = create_variant_files(
                records,
                scope,
                directory,
                variant,
            )
            results.append(result)

            print(
                f"[SAVED] "
                f"{result['manifest_path'].relative_to(PROJECT_ROOT)} "
                f"({result['rows']} rows)"
            )
            print(
                f"[SAVED] "
                f"{result['predictions_path'].relative_to(PROJECT_ROOT)}"
            )
            print(
                f"[SAVED] "
                f"{result['metadata_path'].relative_to(PROJECT_ROOT)}"
            )
            print(
                f"[SAVED] "
                f"{result['api_log_path'].relative_to(PROJECT_ROOT)}"
            )

        verify_ab_alignment(
            directory,
            EXPECTED_COUNTS[scope],
        )
        print(f"[PASS] {scope}: Prompt A/B sample IDs aligned")

    print(f"\n[SAVED] {PROMPT_CATALOG.relative_to(PROJECT_ROOT)}")
    print(f"[SAVED] {EXPERIMENT_CONFIG.relative_to(PROJECT_ROOT)}")
    print(
        f"[SAVED] "
        f"{RUN_METADATA_TEMPLATE.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"[SAVED] "
        f"{API_LOG_TEMPLATE.relative_to(PROJECT_ROOT)}"
    )
    print(f"[SAVED] {README_FILE.relative_to(PROJECT_ROOT)}")

    print("=" * 72)
    print("A/B experiment structure prepared successfully.")
    print("Smoke20: 80 records x 2 prompt variants.")
    print("Pilot: 800 records x 2 prompt variants.")
    print("No API was called and no paid usage occurred.")
    print("Next: configure the API and run Prompt A on smoke20.")
    print("=" * 72)


if __name__ == "__main__":
    main()