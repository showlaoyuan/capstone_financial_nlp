from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROVIDER = "deepseek"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
THINKING_MODE = "disabled"

MAX_OUTPUT_TOKENS = 256
TEMPERATURE = 0
MAX_ATTEMPTS = 3
INITIAL_RETRY_WAIT_SECONDS = 2
REQUEST_INTERVAL_SECONDS = 0.3

CACHE_HIT_INPUT_PRICE_PER_MILLION = 0.003625
CACHE_MISS_INPUT_PRICE_PER_MILLION = 0.435
OUTPUT_PRICE_PER_MILLION = 0.87

LOG_COLUMNS = [
    "request_number",
    "scope",
    "variant",
    "sample_id",
    "dataset",
    "provider",
    "model",
    "attempt",
    "status",
    "http_status",
    "error_type",
    "error_message",
    "input_tokens",
    "cache_hit_input_tokens",
    "cache_miss_input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_ms",
    "estimated_cost_usd",
    "requested_at_utc",
    "completed_at_utc",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {path}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"Line {line_number} is not a JSON object."
                )

            records.append(record)

    return records


def save_jsonl_atomic(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    temporary_path.replace(path)


def save_json_atomic(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)


def validate_manifest(
    records: list[dict[str, Any]],
    expected_variant: str,
) -> None:
    if len(records) != 800:
        raise ValueError(
            f"Pilot test should contain 800 records, "
            f"but found {len(records)}."
        )

    required = {
        "request_number",
        "scope",
        "variant",
        "prompt_name",
        "sample_id",
        "dataset",
        "split",
        "system_prompt",
        "user_prompt",
    }

    sample_ids = []

    for row_number, record in enumerate(records, start=1):
        missing = required - set(record)

        if missing:
            raise ValueError(
                f"Row {row_number} is missing fields: "
                f"{sorted(missing)}"
            )

        if record["scope"] != "pilot":
            raise ValueError(
                f"Row {row_number} has an invalid scope."
            )

        if record["variant"] != expected_variant:
            raise ValueError(
                f"Row {row_number} does not belong to "
                f"Prompt {expected_variant}."
            )

        sample_ids.append(record["sample_id"])

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id found in manifest.")


def initialize_predictions(
    manifest: list[dict[str, Any]],
    predictions_path: Path,
) -> list[dict[str, Any]]:
    if predictions_path.exists():
        predictions = load_jsonl(predictions_path)

        if len(predictions) != len(manifest):
            raise ValueError(
                "Existing predictions file has a different row count."
            )

        manifest_ids = [
            record["sample_id"] for record in manifest
        ]
        prediction_ids = [
            record["sample_id"] for record in predictions
        ]

        if manifest_ids != prediction_ids:
            raise ValueError(
                "Existing predictions are not aligned "
                "with the request manifest."
            )

        return predictions

    return [
        {
            "request_number": record["request_number"],
            "scope": record["scope"],
            "variant": record["variant"],
            "prompt_name": record["prompt_name"],
            "sample_id": record["sample_id"],
            "dataset": record["dataset"],
            "split": record["split"],
            "provider": PROVIDER,
            "model": MODEL,
            "thinking_mode": THINKING_MODE,
            "prediction": None,
            "finish_reason": None,
            "status": "pending",
            "error_type": None,
            "error_message": None,
            "input_tokens": None,
            "cache_hit_input_tokens": None,
            "cache_miss_input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "latency_ms": None,
            "estimated_cost_usd": None,
            "response_id": None,
            "created_at_utc": None,
        }
        for record in manifest
    ]


def initialize_log(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=LOG_COLUMNS,
        )
        writer.writeheader()


def append_log(
    path: Path,
    row: dict[str, Any],
) -> None:
    with path.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=LOG_COLUMNS,
        )
        writer.writerow(row)


def read_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "input_tokens": 0,
            "cache_hit_input_tokens": 0,
            "cache_miss_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    input_tokens = int(
        getattr(usage, "prompt_tokens", 0) or 0
    )
    output_tokens = int(
        getattr(usage, "completion_tokens", 0) or 0
    )
    cache_hit_tokens = int(
        getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    )
    cache_miss_tokens = int(
        getattr(usage, "prompt_cache_miss_tokens", 0) or 0
    )

    if (
        input_tokens > 0
        and cache_hit_tokens == 0
        and cache_miss_tokens == 0
    ):
        cache_miss_tokens = input_tokens

    total_tokens = int(
        getattr(
            usage,
            "total_tokens",
            input_tokens + output_tokens,
        )
        or input_tokens + output_tokens
    )

    return {
        "input_tokens": input_tokens,
        "cache_hit_input_tokens": cache_hit_tokens,
        "cache_miss_input_tokens": cache_miss_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def estimate_cost_usd(
    cache_hit_tokens: int,
    cache_miss_tokens: int,
    output_tokens: int,
) -> float:
    return (
        cache_hit_tokens
        / 1_000_000
        * CACHE_HIT_INPUT_PRICE_PER_MILLION
        + cache_miss_tokens
        / 1_000_000
        * CACHE_MISS_INPUT_PRICE_PER_MILLION
        + output_tokens
        / 1_000_000
        * OUTPUT_PRICE_PER_MILLION
    )


def get_http_status(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)

    if isinstance(status_code, int):
        return status_code

    return None


def is_fatal_error(error: Exception) -> bool:
    return get_http_status(error) in {401, 403, 404}


def clean_error_message(error: Exception) -> str:
    return " ".join(str(error).split())


def build_metadata(
    predictions: list[dict[str, Any]],
    variant: str,
    status: str,
    started_at_utc: str,
    cost_limit_usd: float,
    run_limit: int | None,
) -> dict[str, Any]:
    successful = [
        row for row in predictions
        if row["status"] == "success"
    ]
    failed = [
        row for row in predictions
        if row["status"] == "failed"
    ]
    pending = [
        row for row in predictions
        if row["status"] == "pending"
    ]

    return {
        "experiment_name": "financial_nlp_prompt_ab",
        "scope": "pilot",
        "variant": variant,
        "status": status,
        "api_provider": PROVIDER,
        "model": MODEL,
        "thinking_mode": THINKING_MODE,
        "temperature": TEMPERATURE,
        "seed": None,
        "seed_note": (
            "No seed was sent because this DeepSeek endpoint "
            "does not document a seed parameter."
        ),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "cost_limit_usd": cost_limit_usd,
        "run_limit": run_limit,
        "record_count": len(predictions),
        "successful_calls": len(successful),
        "failed_calls": len(failed),
        "pending_calls": len(pending),
        "input_tokens": sum(
            int(row["input_tokens"] or 0)
            for row in successful
        ),
        "cache_hit_input_tokens": sum(
            int(row["cache_hit_input_tokens"] or 0)
            for row in successful
        ),
        "cache_miss_input_tokens": sum(
            int(row["cache_miss_input_tokens"] or 0)
            for row in successful
        ),
        "output_tokens": sum(
            int(row["output_tokens"] or 0)
            for row in successful
        ),
        "total_tokens": sum(
            int(row["total_tokens"] or 0)
            for row in successful
        ),
        "estimated_cost_usd": round(
            sum(
                float(row["estimated_cost_usd"] or 0)
                for row in successful
            ),
            8,
        ),
        "started_at_utc": started_at_utc,
        "updated_at_utc": utc_now(),
        "finished_at_utc": (
            utc_now()
            if status in {
                "completed",
                "completed_with_errors",
            }
            else None
        ),
        "resume_rule": (
            "Rows with status=success are skipped when rerun."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DeepSeek V4 Pro pilot test "
            "for Prompt A or Prompt B."
        )
    )
    parser.add_argument(
        "--variant",
        choices=["A", "B", "a", "b"],
        default="A",
        help="Prompt variant to run. Default: A",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of new records to process in this run. "
            "Use 5 for the first safety check; omit for all remaining rows."
        ),
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.50,
        help=(
            "Stop before starting another record when the cumulative "
            "estimated cost reaches this value. Default: 0.50"
        ),
    )
    args = parser.parse_args()
    variant = args.variant.upper()

    if args.limit is not None and not 1 <= args.limit <= 800:
        parser.error("--limit must be between 1 and 800")
    if args.max_cost_usd <= 0:
        parser.error("--max-cost-usd must be greater than 0")

    variant_dir = (
        PROJECT_ROOT
        / "outputs"
        / "ab_experiments"
        / "pilot"
        / f"prompt_{variant.lower()}"
    )
    manifest_path = (
        variant_dir / "request_manifest.jsonl"
    )

    run_dir = (
        variant_dir
        / "runs"
        / "deepseek_v4_pro"
    )
    predictions_path = run_dir / "predictions.jsonl"
    log_path = run_dir / "api_call_log.csv"
    metadata_path = run_dir / "run_metadata.json"

    print("=" * 72)
    print("Day 6 - DeepSeek Pilot Test")
    print("=" * 72)
    print(f"Variant: Prompt {variant}")
    print(f"Model: {MODEL}")
    print(f"New-record limit: {args.limit or 'all remaining'}")
    print(f"Cost limit: ${args.max_cost_usd:.2f}")
    print(f"Output: {run_dir.relative_to(PROJECT_ROOT)}")

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[STOP] python-dotenv is not installed.")
        print("Run: python -m pip install python-dotenv")
        return 1

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("[STOP] DEEPSEEK_API_KEY was not found.")
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("[STOP] openai is not installed.")
        print("Run: python -m pip install openai")
        return 1

    try:
        manifest = load_jsonl(manifest_path)
        validate_manifest(manifest, variant)
        predictions = initialize_predictions(
            manifest,
            predictions_path,
        )
    except Exception as error:
        print(f"[STOP] {type(error).__name__}: {error}")
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    initialize_log(log_path)
    save_jsonl_atomic(predictions_path, predictions)

    existing_metadata = {}

    if metadata_path.exists():
        try:
            existing_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            existing_metadata = {}

    started_at_utc = existing_metadata.get(
        "started_at_utc"
    ) or utc_now()

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    completed_before = sum(
        row["status"] == "success"
        for row in predictions
    )

    print(
        f"Already successful: {completed_before}/"
        f"{len(predictions)}"
    )
    print("Starting batch requests...")
    print("-" * 72)

    processed_this_run = 0

    try:
        for index, request_record in enumerate(
            manifest,
            start=1,
        ):
            prediction_record = predictions[index - 1]

            if prediction_record["status"] == "success":
                print(
                    f"[{index:04d}/800] SKIP "
                    f"{request_record['sample_id']}"
                )
                continue

            if (
                args.limit is not None
                and processed_this_run >= args.limit
            ):
                metadata = build_metadata(
                    predictions,
                    variant,
                    "paused_run_limit",
                    started_at_utc,
                    args.max_cost_usd,
                    args.limit,
                )
                save_json_atomic(metadata_path, metadata)
                print("-" * 72)
                print(
                    f"[PAUSED] Reached this run's limit: "
                    f"{args.limit} new records."
                )
                print(
                    "Successful overall: "
                    f"{metadata['successful_calls']}/800"
                )
                print(
                    "Estimated cost overall: "
                    f"${metadata['estimated_cost_usd']:.8f}"
                )
                print("Run the next command when ready to continue.")
                return 0

            current_cost = sum(
                float(row["estimated_cost_usd"] or 0)
                for row in predictions
                if row["status"] == "success"
            )
            if current_cost >= args.max_cost_usd:
                metadata = build_metadata(
                    predictions,
                    variant,
                    "stopped_cost_limit",
                    started_at_utc,
                    args.max_cost_usd,
                    args.limit,
                )
                save_json_atomic(metadata_path, metadata)
                print("-" * 72)
                print(
                    "[STOP] Cumulative estimated cost reached the "
                    f"${args.max_cost_usd:.2f} safety limit."
                )
                print(
                    "Increase --max-cost-usd only after checking "
                    "run_metadata.json."
                )
                return 2

            success = False

            for attempt in range(1, MAX_ATTEMPTS + 1):
                requested_at = utc_now()
                started_clock = time.perf_counter()

                try:
                    response = (
                        client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {
                                    "role": "system",
                                    "content": request_record[
                                        "system_prompt"
                                    ],
                                },
                                {
                                    "role": "user",
                                    "content": request_record[
                                        "user_prompt"
                                    ],
                                },
                            ],
                            max_tokens=MAX_OUTPUT_TOKENS,
                            temperature=TEMPERATURE,
                            stream=False,
                            extra_body={
                                "thinking": {
                                    "type": THINKING_MODE,
                                }
                            },
                        )
                    )

                    latency_ms = round(
                        (
                            time.perf_counter()
                            - started_clock
                        )
                        * 1000,
                        2,
                    )
                    completed_at = utc_now()

                    if not response.choices:
                        raise RuntimeError(
                            "DeepSeek returned no choices."
                        )

                    prediction = (
                        response
                        .choices[0]
                        .message
                        .content
                        or ""
                    ).strip()

                    finish_reason = (
                        response.choices[0].finish_reason
                    )
                    usage = read_usage(response)
                    cost = estimate_cost_usd(
                        usage[
                            "cache_hit_input_tokens"
                        ],
                        usage[
                            "cache_miss_input_tokens"
                        ],
                        usage["output_tokens"],
                    )

                    prediction_record.update({
                        "prediction": prediction,
                        "finish_reason": finish_reason,
                        "status": "success",
                        "error_type": None,
                        "error_message": None,
                        "input_tokens": (
                            usage["input_tokens"]
                        ),
                        "cache_hit_input_tokens": (
                            usage[
                                "cache_hit_input_tokens"
                            ]
                        ),
                        "cache_miss_input_tokens": (
                            usage[
                                "cache_miss_input_tokens"
                            ]
                        ),
                        "output_tokens": (
                            usage["output_tokens"]
                        ),
                        "total_tokens": (
                            usage["total_tokens"]
                        ),
                        "latency_ms": latency_ms,
                        "estimated_cost_usd": round(
                            cost,
                            8,
                        ),
                        "response_id": getattr(
                            response,
                            "id",
                            None,
                        ),
                        "created_at_utc": completed_at,
                    })

                    append_log(
                        log_path,
                        {
                            "request_number": (
                                request_record[
                                    "request_number"
                                ]
                            ),
                            "scope": "pilot",
                            "variant": variant,
                            "sample_id": (
                                request_record["sample_id"]
                            ),
                            "dataset": (
                                request_record["dataset"]
                            ),
                            "provider": PROVIDER,
                            "model": MODEL,
                            "attempt": attempt,
                            "status": "success",
                            "http_status": 200,
                            "error_type": "",
                            "error_message": "",
                            "input_tokens": (
                                usage["input_tokens"]
                            ),
                            "cache_hit_input_tokens": (
                                usage[
                                    "cache_hit_input_tokens"
                                ]
                            ),
                            "cache_miss_input_tokens": (
                                usage[
                                    "cache_miss_input_tokens"
                                ]
                            ),
                            "output_tokens": (
                                usage["output_tokens"]
                            ),
                            "total_tokens": (
                                usage["total_tokens"]
                            ),
                            "latency_ms": latency_ms,
                            "estimated_cost_usd": round(
                                cost,
                                8,
                            ),
                            "requested_at_utc": (
                                requested_at
                            ),
                            "completed_at_utc": (
                                completed_at
                            ),
                        },
                    )

                    save_jsonl_atomic(
                        predictions_path,
                        predictions,
                    )

                    print(
                        f"[{index:04d}/800] SUCCESS "
                        f"{request_record['dataset']} | "
                        f"{request_record['sample_id']} | "
                        f"${cost:.8f}"
                    )

                    success = True
                    break

                except Exception as error:
                    latency_ms = round(
                        (
                            time.perf_counter()
                            - started_clock
                        )
                        * 1000,
                        2,
                    )
                    completed_at = utc_now()
                    http_status = get_http_status(error)
                    error_message = clean_error_message(
                        error
                    )

                    append_log(
                        log_path,
                        {
                            "request_number": (
                                request_record[
                                    "request_number"
                                ]
                            ),
                            "scope": "pilot",
                            "variant": variant,
                            "sample_id": (
                                request_record["sample_id"]
                            ),
                            "dataset": (
                                request_record["dataset"]
                            ),
                            "provider": PROVIDER,
                            "model": MODEL,
                            "attempt": attempt,
                            "status": "failed",
                            "http_status": (
                                http_status or ""
                            ),
                            "error_type": (
                                type(error).__name__
                            ),
                            "error_message": error_message,
                            "input_tokens": "",
                            "cache_hit_input_tokens": "",
                            "cache_miss_input_tokens": "",
                            "output_tokens": "",
                            "total_tokens": "",
                            "latency_ms": latency_ms,
                            "estimated_cost_usd": "",
                            "requested_at_utc": (
                                requested_at
                            ),
                            "completed_at_utc": (
                                completed_at
                            ),
                        },
                    )

                    fatal = is_fatal_error(error)
                    last_attempt = (
                        attempt == MAX_ATTEMPTS
                    )

                    print(
                        f"[{index:04d}/800] FAILED "
                        f"attempt {attempt}: "
                        f"{type(error).__name__}"
                    )
                    print(f"  {error_message}")

                    if fatal or last_attempt:
                        prediction_record.update({
                            "prediction": None,
                            "finish_reason": None,
                            "status": "failed",
                            "error_type": (
                                type(error).__name__
                            ),
                            "error_message": error_message,
                            "input_tokens": None,
                            "cache_hit_input_tokens": None,
                            "cache_miss_input_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                            "latency_ms": latency_ms,
                            "estimated_cost_usd": None,
                            "response_id": None,
                            "created_at_utc": completed_at,
                        })

                        save_jsonl_atomic(
                            predictions_path,
                            predictions,
                        )

                    if fatal:
                        metadata = build_metadata(
                            predictions,
                            variant,
                            "stopped_fatal_error",
                            started_at_utc,
                            args.max_cost_usd,
                            args.limit,
                        )
                        save_json_atomic(
                            metadata_path,
                            metadata,
                        )
                        print(
                            "[STOP] Fatal API error. "
                            "Check the key, model, or account."
                        )
                        return 1

                    if not last_attempt:
                        wait_seconds = (
                            INITIAL_RETRY_WAIT_SECONDS
                            * (2 ** (attempt - 1))
                        )
                        print(
                            f"  Retrying in "
                            f"{wait_seconds} seconds..."
                        )
                        time.sleep(wait_seconds)

            metadata_status = (
                "running"
                if success
                else "running_with_errors"
            )
            metadata = build_metadata(
                predictions,
                variant,
                metadata_status,
                started_at_utc,
                args.max_cost_usd,
                args.limit,
            )
            save_json_atomic(
                metadata_path,
                metadata,
            )

            processed_this_run += 1

            time.sleep(REQUEST_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        metadata = build_metadata(
            predictions,
            variant,
            "interrupted",
            started_at_utc,
            args.max_cost_usd,
            args.limit,
        )
        save_json_atomic(metadata_path, metadata)

        print("\n[INTERRUPTED] Progress has been saved.")
        print("Run the same command to continue.")
        return 130

    successful_count = sum(
        row["status"] == "success"
        for row in predictions
    )
    failed_count = sum(
        row["status"] == "failed"
        for row in predictions
    )

    final_status = (
        "completed"
        if successful_count == len(predictions)
        else "completed_with_errors"
    )

    metadata = build_metadata(
        predictions,
        variant,
        final_status,
        started_at_utc,
        args.max_cost_usd,
        args.limit,
    )
    save_json_atomic(metadata_path, metadata)

    print("-" * 72)
    print("[FINISHED] DeepSeek pilot test completed.")
    print(f"Successful: {successful_count}/800")
    print(f"Failed: {failed_count}/800")
    print(
        "Estimated total cost: "
        f"${metadata['estimated_cost_usd']:.8f}"
    )
    print(
        f"[SAVED] "
        f"{predictions_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"[SAVED] "
        f"{log_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"[SAVED] "
        f"{metadata_path.relative_to(PROJECT_ROOT)}"
    )
    print("=" * 72)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())