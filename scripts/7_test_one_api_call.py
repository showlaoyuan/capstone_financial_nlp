from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ab_experiments"
    / "smoke20"
    / "prompt_a"
    / "request_manifest.jsonl"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "ab_experiments"
    / "smoke20"
    / "prompt_a"
    / "one_record_test_result_deepseek_v4_pro.json"
)

PROVIDER = "deepseek"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
THINKING_MODE = "disabled"
MAX_OUTPUT_TOKENS = 256

# DeepSeek V4 Pro 当前官方价格：美元 / 100万 tokens
CACHE_HIT_INPUT_PRICE_PER_MILLION = 0.003625
CACHE_MISS_INPUT_PRICE_PER_MILLION = 0.435
OUTPUT_PRICE_PER_MILLION = 0.87


def load_first_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Request manifest not found:\n{path}\n\n"
            "Run scripts\\6_prepare_ab_experiment.py first."
        )

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

            required = {
                "request_number",
                "scope",
                "variant",
                "sample_id",
                "dataset",
                "system_prompt",
                "user_prompt",
            }
            missing = required - set(record)

            if missing:
                raise ValueError(
                    f"First request is missing fields: {sorted(missing)}"
                )

            return record

    raise ValueError(f"No usable records found in {path}")


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

    # 如果接口没有返回缓存明细，保守地按未命中缓存计算
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
    cache_hit_input_tokens: int,
    cache_miss_input_tokens: int,
    output_tokens: int,
) -> float:
    cache_hit_cost = (
        cache_hit_input_tokens
        / 1_000_000
        * CACHE_HIT_INPUT_PRICE_PER_MILLION
    )
    cache_miss_cost = (
        cache_miss_input_tokens
        / 1_000_000
        * CACHE_MISS_INPUT_PRICE_PER_MILLION
    )
    output_cost = (
        output_tokens
        / 1_000_000
        * OUTPUT_PRICE_PER_MILLION
    )

    return cache_hit_cost + cache_miss_cost + output_cost


def save_result(data: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def show_error_help(error: Exception) -> None:
    error_name = type(error).__name__
    error_message = str(error)

    print("\n[FAILED] DeepSeek API test failed.")
    print(f"Error type: {error_name}")
    print(f"Message: {error_message}")

    name_lower = error_name.lower()
    message_lower = error_message.lower()

    if (
        "authentication" in name_lower
        or "401" in error_message
        or "api key" in message_lower
    ):
        print(
            "\nPossible cause: DEEPSEEK_API_KEY is missing or invalid."
        )
        print(
            "Check the .env file in the project root. "
            "Do not share or upload the key."
        )

    elif (
        "ratelimit" in name_lower
        or "429" in error_message
        or "quota" in message_lower
        or "billing" in message_lower
        or "balance" in message_lower
    ):
        print(
            "\nPossible cause: insufficient balance, rate limit, "
            "or temporary capacity restriction."
        )

    elif (
        "connection" in name_lower
        or "timeout" in name_lower
    ):
        print(
            "\nPossible cause: network connection, VPN, firewall, "
            "or a temporary DeepSeek service problem."
        )

    elif (
        "notfound" in name_lower
        or "404" in error_message
        or "model" in message_lower
    ):
        print(
            "\nPossible cause: the selected DeepSeek model "
            "is not currently available to this account."
        )

    print(
        "\nNo prediction result was written because the request failed."
    )


def main() -> int:
    print("=" * 72)
    print("Day 6 - Test One DeepSeek API Call")
    print("=" * 72)

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[STOP] python-dotenv is not installed.")
        print("Run: python -m pip install python-dotenv")
        return 1

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("[STOP] DEEPSEEK_API_KEY is not loaded.")
        print(
            "Check the .env file in the project root:\n"
            "DEEPSEEK_API_KEY=your_real_key"
        )
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("[STOP] The OpenAI Python SDK is not installed.")
        print("Run: python -m pip install --upgrade openai")
        return 1

    try:
        request_record = load_first_record(MANIFEST_PATH)
    except Exception as error:
        print(f"[STOP] {type(error).__name__}: {error}")
        return 1

    print("[PASS] DeepSeek API key detected.")
    print(f"Provider: {PROVIDER}")
    print(f"Model: {MODEL}")
    print(f"Thinking mode: {THINKING_MODE}")
    print(f"Scope: {request_record['scope']}")
    print(f"Variant: {request_record['variant']}")
    print(f"Dataset: {request_record['dataset']}")
    print(f"Sample ID: {request_record['sample_id']}")
    print("Calling the DeepSeek API once...")

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": request_record["system_prompt"],
                },
                {
                    "role": "user",
                    "content": request_record["user_prompt"],
                },
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
            stream=False,
            extra_body={
                "thinking": {
                    "type": THINKING_MODE,
                }
            },
        )
    except Exception as error:
        show_error_help(error)
        return 1

    latency_ms = round(
        (time.perf_counter() - started_clock) * 1000,
        2,
    )
    finished_at = datetime.now(timezone.utc)

    if not response.choices:
        print("[FAILED] DeepSeek returned no choices.")
        return 1

    prediction = (
        response.choices[0].message.content or ""
    ).strip()

    usage = read_usage(response)

    estimated_cost_usd = estimate_cost_usd(
        usage["cache_hit_input_tokens"],
        usage["cache_miss_input_tokens"],
        usage["output_tokens"],
    )

    result = {
        "status": "success",
        "provider": PROVIDER,
        "request_number": request_record["request_number"],
        "scope": request_record["scope"],
        "variant": request_record["variant"],
        "sample_id": request_record["sample_id"],
        "dataset": request_record["dataset"],
        "model": MODEL,
        "thinking_mode": THINKING_MODE,
        "prediction": prediction,
        "input_tokens": usage["input_tokens"],
        "cache_hit_input_tokens": (
            usage["cache_hit_input_tokens"]
        ),
        "cache_miss_input_tokens": (
            usage["cache_miss_input_tokens"]
        ),
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_ms": latency_ms,
        "estimated_cost_usd": round(
            estimated_cost_usd,
            8,
        ),
        "response_id": getattr(response, "id", None),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
    }

    save_result(result)

    print("\n[SUCCESS] One DeepSeek API request completed.")
    print(f"Prediction: {prediction}")
    print(f"Input tokens: {usage['input_tokens']}")
    print(
        "Cache-hit input tokens: "
        f"{usage['cache_hit_input_tokens']}"
    )
    print(
        "Cache-miss input tokens: "
        f"{usage['cache_miss_input_tokens']}"
    )
    print(f"Output tokens: {usage['output_tokens']}")
    print(f"Total tokens: {usage['total_tokens']}")
    print(f"Latency: {latency_ms} ms")
    print(f"Estimated cost: ${estimated_cost_usd:.8f}")
    print(
        f"[SAVED] {RESULT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print("=" * 72)
    print(
        "The DeepSeek API connection works. "
        "Next: adapt the batch smoke-test script."
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())