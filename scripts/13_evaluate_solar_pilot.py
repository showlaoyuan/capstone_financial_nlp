from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "pilot"
    / "combined_pilot_eval.jsonl"
)
DEFAULT_PROMPT_A = (
    PROJECT_ROOT
    / "outputs"
    / "ab_experiments"
    / "pilot"
    / "prompt_a"
    / "runs"
    / "solar_pro3"
    / "predictions.jsonl"
)
DEFAULT_PROMPT_B = (
    PROJECT_ROOT
    / "outputs"
    / "ab_experiments"
    / "pilot"
    / "prompt_b"
    / "runs"
    / "solar_pro3"
    / "predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "ab_experiments"
    / "pilot"
    / "evaluation"
    / "solar_pro3"
)

DATASETS = (
    "fpb",
    "fiqa_sa",
    "finqa",
    "financial_mmlu_ko",
)
SENTIMENT_LABELS = ("negative", "neutral", "positive")

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?:[eE][-+]?\d+)?"
)
FULL_NUMBER_PATTERN = re.compile(
    r"^\s*(?:\(\s*)?\$?\s*"
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"\s*%?\s*(?:\))?\s*$"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: list[dict[str, Any]] = []
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
                    f"Line {line_number} is not a JSON object: {path}"
                )
            records.append(record)
    return records


def save_json(path: Path, data: Any) -> None:
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


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
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


def validate_and_index(
    records: list[dict[str, Any]],
    name: str,
) -> dict[str, dict[str, Any]]:
    if len(records) != 800:
        raise ValueError(
            f"{name}: expected 800 records, found {len(records)}"
        )

    indexed: dict[str, dict[str, Any]] = {}
    for row_number, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(
                f"{name}: blank sample_id at row {row_number}"
            )
        if sample_id in indexed:
            raise ValueError(
                f"{name}: duplicate sample_id: {sample_id}"
            )
        indexed[sample_id] = record
    return indexed


def last_number(text: str) -> tuple[float, str, bool] | None:
    normalized = text.replace("−", "-")
    matches = list(NUMBER_PATTERN.finditer(normalized))
    if not matches:
        return None

    match = matches[-1]
    token = match.group().replace(",", "")
    value = float(token)

    if (
        FULL_NUMBER_PATTERN.fullmatch(normalized)
        and normalized.strip().startswith("(")
        and normalized.strip().endswith(")")
        and value > 0
    ):
        value = -value

    percent = "%" in normalized[match.end(): match.end() + 3]
    return value, token, percent


def precision_tolerance(token: str) -> float:
    mantissa = token.lower().split("e", maxsplit=1)[0]
    decimals = (
        len(mantissa.split(".", maxsplit=1)[1])
        if "." in mantissa
        else 0
    )
    return 0.5 * (10 ** (-decimals)) + 1e-9


def evaluate_sentiment(
    prediction: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    raw = prediction.strip()
    normalized = raw.lower()
    target = str(reference["label"]).strip().lower()
    format_valid = normalized in SENTIMENT_LABELS

    if format_valid:
        parsed: str | None = normalized
    else:
        found = [
            label
            for label in SENTIMENT_LABELS
            if re.search(rf"\b{label}\b", normalized)
        ]
        parsed = found[0] if len(found) == 1 else None

    answer_correct = parsed == target
    return {
        "parsed_prediction": parsed,
        "reference_answer": target,
        "answer_correct": answer_correct,
        "format_valid": format_valid,
        "strict_correct": answer_correct and format_valid,
    }


def evaluate_multiple_choice(
    prediction: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    raw = prediction.strip()
    target = str(reference["answer"]).strip()
    format_valid = bool(re.fullmatch(r"[1-5]", raw))

    if format_valid:
        parsed: str | None = raw
    else:
        found = re.findall(r"(?<!\d)[1-5](?!\d)", raw)
        parsed = found[-1] if found else None

    answer_correct = parsed == target
    return {
        "parsed_prediction": parsed,
        "reference_answer": target,
        "answer_correct": answer_correct,
        "format_valid": format_valid,
        "strict_correct": answer_correct and format_valid,
    }


def evaluate_numeric(
    prediction: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    raw = prediction.strip()
    raw_target = reference["answer"]

    # FinQA mainly contains numeric answers, but the full pilot also
    # contains categorical answers such as "yes" and "no".
    try:
        target = float(raw_target)
    except (TypeError, ValueError):
        target_text = str(raw_target).strip().lower()
        normalized = raw.lower()

        if target_text in {"yes", "no"}:
            format_valid = normalized in {"yes", "no"}
            found = re.findall(r"\b(?:yes|no)\b", normalized)
            parsed_text = (
                normalized
                if format_valid
                else found[-1] if found else None
            )
        else:
            format_valid = normalized == target_text
            parsed_text = normalized if normalized else None

        answer_correct = parsed_text == target_text
        return {
            "parsed_prediction": parsed_text,
            "reference_answer": target_text,
            "answer_correct": answer_correct,
            "format_valid": format_valid,
            "strict_correct": answer_correct and format_valid,
        }

    extracted = last_number(raw)
    format_valid = bool(FULL_NUMBER_PATTERN.fullmatch(raw))

    if extracted is None:
        return {
            "parsed_prediction": None,
            "reference_answer": target,
            "answer_correct": False,
            "format_valid": format_valid,
            "strict_correct": False,
        }

    value, token, percent = extracted
    base_tolerance = precision_tolerance(token)
    candidates = [(value, base_tolerance)]

    # A percentage may correspond either to a gold percentage value
    # (146.68% -> 146.67571) or a decimal value (8.2% -> 0.08207).
    if percent:
        candidates.append((value / 100, base_tolerance / 100))

    answer_correct = any(
        abs(candidate - target) <= max(tolerance, 1e-5)
        for candidate, tolerance in candidates
    )

    return {
        "parsed_prediction": value,
        "reference_answer": target,
        "answer_correct": answer_correct,
        "format_valid": format_valid,
        "strict_correct": answer_correct and format_valid,
    }


def evaluate_answer(
    dataset: str,
    prediction: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    if dataset in {"fpb", "fiqa_sa"}:
        return evaluate_sentiment(prediction, reference)
    if dataset == "financial_mmlu_ko":
        return evaluate_multiple_choice(prediction, reference)
    if dataset == "finqa":
        return evaluate_numeric(prediction, reference)
    raise ValueError(f"Unsupported dataset: {dataset}")


def evaluate_variant(
    variant: str,
    references: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if set(references) != set(predictions):
        missing = sorted(set(references) - set(predictions))
        extra = sorted(set(predictions) - set(references))
        raise ValueError(
            f"Prompt {variant} sample IDs are not aligned. "
            f"Missing={missing[:5]}, extra={extra[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for sample_id, source in references.items():
        prediction_record = predictions[sample_id]
        dataset = str(source["dataset"])
        raw_prediction = str(
            prediction_record.get("prediction") or ""
        )
        result = evaluate_answer(
            dataset,
            raw_prediction,
            source["reference"],
        )
        rows.append({
            "variant": variant,
            "sample_id": sample_id,
            "dataset": dataset,
            "task_type": source.get("task_type"),
            "raw_prediction": raw_prediction,
            **result,
            "latency_ms": prediction_record.get("latency_ms"),
            "estimated_cost_usd": prediction_record.get(
                "estimated_cost_usd"
            ),
            "status": prediction_record.get("status"),
        })
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(group)
        correct = sum(bool(row["answer_correct"]) for row in group)
        strict = sum(bool(row["strict_correct"]) for row in group)
        valid = sum(bool(row["format_valid"]) for row in group)
        return {
            "records": total,
            "answer_correct": correct,
            "answer_accuracy": round(correct / total, 4),
            "strict_correct": strict,
            "strict_accuracy": round(strict / total, 4),
            "format_valid": valid,
            "format_compliance": round(valid / total, 4),
        }

    by_dataset = {
        dataset: summarize_group([
            row for row in rows if row["dataset"] == dataset
        ])
        for dataset in DATASETS
    }

    latencies = [
        float(row["latency_ms"])
        for row in rows
        if row["latency_ms"] is not None
    ]
    total_cost = sum(
        float(row["estimated_cost_usd"] or 0)
        for row in rows
    )

    return {
        **summarize_group(rows),
        "by_dataset": by_dataset,
        "failed_api_calls": sum(
            row["status"] != "success" for row in rows
        ),
        "average_latency_ms": round(statistics.mean(latencies), 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "estimated_cost_usd": round(total_cost, 8),
    }


def compare_variants(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    indexed_a = {row["sample_id"]: row for row in rows_a}
    indexed_b = {row["sample_id"]: row for row in rows_b}
    categories: Counter[str] = Counter()
    comparison: list[dict[str, Any]] = []

    for sample_id, row_a in indexed_a.items():
        row_b = indexed_b[sample_id]
        correct_a = bool(row_a["answer_correct"])
        correct_b = bool(row_b["answer_correct"])

        if correct_a and correct_b:
            category = "both_correct"
        elif correct_a:
            category = "prompt_a_only_correct"
        elif correct_b:
            category = "prompt_b_only_correct"
        else:
            category = "both_wrong"

        categories[category] += 1
        comparison.append({
            "sample_id": sample_id,
            "dataset": row_a["dataset"],
            "reference_answer": row_a["reference_answer"],
            "prompt_a_prediction": row_a["raw_prediction"],
            "prompt_a_answer_correct": correct_a,
            "prompt_a_format_valid": row_a["format_valid"],
            "prompt_b_prediction": row_b["raw_prediction"],
            "prompt_b_answer_correct": correct_b,
            "prompt_b_format_valid": row_b["format_valid"],
            "comparison_category": category,
        })

    all_categories = (
        "both_correct",
        "prompt_a_only_correct",
        "prompt_b_only_correct",
        "both_wrong",
    )
    return ({key: categories[key] for key in all_categories}, comparison)


def choose_preliminary_winner(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
) -> dict[str, str]:
    criteria = (
        ("strict_correct", "strict correct answers"),
        ("answer_correct", "answer-correct records"),
        ("format_valid", "format-compliant records"),
    )
    for key, label in criteria:
        if summary_a[key] > summary_b[key]:
            return {
                "winner": "A",
                "reason": f"Prompt A has more {label}.",
            }
        if summary_b[key] > summary_a[key]:
            return {
                "winner": "B",
                "reason": f"Prompt B has more {label}.",
            }

    if summary_a["average_latency_ms"] < summary_b["average_latency_ms"]:
        return {
            "winner": "A",
            "reason": "Accuracy tied; Prompt A has lower average latency.",
        }
    if summary_b["average_latency_ms"] < summary_a["average_latency_ms"]:
        return {
            "winner": "B",
            "reason": "Accuracy tied; Prompt B has lower average latency.",
        }
    return {"winner": "tie", "reason": "All comparison criteria tied."}


def save_comparison_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    pair_counts: dict[str, int],
    recommendation: dict[str, str],
) -> None:
    print("=" * 76)
    print("Solar Pro 3 - Pilot A/B Evaluation")
    print("=" * 76)
    print(
        "Metric                 Prompt A        Prompt B\n"
        f"Answer accuracy        {summary_a['answer_correct']:>3}/800 "
        f"({summary_a['answer_accuracy']:.1%})   "
        f"{summary_b['answer_correct']:>3}/800 "
        f"({summary_b['answer_accuracy']:.1%})\n"
        f"Strict accuracy        {summary_a['strict_correct']:>3}/800 "
        f"({summary_a['strict_accuracy']:.1%})   "
        f"{summary_b['strict_correct']:>3}/800 "
        f"({summary_b['strict_accuracy']:.1%})\n"
        f"Format compliance      {summary_a['format_valid']:>3}/800 "
        f"({summary_a['format_compliance']:.1%})   "
        f"{summary_b['format_valid']:>3}/800 "
        f"({summary_b['format_compliance']:.1%})\n"
        f"Average latency        {summary_a['average_latency_ms']:>8.2f} ms   "
        f"{summary_b['average_latency_ms']:>8.2f} ms\n"
        f"Estimated cost         ${summary_a['estimated_cost_usd']:.8f}   "
        f"${summary_b['estimated_cost_usd']:.8f}"
    )
    print("-" * 76)
    for dataset in DATASETS:
        a = summary_a["by_dataset"][dataset]
        b = summary_b["by_dataset"][dataset]
        print(
            f"{dataset:<22} "
            f"A {a['answer_correct']:>3}/200 | "
            f"B {b['answer_correct']:>3}/200"
        )
    print("-" * 76)
    print(f"Both correct:          {pair_counts['both_correct']}")
    print(f"Prompt A only correct: {pair_counts['prompt_a_only_correct']}")
    print(f"Prompt B only correct: {pair_counts['prompt_b_only_correct']}")
    print(f"Both wrong:            {pair_counts['both_wrong']}")
    print("-" * 76)
    print(
        f"Pilot winner: Prompt {recommendation['winner']}\n"
        f"Reason: {recommendation['reason']}"
    )
    print(
        "Note: this is the formal result for the fixed 800-record pilot set. "
        "It should not be generalized to every financial NLP dataset."
    )
    print("=" * 76)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Solar Pro 3 pilot Prompt A versus Prompt B."
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--prompt-a", type=Path, default=DEFAULT_PROMPT_A)
    parser.add_argument("--prompt-b", type=Path, default=DEFAULT_PROMPT_B)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    try:
        reference_rows = load_jsonl(args.reference)
        prompt_a_rows = load_jsonl(args.prompt_a)
        prompt_b_rows = load_jsonl(args.prompt_b)

        references = validate_and_index(reference_rows, "reference")
        predictions_a = validate_and_index(prompt_a_rows, "Prompt A")
        predictions_b = validate_and_index(prompt_b_rows, "Prompt B")

        evaluated_a = evaluate_variant(
            "A", references, predictions_a
        )
        evaluated_b = evaluate_variant(
            "B", references, predictions_b
        )
    except Exception as error:
        print(f"[STOP] {type(error).__name__}: {error}")
        return 1

    summary_a = summarize_rows(evaluated_a)
    summary_b = summarize_rows(evaluated_b)
    pair_counts, comparison_rows = compare_variants(
        evaluated_a, evaluated_b
    )
    recommendation = choose_preliminary_winner(summary_a, summary_b)

    summary = {
        "experiment": "solar_pro3_pilot_prompt_ab",
        "scoring_rules": {
            "answer_accuracy": (
                "Parsed answer matches the reference. For FinQA, the last "
                "numeric value is evaluated with displayed-precision "
                "tolerance and percent/decimal equivalence."
            ),
            "strict_accuracy": (
                "The answer is correct and the response obeys the required "
                "answer-only format."
            ),
            "format_compliance": (
                "The complete response contains only the requested label, "
                "choice number, or numeric answer."
            ),
        },
        "prompt_a": summary_a,
        "prompt_b": summary_b,
        "paired_comparison": pair_counts,
        "pilot_recommendation": recommendation,
        "limitation": (
            "The fixed pilot contains 200 records per dataset. Results are "
            "specific to this sample, model version, prompts, and decoding "
            "configuration, so they are not a universal model claim."
        ),
    }

    output_dir: Path = args.output_dir
    summary_path = output_dir / "evaluation_summary.json"
    comparison_path = output_dir / "sample_comparison.csv"
    format_errors_path = output_dir / "format_errors.jsonl"

    save_json(summary_path, summary)
    save_comparison_csv(comparison_path, comparison_rows)
    save_jsonl(
        format_errors_path,
        [
            row
            for row in evaluated_a + evaluated_b
            if not row["format_valid"]
        ],
    )

    print_summary(
        summary_a,
        summary_b,
        pair_counts,
        recommendation,
    )
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {comparison_path}")
    print(f"[SAVED] {format_errors_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())