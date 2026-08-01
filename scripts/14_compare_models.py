from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = (
    PROJECT_ROOT / "outputs" / "ab_experiments" / "pilot" / "evaluation"
)
DEEPSEEK_SUMMARY = EVALUATION_ROOT / "deepseek_v4_pro" / "evaluation_summary.json"
SOLAR_SUMMARY = EVALUATION_ROOT / "solar_pro3" / "evaluation_summary.json"
OUTPUT_DIR = EVALUATION_ROOT / "model_comparison"

MODEL_NAMES = {
    "deepseek_v4_pro": "DeepSeek V4 Pro",
    "solar_pro3": "Solar Pro 3",
}
DATASETS = ("fpb", "fiqa_sa", "finqa", "financial_mmlu_ko")
METRICS = ("answer_accuracy", "strict_accuracy", "format_compliance")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rate(count: int, records: int) -> float:
    return count / records if records else 0.0


def normalized_variant(summary: dict[str, Any], variant: str) -> dict[str, Any]:
    raw = summary[f"prompt_{variant.lower()}"]
    records = int(raw["records"])
    result = dict(raw)

    # Recalculate rates from integer counts so comparisons do not depend on
    # the four-decimal values displayed in each evaluation summary.
    result["answer_accuracy"] = rate(int(raw["answer_correct"]), records)
    result["strict_accuracy"] = rate(int(raw["strict_correct"]), records)
    result["format_compliance"] = rate(int(raw["format_valid"]), records)

    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        item = dict(raw["by_dataset"][dataset])
        dataset_records = int(item["records"])
        item["answer_accuracy"] = rate(
            int(item["answer_correct"]), dataset_records
        )
        item["strict_accuracy"] = rate(
            int(item["strict_correct"]), dataset_records
        )
        item["format_compliance"] = rate(
            int(item["format_valid"]), dataset_records
        )
        datasets[dataset] = item
    result["by_dataset"] = datasets
    return result


def recommended_variant(summary: dict[str, Any]) -> str:
    recommendation = summary.get("pilot_recommendation", {})
    winner = str(recommendation.get("winner", "")).strip().upper()
    if winner in {"A", "B"}:
        return winner

    a = normalized_variant(summary, "A")
    b = normalized_variant(summary, "B")
    return "A" if a["strict_correct"] >= b["strict_correct"] else "B"


def validate(summary: dict[str, Any], expected_experiment: str) -> None:
    if summary.get("experiment") != expected_experiment:
        raise ValueError(
            f"Unexpected experiment name: {summary.get('experiment')!r}; "
            f"expected {expected_experiment!r}"
        )
    for variant in ("A", "B"):
        data = summary.get(f"prompt_{variant.lower()}", {})
        if data.get("records") != 800:
            raise ValueError(f"Prompt {variant} must contain exactly 800 records")
        missing = set(DATASETS) - set(data.get("by_dataset", {}))
        if missing:
            raise ValueError(f"Prompt {variant} is missing datasets: {sorted(missing)}")


def pp(value: float) -> float:
    return round(value * 100, 2)


def build_summary(
    deepseek: dict[str, Any], solar: dict[str, Any]
) -> dict[str, Any]:
    chosen_variants = {
        "deepseek_v4_pro": recommended_variant(deepseek),
        "solar_pro3": recommended_variant(solar),
    }
    source = {"deepseek_v4_pro": deepseek, "solar_pro3": solar}
    selected = {
        model: normalized_variant(source[model], variant)
        for model, variant in chosen_variants.items()
    }

    overall: dict[str, Any] = {}
    for metric in METRICS:
        ds_value = selected["deepseek_v4_pro"][metric]
        solar_value = selected["solar_pro3"][metric]
        overall[metric] = {
            "deepseek_v4_pro": ds_value,
            "solar_pro3": solar_value,
            "difference_deepseek_minus_solar_pp": pp(ds_value - solar_value),
            "winner": (
                "deepseek_v4_pro" if ds_value > solar_value
                else "solar_pro3" if solar_value > ds_value
                else "tie"
            ),
        }

    efficiency = {
        "average_latency_ms": {
            model: selected[model]["average_latency_ms"] for model in selected
        },
        "median_latency_ms": {
            model: selected[model]["median_latency_ms"] for model in selected
        },
        "estimated_cost_usd": {
            model: selected[model]["estimated_cost_usd"] for model in selected
        },
    }
    for item in efficiency.values():
        ds_value = float(item["deepseek_v4_pro"])
        solar_value = float(item["solar_pro3"])
        item["solar_reduction_percent"] = round(
            (ds_value - solar_value) / ds_value * 100, 2
        )

    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        ds_value = selected["deepseek_v4_pro"]["by_dataset"][dataset][
            "answer_accuracy"
        ]
        solar_value = selected["solar_pro3"]["by_dataset"][dataset][
            "answer_accuracy"
        ]
        by_dataset[dataset] = {
            "deepseek_v4_pro": ds_value,
            "solar_pro3": solar_value,
            "difference_deepseek_minus_solar_pp": pp(ds_value - solar_value),
            "winner": (
                "deepseek_v4_pro" if ds_value > solar_value
                else "solar_pro3" if solar_value > ds_value
                else "tie"
            ),
        }

    prompt_effects: dict[str, Any] = {}
    for model, model_summary in source.items():
        a = normalized_variant(model_summary, "A")
        b = normalized_variant(model_summary, "B")
        prompt_effects[model] = {
            "recommended_variant": chosen_variants[model],
            "b_minus_a_answer_accuracy_pp": pp(
                b["answer_accuracy"] - a["answer_accuracy"]
            ),
            "b_minus_a_strict_accuracy_pp": pp(
                b["strict_accuracy"] - a["strict_accuracy"]
            ),
            "b_minus_a_format_compliance_pp": pp(
                b["format_compliance"] - a["format_compliance"]
            ),
        }

    return {
        "experiment": "deepseek_v4_pro_vs_solar_pro3_pilot",
        "comparison_policy": (
            "Compare each model using the prompt variant selected by its own "
            "paired A/B pilot recommendation. Rates are recalculated from counts."
        ),
        "selected_prompt": chosen_variants,
        "overall": overall,
        "by_dataset_answer_accuracy": by_dataset,
        "efficiency": efficiency,
        "prompt_effects": prompt_effects,
        "interpretation": {
            "overall_accuracy": "DeepSeek V4 Pro performs better overall.",
            "task_strengths": (
                "Solar Pro 3 leads on FPB and FiQA-SA sentiment tasks; "
                "DeepSeek V4 Pro leads on FinQA numeric reasoning and "
                "financial_mmlu_ko multiple-choice questions."
            ),
            "efficiency": "Solar Pro 3 is faster and cheaper in this pilot.",
        },
        "limitation": (
            "The fixed pilot contains 200 records per dataset. The result is "
            "specific to this sample, model versions, prompts, decoding settings, "
            "API conditions, and the evaluation rules; it is not a universal claim "
            "about Chinese or Korean language models."
        ),
    }


def save_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metric, item in summary["overall"].items():
        rows.append({
            "section": "overall",
            "metric": metric,
            "deepseek_v4_pro": item["deepseek_v4_pro"],
            "solar_pro3": item["solar_pro3"],
            "difference_deepseek_minus_solar_pp": (
                item["difference_deepseek_minus_solar_pp"]
            ),
            "winner": item["winner"],
        })
    for dataset, item in summary["by_dataset_answer_accuracy"].items():
        rows.append({
            "section": "dataset_answer_accuracy",
            "metric": dataset,
            "deepseek_v4_pro": item["deepseek_v4_pro"],
            "solar_pro3": item["solar_pro3"],
            "difference_deepseek_minus_solar_pp": (
                item["difference_deepseek_minus_solar_pp"]
            ),
            "winner": item["winner"],
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(summary: dict[str, Any], path: Path) -> None:
    selected = summary["selected_prompt"]
    overall = summary["overall"]
    datasets = summary["by_dataset_answer_accuracy"]
    efficiency = summary["efficiency"]

    lines = [
        "# DeepSeek V4 Pro vs Solar Pro 3 — Pilot Comparison",
        "",
        f"Selected prompts: DeepSeek **{selected['deepseek_v4_pro']}**, "
        f"Solar **{selected['solar_pro3']}**.",
        "",
        "## Overall results",
        "",
        "| Metric | DeepSeek | Solar | Difference (DeepSeek − Solar) | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for metric in METRICS:
        item = overall[metric]
        lines.append(
            f"| {metric} | {item['deepseek_v4_pro']:.2%} | "
            f"{item['solar_pro3']:.2%} | "
            f"{item['difference_deepseek_minus_solar_pp']:+.2f} pp | "
            f"{MODEL_NAMES.get(item['winner'], 'Tie')} |"
        )

    lines.extend([
        "",
        "## Answer accuracy by dataset",
        "",
        "| Dataset | DeepSeek | Solar | Difference (DeepSeek − Solar) | Winner |",
        "|---|---:|---:|---:|---|",
    ])
    for dataset in DATASETS:
        item = datasets[dataset]
        lines.append(
            f"| {dataset} | {item['deepseek_v4_pro']:.2%} | "
            f"{item['solar_pro3']:.2%} | "
            f"{item['difference_deepseek_minus_solar_pp']:+.2f} pp | "
            f"{MODEL_NAMES.get(item['winner'], 'Tie')} |"
        )

    lines.extend([
        "",
        "## Efficiency",
        "",
        "| Metric | DeepSeek | Solar | Solar reduction |",
        "|---|---:|---:|---:|",
        (
            "| Average latency | "
            f"{efficiency['average_latency_ms']['deepseek_v4_pro']:.2f} ms | "
            f"{efficiency['average_latency_ms']['solar_pro3']:.2f} ms | "
            f"{efficiency['average_latency_ms']['solar_reduction_percent']:.2f}% |"
        ),
        (
            "| Median latency | "
            f"{efficiency['median_latency_ms']['deepseek_v4_pro']:.2f} ms | "
            f"{efficiency['median_latency_ms']['solar_pro3']:.2f} ms | "
            f"{efficiency['median_latency_ms']['solar_reduction_percent']:.2f}% |"
        ),
        (
            "| Estimated cost | "
            f"${efficiency['estimated_cost_usd']['deepseek_v4_pro']:.6f} | "
            f"${efficiency['estimated_cost_usd']['solar_pro3']:.6f} | "
            f"{efficiency['estimated_cost_usd']['solar_reduction_percent']:.2f}% |"
        ),
        "",
        "## Interpretation",
        "",
        "DeepSeek V4 Pro is stronger overall, especially on FinQA numeric reasoning "
        "and financial_mmlu_ko. Solar Pro 3 is stronger on the two sentiment tasks "
        "and is faster and cheaper in this pilot. This supports a task-dependent "
        "conclusion rather than a universal claim that one model is always better.",
        "",
        f"Limitation: {summary['limitation']}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    deepseek = load_json(DEEPSEEK_SUMMARY)
    solar = load_json(SOLAR_SUMMARY)
    validate(deepseek, "deepseek_v4_pro_pilot_prompt_ab")
    validate(solar, "solar_pro3_pilot_prompt_ab")

    summary = build_summary(deepseek, solar)
    json_path = OUTPUT_DIR / "model_comparison_summary.json"
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    markdown_path = OUTPUT_DIR / "MODEL_COMPARISON.md"
    save_json(json_path, summary)
    save_csv(summary, csv_path)
    save_markdown(summary, markdown_path)

    selected = summary["selected_prompt"]
    print("DeepSeek V4 Pro vs Solar Pro 3")
    print("=" * 36)
    print(
        f"Selected prompts: DeepSeek {selected['deepseek_v4_pro']}, "
        f"Solar {selected['solar_pro3']}"
    )
    for metric, item in summary["overall"].items():
        print(
            f"{metric}: DeepSeek={item['deepseek_v4_pro']:.2%}, "
            f"Solar={item['solar_pro3']:.2%}, "
            f"difference={item['difference_deepseek_minus_solar_pp']:+.2f} pp"
        )
    print("\nAnswer accuracy by dataset:")
    for dataset, item in summary["by_dataset_answer_accuracy"].items():
        print(
            f"  {dataset}: DeepSeek={item['deepseek_v4_pro']:.2%}, "
            f"Solar={item['solar_pro3']:.2%}, "
            f"winner={MODEL_NAMES.get(item['winner'], 'Tie')}"
        )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()