from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex",
    "silk", "rayon", "fabric",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange",
)


def flatten(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def count_terms(text: str, vocabulary: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    lowered = text.lower()
    for term in vocabulary:
        count = len(re.findall(rf"\b{re.escape(term)}\b", lowered))
        if count:
            counts[term] += count
    return counts


def type_name(value: object) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def collect_stats(path: Path, sample_limit: int = 3) -> dict[str, Any]:
    row_count = 0
    field_counts: Counter[str] = Counter()
    field_types: defaultdict[str, Counter[str]] = defaultdict(Counter)
    category_counts: Counter[str] = Counter()
    material_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    prices: list[float] = []
    samples: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                product = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error
            if not isinstance(product, dict):
                raise ValueError(f"Expected an object at line {line_number}")

            row_count += 1
            if len(samples) < sample_limit:
                samples.append(product)

            for key, value in product.items():
                field_counts[str(key)] += 1
                field_types[str(key)][type_name(value)] += 1

            categories = product.get("categories")
            if isinstance(categories, list):
                category_counts.update(str(value) for value in categories)
            elif categories not in (None, ""):
                category_counts[str(categories)] += 1

            searchable = flatten(product).lower()
            material_counts.update(count_terms(searchable, MATERIALS))
            color_counts.update(count_terms(searchable, COLORS))

            price = product.get("price")
            if isinstance(price, (int, float)) and not isinstance(price, bool):
                prices.append(float(price))

    fields: dict[str, dict[str, Any]] = {}
    for field_name in sorted(field_counts):
        fields[field_name] = {
            "present": field_counts[field_name],
            "missing": row_count - field_counts[field_name],
            "types": dict(sorted(field_types[field_name].items())),
        }

    price_stats: dict[str, Any] | None = None
    if prices:
        price_stats = {
            "count": len(prices),
            "min": round(min(prices), 2),
            "median": round(statistics.median(prices), 2),
            "mean": round(statistics.mean(prices), 2),
            "max": round(max(prices), 2),
        }

    return {
        "path": str(path),
        "rows": row_count,
        "fields": fields,
        "categories_top": category_counts.most_common(30),
        "materials_mentions": material_counts.most_common(),
        "colors_mentions": color_counts.most_common(),
        "prices": price_stats,
        "samples": samples,
    }


def print_report(stats: dict[str, Any]) -> None:
    print(f"Catalog: {stats['path']}")
    print(f"Rows: {stats['rows']}")

    print("\nFields:")
    for name, info in stats["fields"].items():
        types = ", ".join(f"{key}={value}" for key, value in info["types"].items())
        print(f"  {name}: present={info['present']} missing={info['missing']} types=({types})")

    print("\nTop categories:")
    for value, count in stats["categories_top"]:
        print(f"  {count:>6}  {value}")

    print("\nMaterial mentions:")
    for value, count in stats["materials_mentions"]:
        print(f"  {count:>6}  {value}")

    print("\nColor mentions:")
    for value, count in stats["colors_mentions"]:
        print(f"  {count:>6}  {value}")

    print("\nPrices:")
    if stats["prices"] is None:
        print("  no numeric prices")
    else:
        for key, value in stats["prices"].items():
            print(f"  {key}: {value}")

    print("\nSamples:")
    for index, sample in enumerate(stats["samples"], 1):
        print(f"  sample_{index}: {json.dumps(sample, ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize catalog.jsonl structure and common attributes")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--json", dest="json_path", type=Path, help="Write the full report as JSON")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample rows to include")
    args = parser.parse_args()

    if args.samples < 0:
        parser.error("--samples must be non-negative")
    stats = collect_stats(args.catalog, args.samples)
    print_report(stats)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report written to {args.json_path}")


if __name__ == "__main__":
    main()
