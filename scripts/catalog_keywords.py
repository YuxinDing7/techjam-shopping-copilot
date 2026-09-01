from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*", re.IGNORECASE)

STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "amazon", "available", "brand", "buy", "clothing", "item", "made", "new", "product", "sale",
    "size", "style", "women", "woman", "men", "man",
}

FIELDS = ("title", "features", "description", "details", "categories", "store")


def flatten(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in STOPWORDS and not token.isdigit()
    ]


def phrases(tokens: list[str], size: int) -> list[str]:
    return [" ".join(tokens[index:index + size]) for index in range(len(tokens) - size + 1)]


def add_counts(
    value: object,
    term_counts: Counter[str],
    document_counts: Counter[str],
    phrase_counts: Counter[str],
    phrase_document_counts: Counter[str],
) -> None:
    tokens = tokenize(flatten(value))
    term_counts.update(tokens)
    document_counts.update(set(tokens))
    bigrams = phrases(tokens, 2)
    phrase_counts.update(bigrams)
    phrase_document_counts.update(set(bigrams))


def ranked_terms(
    counts: Counter[str],
    document_counts: Counter[str],
    row_count: int,
    limit: int,
    min_document_frequency: int,
    max_document_coverage: float,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for term, frequency in counts.items():
        doc_frequency = document_counts[term]
        if (
            doc_frequency < min_document_frequency
            or (row_count and doc_frequency / row_count > max_document_coverage)
        ):
            continue
        idf = math.log((1 + row_count) / (1 + doc_frequency)) + 1.0
        score = frequency * idf
        ranked.append({
            "term": term,
            "frequency": frequency,
            "document_frequency": doc_frequency,
            "document_coverage": round(doc_frequency / row_count, 6) if row_count else 0.0,
            "idf": round(idf, 6),
            "tfidf_like_score": round(score, 6),
        })
    ranked.sort(key=lambda item: (-item["tfidf_like_score"], -item["document_frequency"], item["term"]))
    return ranked[:limit]


def collect_keywords(
    path: Path,
    limit: int,
    min_document_frequency: int,
    max_document_coverage: float,
    sample_limit: int,
) -> dict[str, Any]:
    row_count = 0
    field_term_counts: dict[str, Counter[str]] = {field: Counter() for field in FIELDS}
    field_document_counts: dict[str, Counter[str]] = {field: Counter() for field in FIELDS}
    field_phrase_counts: dict[str, Counter[str]] = {field: Counter() for field in FIELDS}
    field_phrase_document_counts: dict[str, Counter[str]] = {field: Counter() for field in FIELDS}
    combined_counts: Counter[str] = Counter()
    combined_document_counts: Counter[str] = Counter()
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
                samples.append({
                    "parent_asin": product.get("parent_asin"),
                    "title": product.get("title"),
                    "categories": product.get("categories"),
                })

            product_tokens: set[str] = set()
            for field in FIELDS:
                value = product.get(field)
                add_counts(
                    value,
                    field_term_counts[field],
                    field_document_counts[field],
                    field_phrase_counts[field],
                    field_phrase_document_counts[field],
                )
                product_tokens.update(tokenize(flatten(value)))
            combined_counts.update(product_tokens)
            combined_document_counts.update(product_tokens)

    fields: dict[str, Any] = {}
    for field in FIELDS:
        fields[field] = {
            "top_terms": ranked_terms(
                field_term_counts[field],
                field_document_counts[field],
                row_count,
                limit,
                min_document_frequency,
                max_document_coverage,
            ),
            "top_phrases": ranked_terms(
                field_phrase_counts[field],
                field_phrase_document_counts[field],
                row_count,
                limit,
                min_document_frequency,
                max_document_coverage,
            ),
        }

    return {
        "path": str(path),
        "rows": row_count,
        "tokenization": {
            "fields": list(FIELDS),
            "min_token_length": 3,
            "stopword_count": len(STOPWORDS),
            "min_document_frequency": min_document_frequency,
            "max_document_coverage": max_document_coverage,
        },
        "top_terms": ranked_terms(
            combined_counts,
            combined_document_counts,
            row_count,
            limit,
            min_document_frequency,
            max_document_coverage,
        ),
        "fields": fields,
        "samples": samples,
    }


def print_report(report: dict[str, Any], limit: int) -> None:
    print(f"Catalog: {report['path']}")
    print(f"Rows: {report['rows']}")
    print(f"Minimum document frequency: {report['tokenization']['min_document_frequency']}")
    print(f"Maximum document coverage: {report['tokenization']['max_document_coverage']}")

    print("\nTop combined terms:")
    for item in report["top_terms"][:limit]:
        print(
            f"  {item['term']:<24} frequency={item['frequency']:<7} "
            f"documents={item['document_frequency']:<7} score={item['tfidf_like_score']}"
        )

    for field, values in report["fields"].items():
        print(f"\nTop terms in {field}:")
        for item in values["top_terms"][:limit]:
            print(
                f"  {item['term']:<24} frequency={item['frequency']:<7} "
                f"documents={item['document_frequency']:<7} score={item['tfidf_like_score']}"
            )
        print(f"Top phrases in {field}:")
        for item in values["top_phrases"][:limit]:
            print(
                f"  {item['term']:<32} frequency={item['frequency']:<7} "
                f"documents={item['document_frequency']:<7} score={item['tfidf_like_score']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract catalog keywords and phrases for attribute design")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/catalog_keywords.json"))
    parser.add_argument("--limit", type=int, default=30, help="Number of terms and phrases per section")
    parser.add_argument("--min-df", type=int, default=5, help="Minimum number of products containing a term")
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=0.8,
        help="Ignore terms appearing in more than this fraction of products",
    )
    parser.add_argument("--samples", type=int, default=3, help="Number of lightweight samples to include")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.min_df < 1:
        parser.error("--min-df must be positive")
    if not 0 < args.max_coverage <= 1:
        parser.error("--max-coverage must be greater than 0 and at most 1")
    if args.samples < 0:
        parser.error("--samples must be non-negative")

    report = collect_keywords(args.catalog, args.limit, args.min_df, args.max_coverage, args.samples)
    print_report(report, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nJSON report written to {args.output}")


if __name__ == "__main__":
    main()
