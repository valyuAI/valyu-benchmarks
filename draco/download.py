#!/usr/bin/env python3
"""Download DRACO dataset from HuggingFace to datasets/draco.jsonl."""

import sys
from pathlib import Path
from collections import Counter

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets")
    sys.exit(1)

import jsonlines


def main():
    output_path = Path(__file__).parent.parent / "datasets" / "draco.jsonl"

    print("Downloading perplexity-ai/draco from HuggingFace...")
    ds = load_dataset("perplexity-ai/draco", split="test")
    print(f"Downloaded {len(ds)} items")

    with jsonlines.open(output_path, "w") as writer:
        for row in ds:
            writer.write(dict(row))

    domains = Counter(row["domain"] for row in ds)
    print(f"\nSaved to {output_path}")
    print("\nDomain distribution:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain:<40} {count}")


if __name__ == "__main__":
    main()
