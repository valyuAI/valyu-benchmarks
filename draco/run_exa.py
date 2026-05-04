#!/usr/bin/env python3
"""
DRACO benchmark runner for Exa Deep Reasoning API.

Usage:
  python3 draco/run_exa.py --type deep-reasoning --workers 10
  python3 draco/run_exa.py --type deep-reasoning --dataset datasets/draco_subset.jsonl
"""

import os
import json
import time
import argparse
import jsonlines
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from exa_py import Exa

API_KEYS = [k for k in [os.environ.get("EXA_API_KEY")] if k]
if not API_KEYS:
    raise SystemExit("EXA_API_KEY not set in environment")


def _format_grounding(groundings) -> str:
    """Append a numbered references section from Exa grounding citations."""
    seen = {}
    order = []
    for g in groundings or []:
        for c in (g.citations or []):
            url = getattr(c, "url", None)
            if not url or url in seen:
                continue
            seen[url] = len(order) + 1
            order.append({"title": getattr(c, "title", "") or "", "url": url})
    if not order:
        return ""
    lines = ["", "## References", ""]
    for i, ref in enumerate(order, 1):
        lines.append(f"{i}. {ref['title']} — {ref['url']}")
    return "\n".join(lines)


def run_item(item: dict, search_type: str) -> dict:
    """Run a single DRACO item on Exa Deep Reasoning."""
    problem = item["problem"]
    client = Exa(API_KEYS[0])

    start = time.time()
    try:
        result = client.search(
            problem,
            num_results=10,
            output_schema={"type": "text"},
            type=search_type,
            contents={"highlights": True},
        )
        elapsed = time.time() - start

        output_obj = getattr(result, "output", None)
        content = ""
        groundings = []
        if output_obj is not None:
            content = getattr(output_obj, "content", "") or ""
            groundings = getattr(output_obj, "grounding", []) or []
        # Append references for downstream rubric grading
        full_md = (content or "") + _format_grounding(groundings)

        cost = getattr(result, "cost_dollars", None)
        cost_total = float(getattr(cost, "total", 0)) if cost else None

        return {
            "id": item["id"],
            "problem": problem,
            "domain": item.get("domain"),
            "answer": item.get("answer"),
            "output": full_md,
            "elapsed": elapsed,
            "provider": "exa",
            "model": search_type,
            "meta": {"cost_usd": cost_total, "request_id": getattr(result, "request_id", None)},
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR {item['id']}: {e}")
        return {
            "id": item["id"],
            "problem": problem,
            "domain": item.get("domain"),
            "answer": item.get("answer"),
            "output": None,
            "elapsed": elapsed,
            "provider": "exa",
            "model": search_type,
            "meta": {},
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run DRACO on Exa Deep Reasoning")
    parser.add_argument("--type", default="deep-reasoning", help="Exa search type tier")
    parser.add_argument("--dataset", default="datasets/draco.jsonl")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", help="Resume from existing output file (skip already-done IDs)")
    args = parser.parse_args()

    items = []
    with open(args.dataset) as f:
        for line in f:
            items.append(json.loads(line))
    if args.limit:
        items = items[: args.limit]

    date_str = datetime.now().strftime("%Y-%m-%d")
    safe_type = args.type.replace("/", "_")
    filename = f"draco_exa_{safe_type}_{date_str}.jsonl"
    output_path = Path("temp-results") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and Path(args.resume).exists():
        with jsonlines.open(args.resume) as r:
            for rec in r:
                if rec.get("error") is None and rec.get("output"):
                    done_ids.add(rec["id"])
        output_path = Path(args.resume)
        items = [it for it in items if it["id"] not in done_ids]

    print(f"Provider:  Exa Deep Reasoning")
    print(f"Tier:      {args.type}")
    print(f"Items:     {len(items)} (skipped {len(done_ids)} already done)")
    print(f"Workers:   {args.workers}")
    print(f"Output:    {output_path}")
    print()

    if not items:
        print("Nothing to do.")
        return

    results = []
    completed = 0

    mode = "a" if args.resume else "w"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_item, item, args.type): item for item in items}
        with jsonlines.open(output_path, mode) as writer:
            for future in as_completed(futures):
                result = future.result()
                writer.write(result)
                results.append(result)
                completed += 1

                status = "OK" if result["error"] is None else "ERR"
                elapsed_str = f"{result['elapsed']:.0f}s" if result["elapsed"] else "ERR"
                dom = (result.get("domain") or "")[:25]
                out_len = len(result.get("output") or "")
                print(f"[{completed:3}/{len(items)}] {status} {dom:<25} {elapsed_str}  {out_len}ch", flush=True)

    successful = [r for r in results if r["error"] is None]
    failed = len(results) - len(successful)
    avg_elapsed = sum(r["elapsed"] for r in successful) / len(successful) if successful else 0
    total_cost = sum(r["meta"].get("cost_usd") or 0 for r in successful)

    print(f"\nResults: {len(successful)}/{len(items)} successful", end="")
    if failed:
        print(f" ({failed} failed)", end="")
    print(f"\nAvg latency: {avg_elapsed:.1f}s")
    print(f"Total cost:  ${total_cost:.3f}")
    print(f"Output: {output_path}")
    print(f"\nEval: python3 draco/eval/rubric_eval.py --input {output_path}")


if __name__ == "__main__":
    main()
