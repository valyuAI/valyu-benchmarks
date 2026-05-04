#!/usr/bin/env python3
"""
DRACO benchmark runner for You.com Research API.

Tiers (highest to lowest quality):
  - exhaustive : <300s  ($450 / 1K req = $0.45)
  - deep       : <120s  ($100 / 1K req = $0.10)
  - standard   : ~10-30s ($50 / 1K req = $0.05)
  - lite       : <10s   ($12 / 1K req = $0.012)

Auth: X-API-Key header. Set YDC_API_KEY in env.

Usage:
  YDC_API_KEY=... python3 draco/run_youcom.py --tier exhaustive --workers 5 \
      --dataset datasets/draco_youcom_25.jsonl
"""

import os
import json
import time
import argparse
import jsonlines
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("YDC_API_KEY")
if not API_KEY:
    raise SystemExit("YDC_API_KEY not set in environment")

ENDPOINT = "https://api.you.com/v1/research"

# Cost per request in USD (per 1K requests pricing from public docs)
TIER_COST_USD = {
    "ulow": 0.0,
    "lite": 0.012,
    "standard": 0.050,
    "deep": 0.100,
    "exhaustive": 0.450,
}

# Per-tier soft client timeouts (seconds). exhaustive is bounded <300s.
TIER_TIMEOUT = {
    "ulow": 60,
    "lite": 60,
    "standard": 90,
    "deep": 240,
    "exhaustive": 600,
}


def _format_sources(sources) -> str:
    """Append a numbered References section from You.com sources[]."""
    if not sources:
        return ""
    lines = ["", "## References", ""]
    for i, src in enumerate(sources, 1):
        title = (src.get("title") or "").strip()
        url = (src.get("url") or "").strip()
        if not url:
            continue
        if title:
            lines.append(f"{i}. {title} - {url}")
        else:
            lines.append(f"{i}. {url}")
    return "\n".join(lines) if len(lines) > 3 else ""


def run_item(item: dict, tier: str, max_retries: int = 2) -> dict:
    """Run a single DRACO item on the You.com Research API."""
    problem = item["problem"]
    timeout = TIER_TIMEOUT.get(tier, 600)

    payload = {"input": problem, "research_effort": tier}
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
    }

    start = time.time()
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                ENDPOINT, json=payload, headers=headers, timeout=timeout
            )
            elapsed = time.time() - start

            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                # 429/5xx -> backoff and retry; 4xx (other) -> hard fail
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(last_err)

            data = resp.json()
            output_obj = data.get("output") or {}
            content = output_obj.get("content") or ""
            sources = output_obj.get("sources") or []
            full_md = content + _format_sources(sources)

            request_id = (
                resp.headers.get("x-request-id")
                or resp.headers.get("X-Request-Id")
                or data.get("request_id")
            )

            return {
                "id": item["id"],
                "problem": problem,
                "domain": item.get("domain"),
                "answer": item.get("answer"),
                "output": full_md,
                "elapsed": elapsed,
                "provider": "youcom",
                "model": f"research-{tier}",
                "meta": {
                    "cost_usd": TIER_COST_USD.get(tier),
                    "request_id": request_id,
                    "num_sources": len(sources),
                },
                "error": None,
            }

        except requests.exceptions.Timeout:
            last_err = f"timeout after {timeout}s"
            if attempt < max_retries:
                continue
            break
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries and "HTTP 4" not in last_err:
                time.sleep(2 ** attempt)
                continue
            break

    elapsed = time.time() - start
    print(f"  ERROR {item['id']}: {last_err}")
    return {
        "id": item["id"],
        "problem": problem,
        "domain": item.get("domain"),
        "answer": item.get("answer"),
        "output": None,
        "elapsed": elapsed,
        "provider": "youcom",
        "model": f"research-{tier}",
        "meta": {"cost_usd": None, "request_id": None},
        "error": last_err,
    }


def main():
    parser = argparse.ArgumentParser(description="Run DRACO on You.com Research API")
    parser.add_argument(
        "--tier",
        default="exhaustive",
        choices=["ulow", "lite", "standard", "deep", "exhaustive"],
        help="research_effort tier (default: exhaustive — highest public quality)",
    )
    parser.add_argument("--dataset", default="datasets/draco.jsonl")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", help="Resume from existing output file (skip done IDs)")
    parser.add_argument("--output", help="Override default output path")
    args = parser.parse_args()

    items = []
    with open(args.dataset) as f:
        for line in f:
            items.append(json.loads(line))
    if args.limit:
        items = items[: args.limit]

    if args.output:
        output_path = Path(args.output)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"draco_youcom_{args.tier}_{date_str}.jsonl"
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

    print(f"Provider:  You.com Research API")
    print(f"Endpoint:  {ENDPOINT}")
    print(f"Tier:      {args.tier} (${TIER_COST_USD.get(args.tier):.3f}/req)")
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
        futures = {executor.submit(run_item, item, args.tier): item for item in items}
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
                print(
                    f"[{completed:3}/{len(items)}] {status} {dom:<25} {elapsed_str}  {out_len}ch",
                    flush=True,
                )

    successful = [r for r in results if r["error"] is None]
    failed = len(results) - len(successful)
    avg_elapsed = sum(r["elapsed"] for r in successful) / len(successful) if successful else 0
    total_cost = sum((r["meta"].get("cost_usd") or 0) for r in successful)

    print(f"\nResults: {len(successful)}/{len(results)} successful", end="")
    if failed:
        print(f" ({failed} failed)", end="")
    print(f"\nAvg latency: {avg_elapsed:.1f}s")
    print(f"Total cost:  ${total_cost:.3f}")
    print(f"Output: {output_path}")
    print(f"\nEval: python3 draco/eval/rubric_eval.py --input {output_path}")


if __name__ == "__main__":
    main()
