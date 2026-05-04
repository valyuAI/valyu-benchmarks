#!/usr/bin/env python3
"""
DRACO benchmark runner for Tavily Research API.

Tavily's `/research` endpoint (GA Jan 2026) is their deepest research tier:
multi-step search + synthesis into a structured markdown report with cited
sources. It is the analogue to Valyu DR Heavy / Parallel Ultra8x / Exa
deep-reasoning. Two model tiers exist: `mini` (4-110 credits) and `pro`
(15-250 credits). We default to `pro` for the highest-compute comparison.

Usage:
  python3 draco/run_tavily.py --model pro --workers 4
  python3 draco/run_tavily.py --model pro --dataset datasets/draco_exa_21.jsonl
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

API_KEY = os.environ.get("TAVILY_API_KEY")
if not API_KEY:
    raise SystemExit("TAVILY_API_KEY not set in environment")

BASE_URL = "https://api.tavily.com"
SUBMIT_URL = f"{BASE_URL}/research"
RETRIEVE_URL = f"{BASE_URL}/research/{{request_id}}"

# Be generous: let Tavily's pro tier take whatever it needs. Most queries land
# in 3-15 min, but we don't aggressively cut them off.
MODEL_TIMEOUTS = {"mini": 600, "pro": 1800, "auto": 1800}
POLL_INTERVAL_S = 10
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

# PAYG credit price ($0.008/credit). Bounds for cost estimate per request.
CREDIT_PRICE_USD = 0.008
CREDIT_BOUNDS = {
    "mini": (4, 110),
    "pro": (15, 250),
    "auto": (4, 250),
}


def _format_sources(sources) -> str:
    """Append a numbered references section from Tavily sources list.

    Tavily's research output already embeds inline `[n]` markers and a
    `### Sources` block in the `content` field, so this is only used as a
    fallback when content is missing the trailing source list.
    """
    if not sources:
        return ""
    lines = ["", "## References", ""]
    for i, src in enumerate(sources, 1):
        title = (src.get("title") or "").strip()
        url = src.get("url") or ""
        if not url:
            continue
        if title:
            lines.append(f"{i}. {title} - {url}")
        else:
            lines.append(f"{i}. {url}")
    return "\n".join(lines)


def _submit(input_text: str, model: str, citation_format: str) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    body = {
        "input": input_text,
        "model": model,
        "citation_format": citation_format,
    }
    # Tavily's submit endpoint can take >60s under load; use 180s and retry on
    # transient network errors before giving up.
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(SUBMIT_URL, headers=headers, json=body, timeout=180)
            r.raise_for_status()
            j = r.json()
            rid = j.get("request_id")
            if not rid:
                raise RuntimeError(f"submit returned no request_id: {j}")
            return rid
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            time.sleep(min(30 * (attempt + 1), 90))
    raise last_err


def _retrieve(request_id: str) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(RETRIEVE_URL.format(request_id=request_id), headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()


def _poll_until_terminal(request_id: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    consecutive_errors = 0
    while time.time() < deadline:
        try:
            j = _retrieve(request_id)
            consecutive_errors = 0
            status = j.get("status")
            if status in TERMINAL_STATUSES:
                return j
        except Exception:
            consecutive_errors += 1
            if consecutive_errors >= 6:
                raise
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"poll deadline exceeded for request {request_id}")


def run_item(item: dict, model: str, citation_format: str) -> dict:
    """Run a single DRACO item on Tavily Research API via submit + poll."""
    problem = item["problem"]
    request_id = None
    start = time.time()
    try:
        request_id = _submit(problem, model, citation_format)
        timeout_s = MODEL_TIMEOUTS.get(model, 420)
        final = _poll_until_terminal(request_id, timeout_s)

        if final.get("status") != "completed":
            raise RuntimeError(f"task ended with status={final.get('status')} error={final.get('error')}")

        elapsed = time.time() - start
        content = (final.get("content") or "").strip()
        sources = final.get("sources") or []

        # Tavily's content already includes inline citations and a sources
        # block. Only append our own References if content lacks any source
        # listing (defensive fallback for rubric grading).
        if content and "Sources" not in content and "References" not in content:
            content = content + _format_sources(sources)

        # Cost estimate: per-request credit usage isn't returned, so report
        # the price-per-request bounds for the chosen model tier.
        lo, hi = CREDIT_BOUNDS.get(model, (0, 0))
        cost_lo = lo * CREDIT_PRICE_USD
        cost_hi = hi * CREDIT_PRICE_USD

        return {
            "id": item["id"],
            "problem": problem,
            "domain": item.get("domain"),
            "answer": item.get("answer"),
            "output": content,
            "elapsed": elapsed,
            "provider": "tavily",
            "model": model,
            "meta": {
                "request_id": request_id,
                "citation_format": citation_format,
                "num_sources": len(sources),
                "cost_usd_min": cost_lo,
                "cost_usd_max": cost_hi,
            },
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
            "provider": "tavily",
            "model": model,
            "meta": {"request_id": request_id} if request_id else {},
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run DRACO on Tavily Research API")
    parser.add_argument("--model", default="pro", choices=["mini", "pro", "auto"], help="Tavily research model tier")
    parser.add_argument("--dataset", default="datasets/draco.jsonl")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    parser.add_argument("--limit", type=int, help="Limit items")
    parser.add_argument("--citation-format", default="numbered", choices=["numbered", "mla", "apa", "chicago"])
    parser.add_argument("--resume", help="Resume from existing output file (skip already-done IDs)")
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
        filename = f"draco_tavily_{args.model}_{date_str}.jsonl"
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

    print(f"Provider:        Tavily Research API")
    print(f"Model:           {args.model}")
    print(f"Citation format: {args.citation_format}")
    print(f"Items:           {len(items)} (skipped {len(done_ids)} already done)")
    print(f"Workers:         {args.workers}")
    print(f"Output:          {output_path}")
    print()

    if not items:
        print("Nothing to do.")
        return

    results = []
    completed = 0

    mode = "a" if args.resume else "w"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_item, item, args.model, args.citation_format): item
            for item in items
        }
        with jsonlines.open(output_path, mode, flush=True) as writer:
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

    # Cost reported as bounds (Tavily doesn't return per-request usage)
    lo, hi = CREDIT_BOUNDS.get(args.model, (0, 0))
    n = len(successful)
    cost_lo_total = n * lo * CREDIT_PRICE_USD
    cost_hi_total = n * hi * CREDIT_PRICE_USD

    print(f"\nResults: {len(successful)}/{len(items)} successful", end="")
    if failed:
        print(f" ({failed} failed)", end="")
    print(f"\nAvg latency: {avg_elapsed:.1f}s")
    print(f"Cost bounds: ${cost_lo_total:.2f} - ${cost_hi_total:.2f} (PAYG, {lo}-{hi} credits/request)")
    print(f"Output: {output_path}")
    print(f"\nEval: python3 draco/eval/rubric_eval.py --input {output_path}")


if __name__ == "__main__":
    main()
