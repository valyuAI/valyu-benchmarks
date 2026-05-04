#!/usr/bin/env python3
"""
DRACO benchmark runner for Parallel Task API.

Usage:
  python3 draco/run_parallel.py --processor ultra2x --workers 50
  python3 draco/run_parallel.py --processor ultra2x --dataset datasets/draco.jsonl
"""

import json
import time
import random
import argparse
import jsonlines
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from parallel import Parallel
from parallel.types import TaskSpecParam, TextSchemaParam

import os

API_KEYS = [k for k in [os.environ.get("PARALLEL_API_KEY")] if k]
if not API_KEYS:
    raise SystemExit("PARALLEL_API_KEY not set in environment")


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
POLL_INTERVAL_S = 20
POLL_DEADLINE_S = 3 * 3600  # 3 hours, ample for ultra8x's 2hr ceiling


def _poll_until_terminal(client: "Parallel", run_id: str) -> str:
    """Poll retrieve() until task hits a terminal status. Returns final status."""
    deadline = time.time() + POLL_DEADLINE_S
    consecutive_errors = 0
    while time.time() < deadline:
        try:
            tr = client.task_run.retrieve(run_id)
            consecutive_errors = 0
            if tr.status in TERMINAL_STATUSES:
                return tr.status
        except Exception:
            consecutive_errors += 1
            if consecutive_errors >= 6:
                raise
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"poll deadline exceeded for run {run_id}")


def run_item(item: dict, processor: str) -> dict:
    """Run a single DRACO item on Parallel Task API via submit + poll."""
    problem = item["problem"]
    key = random.choice(API_KEYS)
    client = Parallel(api_key=key, timeout=120, max_retries=3)

    start = time.time()
    run_id = None
    try:
        task_run = client.task_run.create(
            input=problem,
            processor=processor,
            task_spec=TaskSpecParam(output_schema=TextSchemaParam()),
        )
        run_id = task_run.run_id

        final_status = _poll_until_terminal(client, run_id)
        if final_status != "completed":
            raise RuntimeError(f"task ended with status={final_status}")

        result = client.task_run.result(run_id, api_timeout=60)
        elapsed = time.time() - start

        output = ""
        if hasattr(result, "output") and result.output:
            if hasattr(result.output, "content"):
                output = result.output.content or ""
            else:
                output = str(result.output)

        return {
            "id": item["id"],
            "problem": problem,
            "domain": item.get("domain"),
            "answer": item.get("answer"),
            "output": output,
            "elapsed": elapsed,
            "provider": "parallel",
            "model": processor,
            "meta": {"run_id": task_run.run_id},
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
            "provider": "parallel",
            "model": processor,
            "meta": {"run_id": run_id} if run_id else {},
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run DRACO on Parallel Task API")
    parser.add_argument("--processor", default="ultra2x", help="Parallel processor tier")
    parser.add_argument("--dataset", default="datasets/draco.jsonl")
    parser.add_argument("--workers", type=int, default=50, help="Concurrent workers")
    parser.add_argument("--limit", type=int, help="Limit items")
    parser.add_argument("--resume", help="Resume from existing output file (skip already-done IDs)")
    args = parser.parse_args()

    items = []
    with open(args.dataset) as f:
        for line in f:
            items.append(json.loads(line))
    if args.limit:
        items = items[: args.limit]

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"draco_parallel_{args.processor}_{date_str}.jsonl"
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

    print(f"Provider:  Parallel Task API")
    print(f"Processor: {args.processor}")
    print(f"Items:     {len(items)} (skipped {len(done_ids)} already done)")
    print(f"Workers:   {args.workers}")
    print(f"API keys:  {len(API_KEYS)} (rotating)")
    print(f"Output:    {output_path}")
    print()

    if not items:
        print("Nothing to do.")
        return

    results = []
    completed = 0

    mode = "a" if args.resume else "w"
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_item, item, args.processor): item
            for item in items
        }
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
                print(f"[{completed:3}/{len(items)}] {status} {dom:<25} {elapsed_str}  {out_len}ch")

    successful = [r for r in results if r["error"] is None]
    failed = len(results) - len(successful)
    avg_elapsed = sum(r["elapsed"] for r in successful) / len(successful) if successful else 0

    print(f"\nResults: {len(successful)}/{len(items)} successful", end="")
    if failed:
        print(f" ({failed} failed)", end="")
    print(f"\nAvg latency: {avg_elapsed:.0f}s")
    print(f"Output: {output_path}")
    print(f"\nEval: python3 draco/eval/rubric_eval.py --input {output_path}")


if __name__ == "__main__":
    main()
