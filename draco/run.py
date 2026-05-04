#!/usr/bin/env python3
"""
DRACO benchmark runner.

Evaluates deep research providers (Valyu, Perplexity, OpenAI) on 100 real-world
research tasks across 10 domains. Results saved to temp-results/ for later eval.

Usage:
  python3 draco/run.py --provider valyu --domains Academic Finance
  python3 draco/run.py --provider perplexity --limit 10
  python3 draco/run.py --provider openai-o3 --domains Academic
"""

import os
import sys
import json
import time
import argparse
import jsonlines
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv(".env")

# Patch SDK bug: ImageMetadata.chart_type Literal is too restrictive - API returns
# values like "doughnut", "horizontalBar", "pie" which fail Pydantic validation.
try:
    from valyu.types.deepresearch import ImageMetadata
    from typing import Optional
    ImageMetadata.model_fields["chart_type"].annotation = Optional[str]
    ImageMetadata.model_fields["chart_type"].metadata = []
    ImageMetadata.model_rebuild(force=True)
except Exception:
    pass

PROVIDERS = ["valyu", "perplexity", "openai-o3", "openai-o4-mini", "parallel"]

DOMAINS = [
    "Academic",
    "Finance",
    "Law",
    "Medicine",
    "Technology",
    "General Knowledge",
    "UX Design",
    "Personalized Assistant",
    "Shopping/Product Comparison",
    "Needle in a Haystack",
]


def load_dataset(
    domains: Optional[list[str]] = None, limit: Optional[int] = None, dataset: Optional[str] = None
) -> list[dict]:
    dataset_path = Path(dataset) if dataset else Path(__file__).parent.parent / "datasets" / "draco.jsonl"
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Run: python3 draco/download.py")
        sys.exit(1)

    items = []
    with jsonlines.open(dataset_path) as reader:
        for item in reader:
            if domains and item["domain"] not in domains:
                continue
            items.append(item)

    if limit:
        items = items[:limit]
    return items


def run_valyu(problem: str, model: str = "heavy", search_type: str = "all") -> tuple[str, float, dict]:
    import requests as req
    from valyu import Valyu

    client = Valyu(api_key=os.getenv("VALYU_API_KEY"))
    start = time.time()

    response = client.deepresearch.create(
        input=problem,
        model=model,
        search={"search_type": search_type},
        code_execution=True,
    )
    if not response.success:
        raise RuntimeError(f"DeepResearch create failed: {response}")

    task_id = response.deepresearch_id
    base_url = client.base_url.rstrip("/")
    headers = client.headers

    # Poll raw JSON to avoid SDK Pydantic model rejecting new chart_type values
    terminal = {"completed", "failed", "cancelled"}
    max_wait, poll_interval = 3600, 15
    while time.time() - start < max_wait:
        r = req.get(f"{base_url}/deepresearch/tasks/{task_id}/status", headers=headers)
        if r.status_code in (401, 403, 429, 502, 503, 504):
            time.sleep(poll_interval)
            continue
        r.raise_for_status()
        data = r.json()
        if data.get("status") in terminal:
            break
        time.sleep(poll_interval)

    elapsed = time.time() - start

    if data.get("status") != "completed":
        raise RuntimeError(f"DeepResearch task {task_id} ended with status: {data.get('status')}")

    output = data.get("output") or ""
    if isinstance(output, dict):
        output = json.dumps(output)

    sources = [
        {"title": s.get("title"), "url": s.get("url"), "snippet": s.get("snippet")}
        for s in (data.get("sources") or [])
    ]
    cost = data.get("cost")
    return output, elapsed, {"sources": sources, "cost": cost}


def run_perplexity(problem: str) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai",
        timeout=3600,
    )
    start = time.time()

    response = client.chat.completions.create(
        model="sonar-deep-research",
        messages=[
            {
                "role": "system",
                "content": "You are a thorough research assistant. Provide comprehensive, well-cited research reports.",
            },
            {"role": "user", "content": problem},
        ],
    )
    elapsed = time.time() - start
    output = response.choices[0].message.content or ""
    usage = response.usage.__dict__ if response.usage else {}
    return output, elapsed, {"usage": usage}


def run_openai_dr(
    problem: str, model: str = "o3-deep-research"
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=7200)
    start = time.time()

    response = client.responses.create(
        model=model,
        input=problem,
        background=True,
        tools=[{"type": "web_search_preview"}],
    )

    while response.status not in ("completed", "failed", "cancelled"):
        time.sleep(30)
        response = client.responses.retrieve(response.id)

    elapsed = time.time() - start

    if response.status != "completed":
        raise RuntimeError(f"OpenAI DR failed with status: {response.status}")

    usage = response.usage.__dict__ if response.usage else {}
    return response.output_text, elapsed, {"usage": usage}


def run_parallel(problem: str, processor: str = "pro") -> tuple[str, float, dict]:
    from parallel import Parallel
    from parallel.types import TaskSpecParam, TextSchemaParam

    client = Parallel(api_key=os.getenv("PARALLEL_API_KEY"))
    start = time.time()

    task_run = client.task_run.create(
        input=problem,
        processor=processor,
        task_spec=TaskSpecParam(output_schema=TextSchemaParam()),
    )

    result = client.task_run.result(task_run.run_id, api_timeout=3600)
    elapsed = time.time() - start

    # Extract text content from TaskRunResult
    output = ""
    if hasattr(result, "output") and result.output:
        if hasattr(result.output, "content"):
            output = result.output.content or ""
        else:
            output = str(result.output)

    if not output:
        raise RuntimeError(f"Parallel task returned empty output for run {task_run.run_id}")

    return output, elapsed, {"run_id": task_run.run_id, "processor": processor}


def run_item(item: dict, provider: str, model: str, search_type: str = "all") -> dict:
    problem = item["problem"]
    try:
        if provider == "valyu":
            output, elapsed, meta = run_valyu(problem, model=model, search_type=search_type)
        elif provider == "perplexity":
            output, elapsed, meta = run_perplexity(problem)
        elif provider == "openai-o3":
            output, elapsed, meta = run_openai_dr(problem, "o3-deep-research")
        elif provider == "openai-o4-mini":
            output, elapsed, meta = run_openai_dr(problem, "o4-mini-deep-research")
        elif provider == "parallel":
            output, elapsed, meta = run_parallel(problem, processor=model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return {
            "id": item["id"],
            "problem": problem,
            "domain": item["domain"],
            "output": output,
            "elapsed": elapsed,
            "provider": provider,
            "model": model,
            "meta": meta,
            "error": None,
        }

    except Exception as e:
        print(f"  ERROR on {item['id'][:8]}: {e}")
        return {
            "id": item["id"],
            "problem": problem,
            "domain": item["domain"],
            "output": None,
            "elapsed": None,
            "provider": provider,
            "model": model,
            "meta": {},
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run DRACO benchmark")
    parser.add_argument("--provider", choices=PROVIDERS, default="valyu")
    parser.add_argument(
        "--model",
        default="heavy",
        help="Valyu model size: fast | standard | heavy | max",
    )
    parser.add_argument(
        "--domains", nargs="+", choices=DOMAINS, help="Filter to specific domains"
    )
    parser.add_argument("--limit", type=int, help="Limit number of items to run")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent workers (keep low for deep research APIs)",
    )
    parser.add_argument(
        "--output-dir",
        default="temp-results",
        help="Output directory (temp-results for staging, draco/outputs for publishing)",
    )
    parser.add_argument(
        "--search-type",
        default="all",
        choices=["all", "web", "proprietary"],
        help="Valyu search type: all | web | proprietary",
    )
    parser.add_argument(
        "--dataset",
        help="Path to custom dataset JSONL (default: datasets/draco.jsonl)",
    )
    args = parser.parse_args()

    items = load_dataset(domains=args.domains, limit=args.limit, dataset=args.dataset)
    if not items:
        print("No items found - check --domains filter")
        sys.exit(1)

    print(f"Provider: {args.provider}")
    if args.provider in ("valyu", "parallel"):
        print(f"Model:    {args.model}")
    print(f"Items:    {len(items)}")
    if args.domains:
        print(f"Domains:  {', '.join(args.domains)}")
    print()

    date_str = datetime.now().strftime("%Y-%m-%d")
    suffix = f"_{args.model}" if args.provider in ("valyu", "parallel") else ""
    search_suffix = f"_{args.search_type}" if args.search_type != "all" else ""
    filename = f"draco_{args.provider}{suffix}{search_suffix}_{date_str}.jsonl"
    output_path = Path(args.output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_item, item, args.provider, args.model, args.search_type): item
            for item in items
        }
        with jsonlines.open(output_path, "w") as writer:
            for future in as_completed(futures):
                result = future.result()
                writer.write(result)
                results.append(result)
                completed += 1

                status = "✓" if result["error"] is None else "✗"
                elapsed_str = (
                    f"{result['elapsed']:.0f}s" if result["elapsed"] else "ERR"
                )
                domain = result["domain"][:25]
                print(
                    f"[{completed:3}/{len(items)}] {status} {domain:<25} {elapsed_str}"
                )

    successful = [r for r in results if r["error"] is None]
    failed = len(results) - len(successful)
    avg_elapsed = (
        sum(r["elapsed"] for r in successful) / len(successful) if successful else 0
    )

    print(f"\nResults: {len(successful)}/{len(items)} successful", end="")
    if failed:
        print(f" ({failed} failed)", end="")
    print(f"\nAvg latency: {avg_elapsed:.0f}s")
    print(f"Output: {output_path}")
    print(f"\nNext - run eval:")
    print(
        f"  python3 draco/eval/rubric_eval.py --input {output_path} --workers 20"
    )


if __name__ == "__main__":
    main()
