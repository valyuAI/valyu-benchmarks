#!/usr/bin/env python3
"""
DRACO rubric evaluator using the official rubric library.

Uses the exact same evaluation methodology as the DRACO paper:
- rubric library (https://github.com/The-LLM-Data-Company/rubric)
- PerCriterionGrader with gemini-3-pro-preview judge
- Same system prompt, user prompt format, and scoring formula

Paper baselines (gemini-3-pro-preview judge, 5 runs) can be compared directly:
  Perplexity Deep Research: 70.5%  |  Gemini DR: 59.0%
  OpenAI o3: 52.1%                 |  OpenAI o4-mini: 41.9%

Usage:
  python3 draco/eval/rubric_eval.py --input temp-results/draco_valyu_heavy_2026-03-23.jsonl
  python3 draco/eval/rubric_eval.py --input temp-results/draco_valyu_heavy_2026-03-23.jsonl --runs 5
"""

import os
import json
import asyncio
import argparse
import jsonlines
from collections import defaultdict
from pathlib import Path

import litellm
from dotenv import load_dotenv
from rubric import Rubric, Criterion, EvaluationReport, PerCriterionOutput
from rubric.autograders import PerCriterionGrader

load_dotenv(".env.local")
load_dotenv(".env")

# Match paper's judge model exactly
JUDGE_MODEL = "gemini/gemini-3-pro-preview"
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY", "")


def make_generate_fn(model: str, semaphore: asyncio.Semaphore):
    """Create a generate_fn compatible with PerCriterionGrader."""

    async def generate_fn(system_prompt: str, user_prompt: str, **kwargs) -> PerCriterionOutput:
        for attempt in range(3):
            async with semaphore:
                try:
                    result = await litellm.acompletion(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                        max_tokens=4096,
                    )
                    content = result.choices[0].message.content
                    if not content:
                        raise ValueError("empty response")
                    # Extract JSON (handles markdown code blocks)
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start == -1 or end == 0:
                        raise ValueError(f"no JSON found: {content[:100]}")
                    parsed = json.loads(content[start:end])
                    verdict = parsed.get("criterion_status", "UNMET").upper()
                    if verdict not in ("MET", "UNMET"):
                        verdict = "UNMET"
                    return PerCriterionOutput(
                        criterion_status=verdict,
                        explanation=parsed.get("explanation", ""),
                    )
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2**attempt)
                        continue
                    # Final fallback
                    return PerCriterionOutput(
                        criterion_status="UNMET",
                        explanation=f"judge_error: {e}",
                    )

    return generate_fn


def build_rubric(rubric_dict: dict) -> Rubric:
    criteria = [
        Criterion(weight=c["weight"], requirement=c["requirement"])
        for section in rubric_dict.get("sections", [])
        for c in section.get("criteria", [])
    ]
    return Rubric(criteria)


def compute_pass_rate(report: EvaluationReport) -> float:
    """Pass rate: positive criteria MET + negative criteria UNMET, per paper Section 4.2."""
    if not report.report:
        return 0.0
    n = len(report.report)
    passed = sum(
        1
        for c in report.report
        if (c.weight > 0 and c.verdict == "MET")
        or (c.weight < 0 and c.verdict == "UNMET")
    )
    return passed / n if n > 0 else 0.0


async def evaluate_item(
    item: dict,
    rubric_dict: dict,
    grader: PerCriterionGrader,
    runs: int,
) -> dict:
    rubric = build_rubric(rubric_dict)
    if not rubric.rubric:
        return {"id": item["id"], "domain": item["domain"], "normalized_score": 0.0, "pass_rate": 0.0, "error": "no_criteria"}

    all_runs = []
    for _ in range(runs):
        report = await rubric.grade(
            to_grade=item["output"],
            query=item["problem"],
            autograder=grader,
        )
        pass_rate = compute_pass_rate(report)
        all_runs.append({
            "normalized_score": report.score,
            "pass_rate": pass_rate,
            "raw_score": report.raw_score,
            "criteria_results": [
                {"requirement": c.requirement, "weight": c.weight, "verdict": c.verdict, "explanation": c.reason}
                for c in (report.report or [])
            ],
        })

    avg_score = sum(r["normalized_score"] for r in all_runs) / len(all_runs)
    avg_pass_rate = sum(r["pass_rate"] for r in all_runs) / len(all_runs)

    return {
        "id": item["id"],
        "domain": item["domain"],
        "normalized_score": avg_score,
        "pass_rate": avg_pass_rate,
        "runs": all_runs,
        "elapsed": item.get("elapsed"),
        "provider": item.get("provider"),
        "model": item.get("model"),
    }


async def main_async(args):
    # Load inference results, dedup by id (keep first)
    seen_ids: set[str] = set()
    items = []
    with jsonlines.open(Path(args.input)) as reader:
        for item in reader:
            if item.get("output") is None:
                continue
            if item["id"] in seen_ids:
                continue
            if args.domains and item["domain"] not in args.domains:
                continue
            seen_ids.add(item["id"])
            items.append(item)

    if not items:
        print("No items with output found.")
        return

    print(f"Evaluating {len(items)} items")
    print(f"Judge:     {args.judge_model}")
    print(f"Runs/item: {args.runs}")
    print()

    # Load rubrics
    dataset_path = Path(__file__).parent.parent.parent / "datasets" / "draco.jsonl"
    rubrics: dict[str, dict] = {}
    with jsonlines.open(dataset_path) as reader:
        for row in reader:
            rubrics[row["id"]] = json.loads(row["answer"])

    semaphore = asyncio.Semaphore(args.workers)
    grader = PerCriterionGrader(generate_fn=make_generate_fn(args.judge_model, semaphore))

    total = len(items)
    done = [0]

    async def evaluate_with_log(item):
        r = await evaluate_item(item, rubrics.get(item["id"], {}), grader, args.runs)
        done[0] += 1
        err = f" [ERROR: {r.get('error')}]" if r.get("error") else ""
        print(f"[{done[0]:3}/{total}] {r['domain']:<35} {r['normalized_score']*100:5.1f}%  pass={r['pass_rate']*100:5.1f}%{err}", flush=True)
        return r

    results = await asyncio.gather(*[
        evaluate_with_log(item) for item in items
    ])

    domain_scores: dict[str, list[float]] = defaultdict(list)
    domain_pass_rates: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.get("error"):
            continue
        domain_scores[r["domain"]].append(r["normalized_score"])
        domain_pass_rates[r["domain"]].append(r["pass_rate"])

    overall_score = sum(r["normalized_score"] for r in results) / len(results)
    overall_pass_rate = sum(r["pass_rate"] for r in results) / len(results)
    latencies = [r["elapsed"] for r in results if r.get("elapsed")]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    print(f"\n{'Domain':<35} {'Score':>8} {'Pass Rate':>10} {'N':>4}")
    print("-" * 62)
    for domain in sorted(domain_scores):
        scores = domain_scores[domain]
        prs = domain_pass_rates[domain]
        print(f"{domain:<35} {sum(scores)/len(scores)*100:>7.1f}% {sum(prs)/len(prs)*100:>9.1f}% {len(scores):>4}")
    print("-" * 62)
    print(f"{'Overall':<35} {overall_score*100:>7.1f}% {overall_pass_rate*100:>9.1f}%")
    if avg_latency:
        print(f"Avg latency: {avg_latency:.0f}s")

    output_path = Path(args.input).with_suffix(".eval.json")
    with open(output_path, "w") as f:
        json.dump({
            "overall_normalized_score": overall_score,
            "overall_pass_rate": overall_pass_rate,
            "avg_latency_s": avg_latency,
            "judge_model": args.judge_model,
            "runs_per_item": args.runs,
            "n_items": len(results),
            "domain_scores": {d: sum(s)/len(s) for d, s in domain_scores.items()},
            "domain_pass_rates": {d: sum(s)/len(s) for d, s in domain_pass_rates.items()},
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DRACO benchmark results using official rubric library")
    parser.add_argument("--input", required=True, help="Path to inference JSONL")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="LiteLLM model string")
    parser.add_argument("--domains", nargs="+", choices=[
        "Academic", "Finance", "Law", "Medicine", "Technology",
        "General Knowledge", "UX Design", "Personalized Assistant",
        "Shopping/Product Comparison", "Needle in a Haystack",
    ])
    parser.add_argument("--workers", type=int, default=4, help="Concurrent judge calls")
    parser.add_argument("--runs", type=int, default=1, help="Eval runs per item (paper uses 5)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
