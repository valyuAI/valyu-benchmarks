#!/usr/bin/env python3
"""
ScholarQA-Multi Benchmark Runner - Valyu DeepResearch with Structured Output

Runs 108 multi-domain questions (CS, Bio, Physics) through DeepResearch
and outputs in format compatible with prometheus eval.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env.local"
if env_path.exists():
  load_dotenv(env_path)
else:
  load_dotenv()

from valyu import Valyu


def load_multi_questions(data_path: str) -> list[dict]:
  """Load Scholar-Multi questions from human_answers.json."""
  with open(data_path) as f:
    data = json.load(f)
  return data


def build_schema() -> dict:
  """Generic literature review structured output schema.
  Optimized for prometheus eval: coverage, relevance, organization."""
  return {
    "type": "object",
    "properties": {
      "introduction": {
        "type": "string",
        "description": (
          "2-3 sentences introducing the topic, its significance, and scope of the review. "
          "Include [N] citations to seminal or foundational works."
        ),
      },
      "sections": {
        "type": "array",
        "description": (
          "4-6 thematic sections providing comprehensive coverage of the question. "
          "Each section MUST address a distinct subtopic, perspective, or line of research. "
          "Cover: (1) key methods and approaches, (2) major findings and results, "
          "(3) contrasting viewpoints or debates, (4) recent advances and emerging trends. "
          "Cite diverse papers - aim for 3-5 unique citations per section. "
          "Every factual claim MUST have [N] citations (plain format, NOT [[N]](URL))."
        ),
        "items": {
          "type": "object",
          "properties": {
            "heading": {
              "type": "string",
              "description": "Clear, descriptive section heading that signals the specific subtopic",
            },
            "content": {
              "type": "string",
              "description": (
                "Dense academic prose synthesizing multiple sources. "
                "80-120 words per section. Include specific findings, quantitative results where available, "
                "and cite 3-5 different papers per section using [N] format. "
                "Compare and contrast different studies' findings."
              ),
            },
          },
          "required": ["heading", "content"],
        },
      },
      "conclusion": {
        "type": "string",
        "description": (
          "3-4 sentences synthesizing key findings across sections, identifying consensus, "
          "open questions, and future research directions. Include [N] citations."
        ),
      },
    },
    "required": ["introduction", "sections", "conclusion"],
  }


def build_strategy() -> str:
  """Generic strategy for literature review questions."""
  return "\n".join([
    "You are writing a comprehensive scientific literature review that will be evaluated on COVERAGE, RELEVANCE, and ORGANIZATION.",
    "",
    "COVERAGE (most important): Discuss ALL major lines of research related to the question.",
    "- Include multiple approaches, methods, frameworks, and theoretical perspectives.",
    "- Present contrasting findings or competing hypotheses where they exist.",
    "- Cite diverse sources - at least 10-15 unique papers across the review.",
    "- Go beyond the obvious: include important tangential findings that provide broader context.",
    "",
    "RELEVANCE: Every sentence must directly address the question asked.",
    "- Stay tightly focused on the specific topic.",
    "- Do not include filler or generic statements about the field.",
    "",
    "ORGANIZATION: Structure the review with clear thematic sections.",
    "- Use logical flow from foundational concepts to recent advances.",
    "- Each section should have a clear purpose and not overlap with others.",
    "- Use discourse markers and transitions between sections.",
    "",
    "Use [N] citations (NOT [[N]](URL)) for every factual claim.",
    "Academic tone. 400-600 words total.",
  ])


def structured_to_markdown(output: dict) -> str:
  """Convert structured JSON output to flat markdown."""
  parts = []

  intro = output.get("introduction", "")
  if intro:
    parts.append(intro)
    parts.append("")

  for section in output.get("sections", []):
    heading = section.get("heading", "")
    content = section.get("content", "")
    if heading:
      parts.append(f"## {heading}")
    if content:
      parts.append(content)
    parts.append("")

  conclusion = output.get("conclusion", "")
  if conclusion:
    parts.append("## Conclusion")
    parts.append(conclusion)

  text = "\n".join(parts)
  text = re.sub(r'\[\[(\d+)\]\]\([^)]+\)', r'[\1]', text)
  return text


def extract_sources(response) -> list[dict]:
  """Extract citation sources from DeepResearch response."""
  sources = []
  raw_sources = getattr(response, "sources", None) or []
  if isinstance(raw_sources, list) and raw_sources and isinstance(raw_sources[0], dict):
    for idx, src in enumerate(raw_sources):
      sources.append({
        "text": src.get("snippet", "") or src.get("content", ""),
        "title": src.get("title", ""),
        "url": src.get("url", ""),
      })
  else:
    for idx, src in enumerate(raw_sources):
      sources.append({
        "text": getattr(src, "snippet", "") or getattr(src, "content", ""),
        "title": getattr(src, "title", ""),
        "url": getattr(src, "url", ""),
      })
  return sources


def run_question(client: Valyu, item: dict, model: str, idx: int, total: int) -> dict:
  """Run DeepResearch on a single question."""
  query = item["input"]
  item_id = item.get("id", idx)

  print(f"\n[{idx+1}/{total}] id={item_id} subject={item.get('subject', '?')}")
  print(f"  Q: {query[:100]}...")

  schema = build_schema()
  strategy = build_strategy()
  start = time.time()

  try:
    for create_attempt in range(5):
      response = client.deepresearch.create(
        input=query,
        model=model,
        output_formats=[schema],
        strategy=strategy,
        search={"search_type": "all"},
        code_execution=False,
      )
      if response.success:
        break
      if "Forbidden" in str(response.error) and create_attempt < 4:
        wait = 5 * (create_attempt + 1)
        print(f"  Forbidden, retrying in {wait}s (attempt {create_attempt+2}/5)")
        time.sleep(wait)
        continue
      raise RuntimeError(f"Create failed: {response.error}")

    task_id = response.deepresearch_id
    print(f"  Task: {task_id}")

    try:
      result = client.deepresearch.wait(
        task_id,
        poll_interval=10,
        max_wait_time=1800,
        on_progress=lambda s: None,
      )
    except Exception as wait_err:
      if "validation error" in str(wait_err).lower():
        print(f"  SDK validation error, polling raw API...")
        import requests
        headers = {"x-api-key": os.getenv("VALYU_API_KEY")}
        base_url = os.getenv("VALYU_API_BASE", "https://api.valyu.ai/v1")
        for _ in range(180):
          time.sleep(10)
          resp = requests.get(f"{base_url}/deepresearch/{task_id}", headers=headers)
          data = resp.json()
          if data.get("status") == "completed":
            class _Result:
              pass
            result = _Result()
            result.status = "completed"
            result.output = data.get("output")
            result.sources = data.get("sources", [])
            result.usage = None
            break
          elif data.get("status") in ("failed", "cancelled"):
            raise RuntimeError(f"Task {data['status']}: {data.get('error', 'unknown')}")
        else:
          raise RuntimeError("Task did not complete within 1800 seconds")
      else:
        raise

    if getattr(result, "status", None) != "completed":
      raise RuntimeError(f"Task {result.status}: {getattr(result, 'error', 'unknown')}")

    elapsed = time.time() - start
    cost = getattr(getattr(result, "usage", None), "total_cost", 0) or 0

    raw_output = result.output
    if isinstance(raw_output, str):
      try:
        raw_output = json.loads(raw_output)
      except json.JSONDecodeError:
        pass

    if isinstance(raw_output, dict):
      answer_text = structured_to_markdown(raw_output)
    else:
      answer_text = str(raw_output) if raw_output else ""
      answer_text = re.sub(r'\[\[(\d+)\]\]\([^)]+\)', r'[\1]', answer_text)

    sources = extract_sources(result)
    sources.insert(0, {"text": "", "title": "", "url": ""})

    citation_count = len(re.findall(r'\[\d+\]', answer_text))
    word_count = len(answer_text.split())

    print(f"  Done in {elapsed:.0f}s | ${cost:.4f} | {word_count} words | {citation_count} citations")

    return {
      "input": query,
      "output": answer_text,
      "ctxs": sources,
      "id": item_id,
      "subject": item.get("subject", ""),
      "task_id": task_id,
      "elapsed": elapsed,
      "cost": cost,
    }

  except Exception as e:
    elapsed = time.time() - start
    print(f"  FAILED ({elapsed:.0f}s): {e}")
    return {
      "input": query,
      "output": "",
      "ctxs": [],
      "id": item_id,
      "subject": item.get("subject", ""),
      "error": str(e),
      "elapsed": elapsed,
    }


def main():
  parser = argparse.ArgumentParser(description="ScholarQA-Multi Benchmark Runner")
  parser.add_argument("--data", required=True, help="Path to human_answers.json from ScholarQA-Multi dataset")
  parser.add_argument("--sample", type=int, help="Number of questions to run")
  parser.add_argument("--model", default="fast", choices=["fast", "standard", "heavy", "max"])
  parser.add_argument("--output-dir", default="scholarqa/outputs/multi")
  parser.add_argument("--workers", type=int, default=3)
  args = parser.parse_args()

  api_key = os.getenv("VALYU_API_KEY")
  if not api_key:
    print("ERROR: VALYU_API_KEY not set")
    return

  client = Valyu(api_key=api_key)
  questions = load_multi_questions(args.data)
  print(f"Loaded {len(questions)} questions")

  if args.sample and args.sample < len(questions):
    questions = questions[:args.sample]
    print(f"Sampling first {args.sample} questions")

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  output_file = output_dir / f"valyu_scholarqa_multi_{args.model}.jsonl"

  # Resume: load already-completed questions
  done_inputs = set()
  indexed_results = [None] * len(questions)
  if output_file.exists():
    with open(output_file) as f:
      for line in f:
        if line.strip():
          r = json.loads(line)
          done_inputs.add(r["input"].strip())
          # Place in correct index
          for qi, q in enumerate(questions):
            if q["input"].strip() == r["input"].strip():
              indexed_results[qi] = r
              break
    print(f"Resuming: {len(done_inputs)} already done")

  remaining = [(i, q) for i, q in enumerate(questions) if q["input"].strip() not in done_inputs]

  print(f"\nModel: {args.model}")
  print(f"Workers: {args.workers}")
  print(f"Questions: {len(remaining)} remaining (of {len(questions)})")
  print(f"Output: {output_file}")
  print(f"{'='*60}")

  total_cost = 0.0

  with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {
      executor.submit(run_question, client, q, args.model, i, len(questions)): i
      for i, q in remaining
    }
    for future in as_completed(futures):
      idx = futures[future]
      result = future.result()
      indexed_results[idx] = result
      total_cost += result.get("cost", 0)

      # Write incrementally
      with open(output_file, "w") as f:
        for r in indexed_results:
          if r is not None:
            f.write(json.dumps(r) + "\n")

  # Final write
  with open(output_file, "w") as f:
    for r in indexed_results:
      if r is not None:
        f.write(json.dumps(r) + "\n")

  successes = sum(1 for r in indexed_results if r and not r.get("error"))
  print(f"\n{'='*60}")
  print(f"Complete: {successes}/{len(questions)} successful")
  print(f"Total cost: ${total_cost:.2f}")
  print(f"Output: {output_file}")


if __name__ == "__main__":
  main()
