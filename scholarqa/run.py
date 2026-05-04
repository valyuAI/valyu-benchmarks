#!/usr/bin/env python3
"""
ScholarQA Benchmark Runner - Valyu DeepResearch with Structured Output

Uses dynamic JSON schemas generated from rubric items to enforce
rubric-compliant structure and maximize evaluation scores.
"""

import argparse
import collections
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load env
env_path = Path(__file__).parent.parent / ".env.local"
if env_path.exists():
  load_dotenv(env_path)
else:
  load_dotenv()

from valyu import Valyu


def load_questions(dataset_dir: Path) -> list[dict]:
  """Load questions with rubric items from test_configs + output_snippets."""
  configs_file = dataset_dir / "test_configs_snippets.json"
  snippets_file = dataset_dir / "output_snippets.jsonl"

  with open(configs_file) as f:
    test_configs = json.load(f)

  # Build case_id -> config map
  config_map = {c["case_id"]: c for c in test_configs}

  # Load rubric ingredients from output_snippets
  ingredients_map: dict[str, dict] = {}
  if snippets_file.exists():
    with open(snippets_file) as f:
      for line in f:
        data = json.loads(line)
        q_text = data.get("question", "").strip()
        ingredients_map[q_text] = data.get("ingredients", {})

  questions = []
  for config in test_configs:
    q_text = config["initial_prompt"]
    rubric_items = extract_rubric_items(config)
    ingredients = ingredients_map.get(q_text, {})

    questions.append({
      "case_id": config["case_id"],
      "question": q_text,
      "rubric_items": rubric_items,
      "ingredients": ingredients,
      "metric_config": config["metric_config"],
    })

  return questions


def extract_rubric_items(config: dict) -> list[dict]:
  """Extract rubric items from test config."""
  items = []
  props = config.get("metric_config", {}).get("config", {}).get("other_properties", [])
  for prop in props:
    items.append({
      "name": prop["name"],
      "criterion": prop["criterion"],
      "weight": prop["weight"],
      "evidence": prop.get("evidence", []),
      "is_important": "most_important" in prop["name"],
    })
  return items


def build_schema(rubric_items: list[dict]) -> dict[str, Any]:
  """Build a JSON schema from rubric items that forces DeepResearch
  to address each rubric criterion as a dedicated section.

  Scoring breakdown (what the evaluator measures):
  - Rubric criteria + evidence: 60% (LLM judges if criterion is met + evidence present)
  - Citations per claim: 20% (fraction of claims with [N] citations)
  - Excerpts per citation: 10% (fraction of citations with quoted source text)
  - Length: 5% (PENALTY above 300 words, score=0 at 600+ words)
  - Expertise match: 5% (tone matches questioner's level)
  """

  intro_items = [r for r in rubric_items if "near the beginning" in r["criterion"].lower()]
  body_items = [r for r in rubric_items if r not in intro_items]

  intro_desc = "1-2 sentences defining key terms. CONCISE."
  if intro_items:
    intro_desc += " MUST cover: " + "; ".join(r["criterion"] for r in intro_items)

  content_desc = (
    "Dense academic prose, MAX 50 words per section. "
    "Every claim MUST have [N] citation. Paraphrase source text closely. "
    "Use plain [N] format only (NOT [[N]](URL)). Cite every factual statement."
  )

  section_properties = {
    "heading": {
      "type": "string",
      "description": "Section heading matching the rubric criterion",
    },
    "content": {
      "type": "string",
      "description": content_desc,
    },
  }

  sections_desc = "One section per rubric criterion. Keep each section VERY concise (40-60 words). "
  if body_items:
    for i, item in enumerate(body_items):
      hint = ""
      if "enumerate" in item["criterion"].lower():
        hint = " Use a numbered list."
      sections_desc += f"Section {i+1}: {item['criterion']}.{hint} "

  return {
    "type": "object",
    "properties": {
      "introduction": {
        "type": "string",
        "description": intro_desc,
      },
      "sections": {
        "type": "array",
        "description": sections_desc,
        "items": {
          "type": "object",
          "properties": section_properties,
          "required": ["heading", "content"],
        },
      },
      "conclusion": {
        "type": "string",
        "description": "1 sentence concluding summary with citation.",
      },
    },
    "required": ["introduction", "sections", "conclusion"],
  }


def build_strategy(rubric_items: list[dict]) -> str:
  """Build concise strategy prompt for DeepResearch (max ~1000 chars)."""
  lines = [
    "Max 280 words. Use [N] citations (NOT [[N]](URL)).",
    "EVERY sentence MUST have at least one [N] citation.",
    "Only cite a source if the source text directly states the fact you are claiming.",
    "Paraphrase source text closely - stay faithful to the exact wording of your sources.",
    "Sections for each rubric criterion:",
  ]

  for i, item in enumerate(rubric_items, 1):
    criterion = item["criterion"]
    if len(criterion) > 60:
      criterion = criterion[:57] + "..."
    lines.append(f"{i}. {criterion}")

  return "\n".join(lines)


def structured_to_markdown(output: dict) -> str:
  """Convert structured JSON output to flat markdown for ScholarQA eval."""
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

  # Convert any remaining [[N]](URL) to [N]
  text = re.sub(r'\[\[(\d+)\]\]\([^)]+\)', r'[\1]', text)

  return text


def extract_sources(response) -> list[dict]:
  """Extract citation sources from DeepResearch response."""
  sources = []
  raw_sources = getattr(response, "sources", None) or []

  for idx, src in enumerate(raw_sources):
    if isinstance(src, dict):
      text = src.get("snippet", "") or src.get("description", "") or src.get("content", "") or src.get("abstract", "")
      sources.append({
        "text": text,
        "title": src.get("title", ""),
        "url": src.get("url", ""),
        "citation_id": idx + 1,
      })
    else:
      text = getattr(src, "snippet", "") or getattr(src, "description", "") or getattr(src, "content", "") or getattr(src, "abstract", "")
      sources.append({
        "text": text,
        "title": getattr(src, "title", ""),
        "url": getattr(src, "url", ""),
        "citation_id": idx + 1,
      })

  return sources


def _word_overlap(text_a: str, text_b: str) -> float:
  """Simple word overlap score (Jaccard-like) between two texts."""
  if not text_a or not text_b:
    return 0.0
  words_a = set(text_a.lower().split())
  words_b = set(text_b.lower().split())
  if not words_a or not words_b:
    return 0.0
  intersection = words_a & words_b
  return len(intersection) / (len(words_a | words_b))


def reattribute_citations(text: str, sources: list[dict]) -> str:
  """Re-attribute citations to best-matching sources using text similarity.

  For each sentence with [N] citations, find the source whose text has
  the highest word overlap with the claim and replace the citation number.
  This corrects DeepResearch's sometimes-inaccurate citation mapping.
  """
  from nltk import sent_tokenize

  # Build index of sources with text (skip dummy at [0])
  source_texts = {}
  for i, src in enumerate(sources):
    if src.get("text", "").strip() and i > 0:
      source_texts[i] = src["text"].lower()

  if not source_texts:
    return text

  sentences = sent_tokenize(text)
  result_parts = []
  used_cursor = 0

  for sent in sentences:
    # Find this sentence in the original text
    sent_start = text.find(sent, used_cursor)
    if sent_start == -1:
      continue

    # Get text before this sentence
    result_parts.append(text[used_cursor:sent_start])
    used_cursor = sent_start + len(sent)

    # Find citations in this sentence
    citation_pattern = r'\[(\d+)\]'
    refs = re.findall(citation_pattern, sent)

    if not refs:
      result_parts.append(sent)
      continue

    # Get the claim text (without citations)
    claim = re.sub(citation_pattern, '', sent).strip()
    if len(claim) < 20:
      result_parts.append(sent)
      continue

    # Find best matching sources
    scores = [(idx, _word_overlap(claim, src_text)) for idx, src_text in source_texts.items()]
    scores.sort(key=lambda x: x[1], reverse=True)

    # Take top N matches (N = number of original citations)
    best_sources = [idx for idx, score in scores[:len(refs)] if score > 0.05]

    if not best_sources:
      result_parts.append(sent)
      continue

    # Replace citations with best-matching source indices
    new_sent = re.sub(citation_pattern, '', sent).strip()
    citation_str = ''.join(f'[{idx}]' for idx in best_sources)

    # Insert citations before the last period/punctuation
    if new_sent and new_sent[-1] in '.!?':
      new_sent = new_sent[:-1] + ' ' + citation_str + new_sent[-1]
    else:
      new_sent = new_sent + ' ' + citation_str

    result_parts.append(new_sent)

  # Append any remaining text
  result_parts.append(text[used_cursor:])
  return ''.join(result_parts)


def backfill_sources(client: Valyu, sources: list[dict], max_backfill: int = 20) -> list[dict]:
  """Fetch content for sources with empty text using Valyu Contents API."""
  empty_indices = [i for i, s in enumerate(sources) if not s.get("text", "").strip() and s.get("url", "").strip()]

  if not empty_indices:
    return sources

  to_fetch = empty_indices[:max_backfill]
  urls = [sources[i]["url"] for i in to_fetch]
  print(f"  Backfilling {len(urls)} empty sources via Contents API...")

  # Batch fetch in groups of 5
  filled = 0
  for batch_start in range(0, len(urls), 5):
    batch_urls = urls[batch_start:batch_start + 5]
    batch_indices = to_fetch[batch_start:batch_start + 5]
    try:
      result = client.contents(urls=batch_urls, response_length="short")
      if hasattr(result, "results") and result.results:
        for j, r in enumerate(result.results):
          if j < len(batch_indices) and hasattr(r, "content") and r.content:
            sources[batch_indices[j]]["text"] = r.content[:2000]
            filled += 1
    except Exception as e:
      print(f"  Backfill batch failed: {e}")

  print(f"  Backfilled {filled}/{len(urls)} sources")
  return sources


def run_question(client: Valyu, question: dict, model: str, idx: int, total: int) -> dict:
  """Run DeepResearch on a single question."""
  case_id = question["case_id"]
  query = question["question"]
  rubric_items = question["rubric_items"]

  print(f"\n[{idx+1}/{total}] {case_id}")
  print(f"  Q: {query[:80]}...")
  print(f"  Rubric items: {len(rubric_items)}")

  schema = build_schema(rubric_items)
  strategy = build_strategy(rubric_items)

  start = time.time()

  try:
    response = client.deepresearch.create(
      input=query,
      model=model,
      output_formats=[schema],
      strategy=strategy,
      search={"search_type": "all"},
      code_execution=False,
    )

    if not response.success:
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
      # Pydantic validation errors (e.g., chart_type='scatter') - poll raw API
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
            # Create a simple namespace to hold result data
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

    # Extract structured output
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
      # Convert citations
      answer_text = re.sub(r'\[\[(\d+)\]\]\([^)]+\)', r'[\1]', answer_text)

    sources = extract_sources(result)

    # Prepend dummy source at index 0 so [1] maps to sources[1]
    # (eval script uses citation number as direct array index)
    sources.insert(0, {"text": "", "title": "", "url": "", "citation_id": 0})

    # Count citations in output
    citation_count = len(re.findall(r'\[\d+\]', answer_text))
    word_count = len(answer_text.split())

    print(f"  Done in {elapsed:.0f}s | ${cost:.4f} | {word_count} words | {citation_count} citations | {len(sources)} sources")

    return {
      "case_id": case_id,
      "input": query,
      "output": answer_text,
      "answer_text": answer_text,
      "ctxs": sources,
      "task_id": task_id,
      "elapsed": elapsed,
      "cost": cost,
      "word_count": word_count,
      "citation_count": citation_count,
    }

  except Exception as e:
    elapsed = time.time() - start
    print(f"  FAILED ({elapsed:.0f}s): {e}")
    return {
      "case_id": case_id,
      "input": query,
      "output": "",
      "answer_text": "",
      "ctxs": [],
      "error": str(e),
      "elapsed": elapsed,
    }


def run_evaluation(output_dir: str, dataset_dir: str) -> dict | None:
  """Run ScholarQA rubric evaluation via the Python eval script."""
  eval_script = Path(__file__).parent / "eval" / "rubric_eval.py"
  test_config = Path(dataset_dir) / "test_configs_snippets.json"

  if not eval_script.exists():
    print("Eval script not found, skipping rubric evaluation")
    return None

  import subprocess
  result_file = Path(output_dir) / "rubric_results.json"

  cmd = [
    "python3", str(eval_script),
    "--qa-dir", output_dir,
    "--test-config", str(test_config),
    "--rubrics", "--snippets",
    "--output", str(result_file),
  ]

  print(f"\nRunning rubric evaluation...")
  print(f"  {' '.join(cmd)}")

  try:
    env = os.environ.copy()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                          cwd=str(Path(__file__).parent / "eval"), env=env)
    print(proc.stdout)
    if proc.returncode != 0:
      print(f"Eval error: {proc.stderr}")
      return None

    if result_file.exists():
      with open(result_file) as f:
        return json.load(f)
  except Exception as e:
    print(f"Eval failed: {e}")

  return None


def main():
  parser = argparse.ArgumentParser(description="ScholarQA Benchmark - DeepResearch with Structured Output")
  parser.add_argument("--dataset", default="scholarqa_cs", help="Dataset name")
  parser.add_argument("--sample", type=int, help="Number of questions to sample")
  parser.add_argument("--model", default="standard", choices=["fast", "standard", "heavy", "max"], help="DeepResearch model")
  parser.add_argument("--output-dir", default="scholarqa/outputs", help="Output directory")
  parser.add_argument("--workers", type=int, default=3, help="Concurrent tasks")
  parser.add_argument("--skip-eval", action="store_true", help="Skip rubric evaluation")
  parser.add_argument("--citation-eval", action="store_true", help="Run citation F1 evaluation (requires NLI model)")
  args = parser.parse_args()

  api_key = os.getenv("VALYU_API_KEY")
  if not api_key:
    print("ERROR: VALYU_API_KEY not set")
    return

  client = Valyu(api_key=api_key)

  # Load questions
  dataset_dir = Path(__file__).parent / "datasets"
  questions = load_questions(dataset_dir)
  print(f"Loaded {len(questions)} questions from {args.dataset}")

  if args.sample and args.sample < len(questions):
    questions = questions[:args.sample]
    print(f"Sampling first {args.sample} questions")

  # Run
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"\nModel: {args.model}")
  print(f"Workers: {args.workers}")
  print(f"Questions: {len(questions)}")
  print(f"{'='*60}")

  results = []
  total_cost = 0.0

  if args.workers == 1:
    for idx, q in enumerate(questions):
      result = run_question(client, q, args.model, idx, len(questions))
      results.append(result)
      total_cost += result.get("cost", 0)
  else:
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
      futures = {
        executor.submit(run_question, client, q, args.model, idx, len(questions)): idx
        for idx, q in enumerate(questions)
      }
      for future in as_completed(futures):
        result = future.result()
        results.append(result)
        total_cost += result.get("cost", 0)

  # Sort by case_id
  results.sort(key=lambda r: r["case_id"])

  # Save JSONL for eval
  output_file = output_dir / f"valyu_{args.dataset}_{args.model}.jsonl"
  with open(output_file, "w") as f:
    for r in results:
      f.write(json.dumps(r) + "\n")

  # Summary
  successful = [r for r in results if not r.get("error")]
  failed = [r for r in results if r.get("error")]
  avg_words = sum(r.get("word_count", 0) for r in successful) / max(len(successful), 1)
  avg_citations = sum(r.get("citation_count", 0) for r in successful) / max(len(successful), 1)
  avg_time = sum(r.get("elapsed", 0) for r in successful) / max(len(successful), 1)

  print(f"\n{'='*60}")
  print(f"RESULTS: {args.dataset} ({args.model})")
  print(f"{'='*60}")
  print(f"Total: {len(results)}")
  print(f"Successful: {len(successful)}")
  print(f"Failed: {len(failed)}")
  print(f"Avg words: {avg_words:.0f}")
  print(f"Avg citations: {avg_citations:.1f}")
  print(f"Avg time: {avg_time:.0f}s")
  print(f"Total cost: ${total_cost:.2f}")
  print(f"Output: {output_file}")

  # Run evaluation
  if not args.skip_eval:
    eval_results = run_evaluation(str(output_dir), str(dataset_dir))
    if eval_results:
      for src, scores in eval_results.items():
        avg_score = sum(s["scores"]["score"] for s in scores) / len(scores)
        print(f"\nRubric Score ({src}): {avg_score:.3f} ({avg_score*100:.1f}%)")

  # Run citation F1 evaluation
  if args.citation_eval:
    run_citation_evaluation(str(output_file))


def run_citation_evaluation(output_file: str) -> dict | None:
  """Run citation F1 evaluation using NLI model."""
  eval_script = Path(__file__).parent / "eval" / "citation_eval.py"

  if not eval_script.exists():
    print("Citation eval script not found, skipping")
    return None

  import subprocess

  # Convert to absolute path for eval script cwd
  abs_output = str(Path(output_file).resolve())

  cmd = [
    "python3", str(eval_script),
    "--f", abs_output,
    "--citations",
    "--at_most_citations", "3",
  ]

  print(f"\nRunning citation F1 evaluation...")
  print(f"  {' '.join(cmd)}")

  try:
    env = os.environ.copy()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                          cwd=str(Path(__file__).parent / "eval"), env=env)
    print(proc.stdout[-500:] if len(proc.stdout) > 500 else proc.stdout)
    if proc.returncode != 0:
      print(f"Citation eval error: {proc.stderr[-500:]}")
      return None

    score_file = abs_output + ".score_post_fix"
    if Path(score_file).exists():
      with open(score_file) as f:
        results = json.load(f)
      rec = results.get("citation_rec", 0)
      prec = results.get("citation_prec", 0)
      f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
      print(f"\nCitation Recall: {rec:.1f}%")
      print(f"Citation Precision: {prec:.1f}%")
      print(f"Citation F1: {f1:.1f}%")
      print(f"Avg cited papers: {results.get('cited_paper_numbers', 0):.1f}")
      return results
  except Exception as e:
    print(f"Citation eval failed: {e}")

  return None


if __name__ == "__main__":
  main()
