#!/usr/bin/env python3
"""
Prometheus evaluation for ScholarQA-Multi benchmark.

Uses the Prometheus open-source LLM judge (prometheus-eval/prometheus-bgb-8x7b-v2.0)
to score responses on Organization, Coverage, and Relevance (1-5 scale).

Requirements:
  - vLLM serving the Prometheus model (Mixtral 8x7B, ~94GB bf16)
  - Tested on 4x A100 40GB with tensor parallelism

Setup (on GPU machine):
  pip install vllm
  vllm serve prometheus-eval/prometheus-bgb-8x7b-v2.0 \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 4096

Usage:
  python3 prometheus_eval.py \
    --predictions scholarqa/outputs/multi/valyu_scholarqa_multi_fast_v1.jsonl \
    --references /path/to/human_answers.json \
    --output scholarqa/outputs/multi/prometheus_results.json \
    --vllm-url http://localhost:8000/v1/completions
"""
import argparse
import json
import re
import time

import requests

RUBRICS = {
  "organization": {
    "criteria": "Organization and Coherence: Evaluate if the response is well-organized and logically structured. An acceptable response should be clearly structured, grouping related points together for a logical flow, be coherent, without contradictions or unnecessary repetition.",
    "scores": {
      1: "Poor Organization; disorganized, no clear structure, points scattered, contradictions or irrelevant repetitions.",
      2: "Below Average Organization; some structure but lacks clear grouping, flow is inconsistent.",
      3: "Adequate Organization; reasonably structured with some logical flow, minor issues with grouping or transitions.",
      4: "Well Organized; clearly structured with logical flow, points well-grouped, mostly coherent with minimal issues.",
      5: "Exceptionally Organized; flawless logical structure, points grouped perfectly, seamless flow, clear discourse markers or section headers, no contradictions or repetition.",
    },
  },
  "coverage": {
    "criteria": "Coverage and Amount of Information: Evaluate if the output provides sufficient coverage and amount of information. The output should offer a comprehensive review citing diverse papers and discussing varied sources, and provide enough relevant information to understand each discussion point.",
    "scores": {
      1: "Severely Lacking Coverage; misses core lines of research or focuses on 1-2 papers, lacking holistic view. Greatly limited depth.",
      2: "Limited Coverage; covers some aspects but misses important areas, or depth is insufficient for understanding.",
      3: "Moderate Coverage; covers main topics but may miss some important perspectives or lack depth in places.",
      4: "Good Coverage; covers diverse papers and viewpoints with sufficient depth for understanding key points.",
      5: "Comprehensive Coverage; covers diverse range of papers and viewpoints, thorough overview, includes important discussion points beyond the question. Necessary and sufficient information.",
    },
  },
  "relevance": {
    "criteria": "Relevance and Focus: Does the response stay on topic and maintain a clear focus to provide a useful response to the question?",
    "scores": {
      1: "Off-topic; significantly deviates from the question.",
      2: "Partially relevant; some content relates to the question but includes significant irrelevant material.",
      3: "Mostly relevant; stays on topic with minor digressions.",
      4: "Relevant and focused; stays on topic with clear focus, most information contributes to understanding.",
      5: "Exceptionally focused and entirely on topic; tightly centered on the subject with enough depth, every piece of information contributes directly.",
    },
  },
}

PROMPT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, or explanations.

###The instruction to evaluate:
{question}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[{criteria}]
Score 1: {s1}
Score 2: {s2}
Score 3: {s3}
Score 4: {s4}
Score 5: {s5}

###Feedback:
"""


def evaluate(vllm_url, model, question, response_text, reference, aspect):
  rubric = RUBRICS[aspect]
  prompt = PROMPT_TEMPLATE.format(
    question=question,
    response=response_text[:4000],
    reference=reference[:4000],
    criteria=rubric["criteria"],
    s1=rubric["scores"][1],
    s2=rubric["scores"][2],
    s3=rubric["scores"][3],
    s4=rubric["scores"][4],
    s5=rubric["scores"][5],
  )

  for attempt in range(3):
    try:
      resp = requests.post(vllm_url, json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.0,
      }, timeout=120)
      resp.raise_for_status()
      text = resp.json()["choices"][0]["text"]
      match = re.search(r'\[RESULT\]\s*(\d)', text)
      if match:
        return int(match.group(1)), text
      match = re.search(r'(\d)\s*$', text.strip())
      if match:
        return int(match.group(1)), text
      return 3, text
    except Exception as e:
      if attempt < 2:
        time.sleep(2 ** attempt)
      else:
        print(f"  Failed: {e}")
        return 3, str(e)


def main():
  parser = argparse.ArgumentParser(description="Prometheus evaluation for ScholarQA-Multi")
  parser.add_argument("--predictions", required=True, help="JSONL file with model outputs")
  parser.add_argument("--references", required=True, help="JSON file with human answers")
  parser.add_argument("--output", default="scholarqa/outputs/multi/prometheus_results.json")
  parser.add_argument("--aspects", nargs="+", default=["organization", "coverage", "relevance"])
  parser.add_argument("--sample", type=int, help="Only eval first N")
  parser.add_argument("--vllm-url", default="http://localhost:8000/v1/completions")
  parser.add_argument("--model", default="prometheus-eval/prometheus-bgb-8x7b-v2.0")
  args = parser.parse_args()

  with open(args.predictions) as f:
    preds = [json.loads(line) for line in f if line.strip()]

  with open(args.references) as f:
    refs = json.load(f)

  ref_map = {r["input"].strip(): r["output"] for r in refs}

  if args.sample:
    preds = preds[:args.sample]

  print(f"Evaluating {len(preds)} predictions on {args.aspects}")
  print(f"vLLM endpoint: {args.vllm_url}")
  print(f"Model: {args.model}")

  results = {aspect: [] for aspect in args.aspects}
  all_scores = []

  for i, pred in enumerate(preds):
    if not pred.get("output"):
      continue

    question = pred["input"].strip()
    reference = ref_map.get(question, "")
    if not reference:
      print(f"  [{i+1}] No reference found, skipping")
      continue

    scores = {}
    for aspect in args.aspects:
      score, feedback = evaluate(args.vllm_url, args.model, question, pred["output"], reference, aspect)
      scores[aspect] = score
      results[aspect].append(score)

    avg = sum(scores.values()) / len(scores)
    all_scores.append(avg)

    if (i + 1) % 5 == 0 or i == len(preds) - 1:
      running_avgs = {a: sum(results[a]) / len(results[a]) for a in args.aspects if results[a]}
      running_overall = sum(all_scores) / len(all_scores)
      print(f"  [{i+1}/{len(preds)}] " + " ".join(f"{a}={running_avgs[a]:.2f}" for a in args.aspects) + f" avg={running_overall:.2f}", flush=True)

  summary = {}
  for aspect in args.aspects:
    if results[aspect]:
      summary[aspect] = sum(results[aspect]) / len(results[aspect])
  summary["average"] = sum(summary.values()) / len(summary) if summary else 0

  print(f"\nFINAL SCORES:")
  for k, v in summary.items():
    print(f"  {k}: {v:.2f}")

  from pathlib import Path
  Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  with open(args.output, "w") as f:
    json.dump({"summary": summary, "details": results}, f, indent=2)
  print(f"Saved to {args.output}")


if __name__ == "__main__":
  main()
