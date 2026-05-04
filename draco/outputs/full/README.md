# DRACO benchmark — full 100-item run

This directory contains the canonical artefacts from our DRACO run: the inference outputs we generated for every system we ran ourselves, the per-criterion judge gradings, the consolidated scores, and the reliability notes that document each provider's data-collection nuances.

This README captures the methodology so anyone reading the results can verify how they were produced.

## What DRACO is

DRACO (Deep Research Accuracy, Completeness, and Objectivity) is Perplexity's open expert-rubric benchmark for deep research systems. 100 long-form research tasks across 10 professional knowledge-work domains, each graded against a domain-expert rubric of 30-60 weighted requirements per task. Tasks span Finance, Medicine, Academic research, Law, Technology, plus six broader domains.

- Dataset: <https://huggingface.co/datasets/perplexity-ai/draco>
- Paper: <https://arxiv.org/abs/2602.11685>
- Rubric grader: [github.com/The-LLM-Data-Company/rubric](https://github.com/The-LLM-Data-Company/rubric) (PerCriterionGrader)

## What we did

We ran every commercially available deep research API end-to-end against the same 100 DRACO questions, then graded each output with the same per-criterion judge.

**Five systems we ran ourselves** (full inference outputs + gradings live here):
- Valyu DeepResearch (`Heavy`)
- Parallel Task API (`Ultra8x`)
- You.com Research (`exhaustive`)
- Tavily Research (`pro`)
- Exa Deep Reasoning (`type=deep-reasoning`)

**Six systems whose scores we cite directly from the DRACO paper** (Tables 8, 10, 12):
- Perplexity Deep Research (Opus 4.6)
- Claude Opus 4.6 (web + code)
- Claude Opus 4.5 (web + code)
- Gemini Deep Research
- OpenAI Deep Research (o3)
- OpenAI Deep Research (o4-mini)

The accuracy scores in the headline table for these six systems are taken from the DRACO paper exactly as published — we did not re-run them ourselves and did not modify the numbers.

Cost numbers for these systems are fair estimates. Each provider published per-system input and output token counts in DRACO Table 10; we converted those to a dollar figure using each underlying model's public API pricing. They represent a defensible per-query cost for a developer running the same workflow via standard APIs. They aren't exact — consumer subscription pricing varies — but they're the most credible apples-to-apples cost comparison we could put alongside the systems we ran ourselves with verified per-request prices. Full derivation is in `scores.json::cost_methodology_notes`.

## Methodology — designed to be fair

**Highest publicly-available compute tier per provider.** We picked the strongest tier each provider exposes through their standard API:

| Provider | Tier we ran | Notes |
|---|---|---|
| Valyu DeepResearch | `Heavy` | We offer a higher `Max` tier but did not use it here, since `Max` is more expensive per task than the rest of the field and we wanted a price-comparable comparison. |
| Parallel Task API | `Ultra8x` | The documented top tier. No higher tier exists. |
| You.com Research | `exhaustive` | A `frontier` tier exists but is sales-gated (HTTP 422 on the standard API key). We tested the highest publicly accessible tier. |
| Tavily Research | `pro` | Top tier on the GA `/research` endpoint. |
| Exa Deep Reasoning | `type=deep-reasoning` | The deepest tier on the standard API. |

**Identical conditions across systems.** Each system was contacted with the raw DRACO question, default extraction settings, no custom system prompts, no prompt engineering, no per-task tuning. Exa exposes `system_prompt` for behavioural steering; we did not use it for any system.

**Single run per item, no best-of-N.** `runs_per_item: 1` everywhere. Every per-item record contains exactly one run, and that run's score is what enters the aggregate. Nothing was re-run and selectively kept; nothing was filtered post-hoc.

**Same judge, same rubric, same grader code.** Every system's output is graded by `gemini/gemini-3-pro-preview` against the per-task rubric provided by the DRACO dataset, using the open-source `rubric` library's `PerCriterionGrader`. The judge model is recorded in every eval JSON (`judge_model` field) and is identical across systems.

**Full audit trail per record.** Every inference JSONL line contains the original DRACO `id`, the original `problem`, the gold `answer` (rubric reference), the system's `output` in full, the `elapsed` time, the `provider`, the `model`, and a `meta` block with provider-specific request IDs / source records / cost metadata. No truncation, no summarisation. Every grading JSONL line contains the per-criterion `verdict`, `weight`, `requirement`, and judge `explanation` for every requirement in the rubric.

## Headline result

| | Score | Cost (CPM) |
|---|---:|---:|
| **Valyu DeepResearch (Heavy)** | **72.7%** | **$2,500** |
| Perplexity DR (Opus 4.6)* | 70.5% | ~$5,500 |
| Claude Opus 4.6 (web+code)* | 59.8% | ~$5,500 |
| Gemini Deep Research* | 59.0% | ~$4,500 |
| You.com Research (exhaustive) | 53.0% | $450 |
| OpenAI DR (o3)* | 52.1% | ~$3,500 |
| Parallel Ultra8x | 51.2% | $2,400 |
| Claude Opus 4.5 (web+code)* | 46.7% | ~$4,000 |
| OpenAI DR (o4-mini)* | 41.9% | ~$1,200 |
| Tavily Research (pro) | 38.8% | ~$1,060 |
| Exa Deep Reasoning | 25.9% | $15 |

\* Score reproduced from the DRACO paper. We did not re-run these systems. Cost is a high-end estimate per `scores.json::cost_methodology_notes`.

## Domain breakdown — knowledge-work domains

Among the systems we ran end-to-end, Valyu leads on every knowledge-work domain DRACO covers:

| System | Finance (n=20) | Law (n=6) | Academic (n=12) | Medicine (n=6) | Technology (n=10) |
|---|---:|---:|---:|---:|---:|
| **Valyu (Heavy)** | **67.5%** | **90.5%** | **82.3%** | **80.2%** | **71.1%** |
| You.com (exhaustive) | 49.4% | 70.0% | 63.4% | 51.3% | 47.0% |
| Parallel Ultra8x | 42.3% | 68.9% | 58.1% | 50.9% | 51.2% |
| Tavily (pro) | 27.0% | 59.9% | 41.2% | 44.4% | 38.1% |
| Exa (deep-reason) | 21.4% | 29.8% | 36.9% | 33.2% | 23.6% |

- `n` is the number of DRACO items in that domain (Valyu's full 100-item run).
- Scores are average per-criterion-weighted normalised score across items in the domain, on a 0-100% scale.
- Full per-item scores live in `grading/<system>.eval.json::results[].normalized_score`. Per-domain rollups are in `grading/<system>.eval.json::domain_scores`.

## Directory layout

```
draco/outputs/full/
├── README.md                 (this file)
├── scores.json               (consolidated leaderboard + cost methodology)
├── inference/                (the actual outputs each system produced)
│   ├── valyu_heavy.jsonl
│   ├── parallel_ultra8x.jsonl
│   ├── youcom_exhaustive.jsonl
│   ├── tavily_pro.jsonl
│   └── exa_deep_reasoning.jsonl
├── grading/                  (per-item, per-criterion judge gradings)
│   ├── valyu_heavy.eval.json
│   ├── parallel_ultra8x.eval.json
│   ├── youcom_exhaustive.eval.json
│   ├── tavily_pro.eval.json
│   └── exa_deep_reasoning.eval.json
└── charts/                   (final published charts)
    ├── draco_final_scatter.png
    └── draco_final_leaderboard.png
```

## Reproducing the run

The per-provider runners we used are in [`../../`](../../) (one level above this directory):

- `draco/run.py` — Valyu runner
- `draco/run_parallel.py` — Parallel Task API runner
- `draco/run_youcom.py` — You.com Research API runner
- `draco/run_tavily.py` — Tavily Research API runner
- `draco/run_exa.py` — Exa deep-reasoning runner

Each runner reads the DRACO dataset (`datasets/draco.jsonl`), submits each problem to the provider's highest publicly-available compute tier, and writes one inference record per line to `inference/<system>.jsonl`. The grading step is `draco/eval/rubric_eval.py`, which loads the inference JSONL, fetches the per-task rubric from the DRACO dataset, and grades each output with the `rubric` library against `gemini/gemini-3-pro-preview`.

Required environment variables are documented in each runner. We do not redistribute API keys.

## Why we publish all this

Most published deep-research benchmarks ship without the underlying scripts, the raw outputs, or the per-criterion judge feedback — what some people call "trust me bro" benchmarks. We don't operate that way.

Every score in this directory traces back to:
1. A specific provider tier we documented above
2. A specific run captured in full in `inference/<system>.jsonl`
3. A specific per-criterion judge verdict captured in full in `grading/<system>.eval.json`
4. A specific judge model recorded in every eval file

If something looks wrong, the data is here to argue with. That is the entire point.
