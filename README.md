# Valyu Benchmarking Suite

Comprehensive benchmarking system comparing Valyu API against Google (SerpAPI), Exa, and Parallel AI across three evaluation frameworks.

## Overview

This repository contains three benchmark suites:

1. **Vertical Benchmarks** - Custom QA datasets for Finance, Medical, and Economics domains
2. **FreshQA** - Dynamic questions requiring current world knowledge
3. **SimpleQA** - Straightforward factual questions from OpenAI

## Architecture

- **Response Generation**: Gemini 2.5 Pro with tool-augmented search
- **Evaluation Judges**:
  - Vertical Benchmarks: Gemini 2.5 Pro
  - FreshQA: Claude Sonnet 4 (Anthropic API by default, Vertex AI optional)
  - SimpleQA: OpenAI GPT-4.1
- **Search Tools**: Valyu, Google (SerpAPI), Exa, Parallel AI

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- API keys for search tools and AI models

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

### Environment Setup

Create a `.env` file in each benchmark directory:

```bash
# Search Tool API Keys
VALYU_API_KEY=your_valyu_api_key
SERPAPI_KEY=your_serpapi_key
EXA_API_KEY=your_exa_api_key
PARALLEL_API_KEY=your_parallel_api_key

# Google Gemini API
GOOGLE_GENERATIVE_AI_API_KEY=your_gemini_api_key

# For FreshQA (Claude evaluation via Anthropic API - default)
ANTHROPIC_API_KEY=your_anthropic_api_key

# For FreshQA (Claude evaluation via Vertex AI - optional)
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-east5
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# For SimpleQA (OpenAI grading)
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Vertical Benchmarks

```bash
cd vertical_benchmarks

# Run with Valyu on finance dataset
python benchmark.py --tool valyu --dataset finance --sample 10

# Run with Google on medical dataset
python benchmark.py --tool google --dataset medical

# Resume interrupted benchmark
python benchmark.py --tool valyu --dataset finance --resume
```

**Output**: `results/benchmark_results_{dataset_type}_{tool}.json`

### FreshQA

```bash
cd freshqa

# Run with Valyu (all 600+ questions)
python benchmark.py --tool valyu

# Run with Exa (sample 50 questions)
python benchmark.py --tool exa --sample 50

# Use Vertex AI for Claude evaluation (instead of default Anthropic API)
python benchmark.py --tool valyu --use-vertex true
```

**Output**: `fresheval_results_{tool}.csv`, `fresheval_comprehensive_{tool}.csv`, `fresheval_simple_{tool}.csv`

### SimpleQA

```bash
cd simple-qa

# Run with Valyu (default)
python -m simple-qa.simple_qa

# Run with Google search tool
python -m simple-qa.simple_qa --tool google

# Run with limited examples for testing
python -m simple-qa.simple_qa --tool google --sample 10
```

**Output**: `results/simpleqa_agentsearch_{timestamp}_{tool}.json`

## Benchmark Details

### Vertical Benchmarks
- **Datasets**: Finance, Medical, Economics
- **Features**: Parallel processing (10 workers), checkpoint/resume, domain-specific prompts
- **Evaluation**: Gemini 2.5 Pro judges correctness (correct/partially correct/incorrect)

### FreshQA
- **Dataset**: 600+ weekly-updated questions on current events
- **Features**: Parallel processing (5 workers), relaxed evaluation criteria
- **Evaluation**: Claude Sonnet 4 (Anthropic API default, Vertex AI optional) with detailed reasoning and TRUE/FALSE ratings

### SimpleQA
- **Dataset**: Straightforward factual questions
- **Features**: Multiple model variants, timestamp-based results
- **Evaluation**: OpenAI GPT-4.1 with accuracy metrics

## Search Tools

| Tool | Description |
|------|-------------|
| **Valyu** | Deep search across academic papers, web content, market data, SEC filings |
| **Google** | Organic search results via SerpAPI |
| **Exa** | Live web crawling for up-to-date information |
| **Parallel** | Comprehensive multi-source search |

## Results Format

Each benchmark produces structured results with:
- Response accuracy and correctness metrics
- Judge evaluation reasoning
- Processing time and performance statistics
- Tool-specific metadata and outputs

## Citation

If you use FreshQA in your research:
```
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei,
Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, Thang Luong.
FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.
arXiv:2310.03214, 2023.
```
