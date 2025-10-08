# SimpleQA Benchmark with AgentSearch

Benchmarking framework for evaluating AgentSearch on the SimpleQA dataset.

## Overview

This benchmark evaluates AgentSearch - a question-answering system that uses:
- **Google Gemini 2.5 Pro** as the base model (via Generative AI API)
- **Multiple search tools** (Valyu, Google, Exa, Parallel AI) for information retrieval
- **OpenAI GPT-4.1** as the grading model for evaluation

## Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **API Keys** for:
  - Google Gemini (required for response generation) - Get from https://aistudio.google.com/app/apikey
  - OpenAI (required for grading)
  - Valyu, Exa, SerpAPI, or Parallel AI (depending on which search tool you want to use)

## Setup

### 1. Install Dependencies

```bash
# From the root directory, install Node.js dependencies (shared across all benchmarks)
cd /home/ubuntu/valyu-benchmarking
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual values
nano .env
```

**Required variables:**
```bash
# Google Gemini API (required for response generation)
# Get your API key from https://aistudio.google.com/app/apikey
GOOGLE_GENERATIVE_AI_API_KEY=your_gemini_api_key_here

# OpenAI (required for grading)
OPENAI_API_KEY=sk-...

# Search Tool API Keys (at least one required)
VALYU_API_KEY=your_valyu_key        # For --tool valyu
EXA_API_KEY=your_exa_key            # For --tool exa
SERPAPI_KEY=your_serpapi_key        # For --tool google
PARALLEL_API_KEY=your_parallel_key  # For --tool parallel
```

### 3. Verify Setup

Test that everything is configured correctly:
```bash
# Run with just 2 examples to test
python -m simple-qa.simple_qa --sample 2
```

## Usage

### Basic Usage

Run SimpleQA benchmark with AgentSearch using default Valyu search:
```bash
python -m simple-qa.simple_qa
```

### Search Tool Selection

Specify which search tool to use with the `--tool` parameter:

```bash
# Use Valyu search (default)
python -m simple-qa.simple_qa --tool valyu

# Use Google Search (via SerpAPI)
python -m simple-qa.simple_qa --tool google

# Use Exa search
python -m simple-qa.simple_qa --tool exa

# Use Parallel AI search
python -m simple-qa.simple_qa --tool parallel
```

### Additional Options

```bash
# Run with limited examples (for testing)
python -m simple-qa.simple_qa --sample 5

# Debug mode (runs with 10 examples)
python -m simple-qa.simple_qa --debug

# Run with specific tool and limited examples
python -m simple-qa.simple_qa --tool google --sample 10
```


## Results

Results are saved to the `results/` directory with format: `{eval}_{model}_{timestamp}_{tool}.json`

Example filenames:
- `simpleqa_agentsearch_20250107_143022_valyu.json`
- `simpleqa_agentsearch_20250107_143022_google.json`
- `simpleqa_agentsearch_20250107_143022_exa.json`

Each result file contains:
- Individual question scores
- Aggregate metrics (accuracy, precision, recall, F1)
- Tool usage statistics
- Response metadata

## Components

- **AgentSearch** (`sampler/agentsearch_sampler.py`) - Main sampler interface
- **Query Script** (`query.ts`) - Node.js script that runs Gemini with search tools
- **SimpleQA Eval** (`simpleqa_eval.py`) - Evaluation logic
- **Grading Model** - OpenAI GPT-4.1 for answer grading
