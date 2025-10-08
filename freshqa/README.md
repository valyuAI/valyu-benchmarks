# FreshQA Benchmark

Benchmark system for evaluating AgentSearch against FreshQA dataset using Gemini 2.5 Pro for responses and Claude Sonnet 4 as evaluation judge.

## Features

- **4 Search Tools**: Valyu, Google (SerpAPI), Exa, and Parallel AI
- **Fresh Questions**: Weekly-updated FreshQA dataset with 600+ questions
- **Dual AI System**: Gemini 2.5 Pro for responses + Claude Sonnet 4 for evaluation
- **Parallel Processing**: 5 concurrent workers for fast benchmarking
- **Relaxed Evaluation**: Accepts correct answers even with additional context

## Setup

### 1. Environment Variables

Create a `.env` file:

```bash
# Search Tool API Keys (use the ones you need)
VALYU_API_KEY=your_valyu_api_key_here
SERPAPI_KEY=your_serpapi_key_here
EXA_API_KEY=your_exa_api_key_here
PARALLEL_API_KEY=your_parallel_api_key_here

# Google Gemini API (required for response generation)
GOOGLE_GENERATIVE_AI_API_KEY=your_gemini_api_key_here

# Claude Evaluation - Option 1: Direct Anthropic API (default, omit --use-vertex or use --use-vertex false)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Claude Evaluation - Option 2: Vertex AI (requires --use-vertex true)
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-east5
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

### 2. Install Dependencies

```bash
# From the root directory, install Node.js dependencies (shared across all benchmarks)
cd /home/ubuntu/valyu-benchmarking
npm install

# Install Python dependencies
cd freshqa
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python benchmark.py --tool <tool_name> [--sample <number>]
```

### Available Tools

| Tool | Description |
|------|-------------|
| `valyu` | Valyu deep search for academic papers, web content, market data |
| `google` | Google search via SerpAPI for organic search results |
| `exa` | Exa search with live crawling for up-to-date information |
| `parallel` | Parallel AI search for comprehensive results |

### Examples

```bash
# Benchmark with Valyu search tool (all questions, default Anthropic API)
python benchmark.py --tool valyu

# Benchmark with Google search (sample 50 questions)
python benchmark.py --tool google --sample 50

# Benchmark with Exa search (sample 10 questions for testing)
python benchmark.py --tool exa --sample 10

# Benchmark with Parallel search
python benchmark.py --tool parallel

# Use Vertex AI instead of Anthropic API for Claude evaluation
python benchmark.py --tool valyu --use-vertex true

# Use Anthropic API explicitly (this is the default behavior)
python benchmark.py --tool valyu --use-vertex false
```

## CLI Options

```
--tool TEXT         Search tool to use [default: valyu]
                    Options: valyu, google, exa, parallel
--sample INT        Number of questions to sample [default: all]
--use-vertex BOOL   Use Vertex AI for Claude (true) or direct Anthropic API (false) [default: false]
                    Options: true, false
```

## Output

Results are saved to three CSV files (where `{tool}` is the search tool used):

1. **fresheval_results_{tool}.csv** - FreshEval-compliant format with:
   - Question ID, question text, ground truth answer
   - Model response, rating (TRUE/FALSE), evaluation explanation

2. **fresheval_comprehensive_{tool}.csv** - Complete results with:
   - All original dataset columns
   - Model responses and evaluations

3. **fresheval_simple_{tool}.csv** - Simple format with:
   - Ratings, explanations, responses, questions

Example filenames: `fresheval_results_valyu.csv`, `fresheval_results_google.csv`, `fresheval_results_exa.csv`

## About FreshQA

FreshQA is a dynamic QA benchmark designed to test LLMs on questions requiring current world knowledge. Questions are updated weekly to ensure freshness.

**Citation:**
```
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei,
Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, Thang Luong.
FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.
arXiv:2310.03214, 2023.
```

**Dataset**: Questions are loaded from `dataset/freshqa.csv`

## How It Works

1. **Response Generation**: Gemini 2.5 Pro uses selected search tool to answer questions
2. **Evaluation**: Claude Sonnet 4 judges responses using relaxed criteria
   - Via Direct API (default): Uses `claude-sonnet-4-0` with Anthropic API key
   - Via Vertex AI: Uses `claude-sonnet-4@20250514` with GCP service account
3. **Parallel Processing**: 5 workers process questions simultaneously
4. **Progress Tracking**: Real-time accuracy updates during execution
