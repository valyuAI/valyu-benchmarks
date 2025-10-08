# AgentSearch Benchmarking System

A benchmarking system for evaluating AgentSearch with multiple search tools against QA datasets using Gemini 2.5 Pro as an AI judge.

## Features

- **4 Search Tools**: Valyu, Google (SerpAPI), Exa, and Parallel AI
- **3 Domain Types**: Automatic prompt selection for Finance, Medical, and Economics questions
- **Parallel Processing**: 10 concurrent workers for fast benchmarking
- **Auto Results Naming**: Results automatically saved as `results/benchmark_results_{dataset_type}_{tool}.json`
- **Resume Support**: Checkpoint system to resume interrupted benchmarks

## Setup

### 1. Environment Variables

Create a `.env` file:

```bash
# Search Tool API Keys (use the ones you need)
VALYU_API_KEY=your_valyu_api_key_here
SERPAPI_KEY=your_serpapi_key_here
EXA_API_KEY=your_exa_api_key_here
PARALLEL_API_KEY=your_parallel_api_key_here

# Google Gemini API (required for Gemini AI model)
GOOGLE_GENERATIVE_AI_API_KEY=your_gemini_api_key_here
```

### 2. Install Dependencies

```bash
# From the root directory, install Node.js dependencies (shared across all benchmarks)
cd /home/ubuntu/valyu-benchmarking
npm install

# Install Python dependencies
cd vertical_benchmarks
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python benchmark.py --tool <tool_name> --dataset <dataset_name> --sample <number>
```

### Available Tools

| Tool | Description |
|------|-------------|
| `valyu` | Valyu deep search for academic papers, web content, market data |
| `google` | Google search via SerpAPI for organic search results |
| `exa` | Exa search with live crawling for up-to-date information |
| `parallel` | Parallel AI search for comprehensive results |

### Available Datasets

| Dataset Name | Description | File Location |
|--------------|-------------|---------------|
| `finance` | Financial domain questions | `datasets/finance.json` |
| `medical` | Medical domain questions | `datasets/medical.json` |
| `economics` | Economics domain questions | `datasets/economics.json` |

### Examples

```bash
# Benchmark with Valyu search tool on finance dataset
python benchmark.py --tool valyu --dataset finance --sample 5

# Benchmark with Google search on medical dataset
python benchmark.py --tool google --dataset medical --sample 10

# Custom output file
python benchmark.py --tool valyu --dataset finance --output custom_results.json

# Resume from checkpoint
python benchmark.py --tool valyu --dataset economics --resume
```

## CLI Options

```
Required:
  --dataset TEXT          Dataset name: finance, medical, or economics

Optional:
  --tool TEXT            Search tool to use [default: valyu]
  --sample INTEGER       Number of questions to sample from dataset
  --output TEXT          Custom output file path [default: results/benchmark_results_{dataset_type}_{tool}.json]
  --resume               Resume from checkpoint file
```

## Output

Results are saved to `results/benchmark_results_{dataset_type}_{tool}.json` with:
- Accuracy metrics (correct, partially correct, incorrect)
- Tool usage statistics
- Processing time
- Per-dataset performance
- Detailed judge evaluation reasoning

**Example filenames:**
- `results/benchmark_results_finance_valyu.json`
- `results/benchmark_results_medical_google.json`
- `results/benchmark_results_economics_exa.json`
