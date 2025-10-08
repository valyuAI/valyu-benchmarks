#!/usr/bin/env python3
"""
AgentSearch Benchmarking System

This script evaluates the AgentSearch API against QA datasets using
GCP Gemini (gemini-2.5-pro) as an AI judge for structured evaluation.

Usage:
    python benchmark.py --tool valyu --dataset datasets/finance/finqa.json --sample 5
    python benchmark.py --tool google --dataset datasets/economics/worldbank.json --sample 10
    python benchmark.py --tool exa --dataset datasets/medical/drug_labels.json --resume
    python benchmark.py --tool parallel --dataset datasets/finance/finqa.json --output results/custom_results.json
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from enum import Enum

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from api.agentsearch import AgentSearchAPI

# Load environment variables
load_dotenv()

# Hardcoded configuration defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 300
DEFAULT_SCRIPT_PATH = "query.ts"
DEFAULT_JUDGE_MODEL = "gemini-2.5-pro"
DEFAULT_GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project")
DEFAULT_GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
DEFAULT_CHECKPOINT_INTERVAL = 10
DEFAULT_MAX_WORKERS = 10
DEFAULT_LOG_FILE = "benchmark.log"
DEFAULT_CHECKPOINT_FILE = "benchmark_checkpoint.json"


class CorrectnessLevel(str, Enum):
    """Enum for different levels of correctness."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"


class JudgeEvaluation(BaseModel):
    """Structured output model for GCP Gemini judge evaluation."""
    reasoning: str = Field(
        description="Detailed reasoning explaining why the API response is correct, incorrect, or partially correct compared to the expected answer"
    )
    correctness: CorrectnessLevel = Field(
        description="Level of correctness: 'correct' if the API's answer is semantically equivalent to the expected answer, 'partially_correct' if it contains the right information but is incomplete or has minor issues, 'incorrect' if it's wrong"
    )


class APIResult(BaseModel):
    """Individual API result for one QA pair."""
    response: Optional[str] = None
    judge_evaluation: Optional[JudgeEvaluation] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    tool_outputs: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool outputs and results from the API query")


class BenchmarkResult(BaseModel):
    """Individual benchmark result for one QA pair across multiple APIs."""
    question_id: str
    question: str
    expected_answer: str
    source_dataset: str
    api_results: Dict[str, APIResult] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class BenchmarkSummary(BaseModel):
    """Overall benchmark results summary."""
    benchmark_info: Dict = Field(default_factory=dict)
    results: List[BenchmarkResult] = Field(default_factory=list)

    def calculate_summary(self, api_names: List[str]) -> None:
        """Calculate summary statistics for each API."""
        total = len(self.results)

        api_performance = {}
        
        # Calculate source dataset distribution
        source_datasets = {}
        for result in self.results:
            dataset = result.source_dataset
            if dataset not in source_datasets:
                source_datasets[dataset] = 0
            source_datasets[dataset] += 1

        for api_name in api_names:
            completed = len([r for r in self.results if api_name in r.api_results and r.api_results[api_name].judge_evaluation is not None])
            
            # Count different correctness levels
            fully_correct = len([r for r in self.results if api_name in r.api_results and r.api_results[api_name].judge_evaluation and r.api_results[api_name].judge_evaluation.correctness == CorrectnessLevel.CORRECT])
            partially_correct = len([r for r in self.results if api_name in r.api_results and r.api_results[api_name].judge_evaluation and r.api_results[api_name].judge_evaluation.correctness == CorrectnessLevel.PARTIALLY_CORRECT])
            incorrect = len([r for r in self.results if api_name in r.api_results and r.api_results[api_name].judge_evaluation and r.api_results[api_name].judge_evaluation.correctness == CorrectnessLevel.INCORRECT])
            
            # Count both "correct" and "partially_correct" as correct for accuracy calculation
            correct = fully_correct + partially_correct
            errors = len([r for r in self.results if api_name in r.api_results and r.api_results[api_name].error is not None])

            # Calculate average processing time
            processing_times = [r.api_results[api_name].processing_time for r in self.results
                              if api_name in r.api_results and r.api_results[api_name].processing_time is not None]
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Calculate tool usage statistics
            queries_with_tools = len([r for r in self.results 
                                    if api_name in r.api_results 
                                    and r.api_results[api_name].tool_outputs 
                                    and len(r.api_results[api_name].tool_outputs) > 0])
            total_tool_calls = sum([len(r.api_results[api_name].tool_outputs or []) for r in self.results
                                  if api_name in r.api_results and r.api_results[api_name].tool_outputs])
            avg_tools_per_query = total_tool_calls / completed if completed > 0 else 0

            # Calculate performance by source dataset
            dataset_performance = {}
            for dataset in source_datasets.keys():
                dataset_results = [r for r in self.results if r.source_dataset == dataset]
                dataset_completed = len([r for r in dataset_results if api_name in r.api_results and r.api_results[api_name].judge_evaluation is not None])
                dataset_fully_correct = len([r for r in dataset_results if api_name in r.api_results and r.api_results[api_name].judge_evaluation and r.api_results[api_name].judge_evaluation.correctness == CorrectnessLevel.CORRECT])
                dataset_partially_correct = len([r for r in dataset_results if api_name in r.api_results and r.api_results[api_name].judge_evaluation and r.api_results[api_name].judge_evaluation.correctness == CorrectnessLevel.PARTIALLY_CORRECT])
                dataset_incorrect = len([r for r in dataset_results if api_name in r.api_results and r.api_results[api_name].judge_evaluation and r.api_results[api_name].judge_evaluation.correctness == CorrectnessLevel.INCORRECT])
                dataset_correct = dataset_fully_correct + dataset_partially_correct
                
                dataset_performance[dataset] = {
                    "total": len(dataset_results),
                    "completed": dataset_completed,
                    "fully_correct": dataset_fully_correct,
                    "partially_correct": dataset_partially_correct,
                    "incorrect": dataset_incorrect,
                    "correct": dataset_correct,
                    "accuracy": dataset_correct / dataset_completed if dataset_completed > 0 else 0.0
                }

            api_performance[api_name] = {
                "total_questions": total,
                "completed": completed,
                "fully_correct": fully_correct,
                "partially_correct": partially_correct,
                "incorrect": incorrect,
                "correct_answers": correct,
                "accuracy": correct / completed if completed > 0 else 0.0,
                "error_rate": errors / total if total > 0 else 0.0,
                "completion_rate": completed / total if total > 0 else 0.0,
                "avg_response_time": avg_time,
                "queries_with_tools": queries_with_tools,
                "total_tool_calls": total_tool_calls,
                "avg_tools_per_query": avg_tools_per_query,
                "tool_usage_rate": queries_with_tools / completed if completed > 0 else 0.0,
                "performance_by_dataset": dataset_performance
            }

        self.benchmark_info = {
            "total_questions": total,
            "apis_tested": api_names,
            "run_date": datetime.now().isoformat(),
            "source_dataset_distribution": source_datasets,
            "api_performance": api_performance
        }




class ModularBenchmark:
    """Main AgentSearch benchmarking system."""

    def __init__(self, tool_choice: str = "valyu", dataset_path: Optional[str] = None):
        """Initialize the benchmark system."""
        self.tool_choice = tool_choice
        self.dataset_path = dataset_path
        self._validation_done = False
        self._validation_lock = threading.Lock()

        # Configure logging first
        self._setup_logging()

        # Initialize GCP client
        self._init_gcp_client()

        # Judge prompt template
        self.judge_prompt_template = """
You are a strict and precise judge evaluating AI-generated answers for financial questions. Your goal is to ensure accuracy while allowing for minor acceptable variations.

TASK: Compare the API response with the expected correct answer and determine the level of correctness with a balanced but discerning approach.

EVALUATION PHILOSOPHY - BE FAIR BUT PRECISE:
- The response must match the expected answer exactly or very closely
- Different wording is acceptable if it conveys the exact same meaning
- Small numerical variations are acceptable within reasonable tolerance
- Format differences are acceptable if the substance is correct
- Extra context is acceptable if it doesn't contradict the core answer
- Minor rounding or precision differences should not be heavily penalized

TOLERANCE CRITERIA:
- Numerical precision: Accept numbers within ±2% tolerance for percentages and financial figures
- Currency formats: Different formats acceptable if values match (e.g., "$1.5M" vs "$1,500,000")
- Percentages: Accept variations within ±0.5 percentage points (e.g., 14% vs 13.8% is acceptable)
- Dates: Accept different formats if they represent the same time period
- Text answers: Accept synonymous or equivalent terminology
- Context: Extra information is acceptable if it supports or doesn't contradict the main answer

SCORING LEVELS (BE BALANCED):
- "correct": The API response matches the expected answer exactly or is semantically identical, including responses with minor acceptable variations within tolerance
- "partially_correct": The API response contains the correct core information but has moderate formatting issues, minor inaccuracies outside acceptable tolerance, or unnecessary additions that somewhat detract from clarity
- "incorrect": The API response differs significantly from the expected answer, contains substantial errors, approximations well beyond acceptable tolerance, or fails to provide the requested information

EXAMPLES OF ACCEPTABLE TOLERANCE:
- Expected: "14%", API: "13.8%" → CORRECT (within 0.5% tolerance)
- Expected: "1500", API: "approximately 1520" → CORRECT (within 2% tolerance)
- Expected: "Q2 2023", API: "second quarter of 2023" → CORRECT (semantically identical)
- Expected: "25%", API: "about 25 percent" → CORRECT (acceptable qualification)
- Expected: "500 million", API: "500M" → CORRECT (format difference, same value)
- Expected: "1500", API: "1600" → PARTIALLY_CORRECT (outside 2% tolerance but not drastically wrong)
- Expected: "14%", API: "18%" → INCORRECT (significantly outside tolerance)

QUESTION: {question}

EXPECTED ANSWER: {expected_answer}

API RESPONSE: {api_response}

Evaluate with balanced precision and provide detailed reasoning for your assessment, noting any tolerance considerations.
"""

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(DEFAULT_LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _init_gcp_client(self) -> None:
        """Initialize Gemini API client."""
        try:
            # Check for Gemini API key
            gemini_api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY not set in environment")

            self.gcp_client = genai.Client(
                api_key=gemini_api_key
            )

            self.logger.info("Gemini API client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize GCP client: {e}")
            raise

    def _create_api_instance(self) -> AgentSearchAPI:
        """Create an AgentSearch API instance with current configuration."""
        config = {
            "max_retries": DEFAULT_MAX_RETRIES,
            "timeout": DEFAULT_TIMEOUT,
            "script_path": DEFAULT_SCRIPT_PATH,
            "tool_choice": self.tool_choice,
            "dataset_path": self.dataset_path or ""
        }

        api_instance = AgentSearchAPI(config)

        # Validate configuration (only once)
        with self._validation_lock:
            if not self._validation_done:
                if not api_instance.validate_config():
                    raise ValueError("AgentSearch API configuration validation failed")
                self._validation_done = True

        # Initialize
        api_instance.initialize()

        return api_instance

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    def _judge_response(self, question: str, expected_answer: str, api_response: str) -> JudgeEvaluation:
        """Use GCP Gemini to judge the response."""
        try:
            prompt = self.judge_prompt_template.format(
                question=question,
                expected_answer=expected_answer,
                api_response=api_response
            )

            response = self.gcp_client.models.generate_content(
                model=DEFAULT_JUDGE_MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeEvaluation
                )
            )

            return response.parsed

        except Exception as e:
            self.logger.error(f"GCP judge error: {e}")
            raise

    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load the unified QA dataset."""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            questions = data.get('questions', [])
            self.logger.info(f"Loaded {len(questions)} questions from dataset")
            return questions

        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def _load_checkpoint(self, checkpoint_file: str) -> BenchmarkSummary:
        """Load existing checkpoint file."""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return BenchmarkSummary.model_validate(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.info(f"No valid checkpoint found: {e}")
            return BenchmarkSummary()

    def _save_checkpoint(self, results: BenchmarkSummary, checkpoint_file: str, api_names: List[str]) -> None:
        """Save current progress to checkpoint file."""
        try:
            results.calculate_summary(api_names)
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results.model_dump(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def _process_question_worker(self, question_data: Dict) -> BenchmarkResult:
        """Process a single question with AgentSearch API - worker function for threading."""
        # Construct expected answer with unit if present
        expected_answer = question_data['answer']
        unit = question_data.get('unit')
        if unit and unit.strip():  # Check if unit exists and is not empty/whitespace
            expected_answer = f"{expected_answer} {unit.strip()}"

        result = BenchmarkResult(
            question_id=question_data['id'],
            question=question_data['question'],
            expected_answer=expected_answer,
            source_dataset=question_data.get('source_dataset', 'unknown')
        )

        api_result = APIResult()
        start_time = time.time()

        try:
            # Create API instance for this thread
            api_instance = self._create_api_instance()

            # Query API
            api_result.response = api_instance.query(question_data['question'])

            # Capture tool outputs
            try:
                api_result.tool_outputs = api_instance.get_last_tool_outputs()
                self.logger.debug(f"Captured {len(api_result.tool_outputs or [])} tool outputs")
            except Exception as tool_error:
                self.logger.warning(f"Failed to capture tool outputs: {tool_error}")
                api_result.tool_outputs = []

            # Judge response
            api_result.judge_evaluation = self._judge_response(
                question_data['question'],
                question_data['answer'],
                api_result.response
            )

            api_result.processing_time = time.time() - start_time

            # Clean up API instance
            try:
                api_instance.cleanup()
            except Exception as cleanup_error:
                self.logger.warning(f"API cleanup warning: {cleanup_error}")

        except Exception as e:
            api_result.error = str(e)
            api_result.processing_time = time.time() - start_time
            self.logger.error(f"Error processing {question_data['id']}: {e}")

        result.api_results["agentsearch"] = api_result

        return result

    def run_benchmark(
        self,
        dataset_path: str,
        sample_size: Optional[int] = None,
        resume: bool = False,
        output_file: Optional[str] = None
    ) -> BenchmarkSummary:
        """Run the complete benchmark with AgentSearch API."""

        # Load dataset
        questions = self._load_dataset(dataset_path)

        # Apply sampling if specified
        if sample_size and sample_size < len(questions):
            questions = questions[:sample_size]
            self.logger.info(f"Using sample of {sample_size} questions")

        # Determine dataset type from path
        dataset_path_lower = dataset_path.lower()
        if 'medical' in dataset_path_lower or 'drug' in dataset_path_lower or 'clinical' in dataset_path_lower:
            dataset_type = 'medical'
        elif 'economics' in dataset_path_lower or 'economic' in dataset_path_lower:
            dataset_type = 'economics'
        else:
            # Default to financial for finance datasets and unknown datasets
            dataset_type = 'finance'

        # Get file paths
        checkpoint_file = DEFAULT_CHECKPOINT_FILE

        # Auto-generate results file name based on dataset type and tool choice
        if output_file:
            # User specified custom output file
            results_file = output_file
        else:
            # Auto-name: results/benchmark_results_{dataset_type}_{tool}.json
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            results_file = str(results_dir / f"benchmark_results_{dataset_type}_{self.tool_choice}.json")

        # Load checkpoint if resuming
        if resume:
            results = self._load_checkpoint(checkpoint_file)
            completed_ids = {r.question_id for r in results.results}
            self.logger.info(f"Resuming from checkpoint with {len(completed_ids)} completed")
        else:
            results = BenchmarkSummary()
            completed_ids = set()

        # Validate API configuration
        try:
            test_instance = self._create_api_instance()
            test_instance.cleanup()
            self.logger.info("Validated AgentSearch API configuration")
        except Exception as e:
            self.logger.error(f"Failed to validate AgentSearch API: {e}")
            return results

        # Filter out completed questions
        pending_questions = [q for q in questions if q['id'] not in completed_ids]

        self.logger.info(f"Processing {len(pending_questions)} questions with {DEFAULT_MAX_WORKERS} workers")

        # Process questions in parallel
        completed_count = len(results.results)
        results_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(self._process_question_worker, question_data): question_data
                for question_data in pending_questions
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(pending_questions), desc="Processing questions") as pbar:
                for future in as_completed(future_to_question):
                    question_data = future_to_question[future]

                    try:
                        result = future.result()

                        # Thread-safe result addition
                        with results_lock:
                            results.results.append(result)
                            completed_count += 1

                            # Log completion status
                            api_result = result.api_results.get("agentsearch")
                            if api_result:
                                if api_result.judge_evaluation:
                                    self.logger.info(f"Completed {result.question_id}: {api_result.judge_evaluation.correctness.value}")
                                elif api_result.error:
                                    self.logger.error(f"Failed {result.question_id}: {api_result.error}")

                            # Save checkpoint periodically
                            if completed_count % DEFAULT_CHECKPOINT_INTERVAL == 0:
                                self._save_checkpoint(results, checkpoint_file, ["agentsearch"])
                                self.logger.info(f"Checkpoint saved at {completed_count} completed questions")

                        pbar.update(1)
                        pbar.set_description(f"Completed: {completed_count}")

                    except Exception as e:
                        self.logger.error(f"Error processing question {question_data['id']}: {e}")
                        pbar.update(1)

        # Final save
        self._save_checkpoint(results, checkpoint_file, ["agentsearch"])
        results.calculate_summary(["agentsearch"])

        # Save final results
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results.model_dump(), f, indent=2, ensure_ascii=False)

        # Print summary
        self._print_summary(results, ["agentsearch"], results_file)

        return results

    def _print_summary(self, results: BenchmarkSummary, api_names: List[str], results_file: str) -> None:
        """Print benchmark summary."""
        info = results.benchmark_info

        summary = f"\nBenchmark Complete!\nTotal Questions: {info['total_questions']}\n"
        
        # Print source dataset distribution
        if "source_dataset_distribution" in info:
            summary += "\nSource Dataset Distribution:\n"
            for dataset, count in info["source_dataset_distribution"].items():
                summary += f"  {dataset}: {count}\n"

        for api_name in api_names:
            perf = info["api_performance"][api_name]
            summary += f"\n{api_name.upper()} Performance:\n"
            summary += f"  Completed: {perf['completed']}\n"
            summary += f"  Fully Correct: {perf['fully_correct']}\n"
            summary += f"  Partially Correct: {perf['partially_correct']}\n"
            summary += f"  Incorrect: {perf['incorrect']}\n"
            summary += f"  Total Correct (incl. partial): {perf['correct_answers']}\n"
            summary += f"  Accuracy: {perf['accuracy']:.2%}\n"
            summary += f"  Completion Rate: {perf['completion_rate']:.2%}\n"
            summary += f"  Error Rate: {perf['error_rate']:.2%}\n"
            summary += f"  Avg Response Time: {perf['avg_response_time']:.2f}s\n"
            summary += f"  Tool Usage Rate: {perf['tool_usage_rate']:.2%}\n"
            summary += f"  Total Tool Calls: {perf['total_tool_calls']}\n"
            summary += f"  Avg Tools per Query: {perf['avg_tools_per_query']:.1f}\n"
            
            # Print performance by dataset
            if "performance_by_dataset" in perf:
                summary += f"  Performance by Dataset:\n"
                for dataset, dataset_perf in perf["performance_by_dataset"].items():
                    summary += f"    {dataset}: {dataset_perf['correct']}/{dataset_perf['completed']} ({dataset_perf['accuracy']:.2%}) "
                    summary += f"[Full: {dataset_perf['fully_correct']}, Partial: {dataset_perf['partially_correct']}, Wrong: {dataset_perf['incorrect']}]\n"

        summary += f"\nResults saved to: {results_file}"

        self.logger.info(summary)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run AgentSearch QA Benchmark with parallel processing")
    parser.add_argument("--tool", type=str, default="valyu", help="Search tool to use (valyu, google, exa, parallel)")
    parser.add_argument("--dataset", required=True, help="Dataset name: finance, medical, or economics")
    parser.add_argument("--sample", type=int, help="Number of questions to sample")
    parser.add_argument("--output", type=str, help="Custom output file path (default: results/benchmark_results_{dataset_type}_{tool}.json)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    # Map dataset name to file path
    dataset_name = args.dataset.lower()
    dataset_mapping = {
        'finance': 'datasets/finance.json',
        'medical': 'datasets/medical.json',
        'economics': 'datasets/economics.json'
    }

    if dataset_name not in dataset_mapping:
        print(f"Error: Invalid dataset name '{args.dataset}'. Must be one of: finance, medical, economics")
        return

    dataset_path = dataset_mapping[dataset_name]
    dataset_type = dataset_name  # Use the dataset name directly as the type

    # Initialize benchmark system with tool choice and dataset path
    benchmark = ModularBenchmark(tool_choice=args.tool, dataset_path=dataset_path)

    print(f"Benchmarking with AgentSearch using {args.tool} search tool")
    print(f"Dataset: {dataset_type}")
    print(f"Dataset file: {dataset_path}")
    if args.output:
        print(f"Results will be saved to: {args.output}")
    else:
        print(f"Results will be saved to: results/benchmark_results_{dataset_type}_{args.tool}.json")

    # Run benchmark
    benchmark.run_benchmark(
        dataset_path=dataset_path,
        sample_size=args.sample,
        resume=args.resume,
        output_file=args.output
    )


if __name__ == "__main__":
    main()