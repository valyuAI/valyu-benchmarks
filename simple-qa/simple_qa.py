import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from . import common
from .sampler.agentsearch_sampler import AgentSearchSampler
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)
from .simpleqa_eval import SimpleQAEval


def main() -> List[Dict[str, Any]]:
    """
    Main entry point for running SimpleQA benchmark evaluations.

    This function orchestrates the benchmarking process by:
    1. Parsing command-line arguments
    2. Initializing the AgentSearch sampler with specified search tool
    3. Setting up the grading model (GPT-4.1)
    4. Running evaluations and collecting results
    5. Saving results to JSON and HTML files

    Returns:
        List of dictionaries containing evaluation metrics for each model/eval combination.
        Each dict has keys: 'eval_name', 'model_name', 'metric'
    """
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select a model by name. Also accepts a comma-separated list of models.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="simpleqa",
        help="Select an eval by name. Also accepts a comma-separated list of evals. (default: simpleqa)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats to run. Only supported for certain evals.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="Number of threads to run. Only supported for HealthBench and HealthBenchMeta.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--sample", type=int, help="Number of examples to sample (overrides default)"
    )
    parser.add_argument(
        "--tool",
        type=str,
        default="valyu",
        help="Search tool to use for AgentSearch: valyu, google, exa, parallel (default: valyu)"
    )

    args = parser.parse_args()

    models = {
        # AgentSearch model (using Node.js + Gemini 2.5 Pro + Search Tools)
        "agentsearch": AgentSearchSampler(
            script_path="query.ts",
            source_dataset="unknown",
            tool_type=args.tool,
            timeout=300,
            max_retries=3,
        ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        models_chosen = args.model.split(",")
        for model_name in models_chosen:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found.")
                return
        models = {model_name: models[model_name] for model_name in models_chosen}

    print(f"Running with args {args}")

    # Grading model for SimpleQA evaluation
    grading_sampler = ChatCompletionSampler(
        model="gpt-4.1-2025-04-14",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )

    def get_evals(eval_name: str, debug_mode: bool) -> SimpleQAEval:
        """
        Initialize and return the appropriate evaluation object.

        Args:
            eval_name: Name of the evaluation ('simpleqa')
            debug_mode: Whether running in debug mode (uses fewer examples)

        Returns:
            SimpleQAEval object configured for the specified evaluation

        Raises:
            Exception: If eval_name is not recognized
        """
        num_examples = (
            args.sample if args.sample is not None else (5 if debug_mode else None)
        )
        match eval_name:
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals_list = args.eval.split(",")
    evals = {}
    for eval_name in evals_list:
        try:
            evals[eval_name] = get_evals(eval_name, args.debug)
        except Exception:
            print(f"Error: eval '{eval_name}' not found.")
            return

    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    merge_metrics = []  # Initialize list to track running metrics
    print(f"Running the following evals: {list(evals.keys())}")
    print(f"Running evals for the following models: {list(models.keys())}")

    def print_running_metrics(merge_metrics: List[Dict[str, Any]]) -> None:
        """
        Print current metrics table after each evaluation.

        Args:
            merge_metrics: List of metric dictionaries with keys 'eval_name', 'model_name', 'metric'
        """
        if not merge_metrics:
            return

        temp_df = pd.DataFrame(merge_metrics).pivot(
            index=["model_name"], columns="eval_name"
        )
        print(f"\n=== RUNNING METRICS (Updated after {len(merge_metrics)} evaluations) ===")
        print(temp_df.to_markdown())
        print("=" * 60)

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}"
            # file stem should also include the year, month, day, and time in hours and minutes
            file_stem += f"_{date_str}"
            # Add tool name to file stem
            file_stem += f"_{args.tool}"
            report_filename = os.path.join(results_dir, f"{file_stem}{debug_suffix}.html")
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = os.path.join(results_dir, f"{file_stem}{debug_suffix}.json")
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            full_result_filename = os.path.join(results_dir, f"{file_stem}{debug_suffix}_allresults.json")
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename
            
            # Add to running metrics and display progress
            try:
                result_metrics = json.load(open(result_filename, "r"))
                current_result = result_metrics.get("f1_score", result_metrics.get("score", None))
                if current_result is not None:
                    merge_metrics.append({
                        "eval_name": eval_name, 
                        "model_name": model_name, 
                        "metric": current_result
                    })
                    print_running_metrics(merge_metrics)
            except Exception as e:
                print(f"Error processing running metrics for {file_stem}: {e}")
    
    # Final metrics summary (using already collected merge_metrics)
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    if merge_metrics:
        merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
            index=["model_name"], columns="eval_name"
        )
        print("\nAll results: ")
        print(merge_metrics_df.to_markdown())
    else:
        print("No metrics data available.")
        
    return merge_metrics


if __name__ == "__main__":
    main()
