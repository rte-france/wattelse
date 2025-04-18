#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import re
import typer
import pandas as pd
import os
import sys
from loguru import logger
from pathlib import Path
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from openai import Timeout
from typing import Dict

from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.evaluation_pipeline.config.eval_config import EvalConfig
from wattelse.evaluation_pipeline.utils.file_utils import handle_output_path
from wattelse.evaluation_pipeline import (
    RESULTS_BASE_DIR,
    BASE_OUTPUT_DIR,
    COMPARISON_DATA_DIR,
)

# Column definitions
QUERY_COLUMN = "question"
ANSWER_COLUMN = "answer"
RAG_RELEVANT_EXTRACTS_COLUMN = "rag_relevant_extracts"

app = typer.Typer()


def parse_pairwise_response(
    eval_text: str,
    model1_name: str,
    model2_name: str,
    config: EvalConfig,
    judge_model_name: str = None,
) -> Dict:
    """
    Parse responses from pairwise comparison evaluation.

    Args:
        eval_text: The raw evaluation text from the LLM
        model1_name: Name of the first model
        model2_name: Name of the second model
        config: The evaluation configuration
        judge_model_name: Name of the judge model used for evaluation (optional)

    Returns:
        dict: Dictionary with parsed analysis, winner, and reason
    """
    # Use the regex_patterns from the config instead of creating a new instance
    patterns = config.regex_patterns.get_pairwise_patterns(judge_model_name)

    # Extract analysis section
    analysis_match = re.search(patterns["analysis"], eval_text, re.DOTALL)
    analysis = (
        analysis_match.group(1).strip() if analysis_match else "Analysis not found"
    )

    # Extract winner
    winner_match = re.search(patterns["winner"], eval_text, re.DOTALL)
    winner_raw = winner_match.group(1).strip() if winner_match else "Unknown"

    # Normalize winner to one of the expected values
    if model1_name in winner_raw:
        winner = model1_name
    elif model2_name in winner_raw:
        winner = model2_name
    elif "tie" in winner_raw.lower() or "both" in winner_raw.lower():
        winner = "Tie"
    else:
        winner = "Unknown"

    # Extract reason
    reason_match = re.search(patterns["reason"], eval_text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else "Reason not found"

    return {"analysis": analysis, "winner": winner, "reason": reason}


def merge_datasets(
    df1_path: Path, df2_path: Path, model1_name: str, model2_name: str
) -> pd.DataFrame:
    """
    Merge two evaluation datasets for pairwise comparison.

    Args:
        df1_path: Path to the first evaluation dataset
        df2_path: Path to the second evaluation dataset
        model1_name: Name of the first model
        model2_name: Name of the second model

    Returns:
        pandas.DataFrame: A merged dataframe containing both models' outputs
    """
    logger.info(f"Merging datasets from {df1_path.name} and {df2_path.name}")

    # Load datasets
    df1 = pd.read_excel(df1_path)
    df2 = pd.read_excel(df2_path)

    # Simple check to ensure we have common questions
    common_questions = set(df1[QUERY_COLUMN]).intersection(set(df2[QUERY_COLUMN]))
    if not common_questions:
        logger.error("No common questions found between the two datasets")
        sys.exit(1)

    logger.info(f"Found {len(common_questions)} common questions for comparison")

    # Merge on question column
    # First rename columns to avoid conflicts
    df1_renamed = df1.rename(
        columns={
            ANSWER_COLUMN: f"{ANSWER_COLUMN}_model1",
            RAG_RELEVANT_EXTRACTS_COLUMN: f"{RAG_RELEVANT_EXTRACTS_COLUMN}_model1",
        }
    )

    df2_renamed = df2.rename(
        columns={
            ANSWER_COLUMN: f"{ANSWER_COLUMN}_model2",
            RAG_RELEVANT_EXTRACTS_COLUMN: f"{RAG_RELEVANT_EXTRACTS_COLUMN}_model2",
        }
    )

    # Merge dataframes on question column
    merged_df = pd.merge(
        df1_renamed,
        df2_renamed,
        on=QUERY_COLUMN,
        how="inner",
        suffixes=("_model1", "_model2"),
    )

    # Add model name columns
    merged_df["model1_name"] = model1_name
    merged_df["model2_name"] = model2_name

    logger.info(f"Merged dataset contains {len(merged_df)} questions")
    return merged_df


def evaluate_pairwise_row(row: pd.Series, metric: str, config: EvalConfig) -> Dict:
    """
    Evaluate a single row for pairwise comparison on the specified metric.

    Args:
        row: The DataFrame row containing question, answers, and contexts
        metric: The pairwise metric to evaluate
        config: The evaluation configuration

    Returns:
        dict: Dictionary with evaluation results
    """
    # Initialize the LLM client
    llm_client = OpenAI_Client()

    # Set longer timeout and max tokens for complex evaluations
    llm_client.llm_client.timeout = Timeout(500.0, connect=10.0)
    llm_client.max_tokens = 2048

    try:
        question = row[QUERY_COLUMN]
        model1_name = row["model1_name"]
        model2_name = row["model2_name"]
        answer1 = row[f"{ANSWER_COLUMN}_model1"]
        answer2 = row[f"{ANSWER_COLUMN}_model2"]
        context1 = row[f"{RAG_RELEVANT_EXTRACTS_COLUMN}_model1"]
        context2 = row[f"{RAG_RELEVANT_EXTRACTS_COLUMN}_model2"]

        # Use the EvalConfig to get the appropriate prompt for this metric and model
        model_name = getattr(llm_client, "model_name", config.default_model)

        try:
            prompt = config.get_prompt(metric, model_name)
        except ValueError as e:
            logger.error(f"Error getting prompt for {metric}: {e}")
            return {
                "question": question,
                "metric": metric,
                "analysis": f"Error: Metric '{metric}' not configured",
                "winner": "Error",
                "reason": f"Configuration error: {str(e)}",
                "answer1": answer1,
                "answer2": answer2,
                "context1": context1,
                "context2": context2,
            }

        # Format prompt based on metric type
        if metric == "correctness_pairwise":
            prompt_text = prompt.format(
                question=question,
                model1_name=model1_name,
                answer1=answer1,
                model2_name=model2_name,
                answer2=answer2,
            )
        elif metric == "retrievability_pairwise":
            prompt_text = prompt.format(
                question=question,
                model1_name=model1_name,
                context1=context1,
                model2_name=model2_name,
                context2=context2,
            )
        else:
            logger.warning(f"Unknown pairwise metric type: {metric}")
            return {
                "question": question,
                "metric": metric,
                "analysis": f"Error: Unsupported metric '{metric}'",
                "winner": "Error",
                "reason": "Unsupported metric type",
                "answer1": answer1,
                "answer2": answer2,
                "context1": context1,
                "context2": context2,
            }

        # Generate evaluation
        kwargs = {"max_tokens": 2048, "timeout": Timeout(500.0, connect=10.0)}
        eval_text = llm_client.generate(prompt_text, **kwargs)

        # Parse the response using the judge model name for pattern selection
        # Pass the config to avoid circular imports
        results = parse_pairwise_response(
            eval_text, model1_name, model2_name, config, model_name
        )

        # Add question, metric info, and the model outputs and contexts
        results["question"] = question
        results["metric"] = metric
        results["answer1"] = answer1
        results["answer2"] = answer2
        results["context1"] = context1
        results["context2"] = context2

        logger.info(
            f"Pairwise comparison for question: '{question[:50]}...' on {metric}: Winner: {results['winner']}"
        )

        return results

    except Exception as e:
        logger.error(f"Error in pairwise evaluation: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

        return {
            "question": row[QUERY_COLUMN],
            "metric": metric,
            "analysis": f"Error during evaluation: {str(e)}",
            "winner": "Error",
            "reason": str(e),
            "answer1": row.get(f"{ANSWER_COLUMN}_model1", ""),
            "answer2": row.get(f"{ANSWER_COLUMN}_model2", ""),
            "context1": row.get(f"{RAG_RELEVANT_EXTRACTS_COLUMN}_model1", ""),
            "context2": row.get(f"{RAG_RELEVANT_EXTRACTS_COLUMN}_model2", ""),
        }


def evaluate_pairwise_metrics(
    comparison_df: pd.DataFrame, config: EvalConfig
) -> pd.DataFrame:
    """
    Evaluate the comparison dataset using pairwise metrics.

    Args:
        comparison_df: DataFrame containing merged data from two models
        config: Configuration object for evaluation settings

    Returns:
        pandas.DataFrame: The DataFrame with added evaluation columns
    """
    logger.info(f"LLM Evaluation model: {OpenAI_Client().model_name}")

    # Filter for pairwise metrics from active metrics
    pairwise_metrics = [m for m in config.active_metrics if m.endswith("_pairwise")]

    if not pairwise_metrics:
        logger.warning(
            "No pairwise metrics found in active metrics. Please ensure metrics like 'correctness_pairwise' or 'retrievability_pairwise' are enabled in your config."
        )
        return pd.DataFrame()

    results_dfs = []

    for metric in pairwise_metrics:
        logger.info(f"Running pairwise comparison for metric: {metric}")

        # Wrap the Parallel execution with tqdm_joblib to show progress
        with tqdm_joblib(
            desc=f"Evaluating {metric}", total=comparison_df.shape[0]
        ) as progress_bar:
            evaluations = Parallel(n_jobs=-1)(
                delayed(evaluate_pairwise_row)(row, metric, config)
                for _, row in comparison_df.iterrows()
            )

        # Create a DataFrame from the evaluation results
        metric_df = pd.DataFrame(evaluations)

        # Add model information
        metric_df["model1_name"] = comparison_df["model1_name"].values[
            0
        ]  # Assuming same throughout
        metric_df["model2_name"] = comparison_df["model2_name"].values[
            0
        ]  # Assuming same throughout

        results_dfs.append(metric_df)

    # Combine all metric results
    if results_dfs:
        all_metrics_df = pd.concat(results_dfs, ignore_index=True)

        # Calculate summary statistics
        model1_name = comparison_df["model1_name"].values[0]
        model2_name = comparison_df["model2_name"].values[0]

        for metric in pairwise_metrics:
            metric_results = all_metrics_df[all_metrics_df["metric"] == metric]
            total = len(metric_results)

            if total > 0:
                model1_wins = sum(metric_results["winner"] == model1_name)
                model2_wins = sum(metric_results["winner"] == model2_name)
                ties = sum(metric_results["winner"] == "Tie")
                errors = sum(metric_results["winner"] == "Error") + sum(
                    metric_results["winner"] == "Unknown"
                )

                logger.info(f"\n{metric} Summary:")
                logger.info(
                    f"  {model1_name}: {model1_wins} wins ({model1_wins/total:.1%})"
                )
                logger.info(
                    f"  {model2_name}: {model2_wins} wins ({model2_wins/total:.1%})"
                )
                logger.info(f"  Ties: {ties} ({ties/total:.1%})")
                if errors > 0:
                    logger.warning(f"  Errors/Unknown: {errors} ({errors/total:.1%})")

        return all_metrics_df
    else:
        return pd.DataFrame()


def setup_cloud_model_environment(model_name: str, eval_config: EvalConfig) -> Dict:
    """
    Set up the environment for cloud LLM models for evaluation.

    Args:
        model_name: The name of the model to use for evaluation
        eval_config: The evaluation configuration

    Returns:
        dict: Environment variables for OpenAI client

    Raises:
        ValueError: If the specified model is not a cloud model
    """
    model_config = eval_config.get_model_config(model_name)
    deployment_type = model_config.get("deployment_type", "local")

    if deployment_type != "cloud":
        raise ValueError(
            f"Model {model_name} is not a cloud model. Only cloud models are supported for pairwise evaluation."
        )

    return {
        "OPENAI_API_KEY": model_config["api_key"],
        "OPENAI_ENDPOINT": model_config["api_base"],
        "OPENAI_DEFAULT_MODEL_NAME": model_config["model_name"],
    }


def extract_model_name_from_file(filename: str) -> str:
    """Extract model name from the filename."""
    # Assuming format like CORPUS-A-Mistral-1000.xlsx
    parts = filename.split("-")
    if len(parts) >= 3:
        return parts[2]  # Return the model identifier
    return filename.split(".")[0]  # Fallback to filename without extension


@app.command()
def main(
    set1_filename: str = typer.Argument(
        ..., help="Filename of the first model's evaluation results"
    ),
    set2_filename: str = typer.Argument(
        ..., help="Filename of the second model's evaluation results"
    ),
    config_path: str = typer.Option(
        "config/pair_jury_01.toml", help="Path to the evaluation configuration file"
    ),
    output_dir: str = typer.Option(
        "pairwise_results", help="Directory to save comparison results"
    ),
    overwrite: bool = typer.Option(
        False, help="Whether to overwrite existing output files"
    ),
):
    """
    Run pairwise comparison between two RAG model outputs using cloud-based judge models.
    """
    logger.info(
        f"Starting pairwise comparison between {set1_filename} and {set2_filename}"
    )

    # Set up paths
    set1_path = BASE_OUTPUT_DIR / set1_filename
    set2_path = BASE_OUTPUT_DIR / set2_filename

    # Simple path existence checks with direct errors
    if not set1_path.exists():
        logger.error(f"Input file not found: {set1_path}")
        sys.exit(1)

    if not set2_path.exists():
        logger.error(f"Input file not found: {set2_path}")
        sys.exit(1)

    # Look for the config file in several possible locations
    config_path_obj = Path(config_path)

    # Try different paths to find the config file
    possible_paths = [
        config_path_obj,  # Try as given
        Path(__file__).parent / config_path_obj,  # Relative to script
        Path(__file__).parent / "config" / config_path_obj.name,  # In config directory
        Path.cwd() / config_path_obj,  # Relative to current working directory
    ]

    found_config = None
    for path in possible_paths:
        if path.exists():
            found_config = path
            logger.info(f"Found configuration file at: {path}")
            break

    if not found_config:
        logger.error(f"Configuration file not found: {config_path}")
        logger.error(f"Tried: {[str(p) for p in possible_paths]}")
        sys.exit(1)

    # Ensure comparison data directory exists
    COMPARISON_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Extract model names from filenames
    model1_name = extract_model_name_from_file(set1_filename)
    model2_name = extract_model_name_from_file(set2_filename)

    # Step 1: Merge datasets
    merged_filename = f"comparison_{model1_name}_vs_{model2_name}.xlsx"
    merged_path = COMPARISON_DATA_DIR / merged_filename

    merged_df = merge_datasets(set1_path, set2_path, model1_name, model2_name)
    merged_df.to_excel(merged_path, index=False)
    logger.info(f"Merged dataset saved to {merged_path}")

    # Load configuration
    try:
        eval_config = EvalConfig(found_config)

        # Print active metrics for debugging
        pairwise_metrics = [
            m for m in eval_config.active_metrics if m.endswith("_pairwise")
        ]
        if not pairwise_metrics:
            logger.error("No pairwise metrics found in configuration")
            sys.exit(1)

        logger.info(f"Found pairwise metrics: {', '.join(pairwise_metrics)}")

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Rest of your function remains the same...

    # Select an appropriate cloud model for evaluation
    cloud_models = [
        m
        for m in eval_config.model_configs.keys()
        if eval_config.get_model_config(m).get("deployment_type") == "cloud"
    ]

    if not cloud_models:
        logger.error("No cloud models found in configuration")
        sys.exit(1)

    eval_model = cloud_models[0]  # Use the first cloud model
    logger.info(f"Using cloud model {eval_model} for evaluation")

    # Update the environment with the selected model
    try:
        os.environ.update(setup_cloud_model_environment(eval_model, eval_config))
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Step 2: Run pairwise comparison
    full_output_dir = RESULTS_BASE_DIR / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"pairwise_{model1_name}_vs_{model2_name}.xlsx"
    output_path = full_output_dir / output_filename
    output_path = handle_output_path(output_path, overwrite)

    # Run evaluation using metrics from config
    results_df = evaluate_pairwise_metrics(merged_df, eval_config)

    # Save results
    if not results_df.empty:
        # Rename columns for clarity in the output file
        results_df = results_df.rename(
            columns={
                "answer1": f"{ANSWER_COLUMN}_{model1_name}",
                "answer2": f"{ANSWER_COLUMN}_{model2_name}",
                "context1": f"{RAG_RELEVANT_EXTRACTS_COLUMN}_{model1_name}",
                "context2": f"{RAG_RELEVANT_EXTRACTS_COLUMN}_{model2_name}",
            }
        )

        # Save the complete results
        results_df.to_excel(output_path, index=False)
        logger.success(
            f"Pairwise comparison results with model outputs saved to {output_path}"
        )

        # Create a summary file
        summary_data = []
        pairwise_metrics = [
            m for m in eval_config.active_metrics if m.endswith("_pairwise")
        ]

        for metric in pairwise_metrics:
            metric_results = results_df[results_df["metric"] == metric]
            total = len(metric_results)

            if total > 0:
                model1_wins = sum(metric_results["winner"] == model1_name)
                model2_wins = sum(metric_results["winner"] == model2_name)
                ties = sum(metric_results["winner"] == "Tie")
                errors = sum(metric_results["winner"] == "Error") + sum(
                    metric_results["winner"] == "Unknown"
                )

                summary_data.append(
                    {
                        "metric": metric,
                        f"{model1_name}_wins": model1_wins,
                        f"{model1_name}_win_pct": f"{model1_wins/total:.1%}",
                        f"{model2_name}_wins": model2_wins,
                        f"{model2_name}_win_pct": f"{model2_wins/total:.1%}",
                        "ties": ties,
                        "tie_pct": f"{ties/total:.1%}",
                        "errors": errors,
                        "error_pct": f"{errors/total:.1%}" if errors > 0 else "0.0%",
                        "total_comparisons": total,
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        summary_path = full_output_dir / f"summary_{model1_name}_vs_{model2_name}.xlsx"
        summary_path = handle_output_path(summary_path, overwrite)
        summary_df.to_excel(summary_path, index=False)
        logger.success(f"Summary results saved to {summary_path}")
    else:
        logger.error("No results generated from pairwise comparison")

    logger.success("Pairwise comparison completed!")


if __name__ == "__main__":
    typer.run(main)
