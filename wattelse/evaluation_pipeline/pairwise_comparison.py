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

from wattelse.api.openai.client_openai_api import OpenAI_Client
from wattelse.evaluation_pipeline import (
    BASE_OUTPUT_DIR,
    RESULTS_BASE_DIR,
)
from wattelse.evaluation_pipeline.config.eval_config import EvalConfig
from wattelse.evaluation_pipeline.config.server_config import ServerConfig
from wattelse.evaluation_pipeline.utils.port_manager import PortManager

# Column definitions
QUERY_COLUMN = "question"
ANSWER_COLUMN = "answer"
RAG_RELEVANT_EXTRACTS_COLUMN = "rag_relevant_extracts"

# Directory for comparison data
COMPARISON_DATA_DIR = Path("/DSIA/nlp/experiments/comparaison_data")

# Global port manager
port_manager = PortManager(logger)

app = typer.Typer()


def resolve_config_path(config_path: Path) -> Path:
    """
    Resolve the configuration file path, checking multiple locations.

    Args:
        config_path: Initial path to the config file

    Returns:
        Path: Resolved path to the config file
    """
    # Check if the path is absolute and exists
    if config_path.is_absolute() and config_path.exists():
        return config_path

    # Check relative to current directory
    if Path(config_path.name).exists():
        return Path(config_path.name).absolute()

    # Check in the config directory relative to this file
    config_dir = Path(__file__).parent / "config"
    potential_path = config_dir / config_path.name
    if potential_path.exists():
        return potential_path

    # Check in the parent directory's config
    parent_config_dir = Path(__file__).parent.parent / "config"
    potential_path = parent_config_dir / config_path.name
    if potential_path.exists():
        return potential_path

    # Check in the evaluation_pipeline directory
    eval_pipeline_dir = Path(__file__).parent
    potential_path = eval_pipeline_dir / config_path.name
    if potential_path.exists():
        return potential_path

    # If we can't find it, return the original path (which will fail)
    return config_path


def handle_output_path_custom(path: Path, overwrite: bool) -> Path:
    """
    Custom function to handle output path with proper overwrite behavior.

    Args:
        path: The desired output path
        overwrite: Whether to overwrite existing files

    Returns:
        Path: The final output path
    """
    if not overwrite and path.exists():
        # If overwrite is False and the file exists, create a new filename
        base_name = path.stem
        extension = path.suffix
        directory = path.parent

        counter = 1
        while True:
            new_name = f"{base_name}_{counter}{extension}"
            new_path = directory / new_name
            if not new_path.exists():
                logger.info(f"File {path} exists, saving to {new_path} instead")
                return new_path
            counter += 1

    # If overwrite is True or file doesn't exist, return the original path
    return path


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

    # Verify both datasets have the same questions
    if len(df1) != len(df2):
        logger.warning(f"Datasets have different lengths: {len(df1)} vs {len(df2)}")

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


def parse_pairwise_response(eval_text: str, model1_name: str, model2_name: str) -> dict:
    """Parse responses from pairwise comparison evaluation."""

    # Extract analysis section
    analysis_match = re.search(r"ANALYSIS:\s*(.*?)(?=WINNER:|$)", eval_text, re.DOTALL)
    analysis = (
        analysis_match.group(1).strip() if analysis_match else "Analysis not found"
    )

    # Extract winner
    winner_match = re.search(r"WINNER:\s*(.+?)(?=REASON:|$)", eval_text, re.DOTALL)
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
    reason_match = re.search(r"REASON:\s*(.*?)$", eval_text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else "Reason not found"

    return {"analysis": analysis, "winner": winner, "reason": reason}


def evaluate_pairwise_row(row: pd.Series, metric: str, config: EvalConfig) -> dict:
    """Evaluate a single row for pairwise comparison on the specified metric."""

    # Initialize the LLM client
    llm_client = OpenAI_Client()

    # Override timeout and max_tokens
    llm_client.llm_client.timeout = Timeout(500.0, connect=10.0)
    llm_client.max_tokens = 2048

    try:
        question = row[QUERY_COLUMN]
        model1_name = row["model1_name"]
        model2_name = row["model2_name"]

        if metric == "correctness_pairwise":
            answer1 = row[f"{ANSWER_COLUMN}_model1"]
            answer2 = row[f"{ANSWER_COLUMN}_model2"]

            prompt = config.get_prompt(metric, config.default_model).format(
                question=question,
                model1_name=model1_name,
                answer1=answer1,
                model2_name=model2_name,
                answer2=answer2,
            )
        elif metric == "retrievability_pairwise":
            context1 = row[f"{RAG_RELEVANT_EXTRACTS_COLUMN}_model1"]
            context2 = row[f"{RAG_RELEVANT_EXTRACTS_COLUMN}_model2"]

            prompt = config.get_prompt(metric, config.default_model).format(
                question=question,
                model1_name=model1_name,
                context1=context1,
                model2_name=model2_name,
                context2=context2,
            )
        else:
            raise ValueError(f"Unsupported pairwise metric: {metric}")

        # Generate evaluation
        eval_text = llm_client.generate(
            prompt, max_tokens=2048, timeout=Timeout(500.0, connect=10.0)
        )

        # Parse response
        results = parse_pairwise_response(eval_text, model1_name, model2_name)

        # Add question and metric info
        results["question"] = question
        results["metric"] = metric

        logger.info(
            f"Pairwise comparison for question: '{question[:50]}...' on {metric}: Winner: {results['winner']}"
        )

        return results

    except Exception as e:
        logger.error(f"Error in pairwise evaluation: {e}")
        return {
            "question": row[QUERY_COLUMN],
            "metric": metric,
            "analysis": f"Error during evaluation: {str(e)}",
            "winner": "Error",
            "reason": str(e),
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

    # Get pairwise metrics from config
    metrics = [m for m in config.active_metrics if m.endswith("_pairwise")]

    if not metrics:
        logger.warning(
            "No pairwise metrics found in config. Please ensure you have enabled metrics like 'correctness_pairwise' or 'retrievability_pairwise'"
        )
        return pd.DataFrame()

    results_dfs = []

    for metric in metrics:
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

        for metric in metrics:
            metric_results = all_metrics_df[all_metrics_df["metric"] == metric]
            total = len(metric_results)

            if total > 0:
                model1_wins = sum(metric_results["winner"] == model1_name)
                model2_wins = sum(metric_results["winner"] == model2_name)
                ties = sum(metric_results["winner"] == "Tie")

                logger.info(f"\n{metric} Summary:")
                logger.info(
                    f"  {model1_name}: {model1_wins} wins ({model1_wins/total:.1%})"
                )
                logger.info(
                    f"  {model2_name}: {model2_wins} wins ({model2_wins/total:.1%})"
                )
                logger.info(f"  Ties: {ties} ({ties/total:.1%})")

        return all_metrics_df
    else:
        return pd.DataFrame()


def setup_vllm_environment(
    model_name: str, eval_config: EvalConfig, server_config: ServerConfig
) -> dict:
    """
    Set up the environment for local VLLM models.

    Args:
        model_name: The name of the model to use for evaluation
        eval_config: The evaluation configuration
        server_config: The server configuration

    Returns:
        dict: Environment variables for OpenAI client
    """
    model_config = eval_config.get_model_config(model_name)
    deployment_type = model_config.get("deployment_type", "local")

    if deployment_type == "cloud":
        return {
            "OPENAI_API_KEY": model_config["api_key"],
            "OPENAI_ENDPOINT": model_config["api_base"],
            "OPENAI_DEFAULT_MODEL_NAME": model_config["model_name"],
        }
    else:
        return {
            "OPENAI_ENDPOINT": f"http://localhost:{server_config.port}/v1",
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_DEFAULT_MODEL_NAME": model_name,
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
        "pair_jury_01.toml", help="Path to the evaluation configuration file"
    ),
    server_config_path: str = typer.Option(
        "server_config.toml", help="Path to the server configuration file"
    ),
    output_dir: str = typer.Option(
        "pairwise_results", help="Directory to save comparison results"
    ),
    overwrite: bool = typer.Option(
        False, help="Whether to overwrite existing output files"
    ),
):
    """
    Run pairwise comparison between two RAG model outputs.
    """
    logger.info(
        f"Starting pairwise comparison between {set1_filename} and {set2_filename}"
    )

    # Set up paths
    set1_path = BASE_OUTPUT_DIR / set1_filename
    set2_path = BASE_OUTPUT_DIR / set2_filename

    if not set1_path.exists() or not set2_path.exists():
        logger.error(f"Input files not found: {set1_path} or {set2_path}")
        return

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

    # Resolve configuration paths
    resolved_config_path = resolve_config_path(Path(config_path))
    resolved_server_config_path = resolve_config_path(Path(server_config_path))

    if not resolved_config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error(f"Searched in multiple locations. Please provide a valid path.")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Script directory: {Path(__file__).parent}")
        sys.exit(1)

    if not resolved_server_config_path.exists():
        logger.error(f"Server configuration file not found: {server_config_path}")
        logger.error(f"Searched in multiple locations. Please provide a valid path.")
        sys.exit(1)

    logger.info(f"Using config file: {resolved_config_path}")
    logger.info(f"Using server config file: {resolved_server_config_path}")

    # Load configurations
    eval_config = EvalConfig(resolved_config_path)
    server_config = ServerConfig(resolved_server_config_path)

    # Setup environment for OpenAI client
    os.environ.update(
        setup_vllm_environment(eval_config.default_model, eval_config, server_config)
    )

    # Step 2: Run pairwise comparison
    full_output_dir = RESULTS_BASE_DIR / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"pairwise_{model1_name}_vs_{model2_name}.xlsx"
    output_path = full_output_dir / output_filename
    output_path = handle_output_path_custom(output_path, overwrite)

    # Run evaluation - now reading metrics from config
    results_df = evaluate_pairwise_metrics(merged_df, eval_config)

    # Save results
    if not results_df.empty:
        results_df.to_excel(output_path, index=False)
        logger.success(f"Pairwise comparison results saved to {output_path}")

        # Create a summary file
        summary_data = []
        metrics = [m for m in eval_config.active_metrics if m.endswith("_pairwise")]

        for metric in metrics:
            metric_results = results_df[results_df["metric"] == metric]
            total = len(metric_results)

            if total > 0:
                model1_wins = sum(metric_results["winner"] == model1_name)
                model2_wins = sum(metric_results["winner"] == model2_name)
                ties = sum(metric_results["winner"] == "Tie")

                summary_data.append(
                    {
                        "metric": metric,
                        f"{model1_name}_wins": model1_wins,
                        f"{model1_name}_win_pct": f"{model1_wins/total:.1%}",
                        f"{model2_name}_wins": model2_wins,
                        f"{model2_name}_win_pct": f"{model2_wins/total:.1%}",
                        "ties": ties,
                        "tie_pct": f"{ties/total:.1%}",
                        "total_comparisons": total,
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        summary_path = full_output_dir / f"summary_{model1_name}_vs_{model2_name}.xlsx"
        summary_path = handle_output_path_custom(summary_path, overwrite)
        summary_df.to_excel(summary_path, index=False)
        logger.success(f"Summary results saved to {summary_path}")
    else:
        logger.error("No results generated from pairwise comparison")

    logger.success("Pairwise comparison completed!")


if __name__ == "__main__":
    typer.run(main)
