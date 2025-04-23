"""Data loading and processing utilities."""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import pandas as pd
from pathlib import Path
from .constants import *
from wattelse.evaluation_pipeline import RESULTS_BASE_DIR


def calculate_good_score_percentage(scores):
    """Calculate percentage of Performances (4-5) in the series."""
    if scores is None or len(scores) == 0:
        return 0
    good_scores = scores[scores.isin([4, 5])].count()
    total_scores = scores.count()
    return (good_scores / total_scores * 100) if total_scores > 0 else 0


def load_evaluation_files(eval_dir: str):
    """Load and combine all evaluation Excel files from the directory."""
    if not eval_dir:
        return None

    # Handle both absolute and relative paths
    if not eval_dir.startswith("/"):
        # TODO If relative path, construct path relative to project root
        eval_path = RESULTS_BASE_DIR / eval_dir
    else:
        eval_path = Path(eval_dir)

    if not eval_path.exists():
        return None

    excel_files = list(eval_path.glob("evaluation_*.xlsx"))
    if not excel_files:
        return None

    all_dfs = {}
    combined_df = None
    timing_df = None

    for file in excel_files:
        if "combined" not in file.name:
            model_name = file.stem.split("_")[-1]
            df = pd.read_excel(file)

            # Extract timing columns
            if timing_df is None:
                timing_cols = [QUERY_COLUMN]
                if RAG_QUERY_TIME_COLUMN in df.columns:
                    timing_cols.append(RAG_QUERY_TIME_COLUMN)
                if RAG_RETRIEVER_TIME_COLUMN in df.columns:
                    timing_cols.append(RAG_RETRIEVER_TIME_COLUMN)
                timing_df = df[timing_cols].copy()

            # Process evaluation data
            eval_cols = [
                col
                for col in df.columns
                if col not in [RAG_QUERY_TIME_COLUMN, RAG_RETRIEVER_TIME_COLUMN]
            ]
            all_dfs[model_name] = df[eval_cols]

            if combined_df is None:
                combined_df = df[eval_cols].copy()
                combined_df.columns = [
                    f"{col}_{model_name}" if col != "question" else col
                    for col in combined_df.columns
                ]
            else:
                rename_cols = {
                    col: f"{col}_{model_name}"
                    for col in df[eval_cols].columns
                    if col != "question"
                }
                df_renamed = df[eval_cols].rename(columns=rename_cols)
                combined_df = combined_df.merge(df_renamed, on="question", how="outer")

    return all_dfs, combined_df, timing_df


def load_pairwise_evaluation_files(eval_dir: str):
    """Load and combine all pairwise evaluation Excel files from the directory."""
    if not eval_dir:
        return None

    # Handle both absolute and relative paths
    if not eval_dir.startswith("/"):
        # Construct path relative to pairwise results directory
        eval_path = Path(PAIRWISE_RESULTS_DIR) / eval_dir
    else:
        eval_path = Path(eval_dir)

    if not eval_path.exists():
        return None

    excel_files = list(eval_path.glob("pairwise_*.xlsx"))
    if not excel_files:
        return None

    all_pairwise_dfs = {}

    for file in excel_files:
        judge_name = file.stem.split("_")[-1]
        df = pd.read_excel(file)

        # Store dataframe by judge name
        all_pairwise_dfs[judge_name] = df

    # Calculate combined stats across all judges
    combined_stats = calculate_pairwise_combined_stats(all_pairwise_dfs)

    return all_pairwise_dfs, combined_stats


def calculate_pairwise_combined_stats(pairwise_dfs):
    """Calculate combined statistics from pairwise evaluation dataframes."""
    if not pairwise_dfs:
        return pd.DataFrame()

    # Create empty dataframe to store combined stats
    combined_stats = pd.DataFrame()

    # Extract all unique models from the dataframes
    all_models = set()
    for _, df in pairwise_dfs.items():
        if PAIRWISE_MODEL1_NAME_COLUMN in df.columns:
            all_models.update(df[PAIRWISE_MODEL1_NAME_COLUMN].unique())
        if PAIRWISE_MODEL2_NAME_COLUMN in df.columns:
            all_models.update(df[PAIRWISE_MODEL2_NAME_COLUMN].unique())

    all_models = list(all_models)

    # Initialize win counts for each model and metric
    all_metrics = set()
    for _, df in pairwise_dfs.items():
        if PAIRWISE_METRIC_COLUMN in df.columns:
            all_metrics.update(df[PAIRWISE_METRIC_COLUMN].unique())

    model_win_counts = {
        model: {metric: 0 for metric in all_metrics} for model in all_models
    }
    model_total_comparisons = {
        model: {metric: 0 for metric in all_metrics} for model in all_models
    }

    # Count wins for each model and metric
    for judge_name, df in pairwise_dfs.items():
        if PAIRWISE_WINNER_COLUMN not in df.columns:
            continue

        for _, row in df.iterrows():
            metric = row.get(PAIRWISE_METRIC_COLUMN)
            winner = row.get(PAIRWISE_WINNER_COLUMN)
            model1 = row.get(PAIRWISE_MODEL1_NAME_COLUMN)
            model2 = row.get(PAIRWISE_MODEL2_NAME_COLUMN)

            if not metric or not winner or not model1 or not model2:
                continue

            # Count win for the winning model
            if winner in model_win_counts:
                model_win_counts[winner][metric] += 1

            # Increment total comparisons for both models
            model_total_comparisons[model1][metric] += 1
            model_total_comparisons[model2][metric] += 1

    # Calculate win rates and create summary dataframe
    summary_data = []

    for model in all_models:
        model_data = {"Model": model}

        for metric in all_metrics:
            if model_total_comparisons[model][metric] > 0:
                win_rate = (
                    model_win_counts[model][metric]
                    / model_total_comparisons[model][metric]
                ) * 100
                model_data[f"{metric}_win_rate"] = win_rate
                model_data[f"{metric}_wins"] = model_win_counts[model][metric]
                model_data[f"{metric}_total"] = model_total_comparisons[model][metric]
            else:
                model_data[f"{metric}_win_rate"] = 0
                model_data[f"{metric}_wins"] = 0
                model_data[f"{metric}_total"] = 0

        summary_data.append(model_data)

    combined_stats = pd.DataFrame(summary_data)
    return combined_stats


def calculate_pairwise_combined_stats(pairwise_dfs):
    """Calculate combined statistics from pairwise evaluation dataframes."""
    if not pairwise_dfs:
        return pd.DataFrame()

    # Create empty dataframe to store combined stats
    combined_stats = pd.DataFrame()

    # Extract all unique models from the dataframes
    all_models = set()
    for _, df in pairwise_dfs.items():
        if PAIRWISE_MODEL1_NAME_COLUMN in df.columns:
            all_models.update(df[PAIRWISE_MODEL1_NAME_COLUMN].unique())
        if PAIRWISE_MODEL2_NAME_COLUMN in df.columns:
            all_models.update(df[PAIRWISE_MODEL2_NAME_COLUMN].unique())

    all_models = list(all_models)

    # Initialize win counts for each model and metric
    all_metrics = set()
    for _, df in pairwise_dfs.items():
        if PAIRWISE_METRIC_COLUMN in df.columns:
            all_metrics.update(df[PAIRWISE_METRIC_COLUMN].unique())

    model_win_counts = {
        model: {metric: 0 for metric in all_metrics} for model in all_models
    }
    model_total_comparisons = {
        model: {metric: 0 for metric in all_metrics} for model in all_models
    }

    # Count wins for each model and metric
    for judge_name, df in pairwise_dfs.items():
        if PAIRWISE_WINNER_COLUMN not in df.columns:
            continue

        for _, row in df.iterrows():
            metric = row.get(PAIRWISE_METRIC_COLUMN)
            winner = row.get(PAIRWISE_WINNER_COLUMN)
            model1 = row.get(PAIRWISE_MODEL1_NAME_COLUMN)
            model2 = row.get(PAIRWISE_MODEL2_NAME_COLUMN)

            if not metric or not winner or not model1 or not model2:
                continue

            # Count win for the winning model
            if winner in model_win_counts:
                model_win_counts[winner][metric] += 1

            # Increment total comparisons for both models
            model_total_comparisons[model1][metric] += 1
            model_total_comparisons[model2][metric] += 1

    # Calculate win rates and create summary dataframe
    summary_data = []

    for model in all_models:
        model_data = {"Model": model}

        for metric in all_metrics:
            if model_total_comparisons[model][metric] > 0:
                win_rate = (
                    model_win_counts[model][metric]
                    / model_total_comparisons[model][metric]
                ) * 100
                model_data[f"{metric}_win_rate"] = win_rate
                model_data[f"{metric}_wins"] = model_win_counts[model][metric]
                model_data[f"{metric}_total"] = model_total_comparisons[model][metric]
            else:
                model_data[f"{metric}_win_rate"] = 0
                model_data[f"{metric}_wins"] = 0
                model_data[f"{metric}_total"] = 0

        summary_data.append(model_data)

    combined_stats = pd.DataFrame(summary_data)
    return combined_stats
