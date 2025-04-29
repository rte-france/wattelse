"""Pairwise comparison evaluation utilities."""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import uuid

# Import constants
from .constants import (
    PAIRWISE_WINNER_COLUMN,
    PAIRWISE_REASON_COLUMN,
    PAIRWISE_QUESTION_COLUMN,
    PAIRWISE_METRIC_COLUMN,
    PAIRWISE_MODEL1_NAME_COLUMN,
    PAIRWISE_MODEL2_NAME_COLUMN,
)

# Path constant for pairwise results
PAIRWISE_RESULTS_DIR = "/DSIA/nlp/experiments/results/pairwise_results/"


def load_pairwise_evaluation_files(file_path: str):
    """Load a specific pairwise evaluation file."""
    if not file_path:
        return None

    file_path = Path(file_path)

    if not file_path.exists():
        return None

    try:
        # Load the dataframe
        df = pd.read_excel(file_path)

        # Extract judge name from filename
        judge_name = file_path.stem.split("_")[-1]

        # Store in dict with judge name as key
        all_pairwise_dfs = {judge_name: df}

        # Calculate combined stats
        combined_stats = calculate_pairwise_combined_stats(all_pairwise_dfs)

        return all_pairwise_dfs, combined_stats
    except Exception as e:
        print(f"Error loading pairwise file {file_path}: {str(e)}")
        return None


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


def handle_pairwise_analysis_page(pairwise_experiments_data):
    """Handle pairwise analysis page with visualizations and statistics."""
    st.header("Pairwise Analysis")

    # Create tabs for each pairwise experiment
    experiment_tabs = st.tabs([exp["name"] for exp in pairwise_experiments_data])

    for i, (tab, experiment) in enumerate(
        zip(experiment_tabs, pairwise_experiments_data)
    ):
        with tab:
            st.subheader(f"Experiment: {experiment['name']}")

            # Get all judges
            judges = list(experiment["dfs"].keys())

            if not judges:
                st.warning("No judge data found for this experiment.")
                continue

            # Create tabs for Summary and Individual Judges
            judge_tabs = st.tabs(["Summary"] + judges)

            # Summary tab
            with judge_tabs[0]:
                st.markdown("### Pairwise Comparison Summary")

                # Get combined stats
                combined_stats = experiment["combined_stats"]

                if combined_stats.empty:
                    st.warning("No combined statistics available for this experiment.")
                    continue

                # Display win rate table
                st.markdown("#### Win Rates by Model")

                # Format the win rate table
                display_df = pd.DataFrame()
                display_df["Model"] = combined_stats["Model"]

                # Get all metrics
                metrics = [
                    col.replace("_win_rate", "")
                    for col in combined_stats.columns
                    if col.endswith("_win_rate")
                ]

                # Add win rate columns with formatting
                for metric in metrics:
                    if f"{metric}_win_rate" in combined_stats.columns:
                        display_df[f"{metric.title()} Win Rate"] = combined_stats[
                            f"{metric}_win_rate"
                        ].apply(lambda x: f"{x:.1f}%")

                        # Add wins/total columns
                        if (
                            f"{metric}_wins" in combined_stats.columns
                            and f"{metric}_total" in combined_stats.columns
                        ):
                            display_df[f"{metric.title()} Wins/Total"] = (
                                combined_stats.apply(
                                    lambda row: f"{int(row[f'{metric}_wins'])}/{int(row[f'{metric}_total'])}",
                                    axis=1,
                                )
                            )

                # Display the table
                st.dataframe(display_df, use_container_width=True)

                # Create bar chart for win rates
                st.markdown("#### Win Rate Visualization")

                # Prepare data for the chart
                chart_data = []
                for _, row in combined_stats.iterrows():
                    model = row["Model"]
                    for metric in metrics:
                        if f"{metric}_win_rate" in row:
                            chart_data.append(
                                {
                                    "Model": model,
                                    "Metric": metric.title(),
                                    "Win Rate": row[f"{metric}_win_rate"],
                                }
                            )

                chart_df = pd.DataFrame(chart_data)

                # Create the bar chart
                if not chart_df.empty:
                    fig = go.Figure()

                    for model in chart_df["Model"].unique():
                        model_data = chart_df[chart_df["Model"] == model]
                        fig.add_trace(
                            go.Bar(
                                x=model_data["Metric"],
                                y=model_data["Win Rate"],
                                name=model,
                                hovertemplate="Model: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>",
                            )
                        )

                    fig.update_layout(
                        title="Win Rates by Model and Metric",
                        xaxis_title="Metric",
                        yaxis_title="Win Rate (%)",
                        yaxis_ticksuffix="%",
                        barmode="group",
                        legend_title="Model",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"win_rate_chart_{str(uuid.uuid4())}",
                    )

            # Individual judge tabs
            for j, judge_name in enumerate(judges):
                with judge_tabs[j + 1]:
                    st.markdown(f"### Judge: {judge_name}")

                    # Get judge-specific data
                    judge_df = experiment["dfs"].get(judge_name)

                    if judge_df is None or judge_df.empty:
                        st.warning(f"No data available for judge {judge_name}.")
                        continue

                    # Display judge evaluation data
                    st.markdown("#### Raw Evaluation Data")
                    st.dataframe(judge_df, use_container_width=True)

                    # Calculate statistics by metric
                    metrics = (
                        judge_df[PAIRWISE_METRIC_COLUMN].unique()
                        if PAIRWISE_METRIC_COLUMN in judge_df.columns
                        else []
                    )

                    if metrics:
                        st.markdown("#### Win Statistics by Metric")

                        for metric in metrics:
                            metric_df = judge_df[
                                judge_df[PAIRWISE_METRIC_COLUMN] == metric
                            ]

                            if not metric_df.empty:
                                st.markdown(f"##### {metric.title()}")

                                # Count wins by model
                                if PAIRWISE_WINNER_COLUMN in metric_df.columns:
                                    win_counts = metric_df[
                                        PAIRWISE_WINNER_COLUMN
                                    ].value_counts()

                                    # Create win count dataframe
                                    win_df = pd.DataFrame(
                                        {
                                            "Model": win_counts.index,
                                            "Wins": win_counts.values,
                                            "Win Rate": (
                                                win_counts / len(metric_df) * 100
                                            ).apply(lambda x: f"{x:.1f}%"),
                                        }
                                    )

                                    # Display win statistics
                                    st.dataframe(win_df, use_container_width=True)

                                    # Create win rate pie chart
                                    fig = go.Figure(
                                        go.Pie(
                                            labels=win_df["Model"],
                                            values=win_df["Wins"],
                                            hovertemplate="Model: %{label}<br>Wins: %{value}<br>Percentage: %{percent}<extra></extra>",
                                        )
                                    )

                                    fig.update_layout(
                                        title=f"{metric.title()} Win Distribution"
                                    )

                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"{judge_name}_{metric}_pie_{str(uuid.uuid4())}",
                                    )

                                # Display random sample of evaluation reasons
                                if PAIRWISE_REASON_COLUMN in metric_df.columns:
                                    with st.expander(
                                        "Sample Evaluation Reasons", expanded=False
                                    ):
                                        # Get a random sample of reasons (up to 5)
                                        sample_size = min(5, len(metric_df))
                                        sample_df = metric_df.sample(sample_size)

                                        for _, row in sample_df.iterrows():
                                            winner = row.get(
                                                PAIRWISE_WINNER_COLUMN, "N/A"
                                            )
                                            reason = row.get(
                                                PAIRWISE_REASON_COLUMN,
                                                "No reason provided",
                                            )
                                            question = row.get(
                                                PAIRWISE_QUESTION_COLUMN, ""
                                            )

                                            st.markdown(f"**Question:** {question}")
                                            st.markdown(f"**Winner:** {winner}")
                                            st.markdown(f"**Reason:** {reason}")
                                            st.divider()


def create_pairwise_overview_section(pairwise_data):
    """Create a streamlit section showing pairwise comparison overview for the Performance page.

    Args:
        pairwise_data (list): List of pairwise experiment data dictionaries

    Returns:
        None: Directly renders to Streamlit
    """
    # If there are multiple pairwise datasets, add a selectbox
    if len(pairwise_data) > 1:
        selected_pairwise_name = st.selectbox(
            "Select Pairwise Comparison",
            [exp["name"] for exp in pairwise_data],
            key="overview_pairwise_selector",
        )
        selected_pairwise = next(
            (exp for exp in pairwise_data if exp["name"] == selected_pairwise_name),
            pairwise_data[0],
        )
    else:
        selected_pairwise = pairwise_data[0]

    # Get combined stats
    combined_stats = selected_pairwise["combined_stats"]

    if not combined_stats.empty:
        # Calculate total questions evaluated
        total_questions = 0
        unique_questions = set()

        # Go through all judge dataframes to count unique questions
        for judge_name, df in selected_pairwise["dfs"].items():
            if PAIRWISE_QUESTION_COLUMN in df.columns:
                unique_questions.update(df[PAIRWISE_QUESTION_COLUMN].unique())

        total_questions = len(unique_questions)

        # Create columns for layout matching the criteria evaluation
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display win rate table
            st.markdown(
                f"##### Pairwise Win Statistics (Total Questions: {total_questions})"
            )

            # Format the win rate table
            display_df = pd.DataFrame()
            display_df["Model"] = combined_stats["Model"]

            # Get all metrics
            metrics = [
                col.replace("_win_rate", "")
                for col in combined_stats.columns
                if col.endswith("_win_rate")
            ]

            # Add win rate columns with formatting
            for metric in metrics:
                if f"{metric}_win_rate" in combined_stats.columns:
                    display_df[f"{metric.title()} Win Rate"] = combined_stats[
                        f"{metric}_win_rate"
                    ].apply(lambda x: f"{x:.1f}%")

                    # Add wins/total columns
                    if (
                        f"{metric}_wins" in combined_stats.columns
                        and f"{metric}_total" in combined_stats.columns
                    ):
                        display_df[f"{metric.title()} Wins/Total"] = (
                            combined_stats.apply(
                                lambda row: f"{int(row[f'{metric}_wins'])}/{int(row[f'{metric}_total'])}",
                                axis=1,
                            )
                        )

            # Display the table
            st.dataframe(display_df, use_container_width=True)

        with col2:
            # Create pie chart for a selected metric
            # First look for correctness_pairwise, then use first available metric
            selected_metric = None
            for metric in metrics:
                if "correctness" in metric.lower():
                    selected_metric = metric
                    break

            # If no correctness metric, use the first available
            if not selected_metric and metrics:
                selected_metric = metrics[0]

            if selected_metric:
                # Create win data for pie chart
                win_data = []
                for _, row in combined_stats.iterrows():
                    model = row["Model"]
                    win_rate = row[f"{selected_metric}_win_rate"]
                    win_data.append({"Model": model, "Win Rate": win_rate})

                pie_df = pd.DataFrame(win_data)

                fig = go.Figure(
                    go.Pie(
                        labels=pie_df["Model"],
                        values=pie_df["Win Rate"],
                        hovertemplate="Model: %{label}<br>Win Rate: %{value:.1f}%<extra></extra>",
                    )
                )

                fig.update_layout(
                    title=f"{selected_metric.title()} Win Rates",
                )

                # Display the pie chart
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"pairwise_pie_{str(uuid.uuid4())}",
                )
            else:
                st.info("No metric data available for visualization.")
    else:
        st.info("No statistics available for the selected pairwise comparison.")
