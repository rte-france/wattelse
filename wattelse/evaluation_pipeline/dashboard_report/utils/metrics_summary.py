#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats


def calculate_confidence_interval(scores, confidence=0.95):
    """
    Calculate confidence interval for a series of scores.

    Args:
        scores: Series or list of score values
        confidence: Confidence level (default 0.95 for 95% confidence)

    Returns:
        Tuple of (lower_bound, upper_bound) as percentages
    """
    # Convert to numpy array if not already
    scores_array = np.array(scores)

    # Calculate mean and standard error
    mean = np.mean(scores_array)
    se = stats.sem(scores_array) if len(scores_array) > 1 else 0

    # Calculate confidence interval
    if len(scores_array) > 1:
        # Using normal distribution instead of t-distribution
        h = se * stats.norm.ppf((1 + confidence) / 2)
        lower_bound = max(0, mean - h)  # Ensure lower bound isn't negative
        upper_bound = min(100, mean + h)  # Ensure upper bound doesn't exceed 100%
        return (lower_bound, upper_bound)
    else:
        # If only one score, return the score itself with no interval
        return (mean, mean)


def create_metrics_summary(experiments_data):
    """Create a summary of Performance percentages across metrics and LLMs."""
    # Get all available metrics and judges
    all_metrics = set()
    all_judges = set()

    for exp in experiments_data:
        for judge, df in exp["dfs"].items():
            all_judges.add(judge)
            metrics = [
                col.replace("_score", "")
                for col in df.columns
                if col.endswith("_score")
            ]
            all_metrics.update(metrics)

    # Create dataframes for metric averages
    metric_summary = {metric: [] for metric in all_metrics}

    # Track judge counts for each experiment
    judge_counts = {}
    for exp in experiments_data:
        exp_name = exp["name"]
        judge_counts[exp_name] = len(exp["dfs"])

    # Track the best experiment for each metric according to each judge
    best_counts = {
        metric: {exp["name"]: 0 for exp in experiments_data} for metric in all_metrics
    }

    # Find the best experiments for each metric according to each judge (handling ties)
    for metric in sorted(all_metrics):
        # For each judge, determine the best experiments for this metric
        for judge in all_judges:
            judge_scores = {}

            # Get scores for this judge and metric across all experiments
            for exp in experiments_data:
                if judge in exp["dfs"]:
                    df = exp["dfs"][judge]
                    score_col = f"{metric}_score"
                    if score_col in df.columns:
                        good_score_pct = (
                            df[score_col][df[score_col].isin([4, 5])].count()
                            / df[score_col].count()
                            * 100
                        )
                        judge_scores[exp["name"]] = good_score_pct

            # Find all experiments with the maximum score for this judge and metric
            if judge_scores:
                max_score = max(judge_scores.values())
                # Find all experiments with this max score (handling ties)
                for exp_name, score in judge_scores.items():
                    if score == max_score:
                        best_counts[metric][exp_name] += 1

    # Calculate average Performance percentage for each experiment and metric
    for exp in experiments_data:
        exp_name = exp["name"]

        for metric in sorted(all_metrics):
            judges_values = []
            all_scores = []  # Store all individual scores for CI calculation

            for judge, df in exp["dfs"].items():
                score_col = f"{metric}_score"
                if score_col in df.columns:
                    # Calculate good score percentage
                    good_score_pct = (
                        df[score_col][df[score_col].isin([4, 5])].count()
                        / df[score_col].count()
                        * 100
                    )
                    judges_values.append(good_score_pct)

                    # Collect all individual scores for CI calculation (for all metrics now)
                    binary_scores = [
                        1 if score in [4, 5] else 0 for score in df[score_col].dropna()
                    ]
                    all_scores.extend(binary_scores)

            if judges_values:
                avg_score = sum(judges_values) / len(judges_values)

                # Calculate confidence interval for all metrics
                ci_lower, ci_upper = (0, 0)
                if all_scores:
                    # Convert percentages to proportions (0-1) for CI calculation
                    ci_lower, ci_upper = calculate_confidence_interval(
                        [s * 100 for s in all_scores]
                    )

                metric_summary[metric].append(
                    {
                        "Experiment": exp_name,
                        "Average Performance %": avg_score,
                        "Judges Count": len(judges_values),
                        "Best Count": best_counts[metric][exp_name],
                        "CI Lower": ci_lower,
                        "CI Upper": ci_upper,
                    }
                )

    # Create summary dataframes and figures
    summary_dfs = {}
    summary_figs = {}

    for metric, data in metric_summary.items():
        if data:
            # Create dataframe
            summary_df = pd.DataFrame(data)
            summary_df["Average Performance %"] = summary_df[
                "Average Performance %"
            ].round(1)

            # Round CI values for all metrics
            summary_df["CI Lower"] = summary_df["CI Lower"].round(1)
            summary_df["CI Upper"] = summary_df["CI Upper"].round(1)

            # Format highest score in bold by creating a new display column
            max_score = summary_df["Average Performance %"].max()
            summary_df["Display Score"] = summary_df["Average Performance %"].apply(
                lambda x: f"**{x:.1f}%**" if x == max_score else f"{x:.1f}%"
            )

            # Format CI for all metrics
            summary_df["CI Display"] = summary_df.apply(
                lambda row: f"({row['CI Lower']:.1f}% - {row['CI Upper']:.1f}%)", axis=1
            )

            summary_dfs[metric] = summary_df

            # Create bar chart with error bars for all metrics
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=summary_df["Experiment"],
                    y=summary_df["Average Performance %"],
                    text=summary_df["Average Performance %"].apply(
                        lambda x: f"{x:.1f}%"
                    ),
                    textposition="auto",
                    marker_color="rgb(55, 83, 139)",
                    hovertemplate="Experiment: %{x}<br>Average Performance: %{y:.1f}%<extra></extra>",
                )
            )

            # Add error bars for all metrics using CI
            error_y = []
            for _, row in summary_df.iterrows():
                avg = row["Average Performance %"]
                error_y.append((avg - row["CI Lower"], row["CI Upper"] - avg))

            error_y_array = np.array(error_y).T

            fig.add_trace(
                go.Scatter(
                    x=summary_df["Experiment"],
                    y=summary_df["Average Performance %"],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=error_y_array[1],
                        arrayminus=error_y_array[0],
                        color="rgba(55, 83, 139, 0.6)",
                        thickness=1.5,
                        width=3,
                    ),
                    mode="markers",
                    marker=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            fig.update_layout(
                title=f"Average {metric.title()} Performances Across Judges",
                xaxis_title="Experiment",
                yaxis_title="Average Performance %",
                yaxis=dict(ticksuffix="%", range=[0, 100]),
                height=400,
                showlegend=False,
            )

            summary_figs[metric] = fig

    # Create overall metrics data
    overall_data = []

    for exp in experiments_data:
        exp_name = exp["name"]
        metrics_values = {}
        ci_values = {}  # Store CI values for each metric

        for metric in sorted(all_metrics):
            metric_values = []
            all_scores = []  # For CI calculation

            for judge, df in exp["dfs"].items():
                score_col = f"{metric}_score"
                if score_col in df.columns:
                    good_score_pct = (
                        df[score_col][df[score_col].isin([4, 5])].count()
                        / df[score_col].count()
                        * 100
                    )
                    metric_values.append(good_score_pct)

                    # Collect individual scores for CI for all metrics
                    binary_scores = [
                        1 if score in [4, 5] else 0 for score in df[score_col].dropna()
                    ]
                    all_scores.extend(binary_scores)

            if metric_values:
                metrics_values[metric] = sum(metric_values) / len(metric_values)

                # Calculate CI for all metrics
                if all_scores:
                    ci_lower, ci_upper = calculate_confidence_interval(
                        [s * 100 for s in all_scores]
                    )
                    ci_values[f"{metric}_ci_lower"] = ci_lower
                    ci_values[f"{metric}_ci_upper"] = ci_upper

        if metrics_values:
            overall_data.append(
                {
                    "Experiment": exp_name,
                    "Number of Judges": judge_counts[exp_name],
                    **metrics_values,
                    **ci_values,  # Add CI values for all metrics
                    **{
                        f"{metric}_best_count": best_counts[metric][exp_name]
                        for metric in all_metrics
                    },
                }
            )

    overall_df = pd.DataFrame(overall_data)

    # Round values for better display
    for col in overall_df.columns:
        if (
            col != "Experiment"
            and col != "Number of Judges"
            and not col.endswith("_best_count")
            and not col.endswith("_ci_lower")
            and not col.endswith("_ci_upper")
        ):
            overall_df[col] = overall_df[col].round(1)
        elif col.endswith("_ci_lower") or col.endswith("_ci_upper"):
            overall_df[col] = overall_df[col].round(1)

    return (
        summary_dfs,
        summary_figs,
        overall_df,
    )
