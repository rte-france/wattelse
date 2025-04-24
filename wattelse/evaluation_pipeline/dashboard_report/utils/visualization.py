"""Visualization components for the dashboard."""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import plotly.graph_objects as go
import plotly.colors
import pandas as pd
from utils import (
    PAIRWISE_METRIC_COLUMN,
    PAIRWISE_WINNER_COLUMN,
)


def create_timing_plot(experiments_data, column_name, title):
    """Create timing distribution plot."""
    fig = go.Figure()
    for exp in experiments_data:
        if column_name in exp["timing"].columns:
            fig.add_trace(
                go.Box(y=exp["timing"][column_name], name=exp["name"], boxpoints="all")
            )
    fig.update_layout(title=title, height=500, yaxis_title="Time (seconds)")
    return fig


def create_average_radar_plot(experiments_data):
    """Create a radar plot comparing average performance metrics across experiments."""
    # Get all available metrics
    all_metrics = set()
    for exp in experiments_data:
        for judge, df in exp["dfs"].items():
            metrics = [
                col.replace("_score", "")
                for col in df.columns
                if col.endswith("_score")
            ]
            all_metrics.update(metrics)

    # Sort metrics alphabetically for consistent order
    sorted_metrics = sorted(all_metrics)

    # Use vibrant colors for different experiments
    color_schemes = [
        plotly.colors.qualitative.Plotly,  # Bright default Plotly colors
        plotly.colors.qualitative.D3,  # D3 color scheme
        plotly.colors.qualitative.G10,  # G10 color scheme
        plotly.colors.qualitative.T10,  # Tableau 10 color scheme
    ]

    # Flatten the color schemes to have one large color pool
    all_colors = []
    for scheme in color_schemes:
        all_colors.extend(scheme)

    fig = go.Figure()

    # Create a trace for each experiment's average across judges
    for exp_idx, exp in enumerate(experiments_data):
        exp_name = exp["name"]
        metrics_values = {}

        # Calculate average performance percentage for each metric across judges
        for metric in sorted_metrics:
            metric_values = []

            for judge, df in exp["dfs"].items():
                score_col = f"{metric}_score"
                if score_col in df.columns:
                    good_score_pct = (
                        df[score_col][df[score_col].isin([4, 5])].count()
                        / df[score_col].count()
                        * 100
                    )
                    metric_values.append(good_score_pct)

            if metric_values:
                metrics_values[metric] = sum(metric_values) / len(metric_values)
            else:
                metrics_values[metric] = 0

        # Prepare values for the radar plot
        radar_values = [metrics_values.get(metric, 0) for metric in sorted_metrics]

        # Use distinct color for each experiment
        color = all_colors[exp_idx % len(all_colors)]

        # Add trace with higher line opacity, semi-transparent fill, and visible markers
        fig.add_trace(
            go.Scatterpolar(
                r=radar_values,
                theta=sorted_metrics,
                name=exp_name,
                line=dict(color=color, width=3),  # Thicker lines for better visibility
                fill="toself",  # Fill the area inside the radar
                fillcolor=color,
                opacity=0.4,  # Semi-transparent fill
                marker=dict(
                    size=10,  # Larger markers for better visibility
                    color=color,
                    line=dict(width=2, color="white"),  # White border around markers
                ),
                showlegend=True,
            )
        )

    # Improve radar plot appearance
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                showticklabels=True,
                ticks="outside",
                gridcolor="rgba(200, 200, 200, 1)",  # Lighter grid
                linecolor="rgba(200, 200, 200, 1)",  # Clear axis line
            ),
            angularaxis=dict(
                tickfont=dict(
                    size=12, color="black"
                ),  # Make metric labels more readable
                rotation=90,  # Start from the top
                direction="clockwise",  # Standard radar plot direction
            ),
            bgcolor="rgba(240, 240, 240, 0.1)",  # Light background color
        ),
        showlegend=True,
        title=dict(
            text="Average Performance by Metric (%)",
            font=dict(size=18),
        ),
        height=450,
        width=450,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    return fig


def create_radar_plot(experiments_data):
    """Create a radar plot comparing performance metrics across experiments."""

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

    # Sort metrics alphabetically for consistent order
    sorted_metrics = sorted(all_metrics)

    # Use vibrant color schemes that are more visually distinct
    color_schemes = [
        plotly.colors.qualitative.Plotly,  # Bright default Plotly colors
        plotly.colors.qualitative.D3,  # D3 color scheme
        plotly.colors.qualitative.G10,  # G10 color scheme
        plotly.colors.qualitative.T10,  # Tableau 10 color scheme
    ]

    # Flatten the color schemes to have one large color pool
    all_colors = []
    for scheme in color_schemes:
        all_colors.extend(scheme)

    fig = go.Figure()

    # Create traces for each experiment and judge
    for exp_idx, exp in enumerate(experiments_data):
        for judge in all_judges:
            if judge in exp["dfs"]:
                df = exp["dfs"][judge]
                metrics_values = []

                # Calculate Performance percentages for each metric
                for metric in sorted_metrics:
                    score_col = f"{metric}_score"
                    if score_col in df.columns:
                        good_score_pct = (
                            df[score_col][df[score_col].isin([4, 5])].count()
                            / df[score_col].count()
                            * 100
                        )
                        metrics_values.append(good_score_pct)
                    else:
                        metrics_values.append(0)

                # IMPORTANT: Make the shape close by repeating the first point at the end
                # First, get the list of metrics and values
                display_metrics = sorted_metrics.copy()
                metrics_values_closed = metrics_values.copy()

                # Add the first metric and its value to the end to create a closed loop
                # Only needed when connecting manually, not using fill='toself'
                display_metrics.append(display_metrics[0])
                metrics_values_closed.append(metrics_values_closed[0])

                # Generate a distinct color index for each experiment-judge combination
                color_idx = (
                    exp_idx * len(all_judges) + list(all_judges).index(judge)
                ) % len(all_colors)
                color = all_colors[color_idx]

                # Add trace with higher line opacity, clear fill, and larger markers
                fig.add_trace(
                    go.Scatterpolar(
                        r=metrics_values,
                        theta=sorted_metrics,
                        name=f"{exp['name']} - {judge}",
                        line=dict(
                            color=color, width=3  # Thicker lines for better visibility
                        ),
                        fill="toself",  # Fill the area inside the radar
                        fillcolor=color,
                        opacity=0.4,  # Semi-transparent fill
                        marker=dict(
                            size=10,  # Larger markers for better visibility
                            color=color,
                            line=dict(
                                width=2,
                                color="white",  # White border around markers for contrast
                            ),
                        ),
                        showlegend=True,
                    )
                )

    # Improve radar plot appearance
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                showticklabels=True,
                ticks="outside",
                gridcolor="rgba(200, 200, 200, 1)",  # Lighter grid
                linecolor="rgba(200, 200, 200, 1)",  # Clear axis line
            ),
            angularaxis=dict(
                tickfont=dict(
                    size=12, color="black"
                ),  # Make metric labels more readable
                rotation=90,  # Start from the top
                direction="clockwise",  # Standard radar plot direction
            ),
            bgcolor="rgba(240, 240, 240, 0.1)",  # Light background color
        ),
        showlegend=True,
        title=dict(
            text="Performance Metrics Radar Plot (% of Performances)",
            font=dict(size=20),
        ),
        height=600,
        width=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=80, r=80, t=100, b=80),  # Adjust margins for better fit
    )

    return fig


def create_pairwise_pie_chart(combined_stats, pairwise_dfs=None, metric_name=None):
    """Create a pie chart showing win rates for a specific metric, including all possible outcomes.

    Args:
        combined_stats (pd.DataFrame): Combined statistics DataFrame
        pairwise_dfs (dict, optional): Dictionary of dataframes with raw pairwise data by judge
        metric_name (str, optional): Specific metric to use. If None, first looks
                                    for correctness_pairwise, then uses any available metric.

    Returns:
        go.Figure: Plotly figure with pie chart, or None if no data available
    """
    # First try using the combined stats if provided
    if combined_stats is not None and not combined_stats.empty:
        metric_col = None
        title = ""

        if metric_name:
            metric_col = f"{metric_name}_win_rate"
            title = f"{metric_name.title()} Win Rates"
        else:
            # Look for correctness_pairwise first
            for col in combined_stats.columns:
                if "correctness_pairwise" in col.lower() and col.endswith("_win_rate"):
                    metric_col = col
                    metric_name = col.replace("_win_rate", "")
                    title = f"{metric_name.title()} Win Rates"
                    break

        # If specific metric not found, use any available win_rate column
        if not metric_col:
            for col in combined_stats.columns:
                if col.endswith("_win_rate"):
                    metric_col = col
                    metric_name = col.replace("_win_rate", "")
                    title = f"{metric_name.title()} Win Rates"
                    break

    # If we have raw pairwise data and a metric name, use that for complete statistics
    if pairwise_dfs and metric_name:
        # Create a more comprehensive chart from raw data to include all outcomes
        win_data = []
        all_counts = {}
        total_comparisons = 0

        # Process each judge's dataframe
        for judge_name, df in pairwise_dfs.items():
            if (
                PAIRWISE_METRIC_COLUMN in df.columns
                and PAIRWISE_WINNER_COLUMN in df.columns
            ):
                # Filter for the selected metric
                metric_df = df[df[PAIRWISE_METRIC_COLUMN] == metric_name]

                if not metric_df.empty:
                    # Count occurrences of each winner
                    winners = metric_df[PAIRWISE_WINNER_COLUMN].value_counts()

                    # Add to overall counts
                    for winner, count in winners.items():
                        if winner in all_counts:
                            all_counts[winner] += count
                        else:
                            all_counts[winner] = count

                        total_comparisons += count

        # Calculate percentages and create data for the chart
        for winner, count in all_counts.items():
            percentage = (
                (count / total_comparisons * 100) if total_comparisons > 0 else 0
            )
            win_data.append(
                {"Winner": winner, "Count": count, "Percentage": percentage}
            )

        # Sort by percentage (highest first)
        win_data = sorted(win_data, key=lambda x: x["Percentage"], reverse=True)

        # Create pie chart with all outcomes
        if win_data:
            pie_df = pd.DataFrame(win_data)

            fig = go.Figure(
                go.Pie(
                    labels=pie_df["Winner"],
                    values=pie_df["Percentage"],
                    hovertemplate="Winner: %{label}<br>Percentage: %{value:.1f}%<br>Count: %{customdata[0]}/%{customdata[1]}<extra></extra>",
                    customdata=[
                        (row["Count"], total_comparisons)
                        for _, row in pie_df.iterrows()
                    ],
                )
            )

            fig.update_layout(
                title=f"{metric_name.title()} Win Distribution",
            )

            return fig

    # Fallback to combined stats if we couldn't use raw data
    elif combined_stats is not None and metric_col:
        win_data = []
        for _, row in combined_stats.iterrows():
            model = row["Model"]
            win_rate = row[metric_col]

            # Get wins and total if available
            wins_col = metric_col.replace("_win_rate", "_wins")
            total_col = metric_col.replace("_win_rate", "_total")

            wins = row[wins_col] if wins_col in row else 0
            total = row[total_col] if total_col in row else 0

            win_data.append(
                {"Model": model, "Win Rate": win_rate, "Wins": wins, "Total": total}
            )

        # If we have data, create the pie chart
        if win_data:
            pie_df = pd.DataFrame(win_data)

            fig = go.Figure(
                go.Pie(
                    labels=pie_df["Model"],
                    values=pie_df["Win Rate"],
                    hovertemplate="Model: %{label}<br>Win Rate: %{value:.1f}%<br>Wins: %{customdata[0]}/%{customdata[1]}<extra></extra>",
                    customdata=pie_df[["Wins", "Total"]].values,
                )
            )

            fig.update_layout(
                title=title,
            )

            return fig

    return None
