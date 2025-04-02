"""Main Streamlit application."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import uuid  # Use UUID for Guaranteed Unique Keys for charts
from pathlib import Path
from utils import (
    load_evaluation_files,
    calculate_good_score_percentage,
    create_timing_plot,
    create_radar_plot,
    create_average_radar_plot,
    create_metrics_summary,
    create_pdf_report,
    get_pdf_download_link,
    RAG_QUERY_TIME_COLUMN,
    RAG_RETRIEVER_TIME_COLUMN,
    METRIC_DESCRIPTIONS,
)


def setup_page():
    """Configure page settings and title."""
    st.set_page_config(page_title="RAG Experiment Comparison", layout="wide")
    st.title("RAG Evaluation Pipeline Dashboard")


def get_available_experiments(base_path="/DSIA/nlp/experiments/results"):
    """Get all available experiment directories containing evaluation Excel files."""
    base_path = Path(base_path)

    if not base_path.exists():
        return [], {}, []

    available_experiments = []
    experiment_paths = {}

    # First get all major experiment categories (top-level directories)
    experiment_categories = []
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if this directory contains evaluation files or has subdirectories with them
            has_excel = list(item.glob("**/evaluation_*.xlsx"))
            if has_excel:
                experiment_categories.append(item.name)

    # Also check the base experiments directory itself
    excel_files = list(base_path.glob("evaluation_*.xlsx"))
    if excel_files:
        available_experiments.append("(Base Directory)")
        experiment_paths["(Base Directory)"] = str(base_path)

    # Check all direct subdirectories of the base path for evaluation files
    for item in base_path.iterdir():
        if item.is_dir():
            # Check if this directory contains any evaluation Excel files directly
            excel_files = list(item.glob("evaluation_*.xlsx"))
            if excel_files:
                name = f"({item.name} Base)"
                available_experiments.append(name)
                experiment_paths[name] = str(item)

            # Check subdirectories
            for subitem in item.iterdir():
                if subitem.is_dir():
                    excel_files = list(subitem.glob("evaluation_*.xlsx"))
                    if excel_files:
                        # Name format: subdir (category)
                        name = f"{subitem.name} ({item.name})"
                        available_experiments.append(name)
                        experiment_paths[name] = str(subitem)

    return (
        sorted(available_experiments),
        experiment_paths,
        sorted(experiment_categories),
    )


def handle_experiment_setup():
    """Handle experiment setup page with directory navigation."""
    st.header("Experiment Configuration")

    # Initialize experiments in session state if not already present
    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    # Get available experiments
    base_path = "/DSIA/nlp/experiments/results"
    available_experiments, experiment_paths, experiment_categories = (
        get_available_experiments(base_path)
    )

    st.info(
        """
    Configure the experiments you want to compare:
    1. Select experiment directories from the available options
    2. Give each experiment a meaningful name
    3. Use the move up/down buttons to reorder experiments
    """
    )

    if not available_experiments:
        st.warning(
            f"No experiment directories with evaluation files found in {base_path} or its subdirectories"
        )

        # Add option to check directly for files in current directory
        st.subheader("Available Excel Files")
        st.write(
            "The following evaluation files were found in the current working directory:"
        )

        # List Excel files directly
        excel_files = list(Path().glob("evaluation_*.xlsx"))
        if excel_files:
            for file in excel_files:
                st.text(f"- {file.name}")

            # Add a way to use these files directly
            if st.button("Use Current Directory Files"):
                st.session_state.experiments = [
                    {"dir": "", "name": "Current Directory"}
                ]
                st.rerun()
        else:
            st.error("No evaluation files found in the current directory either.")

        return

    # Add filter for experiment categories
    if experiment_categories:
        selected_category = st.selectbox(
            "Filter by Category",
            ["All Categories"] + experiment_categories,
            key="category_filter",
        )

        # Filter the available experiments based on the selected category
        if selected_category != "All Categories":
            filtered_experiments = [
                exp for exp in available_experiments if f"({selected_category})" in exp
            ]
            display_experiments = filtered_experiments
        else:
            display_experiments = available_experiments
    else:
        display_experiments = available_experiments

    # Display each experiment with reordering controls
    for i, exp in enumerate(st.session_state.experiments):
        with st.container():
            st.markdown(f"### Experiment {i+1}")

            # First row: Move up/down buttons
            move_cols = st.columns([0.5, 0.5, 4])

            with move_cols[0]:
                # Move up button (disabled for first item)
                if i > 0:
                    if st.button("⬆️", key=f"up_{i}", help="Move experiment up"):
                        # Swap with previous experiment
                        (
                            st.session_state.experiments[i],
                            st.session_state.experiments[i - 1],
                        ) = (
                            st.session_state.experiments[i - 1],
                            st.session_state.experiments[i],
                        )
                        st.rerun()
                else:
                    # Disabled button (placeholder to maintain layout)
                    st.empty()

            with move_cols[1]:
                # Move down button (disabled for last item)
                if i < len(st.session_state.experiments) - 1:
                    if st.button("⬇️", key=f"down_{i}", help="Move experiment down"):
                        # Swap with next experiment
                        (
                            st.session_state.experiments[i],
                            st.session_state.experiments[i + 1],
                        ) = (
                            st.session_state.experiments[i + 1],
                            st.session_state.experiments[i],
                        )
                        st.rerun()
                else:
                    # Disabled button (placeholder to maintain layout)
                    st.empty()

            # Main experiment configuration row
            config_cols = st.columns([2, 2, 0.5])

            with config_cols[0]:
                # Get the current directory value
                current_dir = exp["dir"]

                # Find the correct index based on the current directory
                dir_index = 0  # Default to empty selection
                for idx, option in enumerate(display_experiments):
                    option_dir = ""
                    if option == "(Base Directory)":
                        option_dir = ""
                    elif option.startswith("(") and option.endswith(" Base)"):
                        option_dir = option[1:-6]  # Remove "(" and " Base)"
                    elif " (" in option and ")" in option:
                        parts = option.split(" (")
                        exp_name = parts[0]
                        category = parts[1][:-1]  # Remove the closing ")"
                        option_dir = f"{category}/{exp_name}"
                    else:
                        option_dir = option

                    if option_dir == current_dir:
                        dir_index = (
                            idx + 1
                        )  # +1 because we add an empty option at index 0
                        break

                selected_exp = st.selectbox(
                    "📁 Directory",
                    [""] + display_experiments,
                    index=dir_index,
                    key=f"dir_{i}",
                )

                # Update the experiment directory based on the selection
                new_dir = ""
                if selected_exp == "(Base Directory)":
                    new_dir = ""
                elif selected_exp.startswith("(") and selected_exp.endswith(" Base)"):
                    # Extract the category name and set as dir
                    category = selected_exp[1:-6]  # Remove "(" and " Base)"
                    new_dir = category
                elif selected_exp:
                    # For experiment with category format: "name (category)"
                    if " (" in selected_exp and ")" in selected_exp:
                        parts = selected_exp.split(" (")
                        exp_name = parts[0]
                        category = parts[1][:-1]  # Remove the closing ")"
                        new_dir = f"{category}/{exp_name}"
                    else:
                        new_dir = selected_exp

                # Only update if the directory has changed
                if new_dir != current_dir:
                    exp["dir"] = new_dir

                # Show the full path for clarity
                if selected_exp and selected_exp in experiment_paths:
                    st.caption(f"Full path: {experiment_paths[selected_exp]}")

            with config_cols[1]:
                exp["name"] = st.text_input(
                    "📝 Name", value=exp["name"], key=f"name_{i}"
                )

            with config_cols[2]:
                if st.button("🗑️", key=f"remove_{i}", help="Remove this experiment"):
                    st.session_state.experiments.pop(i)
                    st.rerun()

            # Preview available files in the selected directory
            if selected_exp and selected_exp in experiment_paths:
                preview_path = Path(experiment_paths[selected_exp])

                excel_files = list(preview_path.glob("evaluation_*.xlsx"))
                if excel_files:
                    with st.expander("Preview Available Files", expanded=False):
                        for file in excel_files:
                            st.text(f"- {file.name}")
                else:
                    st.warning("No evaluation files found in this directory")

            # Insert button
            if st.button(
                "➕ Insert Experiment",
                key=f"insert_{i}",
                help="Insert experiment below",
            ):
                st.session_state.experiments.insert(
                    i + 1,
                    {
                        "dir": "",
                        "name": f"Experiment {len(st.session_state.experiments) + 1}",
                    },
                )
                st.rerun()

            st.divider()

    # Add a final "Add Experiment" button at the bottom for convenience
    if st.button("➕ Add Experiment", key="add_exp_bottom", use_container_width=True):
        st.session_state.experiments.append(
            {"dir": "", "name": f"Experiment {len(st.session_state.experiments) + 1}"}
        )
        st.rerun()


# Helper functions moved to module level
def normalize_score(score):
    """Normalize scores for comparison (handles float vs int equality)."""
    if pd.isna(score):
        return None
    return float(score)  # Convert both ints and floats to float for comparison


def is_good_score(score):
    """Check if a score is good (4-5) or bad (1-3)."""
    if pd.isna(score):
        return None
    return float(score) >= 4


def highlight_differences(
    row, display_df, comparison_df, selected_exps, filtered_score_columns
):
    """Highlight differences between experiment scores."""
    # Start with default styling
    styles = [""] * len(display_df.columns)

    # Skip styling for the question column
    styles[0] = ""  # First column is always 'question'

    # Group columns by metric
    for score_col in filtered_score_columns:
        # Get all columns for this metric
        exp_cols = []
        for exp_name in selected_exps:
            col_name = f"{score_col} ({exp_name})"
            if col_name in display_df.columns:
                exp_cols.append((col_name, exp_name))

        if len(exp_cols) > 1:
            # Get values and performance categories
            norm_values = []
            perf_values = []

            for col_name, exp_name in exp_cols:
                val = row[col_name]
                if pd.notna(val):
                    norm_values.append(normalize_score(val))

                    # Get performance classification for this score
                    perf_col = f"{score_col}_perf_{exp_name}"
                    if perf_col in comparison_df.columns:
                        perf_val = comparison_df.loc[row.name, perf_col]
                        if pd.notna(perf_val):
                            perf_values.append(perf_val)

            # Check if values differ after normalization
            if len(norm_values) > 1 and len(set(norm_values)) > 1:
                # Apply styling to these columns
                for col_name, _ in exp_cols:
                    col_idx = display_df.columns.get_loc(col_name)

                    # Check if performance categories differ
                    if len(perf_values) > 1 and len(set(perf_values)) > 1:
                        # Red for performance category changes
                        styles[col_idx] = (
                            "background-color: #ff9999; font-weight: bold;"
                        )
                    else:
                        # Light pink for score differences without category change
                        styles[col_idx] = "background-color: #ffcccb;"

    return styles


def calculate_metric_improvements(comparison_df, selected_exps, filtered_score_columns):
    """Calculate improvements between experiments for each metric."""
    improvements = {}

    # Initialize counters for each metric
    for score_col in filtered_score_columns:
        metric_name = score_col.replace("_score", "").title()
        improvements[metric_name] = {
            "improved": 0,
            "worsened": 0,
            "unchanged": 0,
            "performance_improved": 0,
            "performance_worsened": 0,
        }

    # Compare first experiment to each other experiment
    base_exp = selected_exps[0]
    for comp_exp in selected_exps[1:]:
        for score_col in filtered_score_columns:
            metric_name = score_col.replace("_score", "").title()
            base_col = f"{score_col} ({base_exp})"
            comp_col = f"{score_col} ({comp_exp})"

            # Get performance columns
            base_perf_col = f"{score_col}_perf_{base_exp}"
            comp_perf_col = f"{score_col}_perf_{comp_exp}"

            # Compare rows where both values exist
            valid_rows = comparison_df[
                comparison_df[base_col].notna() & comparison_df[comp_col].notna()
            ]

            for _, row in valid_rows.iterrows():
                base_score = normalize_score(row[base_col])
                comp_score = normalize_score(row[comp_col])

                if base_score is not None and comp_score is not None:
                    if comp_score > base_score:
                        improvements[metric_name]["improved"] += 1
                    elif comp_score < base_score:
                        improvements[metric_name]["worsened"] += 1
                    else:
                        improvements[metric_name]["unchanged"] += 1

                    # Check for performance category changes
                    if (
                        base_perf_col in comparison_df.columns
                        and comp_perf_col in comparison_df.columns
                    ):
                        base_perf = (
                            row[base_perf_col]
                            if base_perf_col in row and pd.notna(row[base_perf_col])
                            else None
                        )
                        comp_perf = (
                            row[comp_perf_col]
                            if comp_perf_col in row and pd.notna(row[comp_perf_col])
                            else None
                        )

                        if base_perf == "Bad" and comp_perf == "Good":
                            improvements[metric_name]["performance_improved"] += 1
                        elif base_perf == "Good" and comp_perf == "Bad":
                            improvements[metric_name]["performance_worsened"] += 1

    return improvements


def handle_raw_data_page(experiments_data):
    """Enhanced Raw Data page with row-by-row comparison functionality, metric filtering, and judge justifications."""
    st.header("Raw Data")

    # Create tabs for different data views
    tab1, tab2 = st.tabs(["🔍 Row Comparison", "⏱️ Timing Data"])

    # Tab 1: Row Comparison View
    with tab1:
        st.subheader("Row-by-Row Experiment Comparison")

        # Get all judges from all experiments
        all_judges = set()
        for exp in experiments_data:
            all_judges.update(exp["dfs"].keys())

        # Select a judge for comparison
        selected_judge = st.selectbox(
            "Select Judge for Comparison",
            sorted(all_judges),
            key="row_compare_judge_selector",
        )

        # Get experiments that have data for this judge
        valid_experiments = [
            exp["name"] for exp in experiments_data if selected_judge in exp["dfs"]
        ]

        if not valid_experiments:
            st.warning(f"No experiments found with data from {selected_judge}")
        else:
            # Select experiments for comparison (multi-select)
            selected_exps = st.multiselect(
                "Select Experiments to Compare",
                valid_experiments,
                default=valid_experiments[: min(2, len(valid_experiments))],
                key="row_compare_exp_selector",
            )

            if len(selected_exps) < 2:
                st.info("Please select at least 2 experiments to compare")
            else:
                # Collect dataframes for selected experiments
                exp_dfs = {}
                for exp_name in selected_exps:
                    exp_data = next(
                        exp for exp in experiments_data if exp["name"] == exp_name
                    )
                    if selected_judge in exp_data["dfs"]:
                        exp_dfs[exp_name] = exp_data["dfs"][selected_judge]

                # Get common questions across all selected experiments
                common_questions = set(exp_dfs[selected_exps[0]]["question"])
                for exp_name in selected_exps[1:]:
                    common_questions &= set(exp_dfs[exp_name]["question"])

                if not common_questions:
                    st.warning(
                        "No common questions found across the selected experiments"
                    )
                else:
                    # Get all score columns from first experiment
                    score_columns = [
                        col
                        for col in exp_dfs[selected_exps[0]].columns
                        if col.endswith("_score")
                    ]

                    # Create metrics list with friendly names and get justification columns
                    metrics = ["All Metrics"] + [
                        col.replace("_score", "").title() for col in score_columns
                    ]
                    justification_columns = [
                        col.replace("_score", "") for col in score_columns
                    ]

                    # Add metric filter
                    col1, col2, col3 = st.columns([1, 1.5, 1])
                    with col1:
                        selected_metric = st.selectbox(
                            "Filter by Metric", metrics, key="metric_filter_selector"
                        )

                    with col2:
                        # Option to filter by score differences or view all rows
                        comparison_mode = st.radio(
                            "View Mode",
                            [
                                "Show All Questions",
                                "Show Only Differences",
                                "Show Performance Changes",
                            ],
                            key="comparison_mode",
                            horizontal=True,
                        )

                    with col3:
                        # Add option to show significant differences (performance changes)
                        if comparison_mode == "Show Performance Changes":
                            st.info(
                                "Showing rows where scores cross the performance threshold (3→4 or 4→3)"
                            )

                    # Get filtered score columns based on selected metric
                    filtered_score_columns = score_columns
                    if selected_metric != "All Metrics":
                        metric_name = selected_metric.lower()
                        filtered_score_columns = [
                            col for col in score_columns if col.startswith(metric_name)
                        ]

                    # Create comparison dataframe
                    comparison_data = []

                    for question in sorted(common_questions):
                        # Get rows for this question from each experiment
                        question_rows = {}
                        for exp_name in selected_exps:
                            df = exp_dfs[exp_name]
                            question_rows[exp_name] = df[
                                df["question"] == question
                            ].iloc[0]

                        # Check if there are differences in scores for the selected metric
                        has_differences = False
                        has_performance_change = False

                        for score_col in filtered_score_columns:
                            if score_col in question_rows[selected_exps[0]]:
                                # Get normalized scores for accurate comparison
                                base_score = normalize_score(
                                    question_rows[selected_exps[0]][score_col]
                                )
                                base_is_good = (
                                    is_good_score(base_score)
                                    if base_score is not None
                                    else None
                                )

                                for exp_name in selected_exps[1:]:
                                    if score_col in question_rows[exp_name]:
                                        comp_score = normalize_score(
                                            question_rows[exp_name][score_col]
                                        )
                                        comp_is_good = (
                                            is_good_score(comp_score)
                                            if comp_score is not None
                                            else None
                                        )

                                        # Check for numeric differences
                                        if (
                                            base_score is not None
                                            and comp_score is not None
                                        ):
                                            # Check if scores are different (after normalization)
                                            if (
                                                abs(base_score - comp_score) > 0.001
                                            ):  # Use small epsilon for float comparison
                                                has_differences = True

                                                # Check if performance category changed (bad<->good)
                                                if (
                                                    base_is_good is not None
                                                    and comp_is_good is not None
                                                ):
                                                    if base_is_good != comp_is_good:
                                                        has_performance_change = True

                        # Skip based on selected filter mode
                        if (
                            comparison_mode == "Show Only Differences"
                            and not has_differences
                        ):
                            continue
                        elif (
                            comparison_mode == "Show Performance Changes"
                            and not has_performance_change
                        ):
                            continue

                        # Create row for comparison table
                        row = {"question": question}

                        # Add score columns for each experiment (filtered by metric if selected)
                        for exp_name in selected_exps:
                            for score_col in filtered_score_columns:
                                if score_col in question_rows[exp_name]:
                                    # Store original score
                                    row[f"{score_col} ({exp_name})"] = question_rows[
                                        exp_name
                                    ][score_col]

                                    # Add performance classification (store separately for styling)
                                    score = normalize_score(
                                        question_rows[exp_name][score_col]
                                    )
                                    if score is not None:
                                        is_good = is_good_score(score)
                                        # Store performance classification as a separate column that won't be displayed
                                        row[f"{score_col}_perf_{exp_name}"] = (
                                            "Good" if is_good else "Bad"
                                        )

                        comparison_data.append(row)

                    # Create and display the comparison dataframe
                    if comparison_data:
                        # Add metrics counter and summary
                        total_questions = len(common_questions)
                        diff_questions = len(comparison_data)
                        diff_percent = (
                            (diff_questions / total_questions) * 100
                            if total_questions > 0
                            else 0
                        )

                        comparison_df = pd.DataFrame(comparison_data)

                        # Calculate metric improvements between experiments
                        improvements = calculate_metric_improvements(
                            comparison_df, selected_exps, filtered_score_columns
                        )

                        # Show metric improvements summary
                        st.subheader("Metric Improvements Summary")

                        # Create comparison basis text
                        if len(selected_exps) > 1:
                            comparison_basis = f"Comparing {selected_exps[0]} (baseline) to {selected_exps[1]}"
                            if len(selected_exps) > 2:
                                comparison_basis += (
                                    f" and {len(selected_exps)-2} more experiment(s)"
                                )

                        st.markdown(f"**{comparison_basis}**")

                        # Create improvement summary table
                        improvement_data = []
                        for metric, stats in improvements.items():
                            improvement_data.append(
                                {
                                    "Metric": metric,
                                    "Improved": stats["improved"],
                                    "Worsened": stats["worsened"],
                                    "Unchanged": stats["unchanged"],
                                    "Performance Improved": stats[
                                        "performance_improved"
                                    ],
                                    "Performance Worsened": stats[
                                        "performance_worsened"
                                    ],
                                }
                            )

                        # Display improvements table
                        if improvement_data:
                            improvement_df = pd.DataFrame(improvement_data)
                            st.dataframe(improvement_df, use_container_width=True)

                        # Display filter info
                        if comparison_mode == "Show Only Differences":
                            st.info(
                                f"Showing {diff_questions} questions with differences out of {total_questions} total "
                                f"({diff_percent:.1f}% have differences in "
                                f"{'the selected metric' if selected_metric != 'All Metrics' else 'at least one metric'})"
                            )
                        elif comparison_mode == "Show Performance Changes":
                            st.info(
                                f"Showing {diff_questions} questions with performance changes (crossing the good/bad threshold) "
                                f"out of {total_questions} total ({diff_percent:.1f}%)"
                            )

                        # Create a simplified DataFrame for display (excluding the _perf columns)
                        display_cols = ["question"]
                        for score_col in filtered_score_columns:
                            for exp_name in selected_exps:
                                col_name = f"{score_col} ({exp_name})"
                                if col_name in comparison_df.columns:
                                    display_cols.append(col_name)

                        # Create the display DataFrame (ensure all columns exist)
                        display_df = comparison_df[display_cols].copy()

                        # Apply styling and display
                        try:
                            # Use axis=1 to apply styling row by row
                            styled_df = display_df.style.apply(
                                lambda row: highlight_differences(
                                    row,
                                    display_df,
                                    comparison_df,
                                    selected_exps,
                                    filtered_score_columns,
                                ),
                                axis=1,
                            )
                            st.dataframe(styled_df, use_container_width=True)
                        except Exception as e:
                            # Fallback to unstyled display if styling fails
                            st.error(f"Error applying styling: {str(e)}")
                            st.dataframe(display_df, use_container_width=True)

                        # Add export to CSV option
                        csv = display_df.to_csv(index=False)
                        metric_name = selected_metric.replace(" ", "_").lower()
                        export_mode = comparison_mode.replace(" ", "_").lower()
                        st.download_button(
                            label="Download Comparison as CSV",
                            data=csv,
                            file_name=f"{selected_judge}_{metric_name}_{export_mode}.csv",
                            mime="text/csv",
                        )

                        # Add option to view detailed comparison for a specific question
                        st.subheader("Detailed Question Comparison")

                        # Get the filtered questions based on current settings
                        filtered_questions = comparison_df["question"].tolist()

                        if filtered_questions:
                            # Select a question to examine in detail (only from filtered questions)
                            selected_question = st.selectbox(
                                "Select Question to Examine",
                                filtered_questions,
                                key="detailed_question_selector",
                            )

                            # Show detailed view of the selected question across experiments
                            st.markdown("##### Question")
                            st.markdown(f"**{selected_question}**")

                            # Create columns for each experiment
                            exp_cols = st.columns(len(selected_exps))

                            for i, exp_name in enumerate(selected_exps):
                                with exp_cols[i]:
                                    st.markdown(f"##### {exp_name}")
                                    df = exp_dfs[exp_name]
                                    question_row = df[
                                        df["question"] == selected_question
                                    ].iloc[0]

                                    # Display scores with colored backgrounds based on value and add justifications
                                    # Use filtered metrics if a specific metric is selected
                                    display_score_cols = (
                                        filtered_score_columns
                                        if selected_metric != "All Metrics"
                                        else score_columns
                                    )

                                    for score_col in display_score_cols:
                                        if score_col in question_row and pd.notna(
                                            question_row[score_col]
                                        ):
                                            score = normalize_score(
                                                question_row[score_col]
                                            )
                                            metric_name = score_col.replace(
                                                "_score", ""
                                            ).title()
                                            justification_col = score_col.replace(
                                                "_score", ""
                                            )

                                            # Determine color based on score (1-5 scale)
                                            if score is not None:
                                                if score >= 4:  # Good scores (4-5)
                                                    color = "#c6efce"  # Light green
                                                    performance = "✅ Good"
                                                else:  # Poor scores (1-3)
                                                    color = "#ffc7ce"  # Light red
                                                    performance = "❌ Bad"

                                                # Display the score with performance indicator
                                                st.markdown(
                                                    f"<div style='background-color: {color}; padding: 5px; border-radius: 5px; margin-bottom: 5px;'>"
                                                    f"<b>{metric_name}:</b> {question_row[score_col]} ({performance})"
                                                    f"</div>",
                                                    unsafe_allow_html=True,
                                                )

                                                # Add justification if available
                                                if (
                                                    justification_col in question_row
                                                    and pd.notna(
                                                        question_row[justification_col]
                                                    )
                                                ):
                                                    with st.expander(
                                                        f"{metric_name} Justification",
                                                        expanded=True,
                                                    ):
                                                        st.markdown(
                                                            f"_{str(question_row[justification_col])}_"
                                                        )

                                    # Add expandable sections for answer, context, extracts and source docs
                                    st.markdown("---")

                                    # Show answers if available
                                    if "answer" in question_row and pd.notna(
                                        question_row["answer"]
                                    ):
                                        with st.expander("📝 Answer", expanded=False):
                                            st.markdown(str(question_row["answer"]))

                                    # Show relevant extracts if available (separate from context)
                                    if (
                                        "rag_relevant_extracts" in question_row
                                        and pd.notna(
                                            question_row["rag_relevant_extracts"]
                                        )
                                    ):
                                        with st.expander(
                                            "📑 Relevant Extracts", expanded=False
                                        ):
                                            st.markdown(
                                                str(
                                                    question_row[
                                                        "rag_relevant_extracts"
                                                    ]
                                                )
                                            )

                                    # Show context if available
                                    if "context" in question_row and pd.notna(
                                        question_row["context"]
                                    ):
                                        with st.expander("📄 Context", expanded=False):
                                            st.markdown(str(question_row["context"]))

                                    # Alternative name for relevant extracts
                                    if (
                                        "relevant_extracts" in question_row
                                        and pd.notna(question_row["relevant_extracts"])
                                    ):
                                        with st.expander(
                                            "� Relevant Extracts", expanded=False
                                        ):
                                            st.markdown(
                                                str(question_row["relevant_extracts"])
                                            )

                                    # Show source documents if available
                                    if "source_doc" in question_row and pd.notna(
                                        question_row["source_doc"]
                                    ):
                                        with st.expander(
                                            "� Source Documents", expanded=False
                                        ):
                                            st.markdown(str(question_row["source_doc"]))
                        else:
                            st.info("No questions match the current filter criteria")
                    else:
                        st.info("No rows match the current filter criteria")

    # Tab 2: Timing Data
    with tab2:
        for exp in experiments_data:
            st.subheader(f"{exp['name']}")
            if exp["timing"] is not None:
                st.dataframe(exp["timing"], use_container_width=True)
            else:
                st.info(f"No timing data available for {exp['name']}")


def display_metric_descriptions():
    """Display metric descriptions in an expandable section."""
    with st.expander("ℹ️ Metric Descriptions", expanded=False):
        for metric, description in METRIC_DESCRIPTIONS.items():
            st.markdown(f"**{metric.title()}**: {description}")


def handle_pdf_export(experiments_data):
    """Handle PDF report generation."""
    st.header("PDF Report Generation")

    st.info(
        """
    Generate a PDF report of your experiment results with structured description sections and formatting.
    The report will include:
    - Experiment configuration information
    - Performance overview with summary tables
    - Judge-specific analysis tables
    - Timing analysis tables
    
    The Raw Data section will not be included in the report.
    """
    )

    # Initialize session state variables if they don't exist
    if "report_title" not in st.session_state:
        st.session_state.report_title = "RAG Evaluation Report"

    if "report_author" not in st.session_state:
        st.session_state.report_author = ""

    # Default template
    default_template = """## Experiment Objectives
This experiment aims to evaluate the performance of different RAG configurations...

## Key Findings
- The highest faithfulness score was achieved by...
- Response quality improved when...

## Conclusion
Based on the results, we found that...

## Limitations
Some limitations of this approach include..."""

    # Initialize description with template if empty
    if "report_description" not in st.session_state:
        st.session_state.report_description = default_template

    # Initialize filename if not present
    if "filename" not in st.session_state:
        st.session_state.filename = "rag_evaluation_report.pdf"

    # Add a title field (using lambda for callback)
    st.text_input(
        "Report Title",
        value=st.session_state.report_title,
        key="title_input",
        on_change=lambda: setattr(
            st.session_state, "report_title", st.session_state.title_input
        ),
        help="Custom title for your PDF report",
    )

    # Add author field (using lambda for callback)
    st.text_input(
        "Author",
        value=st.session_state.report_author,
        key="author_input",
        on_change=lambda: setattr(
            st.session_state, "report_author", st.session_state.author_input
        ),
        placeholder="Enter your name or organization",
        help="Author name to be displayed in the report",
    )

    # Add description guidelines
    st.subheader("Report Description")
    st.markdown(
        """
    You can use markdown formatting in your description:
    - **Bold text**: Use `**bold**` → **bold**
    - *Italicized text*: Use `*italics*` → *italicized*
    - **Headings**:
    - `# Main Section` → **Large Title**
    - `## Subsection` → **Medium Title**
    - `### Sub-subsection` → **Smaller Title**
    - **Bullet points**: Use `- item`
    - **Numbered lists**: Use `1. item`
    - **Blockquotes**: Use `> Quote` → *"Formatted Quote"*
    - **Links**: `[Text](URL)` → [Example Link](https://example.com)
    """
    )

    # Add a description field with session state persistence (using lambda for callback)
    st.text_area(
        "Description",
        value=st.session_state.report_description,
        key="description_input",
        on_change=lambda: setattr(
            st.session_state, "report_description", st.session_state.description_input
        ),
        height=300,
        help="This description will appear at the beginning of the PDF report with proper formatting.",
    )

    # Preview formatted description in an expander
    with st.expander("Preview Formatted Description", expanded=True):
        st.markdown(st.session_state.report_description)

    # PDF generation options
    col1, col2 = st.columns(2)
    with col1:
        include_tables = st.checkbox(
            "Include tables",
            value=True,
            help="Include all performance and timing tables in the report",
        )

    with col2:
        st.text_input(
            "Filename",
            value=st.session_state.filename,
            key="filename_input",
            on_change=lambda: setattr(
                st.session_state, "filename", st.session_state.filename_input
            ),
            help="Name of the PDF file that will be downloaded",
        )

    if st.button("Generate PDF Report", type="primary"):
        if not experiments_data:
            st.error(
                "No experiment data available. Please configure and run experiments first."
            )
            return

        # Create spinner while generating PDF
        with st.spinner("Generating PDF report..."):
            # Get experiment configuration from session state
            experiment_configs = (
                st.session_state.experiments
                if "experiments" in st.session_state
                else None
            )

            # Generate the PDF
            pdf_bytes = create_pdf_report(
                experiments_data,
                experiment_configs,
                st.session_state.report_description,
                include_tables,
                st.session_state.report_title,
                st.session_state.report_author,
            )

            # Provide download link
            st.success("PDF report generated successfully! Click below to download.")

            # Add some CSS to style the download button
            st.markdown(
                """
            <style>
            .download-button {
                display: inline-block;
                padding: 0.5em 1em;
                margin: 1em 0;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                text-align: center;
                font-weight: bold;
            }
            .download-button:hover {
                background-color: #45a049;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                get_pdf_download_link(pdf_bytes, st.session_state.filename),
                unsafe_allow_html=True,
            )


def main():
    setup_page()

    # Initialize session state with an empty list
    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    # Sidebar navigation
    page = st.sidebar.radio(
        "Select a Page",
        [
            "Experiment Setup",
            "Performance Overview",
            "Timing Analysis",
            "PDF Export",
            "Raw Data",
        ],
    )

    if page == "Experiment Setup":
        handle_experiment_setup()
        return

    # Check if experiments are configured
    if len(st.session_state.experiments) == 0:
        st.info(
            "No experiments are configured. Please go to the Experiment Setup page to add experiments."
        )
        return

    # Load experiments data
    experiments_data = []
    has_invalid_paths = False

    for exp in st.session_state.experiments:
        data = load_evaluation_files(exp["dir"])
        if data is not None:
            experiments_data.append(
                {
                    "name": exp["name"],
                    "dfs": data[0],
                    "combined": data[1],
                    "timing": data[2],
                }
            )
        else:
            has_invalid_paths = True

    # Handle various error states
    if not st.session_state.experiments:
        st.error("No experiments configured. Please add experiments in the Setup page.")
        return
    elif has_invalid_paths:
        st.error(
            "Some experiment paths are invalid or empty. Please check the configuration."
        )
        return
    elif not experiments_data:
        st.error("No valid evaluation files found in the specified directories.")
        return

    if page == "Performance Overview":
        st.header("Performance Overview")

        # Display metric descriptions at the top of the performance page
        display_metric_descriptions()

        # Get all judges
        all_judges = set()
        for exp in experiments_data:
            all_judges.update(exp["dfs"].keys())

        # Create tabs for Summary and Individual Judges
        tab_summary, *judge_tabs = st.tabs(["Summary"] + list(sorted(all_judges)))

        with tab_summary:
            st.subheader("Evaluation Summary")
            st.caption(
                "Average Performance percentages (Judgements of 4-5) across all LLM judges"
            )

            # Generate summary metrics
            summary_dfs, summary_figs, overall_df, overall_fig = create_metrics_summary(
                experiments_data
            )

            # Display overall summary
            st.plotly_chart(
                overall_fig,
                use_container_width=True,
                key=f"overall_summary_chart_{str(uuid.uuid4())}",
            )

            # Display summary table and radar plot side-by-side
            st.subheader("Performance Summary")
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("##### Metric Averages Table")
                formatted_df = overall_df.copy()

                # Get all metric columns (excluding special columns)
                metric_columns = [
                    col
                    for col in formatted_df.columns
                    if col != "Experiment"
                    and col != "Number of Judges"
                    and not col.endswith("_best_count")
                    and not col.endswith("_ci_lower")
                    and not col.endswith("_ci_upper")
                ]

                # Format each column with highest value in bold and add stars for judge agreement
                for col in metric_columns:
                    # Find max value in this column
                    max_val = formatted_df[col].max()

                    # Add stars based on how many judges rated this experiment as best for this metric
                    best_count_col = f"{col}_best_count"
                    if best_count_col in formatted_df.columns:
                        formatted_df[f"{col}_display"] = formatted_df.apply(
                            lambda row: (
                                f"**{row[col]:.1f}%** {'*' * row[best_count_col]}"
                                if row[col] == max_val
                                else (
                                    f"{row[col]:.1f}% {'*' * row[best_count_col]}"
                                    if row[best_count_col] > 0
                                    else f"{row[col]:.1f}%"
                                )
                            ),
                            axis=1,
                        )
                    else:
                        # If best_count_col doesn't exist, just format without stars
                        formatted_df[f"{col}_display"] = formatted_df.apply(
                            lambda row: (
                                f"**{row[col]:.1f}%**"
                                if row[col] == max_val
                                else f"{row[col]:.1f}%"
                            ),
                            axis=1,
                        )

                # MAIN TABLE - Create a new DataFrame with just the display columns
                display_df = pd.DataFrame()
                display_df["Experiment"] = formatted_df["Experiment"]

                # Add just the metric columns (no CI)
                for metric in sorted(metric_columns):
                    if f"{metric}_display" in formatted_df.columns:
                        display_df[metric] = formatted_df[f"{metric}_display"]

                # Add Number of Judges column
                display_df["Number of Judges"] = formatted_df["Number of Judges"]

                # Use st.markdown to render the bold formatting
                st.markdown(display_df.to_markdown(index=False), unsafe_allow_html=True)

                # Add a note explaining the stars
                st.caption(
                    "Note: Stars (*) indicate how many judges rated this experiment as the best for that metric. "
                    "Bold values indicate the highest score in each column."
                )

                # CONFIDENCE INTERVAL TABLE - Create a separate table just for CIs
                st.markdown("##### 95% Confidence Intervals")

                ci_df = pd.DataFrame()
                ci_df["Experiment"] = formatted_df["Experiment"]

                # Add CI columns for all metrics
                for metric in sorted(metric_columns):
                    # Add CI column if available
                    ci_lower_col = f"{metric}_ci_lower"
                    ci_upper_col = f"{metric}_ci_upper"
                    if (
                        ci_lower_col in formatted_df.columns
                        and ci_upper_col in formatted_df.columns
                    ):
                        # Find row with highest upper bound for this metric's CI
                        max_upper_bound = formatted_df[ci_upper_col].max()

                        # Format CI column with highest value in bold
                        ci_df[f"{metric.title()}"] = formatted_df.apply(
                            lambda row: (
                                f"**({row[ci_lower_col]:.1f}% - {row[ci_upper_col]:.1f}%)**"
                                if row[ci_upper_col] == max_upper_bound
                                else f"({row[ci_lower_col]:.1f}% - {row[ci_upper_col]:.1f}%)"
                            ),
                            axis=1,
                        )

                # Use st.markdown to render the bold formatting for CI table
                st.markdown(ci_df.to_markdown(index=False), unsafe_allow_html=True)

                # Add a note explaining the CI table
                st.caption(
                    "Note: This table shows the 95% confidence intervals for each metric. "
                    "Bold values indicate the highest upper bound in each column."
                )

            with col2:
                # Create and display the radar plot
                avg_radar_fig = create_average_radar_plot(experiments_data)
                st.plotly_chart(
                    avg_radar_fig,
                    use_container_width=True,
                    key=f"avg_radar_chart_{str(uuid.uuid4())}",
                )

            # Display individual metric summaries in expandable sections
            st.subheader("Individual Metric Summaries")
            for metric in sorted(summary_figs.keys()):
                with st.expander(f"{metric.title()} Metric Details", expanded=False):
                    st.plotly_chart(
                        summary_figs[metric],
                        use_container_width=True,
                        key=f"summary_{metric}_chart_{str(uuid.uuid4())}",
                    )

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        # Create display DataFrame with bold formatting
                        display_df = pd.DataFrame()
                        display_df["Experiment"] = summary_dfs[metric]["Experiment"]
                        display_df["Average Performance %"] = summary_dfs[metric][
                            "Display Score"
                        ]
                        display_df["Judges Count"] = summary_dfs[metric]["Judges Count"]

                        # Use st.markdown to render the bold formatting
                        st.markdown(
                            display_df.to_markdown(index=False), unsafe_allow_html=True
                        )
                    with col2:
                        if metric in METRIC_DESCRIPTIONS:
                            st.info(METRIC_DESCRIPTIONS[metric])

        # Create individual judge tabs
        for judge_tab, judge_name in zip(judge_tabs, sorted(all_judges)):
            with judge_tab:
                st.subheader(f"Analysis by {judge_name}")

                # Filter data for this judge
                judge_data = []
                for exp in experiments_data:
                    if judge_name in exp["dfs"]:
                        judge_data.append(
                            {
                                "name": exp["name"],
                                "dfs": {judge_name: exp["dfs"][judge_name]},
                                "combined": exp["combined"],
                                "timing": exp["timing"],
                            }
                        )

                if judge_data:
                    # Create radar plot for this judge
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = create_radar_plot(judge_data)
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"{judge_name}_radar_chart_{str(uuid.uuid4())}",
                        )

                    # Add experiment comparison
                    st.subheader("Experiment Comparison")
                    metrics = [
                        col.replace("_score", "")
                        for col in judge_data[0]["dfs"][judge_name].columns
                        if col.endswith("_score")
                    ]

                    # Add a brief explanation of the metrics expandable sections
                    st.info(
                        "Click on each metric below to see detailed comparison across experiments"
                    )

                    for metric in sorted(metrics):
                        # Use markdown with HTML for larger, more prominent title in the expander
                        with st.expander(
                            f"### {metric.title()} Metric", expanded=False
                        ):
                            if metric in METRIC_DESCRIPTIONS:
                                st.info(METRIC_DESCRIPTIONS[metric])

                            # Prepare data for both table and plot
                            metric_data = []
                            plot_data = []
                            for exp_data in judge_data:
                                df = exp_data["dfs"][judge_name]
                                score_col = f"{metric}_score"
                                if score_col in df.columns:
                                    good_score_pct = calculate_good_score_percentage(
                                        df[score_col]
                                    )
                                    metric_data.append(
                                        {
                                            "Experiment": exp_data["name"],
                                            "Performance %": f"{good_score_pct:.1f}%",
                                        }
                                    )
                                    # Add all scores for the plot
                                    scores = df[score_col].value_counts().sort_index()
                                    for score, count in scores.items():
                                        plot_data.append(
                                            {
                                                "Experiment": exp_data["name"],
                                                "Score": score,
                                                "Count": count,
                                                "Percentage": (
                                                    count / len(df[score_col])
                                                )
                                                * 100,
                                            }
                                        )

                            # Create and display table with Performance plot side by side
                            metric_df = pd.DataFrame(metric_data)

                            # Format with bold for highest score
                            display_df = pd.DataFrame()
                            display_df["Experiment"] = metric_df["Experiment"]

                            # Get the highest score percentage
                            max_score = max(
                                [
                                    float(score.rstrip("%"))
                                    for score in metric_df["Performance %"]
                                ]
                            )

                            # Create formatted column with bold for maximum values
                            display_df["Performance %"] = metric_df[
                                "Performance %"
                            ].apply(
                                lambda x: (
                                    f"**{x}**"
                                    if float(x.rstrip("%")) == max_score
                                    else x
                                )
                            )

                            # Use H3 tag for consistent heading style - same size as the main metric title
                            st.markdown(f"##### {metric.title()} Metric Analysis")
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                # Use st.markdown to render the bold formatting
                                st.markdown(
                                    display_df.to_markdown(index=False),
                                    unsafe_allow_html=True,
                                )

                            with col2:
                                # Create Performance percentage plot
                                good_score_fig = go.Figure()

                                # Extract experiment names and good judgment percentages
                                experiments = [row["Experiment"] for row in metric_data]
                                good_judgments = [
                                    float(row["Performance %"].rstrip("%"))
                                    for row in metric_data
                                ]

                                good_score_fig.add_trace(
                                    go.Scatter(
                                        x=experiments,
                                        y=good_judgments,
                                        mode="lines+markers",
                                        name="Good Judgments %",
                                        line=dict(width=3),
                                        marker=dict(size=10),
                                        hovertemplate="Experiment: %{x}<br>Good Judgments: %{y:.1f}%<extra></extra>",
                                    )
                                )

                                good_score_fig.update_layout(
                                    yaxis_title="Good Judgments Percentage",
                                    yaxis_ticksuffix="%",
                                    yaxis_range=[0, 100],
                                    showlegend=False,
                                    height=200,
                                    margin=dict(t=0, b=0, l=0, r=0),
                                )
                                st.plotly_chart(
                                    good_score_fig,
                                    use_container_width=True,
                                    key=f"{judge_name}_{metric}_good_score_chart_{str(uuid.uuid4())}",
                                )

                            # Create and display score distribution plot
                            # Use H3 tag for consistent heading size
                            st.markdown(f"##### {metric.title()} Score Distribution")
                            plot_df = pd.DataFrame(plot_data)
                            if not plot_df.empty:
                                fig = go.Figure()

                                # Add traces for each experiment
                                for exp_name in plot_df["Experiment"].unique():
                                    exp_data = plot_df[
                                        plot_df["Experiment"] == exp_name
                                    ]
                                    fig.add_trace(
                                        go.Scatter(
                                            x=exp_data["Score"],
                                            y=exp_data["Percentage"],
                                            mode="lines+markers",
                                            name=exp_name,
                                            hovertemplate="Score: %{x}<br>Percentage: %{y:.1f}%<extra></extra>",
                                        )
                                    )

                                fig.update_layout(
                                    xaxis_title="Judgment (1-5)",
                                    yaxis_title="Percentage of Responses",
                                    yaxis_ticksuffix="%",
                                    xaxis=dict(
                                        tickmode="linear",
                                        tick0=1,
                                        dtick=1,
                                        range=[0.5, 5.5],
                                    ),
                                    hovermode="x unified",
                                    showlegend=True,
                                    height=400,
                                    margin=dict(t=0, b=0, l=0, r=0),
                                )
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"{judge_name}_{metric}_distribution_chart_{str(uuid.uuid4())}",
                                )
                else:
                    st.warning(f"No data available for {judge_name}")

    elif page == "Timing Analysis":
        st.header("Timing Analysis")
        tab1, tab2 = st.tabs(["📊 Total Time", "🔄 Retriever Time"])

        with tab1:
            fig = create_timing_plot(
                experiments_data, RAG_QUERY_TIME_COLUMN, "Total Time Distribution"
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"total_time_chart_{str(uuid.uuid4())}",
            )

        with tab2:
            fig = create_timing_plot(
                experiments_data,
                RAG_RETRIEVER_TIME_COLUMN,
                "Retriever Time Distribution",
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"retriever_time_chart_{str(uuid.uuid4())}",
            )

    elif page == "PDF Export":
        handle_pdf_export(experiments_data)

    elif page == "Raw Data":
        handle_raw_data_page(experiments_data)


if __name__ == "__main__":
    main()
