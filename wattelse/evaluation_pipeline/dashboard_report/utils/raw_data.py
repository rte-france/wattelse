import streamlit as st
import pandas as pd


def highlight_differences(
    row, display_df, comparison_df, selected_exps, filtered_score_columns
):
    """Highlight differences between experiment scores, showing which experiment performs better."""
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
            exp_values = {}
            exp_perfs = {}

            for col_name, exp_name in exp_cols:
                val = row[col_name]
                if pd.notna(val):
                    # Normalize score - inline function
                    norm_val = float(val) if not pd.isna(val) else None
                    exp_values[exp_name] = norm_val

                    # Get performance classification for this score
                    perf_col = f"{score_col}_perf_{exp_name}"
                    if perf_col in comparison_df.columns:
                        perf_val = comparison_df.loc[row.name, perf_col]
                        if pd.notna(perf_val):
                            exp_perfs[exp_name] = perf_val

            # Check if values differ after normalization
            if len(exp_values) > 1 and len(set(exp_values.values())) > 1:
                # Apply styling to these columns based on which value is better
                for col_name, exp_name in exp_cols:
                    if exp_name in exp_values:
                        col_idx = display_df.columns.get_loc(col_name)

                        # Check if performance categories differ
                        if len(exp_perfs) > 1 and len(set(exp_perfs.values())) > 1:
                            # Performance categories differ (Good vs Bad)
                            if exp_perfs[exp_name] == "Good":
                                # Green for good performance
                                styles[col_idx] = (
                                    "background-color: #c6efce; font-weight: bold;"
                                )
                            else:
                                # Red for bad performance
                                styles[col_idx] = (
                                    "background-color: #ffc7ce; font-weight: bold;"
                                )
                        else:
                            # Performance categories are the same, but scores differ
                            # Find the max value (higher is better)
                            max_val = max(exp_values.values())
                            if exp_values[exp_name] == max_val:
                                # Light green for better score
                                styles[col_idx] = "background-color: #e6ffe6;"
                            else:
                                # Light red for worse score
                                styles[col_idx] = "background-color: #ffebeb;"

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
            "best_experiment": None,
            "worst_experiment": None,
        }

    # Calculate scores for each experiment pair
    pair_improvements = {}

    # For each pair of experiments
    for i, exp1 in enumerate(selected_exps):
        for j, exp2 in enumerate(selected_exps):
            if i < j:  # Only process each pair once
                pair_key = f"{exp1} vs {exp2}"
                pair_improvements[pair_key] = {}

                # For each metric
                for score_col in filtered_score_columns:
                    metric_name = score_col.replace("_score", "").title()
                    pair_improvements[pair_key][metric_name] = {
                        "improved": 0,
                        "worsened": 0,
                        "unchanged": 0,
                        "performance_improved": 0,
                        "performance_worsened": 0,
                    }

                    # Get columns
                    exp1_col = f"{score_col} ({exp1})"
                    exp2_col = f"{score_col} ({exp2})"

                    # Get performance columns
                    exp1_perf_col = f"{score_col}_perf_{exp1}"
                    exp2_perf_col = f"{score_col}_perf_{exp2}"

                    # Compare rows where both values exist
                    valid_rows = comparison_df[
                        comparison_df[exp1_col].notna()
                        & comparison_df[exp2_col].notna()
                    ]

                    for _, row in valid_rows.iterrows():
                        # Inline normalize_score function
                        exp1_score = (
                            float(row[exp1_col]) if not pd.isna(row[exp1_col]) else None
                        )
                        exp2_score = (
                            float(row[exp2_col]) if not pd.isna(row[exp2_col]) else None
                        )

                        if exp1_score is not None and exp2_score is not None:
                            if exp2_score > exp1_score:
                                pair_improvements[pair_key][metric_name][
                                    "improved"
                                ] += 1
                            elif exp2_score < exp1_score:
                                pair_improvements[pair_key][metric_name][
                                    "worsened"
                                ] += 1
                            else:
                                pair_improvements[pair_key][metric_name][
                                    "unchanged"
                                ] += 1

                            # Check for performance category changes
                            if (
                                exp1_perf_col in comparison_df.columns
                                and exp2_perf_col in comparison_df.columns
                            ):
                                exp1_perf = (
                                    row[exp1_perf_col]
                                    if exp1_perf_col in row
                                    and pd.notna(row[exp1_perf_col])
                                    else None
                                )
                                exp2_perf = (
                                    row[exp2_perf_col]
                                    if exp2_perf_col in row
                                    and pd.notna(row[exp2_perf_col])
                                    else None
                                )

                                if exp1_perf == "Bad" and exp2_perf == "Good":
                                    pair_improvements[pair_key][metric_name][
                                        "performance_improved"
                                    ] += 1
                                elif exp1_perf == "Good" and exp2_perf == "Bad":
                                    pair_improvements[pair_key][metric_name][
                                        "performance_worsened"
                                    ] += 1

    # Determine the best and worst experiments for each metric
    for score_col in filtered_score_columns:
        metric_name = score_col.replace("_score", "").title()

        # Calculate total score improvements for each experiment
        exp_scores = {exp: 0 for exp in selected_exps}

        # For each experiment pair
        for pair_key, metrics in pair_improvements.items():
            exps = pair_key.split(" vs ")
            exp1, exp2 = exps[0], exps[1]

            # Add score based on improvements
            if metric_name in metrics:
                # Positive points for improvements, negative for worsenings
                imp_score = (
                    metrics[metric_name]["improved"] - metrics[metric_name]["worsened"]
                )

                # Performance changes count more
                perf_score = (metrics[metric_name]["performance_improved"] * 3) - (
                    metrics[metric_name]["performance_worsened"] * 3
                )

                # If exp2 is better than exp1, exp2 gets positive, exp1 gets negative
                exp_scores[exp2] += imp_score + perf_score
                exp_scores[exp1] -= imp_score + perf_score

        # Find best and worst experiments
        if exp_scores:
            best_exp = max(exp_scores.items(), key=lambda x: x[1])
            worst_exp = min(exp_scores.items(), key=lambda x: x[1])

            improvements[metric_name]["best_experiment"] = (
                best_exp[0] if best_exp[1] > 0 else None
            )
            improvements[metric_name]["worst_experiment"] = (
                worst_exp[0] if worst_exp[1] < 0 else None
            )

    # Add overall improvement metrics based on first experiment as baseline
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
                # Inline normalize_score function
                base_score = (
                    float(row[base_col]) if not pd.isna(row[base_col]) else None
                )
                comp_score = (
                    float(row[comp_col]) if not pd.isna(row[comp_col]) else None
                )

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

    return improvements, pair_improvements


def calculate_extract_similarity(extract1, extract2):
    """Calculate similarity between two text extracts using simple token-based comparison."""
    if pd.isna(extract1) or pd.isna(extract2):
        return 0.0

    # Convert to strings
    str1 = str(extract1).lower()
    str2 = str(extract2).lower()

    # Tokenize (simple whitespace tokenization)
    tokens1 = set(str1.split())
    tokens2 = set(str2.split())

    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    if union == 0:
        return 0.0

    return intersection / union


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
                                # Inline normalize_score and is_good_score functions
                                base_score = (
                                    float(question_rows[selected_exps[0]][score_col])
                                    if not pd.isna(
                                        question_rows[selected_exps[0]][score_col]
                                    )
                                    else None
                                )
                                base_is_good = (
                                    base_score >= 4 if base_score is not None else None
                                )

                                for exp_name in selected_exps[1:]:
                                    if score_col in question_rows[exp_name]:
                                        comp_score = (
                                            float(question_rows[exp_name][score_col])
                                            if not pd.isna(
                                                question_rows[exp_name][score_col]
                                            )
                                            else None
                                        )
                                        comp_is_good = (
                                            comp_score >= 4
                                            if comp_score is not None
                                            else None
                                        )

                                        # Check for numeric differences
                                        if (
                                            base_score is not None
                                            and comp_score is not None
                                        ):
                                            # Check if scores are different
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
                                    # Inline normalize_score and is_good_score
                                    score = (
                                        float(question_rows[exp_name][score_col])
                                        if not pd.isna(
                                            question_rows[exp_name][score_col]
                                        )
                                        else None
                                    )
                                    if score is not None:
                                        is_good = score >= 4
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
                        improvements, pair_improvements = calculate_metric_improvements(
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

                        # Create improvement summary table with best/worst experiment indicators
                        improvement_data = []
                        for metric, stats in improvements.items():
                            row_data = {
                                "Metric": metric,
                                "Improved": stats["improved"],
                                "Worsened": stats["worsened"],
                                "Unchanged": stats["unchanged"],
                                "Performance Improved": stats["performance_improved"],
                                "Performance Worsened": stats["performance_worsened"],
                                "Best Experiment": (
                                    stats["best_experiment"]
                                    if stats["best_experiment"]
                                    else "No clear winner"
                                ),
                                "Worst Experiment": (
                                    stats["worst_experiment"]
                                    if stats["worst_experiment"]
                                    else "No clear loser"
                                ),
                            }
                            improvement_data.append(row_data)

                        # Display improvements table
                        if improvement_data:
                            improvement_df = pd.DataFrame(improvement_data)
                            st.dataframe(improvement_df, use_container_width=True)

                            # For multiple experiments, show pair-wise comparisons
                            if len(selected_exps) > 2:
                                st.subheader("Pair-wise Comparison")

                                # Create tabs for each pair
                                pair_tabs = []
                                for i, exp1 in enumerate(selected_exps):
                                    for j, exp2 in enumerate(selected_exps):
                                        if i < j:  # Only process each pair once
                                            pair_key = f"{exp1} vs {exp2}"
                                            pair_tabs.append(pair_key)

                                if pair_tabs:
                                    tabs = st.tabs(pair_tabs)

                                    for tab_idx, pair_key in enumerate(pair_tabs):
                                        with tabs[tab_idx]:
                                            if pair_key in pair_improvements:
                                                # Create pair comparison data
                                                pair_data = []
                                                for metric, stats in pair_improvements[
                                                    pair_key
                                                ].items():
                                                    pair_data.append(
                                                        {
                                                            "Metric": metric,
                                                            "Improved": stats[
                                                                "improved"
                                                            ],
                                                            "Worsened": stats[
                                                                "worsened"
                                                            ],
                                                            "Unchanged": stats[
                                                                "unchanged"
                                                            ],
                                                            "Performance Improved": stats[
                                                                "performance_improved"
                                                            ],
                                                            "Performance Worsened": stats[
                                                                "performance_worsened"
                                                            ],
                                                        }
                                                    )

                                                if pair_data:
                                                    pair_df = pd.DataFrame(pair_data)
                                                    st.dataframe(
                                                        pair_df,
                                                        use_container_width=True,
                                                    )

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

                            # Collect relevant extracts for similarity calculation
                            extracts_data = {}
                            for exp_name in selected_exps:
                                df = exp_dfs[exp_name]
                                question_row = df[
                                    df["question"] == selected_question
                                ].iloc[0]

                                # Check for different extract fields
                                if (
                                    "rag_relevant_extracts" in question_row
                                    and pd.notna(question_row["rag_relevant_extracts"])
                                ):
                                    extracts_data[exp_name] = str(
                                        question_row["rag_relevant_extracts"]
                                    )
                                elif "relevant_extracts" in question_row and pd.notna(
                                    question_row["relevant_extracts"]
                                ):
                                    extracts_data[exp_name] = str(
                                        question_row["relevant_extracts"]
                                    )

                            # Calculate extract similarities if we have extracts for at least 2 experiments
                            extract_similarities = {}
                            if len(extracts_data) >= 2:
                                # For each pair of experiments
                                for i, exp1 in enumerate(selected_exps):
                                    for j, exp2 in enumerate(selected_exps):
                                        if (
                                            i < j
                                            and exp1 in extracts_data
                                            and exp2 in extracts_data
                                        ):  # Only process each pair once
                                            pair_key = f"{exp1} vs {exp2}"
                                            similarity = calculate_extract_similarity(
                                                extracts_data[exp1], extracts_data[exp2]
                                            )
                                            extract_similarities[pair_key] = similarity

                            # Show extract similarities if we have any
                            if extract_similarities:
                                st.subheader("Relevant Extracts Similarity")
                                similarity_data = [
                                    {
                                        "Experiment Pair": k,
                                        "Similarity Score": f"{v:.2f}",
                                    }
                                    for k, v in extract_similarities.items()
                                ]
                                st.dataframe(
                                    pd.DataFrame(similarity_data),
                                    use_container_width=True,
                                )

                            # Display experiment details
                            for i, exp_name in enumerate(selected_exps):
                                with exp_cols[i]:
                                    st.markdown(f"##### {exp_name}")
                                    df = exp_dfs[exp_name]
                                    question_row = df[
                                        df["question"] == selected_question
                                    ].iloc[0]

                                    # REORDERED: First show answers, extracts, context and source docs
                                    # Show answers if available
                                    if "answer" in question_row and pd.notna(
                                        question_row["answer"]
                                    ):
                                        with st.expander("📝 Answer", expanded=True):
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

                                    # Show source documents if available
                                    if "source_doc" in question_row and pd.notna(
                                        question_row["source_doc"]
                                    ):
                                        with st.expander(
                                            "� Source Documents", expanded=True
                                        ):
                                            st.markdown(str(question_row["source_doc"]))

                                    # Add separator
                                    st.markdown("---")

                                    # THEN show scores and justifications
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
                                            # Inline normalize_score
                                            score = (
                                                float(question_row[score_col])
                                                if not pd.isna(question_row[score_col])
                                                else None
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

                                                # Add justification if available - NOT expanded by default
                                                if (
                                                    justification_col in question_row
                                                    and pd.notna(
                                                        question_row[justification_col]
                                                    )
                                                ):
                                                    with st.expander(
                                                        f"{metric_name} Justification",
                                                        expanded=False,
                                                    ):
                                                        st.markdown(
                                                            f"_{str(question_row[justification_col])}_"
                                                        )
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
