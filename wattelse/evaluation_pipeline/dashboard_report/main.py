"""Main Streamlit application."""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

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
    handle_raw_data_page,
    load_pairwise_evaluation_files,
    create_pairwise_pie_chart,
    PAIRWISE_RESULTS_DIR,
    RAG_QUERY_TIME_COLUMN,
    RAG_RETRIEVER_TIME_COLUMN,
    METRIC_DESCRIPTIONS,
    PAIRWISE_METRIC_COLUMN,
    PAIRWISE_WINNER_COLUMN,
    PAIRWISE_QUESTION_COLUMN,
)
from wattelse.evaluation_pipeline import RESULTS_BASE_DIR


def setup_page():
    """Configure page settings and title."""
    st.set_page_config(page_title="RAG Experiment Comparison", layout="wide")
    st.title("RAG Evaluation Pipeline Dashboard")


def get_available_experiments(base_path=RESULTS_BASE_DIR):
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


def get_available_pairwise_files(base_path):
    """Get all available pairwise Excel files."""
    base_path = Path(base_path)

    if not base_path.exists():
        return []

    # Find all pairwise Excel files in the base directory and subdirectories
    all_files = list(base_path.glob("**/pairwise_*.xlsx"))

    # Convert to strings for display
    return [str(file) for file in all_files]


def handle_experiment_setup():
    """Handle experiment setup page with directory navigation."""
    st.header("Experiment Configuration")

    # Initialize experiments in session state if not already present
    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    # Initialize pairwise experiments in session state if not already present
    if "pairwise_experiments" not in st.session_state:
        st.session_state.pairwise_experiments = []

    # Create tabs for regular and pairwise experiments
    tab1, tab2 = st.tabs(["Standard Evaluation", "Pairwise Evaluation"])

    with tab1:
        # CRITERIA-BASED EVALUATION
        st.subheader("Standard Criteria-Based Evaluation")

        # Get available experiments
        base_path = RESULTS_BASE_DIR
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
                    exp
                    for exp in available_experiments
                    if f"({selected_category})" in exp
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
                    # Get current directory value
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
                    default_name = ""  # Added this line to track the default name

                    if selected_exp == "(Base Directory)":
                        new_dir = ""
                        default_name = "Base Directory"  # Add default name
                    elif selected_exp.startswith("(") and selected_exp.endswith(
                        " Base)"
                    ):
                        # Extract the category name and set as dir
                        category = selected_exp[1:-6]  # Remove "(" and " Base)"
                        new_dir = category
                        default_name = category  # Use category as default name
                    elif selected_exp:
                        # For experiment with category format: "name (category)"
                        if " (" in selected_exp and ")" in selected_exp:
                            parts = selected_exp.split(" (")
                            exp_name = parts[0]
                            category = parts[1][:-1]  # Remove the closing ")"
                            new_dir = f"{category}/{exp_name}"
                            default_name = exp_name  # Use experiment name as default
                        else:
                            new_dir = selected_exp
                            default_name = selected_exp  # Use full name as default

                    # Only update if the directory has changed
                    if new_dir != current_dir:
                        exp["dir"] = new_dir
                        # Auto-populate name if it's empty or matches old directory structure
                        if (
                            not exp["name"]
                            or exp["name"] == f"Experiment {i+1}"
                            or exp["name"].startswith(current_dir)
                        ):
                            exp["name"] = default_name

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
        if st.button(
            "➕ Add Experiment", key="add_exp_bottom", use_container_width=True
        ):
            st.session_state.experiments.append(
                {
                    "dir": "",
                    "name": f"Experiment {len(st.session_state.experiments) + 1}",
                }
            )
            st.rerun()

    # PAIRWISE EVALUATION TAB
    with tab2:
        # Pairwise evaluation setup
        st.subheader("Pairwise Comparison Evaluation")

        st.info(
            """
        Configure pairwise comparison experiments:
        1. Select pairwise comparison Excel files
        2. Give each comparison a meaningful name
        """
        )

        # Get available pairwise files
        base_path = PAIRWISE_RESULTS_DIR
        available_pairwise_files = get_available_pairwise_files(base_path)

        if not available_pairwise_files:
            st.warning(
                f"No pairwise comparison files found in {base_path} or its subdirectories"
            )
            return

        # Display each pairwise experiment configuration
        for i, exp in enumerate(st.session_state.pairwise_experiments):
            with st.container():
                st.markdown(f"### Pairwise Comparison {i+1}")

                # First row: Move up/down buttons
                move_cols = st.columns([0.5, 0.5, 4])

                with move_cols[0]:
                    # Move up button (disabled for first item)
                    if i > 0:
                        if st.button(
                            "⬆️", key=f"pairwise_up_{i}", help="Move comparison up"
                        ):
                            # Swap with previous experiment
                            (
                                st.session_state.pairwise_experiments[i],
                                st.session_state.pairwise_experiments[i - 1],
                            ) = (
                                st.session_state.pairwise_experiments[i - 1],
                                st.session_state.pairwise_experiments[i],
                            )
                            st.rerun()
                    else:
                        # Disabled button (placeholder to maintain layout)
                        st.empty()

                with move_cols[1]:
                    # Move down button (disabled for last item)
                    if i < len(st.session_state.pairwise_experiments) - 1:
                        if st.button(
                            "⬇️", key=f"pairwise_down_{i}", help="Move comparison down"
                        ):
                            # Swap with next experiment
                            (
                                st.session_state.pairwise_experiments[i],
                                st.session_state.pairwise_experiments[i + 1],
                            ) = (
                                st.session_state.pairwise_experiments[i + 1],
                                st.session_state.pairwise_experiments[i],
                            )
                            st.rerun()
                    else:
                        # Disabled button (placeholder to maintain layout)
                        st.empty()

                # Main configuration row
                config_cols = st.columns([2, 2, 0.5])

                with config_cols[0]:
                    # Get current file path
                    current_file = exp.get("file", "")

                    # Find index for the current file
                    file_index = 0  # Default to empty selection
                    for idx, file_path in enumerate(available_pairwise_files):
                        if file_path == current_file:
                            file_index = (
                                idx + 1
                            )  # +1 because we add empty option at index 0
                            break

                    # File selection dropdown
                    selected_file = st.selectbox(
                        "📄 Pairwise Comparison File",
                        [""] + available_pairwise_files,
                        index=file_index,
                        key=f"pairwise_file_{i}",
                        format_func=lambda x: Path(x).name if x else "Select a file",
                    )

                    # Update the file path
                    if selected_file != current_file:
                        exp["file"] = selected_file

                    # Show full path
                    if selected_file:
                        st.caption(f"Full path: {selected_file}")

                with config_cols[1]:
                    # Name field with default from filename if empty
                    default_name = exp.get("name", "")
                    if not default_name and selected_file:
                        file_name = Path(selected_file).stem
                        default_name = file_name.replace("pairwise_", "")

                    exp["name"] = st.text_input(
                        "📝 Name",
                        value=default_name,
                        key=f"pairwise_name_{i}",
                    )

                with config_cols[2]:
                    if st.button(
                        "🗑️", key=f"pairwise_remove_{i}", help="Remove this comparison"
                    ):
                        st.session_state.pairwise_experiments.pop(i)
                        st.rerun()

                # Preview selected file
                if selected_file:
                    try:
                        preview_df = pd.read_excel(selected_file, nrows=1)
                        with st.expander("Preview File Structure", expanded=False):
                            st.write("File columns:")
                            st.write(", ".join(preview_df.columns.tolist()))
                    except Exception as e:
                        st.warning(f"Could not preview file: {str(e)}")

                st.divider()

        # Add button for new pairwise comparison
        if st.button(
            "➕ Add Pairwise Comparison",
            key="add_pairwise_comp",
            use_container_width=True,
        ):
            st.session_state.pairwise_experiments.append({"file": "", "name": ""})
            st.rerun()


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

    if "pairwise_experiments" not in st.session_state:
        st.session_state.pairwise_experiments = []

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

    # Check if regular or pairwise experiments are configured based on the page
    elif page == "Pairwise Analysis":
        if len(st.session_state.pairwise_experiments) == 0:
            st.info(
                "No pairwise experiments are configured. Please go to the Experiment Setup page to add pairwise comparisons."
            )
            return

        # Load pairwise experiments data
        pairwise_experiments_data = []
        has_invalid_files = False

        for exp in st.session_state.pairwise_experiments:
            file_path = exp.get("file", "")
            if not file_path:
                has_invalid_files = True
                continue

            data = load_pairwise_evaluation_files(file_path)
            if data is not None:
                pairwise_experiments_data.append(
                    {
                        "name": exp["name"],
                        "file": file_path,
                        "dfs": data[0],  # Judge-specific dataframes
                        "combined_stats": data[1],  # Combined statistics
                    }
                )
            else:
                has_invalid_files = True

        # Handle various error states
        if has_invalid_files:
            st.warning(
                "Some pairwise comparison files are invalid or empty. Please check the configuration."
            )

        if not pairwise_experiments_data:
            st.error(
                "No valid pairwise evaluation files found. Please configure valid files."
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
            _, _, overall_df = create_metrics_summary(experiments_data)

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

            # PAIRWISE COMPARISON SECTION - Below the confidence intervals
            # In the Performance Overview section where you display pairwise data:
            st.subheader("Pairwise Comparison Evaluation")

            # Check if pairwise data is available
            if (
                "pairwise_experiments" in st.session_state
                and st.session_state.pairwise_experiments
            ):
                # Load pairwise data
                pairwise_data = []
                for exp in st.session_state.pairwise_experiments:
                    file_path = exp.get("file", "")
                    if file_path:
                        data = load_pairwise_evaluation_files(file_path)
                        if data is not None:
                            pairwise_data.append(
                                {
                                    "name": exp["name"],
                                    "file": file_path,
                                    "dfs": data[0],
                                    "combined_stats": data[1],
                                }
                            )

                if pairwise_data:
                    # If there are multiple pairwise datasets, add a selectbox
                    if len(pairwise_data) > 1:
                        selected_pairwise_name = st.selectbox(
                            "Select Pairwise Comparison",
                            [exp["name"] for exp in pairwise_data],
                            key="overview_pairwise_selector",
                        )
                        selected_pairwise = next(
                            (
                                exp
                                for exp in pairwise_data
                                if exp["name"] == selected_pairwise_name
                            ),
                            pairwise_data[0],
                        )
                    else:
                        selected_pairwise = pairwise_data[0]

                    # Get combined stats
                    combined_stats = selected_pairwise["combined_stats"]

                    if not combined_stats.empty:
                        # Create columns for layout
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            # Calculate total unique questions across all judges
                            total_questions = 0
                            unique_questions = set()

                            # Go through all judge dataframes to count unique questions
                            for judge_name, df in selected_pairwise["dfs"].items():
                                if PAIRWISE_QUESTION_COLUMN in df.columns:
                                    unique_questions.update(
                                        df[PAIRWISE_QUESTION_COLUMN].unique()
                                    )

                            total_questions = len(unique_questions)

                            # Display comprehensive table with all metrics and all information
                            st.markdown(
                                f"##### Pairwise Win Statistics (Total Questions: {total_questions})"
                            )

                            # Get all metrics and possible winners (including Tie, Error, etc.)
                            metrics = [
                                col.replace("_win_rate", "")
                                for col in combined_stats.columns
                                if col.endswith("_win_rate")
                            ]

                            # Create a table that includes all possible outcomes in the Winner column
                            all_results_data = []

                            # Process raw judge data to get complete winner statistics
                            for judge_name, df in selected_pairwise["dfs"].items():
                                if (
                                    PAIRWISE_METRIC_COLUMN in df.columns
                                    and PAIRWISE_WINNER_COLUMN in df.columns
                                ):
                                    # Group by metric and winner to get counts
                                    grouped = (
                                        df.groupby(
                                            [
                                                PAIRWISE_METRIC_COLUMN,
                                                PAIRWISE_WINNER_COLUMN,
                                            ]
                                        )
                                        .size()
                                        .reset_index(name="Count")
                                    )

                                    # Calculate percentages by metric
                                    for metric, metric_df in grouped.groupby(
                                        PAIRWISE_METRIC_COLUMN
                                    ):
                                        total = metric_df["Count"].sum()

                                        for _, row in metric_df.iterrows():
                                            winner = row[PAIRWISE_WINNER_COLUMN]
                                            count = row["Count"]
                                            percentage = (
                                                (count / total * 100)
                                                if total > 0
                                                else 0
                                            )

                                            all_results_data.append(
                                                {
                                                    "Metric": metric,
                                                    "Winner": winner,
                                                    "Count": count,
                                                    "Percentage": percentage,
                                                    "Judge": judge_name,
                                                }
                                            )

                            # Create a combined summary across all judges
                            if all_results_data:
                                results_df = pd.DataFrame(all_results_data)
                                summary = (
                                    results_df.groupby(["Metric", "Winner"])
                                    .agg({"Count": "sum", "Judge": "count"})
                                    .reset_index()
                                )

                                # Calculate overall percentages
                                for metric, metric_df in summary.groupby("Metric"):
                                    total = metric_df["Count"].sum()
                                    summary.loc[
                                        summary["Metric"] == metric, "Percentage"
                                    ] = (
                                        summary.loc[
                                            summary["Metric"] == metric, "Count"
                                        ]
                                        / total
                                        * 100
                                    )

                                # Find max percentage for each metric for bold formatting
                                max_percentages = (
                                    summary.groupby("Metric")["Percentage"]
                                    .max()
                                    .to_dict()
                                )

                                # Format for display with bold for highest values
                                summary["Display"] = summary.apply(
                                    lambda row: (
                                        f"**{row['Count']} ({row['Percentage']:.1f}%)**"
                                        if row["Percentage"]
                                        == max_percentages.get(row["Metric"], 0)
                                        else f"{row['Count']} ({row['Percentage']:.1f}%)"
                                    ),
                                    axis=1,
                                )

                                # Pivot table for better display
                                pivot_table = summary.pivot(
                                    index="Winner", columns="Metric", values="Display"
                                ).reset_index()

                                # Format the column headers nicely
                                pivot_table.columns = [
                                    col.title() if col != "Winner" else col
                                    for col in pivot_table.columns
                                ]

                                # Display the pivot table with markdown to maintain bold formatting
                                st.markdown(
                                    pivot_table.to_markdown(index=False),
                                    unsafe_allow_html=True,
                                )

                                # Add caption explaining bold values
                                st.caption(
                                    "**Bold values** indicate the highest percentage for each metric category."
                                )
                            else:
                                st.info("No detailed winner statistics available.")

                            # And then replace the pie chart creation code with:
                            with col2:
                                # Create a PIE CHART for a selected metric
                                # Look for correctness_pairwise first, then use first available metric
                                selected_metric = None

                                for metric in metrics:
                                    if "correctness" in metric.lower():
                                        selected_metric = metric
                                        break

                                # If no correctness metric found, use the first available
                                if not selected_metric and metrics:
                                    selected_metric = metrics[0]

                                if selected_metric:
                                    # Create the pie chart using the imported function
                                    fig = create_pairwise_pie_chart(
                                        combined_stats=selected_pairwise[
                                            "combined_stats"
                                        ],
                                        pairwise_dfs=selected_pairwise["dfs"],
                                        metric_name=selected_metric,
                                    )

                                    if fig:
                                        # Display the pie chart
                                        st.plotly_chart(
                                            fig,
                                            use_container_width=True,
                                            key=f"pairwise_pie_{str(uuid.uuid4())}",
                                        )
                                    else:
                                        st.info(
                                            "No metric data available for visualization."
                                        )
                                else:
                                    st.info(
                                        "No metric data available for visualization."
                                    )
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
        # Check if pairwise experiments exist
        pairwise_data = []
        if (
            "pairwise_experiments" in st.session_state
            and st.session_state.pairwise_experiments
        ):
            # Load pairwise experiments data
            for exp in st.session_state.pairwise_experiments:
                file_path = exp.get("file", "")
                if file_path:
                    data = load_pairwise_evaluation_files(file_path)
                    if data is not None:
                        pairwise_data.append(
                            {
                                "name": exp["name"],
                                "file": file_path,
                                "dfs": data[0],
                                "combined_stats": data[1],
                            }
                        )

        # Call the enhanced raw data page with pairwise data
        handle_raw_data_page(experiments_data, pairwise_data)


if __name__ == "__main__":
    main()
