"""PDF generation utilities for the RAG Evaluation Dashboard with proper Markdown support."""

import io
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    ListFlowable,
    ListItem,
)
from reportlab.lib.styles import ListStyle
from reportlab.lib.units import inch
from .constants import METRIC_DESCRIPTIONS
import base64
from datetime import datetime
import markdown
from scipy import stats
import numpy as np
from bs4 import BeautifulSoup
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_markdown_to_reportlab(text):
    """
    Convert markdown text to ReportLab elements.

    Args:
        text: Text with markdown formatting

    Returns:
        List of ReportLab flowable elements
    """
    if not text:
        return []

    # Create styles
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    heading1_style = ParagraphStyle(
        "Heading1", parent=styles["Heading1"], spaceBefore=14, spaceAfter=10
    )
    heading2_style = ParagraphStyle(
        "Heading2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=8
    )
    heading3_style = ParagraphStyle(
        "Heading3", parent=styles["Heading3"], spaceBefore=10, spaceAfter=6
    )
    bullet_style = ParagraphStyle("BulletPoint", parent=normal_style, leftIndent=20)

    # Create list styles
    bullet_list_style = ListStyle(
        name="BulletList",
        leftIndent=20,
        rightIndent=0,
        bulletAlign="left",
        bulletType="bullet",
        bulletColor=colors.black,
        bulletFontName="Helvetica",
        bulletFontSize=10,
        bulletOffsetY=0,
        bulletDedent="auto",
        bulletDir="ltr",
        bulletFormat="%s",
        start=None,
    )

    numbered_list_style = ListStyle(
        name="NumberedList",
        leftIndent=20,
        rightIndent=0,
        bulletAlign="left",
        bulletType="1",
        bulletColor=colors.black,
        bulletFontName="Helvetica",
        bulletFontSize=10,
        bulletOffsetY=0,
        bulletDedent="auto",
        bulletDir="ltr",
        bulletFormat="%s.",
        start=1,
    )

    # Convert markdown to HTML
    html_content = markdown.markdown(text, extensions=["extra"])

    # Parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # List to store ReportLab elements
    elements = []

    # Process each element
    for element in soup.children:
        if element.name is None:
            continue

        # Handle headings
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(element.name[1])
            style = (
                heading1_style
                if level == 1
                else heading2_style if level == 2 else heading3_style
            )
            elements.append(Paragraph(element.text, style))

        # Handle paragraphs
        elif element.name == "p":
            # Convert all internal HTML to ReportLab-compatible format
            paragraph_text = str(element)
            # Replace <strong> with <b> and <em> with <i>
            paragraph_text = paragraph_text.replace("<strong>", "<b>").replace(
                "</strong>", "</b>"
            )
            paragraph_text = paragraph_text.replace("<em>", "<i>").replace(
                "</em>", "</i>"
            )
            # Strip <p> tags
            paragraph_text = paragraph_text.replace("<p>", "").replace("</p>", "")
            elements.append(Paragraph(paragraph_text, normal_style))

        # Handle lists
        elif element.name in ["ul", "ol"]:
            list_items = []
            for li in element.find_all("li", recursive=False):
                list_items.append(ListItem(Paragraph(li.text, bullet_style)))

            # Use the appropriate list style
            list_style = (
                bullet_list_style if element.name == "ul" else numbered_list_style
            )
            elements.append(ListFlowable(list_items, style=list_style))

        # Handle blockquotes
        elif element.name == "blockquote":
            quote_style = ParagraphStyle(
                "Quote",
                parent=normal_style,
                leftIndent=30,
                fontName="Helvetica-Oblique",
            )
            for p in element.find_all("p"):
                elements.append(Paragraph(f'"{p.text}"', quote_style))

    return elements


def create_pdf_report(
    experiments_data,
    experiment_configs=None,
    description="",
    include_tables=True,
    custom_title="RAG Evaluation Report",
    author="",
):
    """
    Generate a PDF report from the RAG Evaluation Dashboard.

    Args:
        experiments_data: List of experiment data dictionaries
        experiment_configs: Configuration information for each experiment
        description: Text description to include in the PDF (in Markdown format)
        include_tables: Whether to include tables in the PDF
        custom_title: Custom title for the report
        author: Report author name

    Returns:
        BytesIO object containing the PDF
    """
    # Create a BytesIO object to store the PDF
    buffer = io.BytesIO()

    # Create a PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading1"]
    subheading_style = styles["Heading2"]
    normal_style = styles["Normal"]
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=9,
        fontName="Helvetica-Oblique",
        textColor=colors.gray,
        leading=12,
        spaceAfter=12,
    )

    # Build the content
    content = []

    # Add title and date
    content.append(Paragraph(custom_title, title_style))
    content.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style
        )
    )

    # Add author if provided
    if author:
        content.append(Paragraph(f"Author: {author}", normal_style))

    content.append(Spacer(1, 0.25 * inch))

    # Add description with proper Markdown formatting
    if description:
        content.append(Paragraph("Description", heading_style))

        try:
            # Convert markdown to ReportLab elements
            markdown_elements = convert_markdown_to_reportlab(description)
            content.extend(markdown_elements)
        except Exception as e:
            logger.error(f"Error processing markdown: {e}")
            # Fallback to plain text if markdown processing fails
            content.append(Paragraph(description, normal_style))

        content.append(Spacer(1, 0.15 * inch))

    # Add Experiment Configuration section on a new page
    content.append(PageBreak())
    if experiment_configs and len(experiment_configs) > 0:
        content.append(Paragraph("Experiment Configuration", heading_style))

        # Create a table for experiment configurations
        config_table_data = [["Experiment Name", "Directory"]]

        for config in experiment_configs:
            config_table_data.append(
                [config.get("name", "Unnamed"), config.get("dir", "N/A")]
            )

        if len(config_table_data) > 1:  # Only create if we have data rows
            table = Table(config_table_data, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("ALIGN", (0, 1), (0, -1), "LEFT"),
                        ("ALIGN", (1, 1), (1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            content.append(table)
            content.append(Spacer(1, 0.25 * inch))

        # Add Metrics Descriptions subsection
        if METRIC_DESCRIPTIONS:
            content.append(Paragraph("Metrics Descriptions", subheading_style))
            content.append(Spacer(1, 0.05 * inch))

            for metric, description in METRIC_DESCRIPTIONS.items():
                metric_name = Paragraph(f"<b>{metric.title()}</b>", normal_style)
                content.append(metric_name)

                # Create paragraph with proper wrapping for the description
                metric_desc = Paragraph(
                    description,
                    ParagraphStyle(
                        "MetricDescription",
                        parent=normal_style,
                        leftIndent=0.25 * inch,
                        spaceBefore=2,
                        spaceAfter=8,
                    ),
                )
                content.append(metric_desc)
                content.append(Spacer(1, 0.05 * inch))

    # Add Performance Overview section (on a new page)
    content.append(PageBreak())
    content.append(Paragraph("Performance Overview", heading_style))
    content.append(Spacer(1, 0.1 * inch))

    # Get all judges and metrics
    all_judges = set()
    all_metrics = set()
    for exp in experiments_data:
        for judge, df in exp["dfs"].items():
            all_judges.add(judge)
            metrics = [
                col.replace("_score", "")
                for col in df.columns
                if col.endswith("_score")
            ]
            all_metrics.update(metrics)

    # Track judge counts for each experiment
    judge_counts = {}
    for exp in experiments_data:
        judge_counts[exp["name"]] = len(exp["dfs"])

    # Track the best experiment for each metric according to each judge
    best_counts = {
        metric: {exp["name"]: 0 for exp in experiments_data} for metric in all_metrics
    }

    # Find the best experiment for each metric according to each judge (properly handling ties)
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

    # Create a summary table
    if include_tables:
        # Create overall summary table
        content.append(Paragraph("Summary of Performances (%)", subheading_style))

        metrics_header_row = ["Experiment"] + sorted(all_metrics) + ["Number of Judges"]
        metrics_table_data = [metrics_header_row]

        # CI TABLE
        # Build CI table data
        ci_header_row = ["Experiment"]
        for metric in sorted(all_metrics):
            ci_header_row.append(f"{metric.title()} (95% CI)")
        ci_table_data = [ci_header_row]

        for exp in experiments_data:
            exp_name = exp["name"]
            metrics_values = {}
            ci_values = {}  # To store CI values for all metrics

            # Calculate metrics values and CI for all metrics
            for metric in all_metrics:
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

                        # Collect individual scores for CI calculation for all metrics
                        binary_scores = [
                            1 if score in [4, 5] else 0
                            for score in df[score_col].dropna()
                        ]
                        all_scores.extend(binary_scores)

                if metric_values:
                    metrics_values[metric] = sum(metric_values) / len(metric_values)

                    # Calculate CI for all metrics
                    if all_scores:
                        from scipy import stats
                        import numpy as np

                        scores_array = np.array(
                            [s * 100 for s in all_scores]
                        )  # Convert to percentages
                        mean = np.mean(scores_array)
                        se = stats.sem(scores_array) if len(scores_array) > 1 else 0

                        if len(scores_array) > 1:
                            h = se * stats.t.ppf((1 + 0.95) / 2, len(scores_array) - 1)
                            ci_lower = max(0, mean - h)
                            ci_upper = min(100, mean + h)
                        else:
                            ci_lower = mean
                            ci_upper = mean

                        ci_values[f"{metric}_ci"] = (
                            f"({ci_lower:.1f}% - {ci_upper:.1f}%)"
                        )
                    else:
                        ci_values[f"{metric}_ci"] = "N/A"
                else:
                    metrics_values[metric] = "N/A"
                    ci_values[f"{metric}_ci"] = "N/A"

            # METRICS TABLE ROW
            metrics_row = [exp_name]

            # Add metric values with stars
            for metric in sorted(all_metrics):
                if metric in metrics_values and metrics_values[metric] != "N/A":
                    stars = "*" * best_counts[metric][exp_name]
                    metrics_row.append(
                        f"{metrics_values[metric]:.1f}%{' ' + stars if stars else ''}"
                    )
                else:
                    metrics_row.append("N/A")

            # Add judge count
            metrics_row.append(str(judge_counts[exp_name]))
            metrics_table_data.append(metrics_row)

            # CI TABLE ROW
            ci_row = [exp_name]

            # Add CI values
            for metric in sorted(all_metrics):
                if f"{metric}_ci" in ci_values:
                    ci_row.append(ci_values[f"{metric}_ci"])
                else:
                    ci_row.append("N/A")

            ci_table_data.append(ci_row)

        # METRICS TABLE - Create and style the table
        if len(metrics_table_data) > 1:  # Only create if we have data rows
            metrics_table = Table(metrics_table_data, repeatRows=1)

            # Find all cells with maximum values in metrics table
            max_metric_cells = (
                {}
            )  # Dictionary to store column_idx -> list of row indices with max value

            # Process each metric column to find highest values
            for col_idx in range(
                1, len(metrics_table_data[0]) - 1
            ):  # Skip first (Experiment) and last (Number of Judges) columns
                col_values = []

                for row_idx in range(1, len(metrics_table_data)):
                    value = metrics_table_data[row_idx][col_idx]
                    if value != "N/A":
                        try:
                            # Extract numeric value from metrics format
                            numeric_value = float(value.split("%")[0])
                            col_values.append((row_idx, numeric_value))
                        except (ValueError, IndexError, AttributeError):
                            continue

                if col_values:
                    # Find max value for this column
                    max_value = max(col_values, key=lambda x: x[1])[1]
                    # Find all rows with this max value (handling ties)
                    max_metric_cells[col_idx] = [
                        row_idx for row_idx, value in col_values if value == max_value
                    ]

            # Create base style for metrics table
            metrics_style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]

            # Add bold formatting and background color for cells with highest values
            for col_idx, row_indices in max_metric_cells.items():
                for row_idx in row_indices:
                    metrics_style.append(
                        (
                            "FONTNAME",
                            (col_idx, row_idx),
                            (col_idx, row_idx),
                            "Helvetica-Bold",
                        )
                    )
                    metrics_style.append(
                        (
                            "BACKGROUND",
                            (col_idx, row_idx),
                            (col_idx, row_idx),
                            colors.lightblue,
                        )
                    )

            metrics_table.setStyle(TableStyle(metrics_style))
            content.append(metrics_table)

            # Add a note explaining what the stars mean
            content.append(Spacer(1, 0.1 * inch))
            content.append(
                Paragraph(
                    "Note: Stars (*) indicate how many judges rated this experiment as the best for that metric. "
                    "Multiple experiments may be rated as best in case of ties.",
                    caption_style,
                )
            )

            # Add some space between tables
            content.append(Spacer(1, 0.3 * inch))
            content.append(Paragraph("95% Confidence Intervals", subheading_style))
            content.append(Spacer(1, 0.1 * inch))

            # CI TABLE - Create and style the table
            ci_table = Table(ci_table_data, repeatRows=1)

            # Find all cells with maximum values in CI table
            max_ci_cells = (
                {}
            )  # Dictionary to store column_idx -> list of row indices with max value

            # Process each CI column to find highest upper bounds
            for col_idx in range(
                1, len(ci_table_data[0])
            ):  # Skip first (Experiment) column
                col_values = []

                for row_idx in range(1, len(ci_table_data)):
                    value = ci_table_data[row_idx][col_idx]
                    if value != "N/A":
                        try:
                            # Extract upper bound from CI format "(X.X% - Y.Y%)"
                            upper_bound = float(value.split(" - ")[1].rstrip("%)"))
                            col_values.append((row_idx, upper_bound))
                        except (ValueError, IndexError, AttributeError):
                            continue

                if col_values:
                    # Find max value for this column
                    max_value = max(col_values, key=lambda x: x[1])[1]
                    # Find all rows with this max value (handling ties)
                    max_ci_cells[col_idx] = [
                        row_idx for row_idx, value in col_values if value == max_value
                    ]

            # Create base style for CI table
            ci_style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]

            # Add bold formatting and background color for cells with highest CI values
            for col_idx, row_indices in max_ci_cells.items():
                for row_idx in row_indices:
                    ci_style.append(
                        (
                            "FONTNAME",
                            (col_idx, row_idx),
                            (col_idx, row_idx),
                            "Helvetica-Bold",
                        )
                    )
                    ci_style.append(
                        (
                            "BACKGROUND",
                            (col_idx, row_idx),
                            (col_idx, row_idx),
                            colors.lightblue,
                        )
                    )

            ci_table.setStyle(TableStyle(ci_style))
            content.append(ci_table)

            # Add a note explaining the CI table
            content.append(Spacer(1, 0.1 * inch))
            content.append(
                Paragraph(
                    "Note: This table shows the 95% confidence intervals for each metric. "
                    "The confidence interval indicates the range where we can be 95% confident the true performance percentage lies.",
                    caption_style,
                )
            )
            content.append(Spacer(1, 0.15 * inch))

    # Add judge-specific tables
    if include_tables:
        # Add page break before judge analysis section
        content.append(PageBreak())

        # Process each judge
        judge_count = 0
        for judge_name in sorted(all_judges):
            # Add page breaks between judges (not before the first one)
            if judge_count > 0:
                content.append(PageBreak())

            content.append(Paragraph(f"Analysis by {judge_name}", subheading_style))

            # Build judge-specific data for each metric
            for metric in sorted(all_metrics):
                # Create metric-specific table
                metric_table_data = [["Experiment", "Performance %"]]

                for exp in experiments_data:
                    if judge_name in exp["dfs"]:
                        df = exp["dfs"][judge_name]
                        score_col = f"{metric}_score"
                        if score_col in df.columns:
                            good_score_pct = (
                                df[score_col][df[score_col].isin([4, 5])].count()
                                / df[score_col].count()
                                * 100
                            )
                            metric_table_data.append(
                                [exp["name"], f"{good_score_pct:.1f}%"]
                            )

                if len(metric_table_data) > 1:  # Only create if we have data rows
                    content.append(Paragraph(f"{metric.title()} Metric", normal_style))
                    table = Table(metric_table_data, repeatRows=1)
                    # Find all cells with the maximum value to highlight (for ties)
                    max_row_indices = []
                    max_value = -1

                    for row_idx in range(1, len(metric_table_data)):
                        value = metric_table_data[row_idx][1]
                        try:
                            # Extract numeric value from the formatted string
                            numeric_value = float(value.rstrip("%"))
                            if numeric_value > max_value:
                                max_value = numeric_value
                                max_row_indices = [row_idx]
                            elif numeric_value == max_value:
                                max_row_indices.append(row_idx)
                        except (ValueError, AttributeError):
                            continue

                    # Create base style
                    style = [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]

                    # Add bold formatting for all highest values (handling ties)
                    for row_idx in max_row_indices:
                        style.append(
                            ("FONTNAME", (1, row_idx), (1, row_idx), "Helvetica-Bold")
                        )
                        style.append(
                            ("BACKGROUND", (1, row_idx), (1, row_idx), colors.lightblue)
                        )

                    table.setStyle(TableStyle(style))
                    content.append(table)
                    content.append(Spacer(1, 0.15 * inch))

            judge_count += 1  # Increment judge counter

    # Add Timing Analysis section on a new page
    content.append(PageBreak())
    content.append(Paragraph("Timing Analysis", heading_style))
    content.append(Spacer(1, 0.1 * inch))

    if include_tables:
        # Create timing table
        content.append(Paragraph("Timing Summary", subheading_style))

        # Check if timing data is available
        has_timing_data = False
        for exp in experiments_data:
            if exp["timing"] is not None and len(exp["timing"]) > 0:
                has_timing_data = True
                break

        if has_timing_data:
            # Create query time table
            query_time_data = [["Experiment", "Min", "Max", "Mean", "Median"]]

            for exp in experiments_data:
                if (
                    exp["timing"] is not None
                    and "rag_query_time_seconds" in exp["timing"].columns
                ):
                    times = exp["timing"]["rag_query_time_seconds"].dropna()
                    if len(times) > 0:
                        query_time_data.append(
                            [
                                exp["name"],
                                f"{times.min():.2f}s",
                                f"{times.max():.2f}s",
                                f"{times.mean():.2f}s",
                                f"{times.median():.2f}s",
                            ]
                        )

            if len(query_time_data) > 1:  # Only create if we have data rows
                content.append(Paragraph("Total Query Time", normal_style))
                table = Table(query_time_data, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )
                content.append(table)
                content.append(Spacer(1, 0.15 * inch))

            # Create retriever time table
            retriever_time_data = [["Experiment", "Min", "Max", "Mean", "Median"]]

            for exp in experiments_data:
                if (
                    exp["timing"] is not None
                    and "rag_retriever_time_seconds" in exp["timing"].columns
                ):
                    times = exp["timing"]["rag_retriever_time_seconds"].dropna()
                    if len(times) > 0:
                        retriever_time_data.append(
                            [
                                exp["name"],
                                f"{times.min():.2f}s",
                                f"{times.max():.2f}s",
                                f"{times.mean():.2f}s",
                                f"{times.median():.2f}s",
                            ]
                        )

            if len(retriever_time_data) > 1:  # Only create if we have data rows
                content.append(Paragraph("Retriever Time", normal_style))
                table = Table(retriever_time_data, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )
                content.append(table)
        else:
            content.append(Paragraph("No timing data available", normal_style))

    def myFirstPage(canvas, doc):
        canvas.saveState()
        # Footer with page number
        canvas.setFont("Helvetica", 9)
        canvas.drawString(inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    def myLaterPages(canvas, doc):
        canvas.saveState()
        # Footer with page number
        canvas.setFont("Helvetica", 9)
        canvas.drawString(inch, 0.5 * inch, f"Page {doc.page}")
        canvas.restoreState()

    # Build the PDF with page templates
    doc.build(content, onFirstPage=myFirstPage, onLaterPages=myLaterPages)
    buffer.seek(0)
    return buffer


def get_pdf_download_link(
    pdf_bytes, filename="rag_evaluation_report.pdf", text="Download PDF Report"
):
    """
    Generate a download link for the PDF.

    Args:
        pdf_bytes: BytesIO object containing the PDF
        filename: Name of the file to download
        text: Text to display on the download button

    Returns:
        HTML string with the download link
    """
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href
