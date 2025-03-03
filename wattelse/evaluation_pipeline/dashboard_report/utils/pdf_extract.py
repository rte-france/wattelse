"""PDF generation utilities for the RAG Evaluation Dashboard."""

import io
import re
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
)
from reportlab.lib.units import inch
from .constants import METRIC_DESCRIPTIONS
import base64
from datetime import datetime

import re


def convert_markdown_to_html(text):
    """
    Convert markdown-style text to ReportLab-compatible HTML for scientific reports.

    Args:
        text: Text with markdown formatting

    Returns:
        Text with HTML tags for ReportLab
    """
    if not text:
        return ""

    # Convert bold: **text** to <b>text</b>
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # Convert italics: *text* to <i>text</i> (excluding bold cases)
    text = re.sub(r"(?<!\*)\*(.*?)\*(?!\*)", r"<i>\1</i>", text)

    # Convert headers with proper spacing
    text = re.sub(
        r"^#\s+(.*?)$",
        r'<br/><font size="16"><b>\1</b></font><br/>',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^##\s+(.*?)$",
        r'<br/><font size="14"><b>\1</b></font><br/>',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^###\s+(.*?)$",
        r'<br/><font size="12"><b>\1</b></font><br/>',
        text,
        flags=re.MULTILINE,
    )

    # Convert bullet points: - item → • item
    text = re.sub(r"^\-\s+(.*?)$", r"• \1", text, flags=re.MULTILINE)

    # Convert numbered lists: 1. item → 1. item
    text = re.sub(r"^\d+\.\s+(.*?)$", r"\g<0>", text, flags=re.MULTILINE)

    # Convert blockquotes: > Quote → <i>"Quote"</i>
    text = re.sub(r"^>\s+(.*?)$", r'<i>"\1"</i>', text, flags=re.MULTILINE)

    # Convert links: [text](url) → <a href="url">text</a>
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)

    # Ensure paragraphs are wrapped properly
    text = re.sub(r"([^\n]+)", r"<p>\1</p>", text)

    # Handle proper line breaks
    text = text.replace("\n", "<br/>")

    return text


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
        description: Text description to include in the PDF
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

    # Add a custom style for the description that supports HTML formatting
    description_style = ParagraphStyle(
        "Description", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=10
    )

    # Add a style for notes and captions
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

    # Add description with proper formatting
    if description:
        content.append(Paragraph("Description", heading_style))

        # Convert markdown to HTML and split into sections if they exist
        if "## " in description:
            # Description has section headers, process each section
            sections = re.split(r"(?=^##\s+)", description, flags=re.MULTILINE)

            for section in sections:
                if section.strip():
                    # If it's a section with a header
                    if section.strip().startswith("## "):
                        # Process the section
                        html_text = convert_markdown_to_html(section)
                        content.append(Paragraph(html_text, description_style))
                        content.append(Spacer(1, 0.1 * inch))
                    else:
                        # It's an intro text before any section
                        html_text = convert_markdown_to_html(section)
                        content.append(Paragraph(html_text, description_style))
                        content.append(Spacer(1, 0.1 * inch))
        else:
            # No section headers, process as a single block
            html_text = convert_markdown_to_html(description)
            content.append(Paragraph(html_text, description_style))

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

        # Build overall summary data
        table_data = [
            ["Experiment", "Overall"] + sorted(all_metrics) + ["Number of Judges"]
        ]

        for exp in experiments_data:
            exp_name = exp["name"]
            metrics_values = {}

            for metric in sorted(all_metrics):
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

            if metrics_values:
                overall_avg = sum(metrics_values.values()) / len(metrics_values)
                row_data = [exp_name, f"{overall_avg:.1f}%"]

                for metric in sorted(all_metrics):
                    if metric in metrics_values:
                        # Add stars based on how many judges rated this experiment as best for this metric
                        stars = "*" * best_counts[metric][exp_name]
                        row_data.append(
                            f"{metrics_values[metric]:.1f}%{' ' + stars if stars else ''}"
                        )
                    else:
                        row_data.append("N/A")

                # Add judge count
                row_data.append(str(judge_counts[exp_name]))

                table_data.append(row_data)

        # Create and style the table
        if len(table_data) > 1:  # Only create if we have data rows
            table = Table(table_data, repeatRows=1)

            # Find all cells with maximum values in each column to highlight
            max_value_cells = (
                {}
            )  # Dictionary to store column_idx -> list of row indices with max value
            for col_idx in range(
                1, len(table_data[0]) - 1
            ):  # Skip last column (Number of Judges)
                col_values = []
                for row_idx in range(1, len(table_data)):
                    value = table_data[row_idx][col_idx]
                    if value != "N/A":
                        try:
                            # Extract numeric value from the formatted string (ignoring stars)
                            numeric_value = float(value.split("%")[0])
                            col_values.append((row_idx, numeric_value))
                        except (ValueError, AttributeError):
                            continue

                if col_values:
                    # Find max value for this column
                    max_value = max(col_values, key=lambda x: x[1])[1]
                    # Find all rows with this max value (handling ties)
                    max_value_cells[col_idx] = [
                        row_idx for row_idx, value in col_values if value == max_value
                    ]

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
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]

            # Add bold formatting and background color for ALL cells with highest values in each column
            for col_idx, row_indices in max_value_cells.items():
                for row_idx in row_indices:
                    style.append(
                        (
                            "FONTNAME",
                            (col_idx, row_idx),
                            (col_idx, row_idx),
                            "Helvetica-Bold",
                        )
                    )
                    style.append(
                        (
                            "BACKGROUND",
                            (col_idx, row_idx),
                            (col_idx, row_idx),
                            colors.lightblue,
                        )
                    )

            table.setStyle(TableStyle(style))
            content.append(table)

            # Add a note explaining what the stars mean
            content.append(Spacer(1, 0.1 * inch))
            content.append(
                Paragraph(
                    "Note: Stars (*) indicate how many judges rated this experiment as the best for that metric. Multiple experiments may be rated as best in case of ties.",
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

    # Create a custom page template with page numbers
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
