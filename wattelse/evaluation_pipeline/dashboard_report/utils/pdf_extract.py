"""PDF generation utilities for the RAG Evaluation Dashboard."""

import io
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
import pandas as pd
import base64
import streamlit as st
from datetime import datetime

def create_pdf_report(experiments_data, description="", include_tables=True):
    """
    Generate a PDF report from the RAG Evaluation Dashboard.
    
    Args:
        experiments_data: List of experiment data dictionaries
        description: Text description to include in the PDF
        include_tables: Whether to include tables in the PDF
    
    Returns:
        BytesIO object containing the PDF
    """
    # Create a BytesIO object to store the PDF
    buffer = io.BytesIO()
    
    # Create a PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Add a custom style for the description
    description_style = ParagraphStyle(
        'Description',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=10
    )
    
    # Build the content
    content = []
    
    # Add title and date
    content.append(Paragraph("RAG Evaluation Report", title_style))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Add description
    if description:
        content.append(Paragraph("Description:", heading_style))
        content.append(Paragraph(description, description_style))
        content.append(Spacer(1, 0.25*inch))
    
    # Add Performance Overview section
    content.append(Paragraph("Performance Overview", heading_style))
    content.append(Spacer(1, 0.1*inch))
    
    # Get all judges and metrics
    all_judges = set()
    all_metrics = set()
    for exp in experiments_data:
        for judge, df in exp['dfs'].items():
            all_judges.add(judge)
            metrics = [col.replace('_score', '') for col in df.columns if col.endswith('_score')]
            all_metrics.update(metrics)
    
    # Create a summary table
    if include_tables:
        # Create overall summary table
        content.append(Paragraph("Summary of Good Scores (%)", subheading_style))
        
        # Build overall summary data
        table_data = [['Experiment'] + ['Overall'] + sorted(all_metrics)]
        
        for exp in experiments_data:
            exp_name = exp['name']
            metrics_values = {}
            
            for metric in sorted(all_metrics):
                metric_values = []
                
                for judge, df in exp['dfs'].items():
                    score_col = f'{metric}_score'
                    if score_col in df.columns:
                        good_score_pct = (df[score_col][df[score_col].isin([4, 5])].count() / 
                                        df[score_col].count() * 100)
                        metric_values.append(good_score_pct)
                
                if metric_values:
                    metrics_values[metric] = sum(metric_values) / len(metric_values)
            
            if metrics_values:
                overall_avg = sum(metrics_values.values()) / len(metrics_values)
                row_data = [exp_name, f"{overall_avg:.1f}%"]
                
                for metric in sorted(all_metrics):
                    if metric in metrics_values:
                        row_data.append(f"{metrics_values[metric]:.1f}%")
                    else:
                        row_data.append("N/A")
                
                table_data.append(row_data)
        
        # Create and style the table
        if len(table_data) > 1:  # Only create if we have data rows
            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            content.append(table)
            content.append(Spacer(1, 0.25*inch))
    
    # Add judge-specific tables
    if include_tables:
        for judge_name in sorted(all_judges):
            content.append(Paragraph(f"Analysis by {judge_name}", subheading_style))
            
            # Build judge-specific data for each metric
            for metric in sorted(all_metrics):
                # Create metric-specific table
                metric_table_data = [['Experiment', 'Good Score %']]
                
                for exp in experiments_data:
                    if judge_name in exp['dfs']:
                        df = exp['dfs'][judge_name]
                        score_col = f'{metric}_score'
                        if score_col in df.columns:
                            good_score_pct = (df[score_col][df[score_col].isin([4, 5])].count() / 
                                            df[score_col].count() * 100)
                            metric_table_data.append([exp['name'], f"{good_score_pct:.1f}%"])
                
                if len(metric_table_data) > 1:  # Only create if we have data rows
                    content.append(Paragraph(f"{metric.title()} Metric", normal_style))
                    table = Table(metric_table_data, repeatRows=1)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    content.append(table)
                    content.append(Spacer(1, 0.15*inch))
            
            content.append(PageBreak())
    
    # Add Timing Analysis section
    content.append(Paragraph("Timing Analysis", heading_style))
    content.append(Spacer(1, 0.1*inch))
    
    if include_tables:
        # Create timing table
        content.append(Paragraph("Timing Summary", subheading_style))
        
        # Check if timing data is available
        has_timing_data = False
        for exp in experiments_data:
            if exp['timing'] is not None and len(exp['timing']) > 0:
                has_timing_data = True
                break
        
        if has_timing_data:
            # Create query time table
            query_time_data = [['Experiment', 'Min', 'Max', 'Mean', 'Median']]
            
            for exp in experiments_data:
                if exp['timing'] is not None and 'rag_query_time_seconds' in exp['timing'].columns:
                    times = exp['timing']['rag_query_time_seconds'].dropna()
                    if len(times) > 0:
                        query_time_data.append([
                            exp['name'],
                            f"{times.min():.2f}s",
                            f"{times.max():.2f}s",
                            f"{times.mean():.2f}s",
                            f"{times.median():.2f}s"
                        ])
            
            if len(query_time_data) > 1:  # Only create if we have data rows
                content.append(Paragraph("Total Query Time", normal_style))
                table = Table(query_time_data, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                content.append(table)
                content.append(Spacer(1, 0.15*inch))
            
            # Create retriever time table
            retriever_time_data = [['Experiment', 'Min', 'Max', 'Mean', 'Median']]
            
            for exp in experiments_data:
                if exp['timing'] is not None and 'rag_retriever_time_seconds' in exp['timing'].columns:
                    times = exp['timing']['rag_retriever_time_seconds'].dropna()
                    if len(times) > 0:
                        retriever_time_data.append([
                            exp['name'],
                            f"{times.min():.2f}s",
                            f"{times.max():.2f}s",
                            f"{times.mean():.2f}s",
                            f"{times.median():.2f}s"
                        ])
            
            if len(retriever_time_data) > 1:  # Only create if we have data rows
                content.append(Paragraph("Retriever Time", normal_style))
                table = Table(retriever_time_data, repeatRows=1)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                content.append(table)
        else:
            content.append(Paragraph("No timing data available", normal_style))
    
    # Build the PDF
    doc.build(content)
    buffer.seek(0)
    return buffer

def get_pdf_download_link(pdf_bytes, filename="rag_evaluation_report.pdf", text="Download PDF Report"):
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