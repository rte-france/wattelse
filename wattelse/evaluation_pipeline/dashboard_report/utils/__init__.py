"""Utils module initialization."""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

# Import constants
from .constants import (
    QUERY_COLUMN,
    ANSWER_COLUMN,
    DOC_LIST_COLUMN,
    CONTEXT_COLUMN,
    COMPLEXITY_COLUMN,
    RAG_RELEVANT_EXTRACTS_COLUMN,
    RAG_QUERY_TIME_COLUMN,
    RAG_RETRIEVER_TIME_COLUMN,
    METRIC_DESCRIPTIONS,
    PAIRWISE_WINNER_COLUMN,
    PAIRWISE_METRIC_COLUMN,
)

# Import data loading functions
from .data_loader import load_evaluation_files, calculate_good_score_percentage

# Import visualization functions
from .visualization import (
    create_timing_plot,
    create_radar_plot,
    create_average_radar_plot,
    create_pairwise_pie_chart,
)

# Import the new metrics summary function
from .metrics_summary import create_metrics_summary

# Import PDF export functions
from .pdf_extract import create_pdf_report, get_pdf_download_link

# Import raw data handling functions
from .raw_data import handle_raw_data_page

# Import pairwise functions
from .pairwise import (
    load_pairwise_evaluation_files,
    handle_pairwise_analysis_page,
    PAIRWISE_RESULTS_DIR,
)
