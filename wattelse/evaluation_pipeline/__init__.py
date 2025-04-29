"""
WattElse Evaluation Pipeline

Core module for evaluating RAG (Retrieval Augmented Generation) systems.
"""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import os
from pathlib import Path

# Define base paths
BASE_DIR = Path(os.getenv("RAG_PIPELINE_EVAL_PATH", "/DSIA/nlp/experiments"))
CONFIG_EVAL = BASE_DIR / "eval_config.toml"
REPORT_PATH = BASE_DIR / "report_output.xlsx"
RESULTS_BASE_DIR = BASE_DIR / "results"
BASE_DATA_DIR = BASE_DIR / "data"
BASE_DOCS_DIR = BASE_DIR / "docs"
BASE_OUTPUT_DIR = BASE_DIR / "data_predictions"
COMPARISON_DATA_DIR = BASE_DIR / "comparison_data"
PAIRWISE_RESULTS_DIR = RESULTS_BASE_DIR / "pairwise_results"

# Import main functions for easier access
from wattelse.evaluation_pipeline.evaluation import evaluate_rag_metrics
from wattelse.evaluation_pipeline.run_jury import main as run_jury

# Expose key components at the package level
__all__ = [
    "evaluate_rag_metrics",
    "run_jury",
]
