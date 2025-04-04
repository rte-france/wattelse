"""
WattElse Evaluation Pipeline

Core module for evaluating RAG (Retrieval Augmented Generation) systems.
"""

import os
from pathlib import Path

# Import main functions for easier access
from wattelse.evaluation_pipeline.evaluation import evaluate_rag_metrics
from wattelse.evaluation_pipeline.run_jury import main as run_jury

# Define base paths
BASE_DIR = Path(os.getenv("RAG_PIPELINE_EVAL_PATH", "/DSIA/nlp/experiments"))
CONFIG_EVAL = BASE_DIR / "eval_config.cfg"
REPORT_PATH = BASE_DIR / "report_output.xlsx"
RESULTS_BASE_DIR = BASE_DIR / "results"
BASE_DATA_DIR = BASE_DIR / "data"
BASE_DOCS_DIR = BASE_DIR / "docs"
BASE_OUTPUT_DIR = BASE_DIR / "data_predictions"

# Expose key components at the package level
__all__ = [
    "evaluate_rag_metrics",
    "run_jury",
]
