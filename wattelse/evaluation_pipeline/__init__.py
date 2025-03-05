"""
WattElse Evaluation Pipeline

Core module for evaluating RAG (Retrieval Augmented Generation) systems.
"""

# Import main functions for easier access
from wattelse.evaluation_pipeline.evaluation import evaluate_rag_metrics
from wattelse.evaluation_pipeline.run_jury import main as run_jury

# Expose key components at the package level
__all__ = [
    "evaluate_rag_metrics",
    "run_jury",
]
