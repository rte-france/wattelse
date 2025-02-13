"""Data loading and processing utilities."""

import pandas as pd
from pathlib import Path
from .constants import *

def get_available_metrics(df: pd.DataFrame) -> list:
    """Get list of available metrics from DataFrame columns."""
    metrics = []
    for metric in ['faithfulness', 'correctness', 'retrievability']:
        if f'{metric}_score' in df.columns:
            metrics.append(metric)
    return metrics

def load_evaluation_files(eval_dir: str):
    """Load and combine all evaluation Excel files from the directory."""
    if not eval_dir:
        return None
    
    # Handle both absolute and relative paths
    if not eval_dir.startswith('/'):
        # If relative path, construct path relative to project root
        eval_path = Path(__file__).parents[2] / 'retriever-testing' / eval_dir
    else:
        eval_path = Path(eval_dir)
    
    if not eval_path.exists():
        return None
        
    excel_files = list(eval_path.glob("evaluation_*.xlsx"))
    if not excel_files:
        return None
    
    all_dfs = {}
    combined_df = None
    timing_df = None
    
    for file in excel_files:
        if "combined" not in file.name:
            model_name = file.stem.split('_')[-1]
            df = pd.read_excel(file)
            
            # Extract timing columns
            if timing_df is None:
                timing_cols = [QUERY_COLUMN]
                if RAG_QUERY_TIME_COLUMN in df.columns:
                    timing_cols.append(RAG_QUERY_TIME_COLUMN)
                if RAG_RETRIEVER_TIME_COLUMN in df.columns:
                    timing_cols.append(RAG_RETRIEVER_TIME_COLUMN)
                timing_df = df[timing_cols].copy()
            
            # Process evaluation data
            eval_cols = [col for col in df.columns if col not in 
                        [RAG_QUERY_TIME_COLUMN, RAG_RETRIEVER_TIME_COLUMN]]
            all_dfs[model_name] = df[eval_cols]
            
            if combined_df is None:
                combined_df = df[eval_cols].copy()
                combined_df.columns = [f"{col}_{model_name}" if col != "question" else col 
                                    for col in combined_df.columns]
            else:
                rename_cols = {col: f"{col}_{model_name}" for col in df[eval_cols].columns 
                             if col != "question"}
                df_renamed = df[eval_cols].rename(columns=rename_cols)
                combined_df = combined_df.merge(df_renamed, on="question", how="outer")
    
    return all_dfs, combined_df, timing_df