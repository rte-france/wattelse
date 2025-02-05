"""Analysis utilities for experiment data."""

import pandas as pd
from .constants import JUDGE_COLORS

def calculate_good_score_percentage(scores):
    """Calculate percentage of good scores (4-5) in the series."""
    if scores is None or len(scores) == 0:
        return 0
    good_scores = scores[scores.isin([4, 5])].count()
    total_scores = scores.count()
    return (good_scores / total_scores * 100) if total_scores > 0 else 0

def create_performance_summary(experiments_data):
    """Create a formatted performance summary dataframe."""
    metrics_data = []
    
    # Get all available metrics
    all_metrics = set()
    for exp in experiments_data:
        for df in exp['dfs'].values():
            all_metrics.update([col.replace('_score', '') 
                              for col in df.columns if col.endswith('_score')])
    
    # Create summary for each judge
    for judge in JUDGE_COLORS.keys():
        for metric in sorted(all_metrics):
            row = {'Judge': judge, 'Metric': metric}
            
            # Add data for each experiment
            for exp in experiments_data:
                if judge in exp['dfs']:
                    df = exp['dfs'][judge]
                    score_col = f'{metric}_score'
                    if score_col in df.columns:
                        row[f"{exp['name']} (Avg)"] = f"{df[score_col].mean():.2f}"
                        row[f"{exp['name']} (Good %)"] = f"{calculate_good_score_percentage(df[score_col]):.1f}%"
                
            metrics_data.append(row)
    
    return pd.DataFrame(metrics_data)