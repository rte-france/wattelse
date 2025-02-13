"""Analysis utilities for experiment data."""

def calculate_good_score_percentage(scores):
    """Calculate percentage of good scores (4-5) in the series."""
    if scores is None or len(scores) == 0:
        return 0
    good_scores = scores[scores.isin([4, 5])].count()
    total_scores = scores.count()
    return (good_scores / total_scores * 100) if total_scores > 0 else 0