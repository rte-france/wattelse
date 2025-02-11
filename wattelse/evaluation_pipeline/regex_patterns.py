from dataclasses import dataclass
from typing import Dict

# This dataclass is the one that is able to define regex for each model (parsing function)
@dataclass
class RegexPatterns:
    """Class to manage regex patterns for different models."""
    patterns: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        self.patterns = {
            "default": {
                "evaluation": r"Evaluation:\s*(.*)",
                "judgment": r"Judgment:\s*([1-5])"
            },
            "re_llama3": {
                "evaluation": r"Evaluation:\s*(.*)",
                "judgment": r"Judgment:\s*([1-5])"
            },
            "re_deepseek": {
                "evaluation": r"Evaluation:\s*(.*)",
                "judgment": r"(?:\*\*)?Judgment:(?:\*\*)?\s*([1-5])"
            },
            "re_selene": {
                "evaluation": r"Reasoning:\s*(.*)",
                "judgment": r"\*\*Result:\*\*\s*([1-5])\'?"
            },
            "re_prometheus": {
                "evaluation": r"(.*?)(?=\[SCORE\])",
                "judgment": r"\[SCORE\]\s*([1-5])"
            }
        }
    
    def get_patterns(self, regex_type: str) -> Dict[str, str]:
        """Get regex patterns for a specific model type."""
        return self.patterns.get(regex_type, self.patterns["default"])  # Default to llama3 patterns
    
    def add_pattern(self, model_type: str, evaluation_pattern: str, judgment_pattern: str) -> None:
        """Add new regex patterns for a model type."""
        self.patterns[model_type] = {
            "evaluation": evaluation_pattern,
            "judgment": judgment_pattern
        }