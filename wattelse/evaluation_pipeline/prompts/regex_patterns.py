#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RegexPatterns:
    """Class to manage regex patterns for different models and evaluation types."""

    patterns: Dict[str, Dict[str, str]] = None

    def __post_init__(self):
        self.patterns = {
            # Single model evaluation patterns
            "default": {
                "evaluation": r"Evaluation:\s*(.*)",
                "judgment": r"Judgment:\s*([1-5])",
            },
            "re_llama3": {
                "evaluation": r"Evaluation:\s*(.*)",
                "judgment": r"Judgment:\s*([1-5])",
            },
            "re_deepseek": {
                "evaluation": r"Evaluation:\s*(.*)",
                "judgment": r"(?:\*\*)?Judgment:(?:\*\*)?\s*([1-5])",
            },
            "re_selene": {
                "evaluation": r"Reasoning:\s*(.*)",
                "judgment": r"\*\*Result:\*\*\s*([1-5])\'?",
            },
            "re_prometheus": {
                "evaluation": r"(.*?)(?=\[SCORE\])",
                "judgment": r"\[SCORE\]\s*([1-5])",
            },
            # Pairwise comparison patterns
            "pairwise": {
                "analysis": r"ANALYSIS:\s*(.*?)(?=WINNER:|$)",
                "winner": r"WINNER:\s*(.+?)(?=REASON:|$)",
                "reason": r"REASON:\s*(.*?)$",
            },
        }

    def get_patterns(self, regex_type: str) -> Dict[str, str]:
        """
        Get regex patterns for a specific model type.

        Args:
            regex_type: The model type or evaluation type to get patterns for

        Returns:
            Dict[str, str]: Dictionary of pattern names and their regex patterns
        """
        return self.patterns.get(regex_type, self.patterns["default"])

    def get_pairwise_patterns(self, model_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get pairwise comparison patterns for a specific model.

        Args:
            model_name: The name of the model to get pairwise patterns for

        Returns:
            Dict[str, str]: Dictionary of pairwise pattern names and their regex patterns
        """
        if model_name:
            # Try to get model-specific pairwise patterns
            pattern_key = f"pairwise_{model_name.lower()}"
            if pattern_key in self.patterns:
                return self.patterns[pattern_key]

        # Default to generic pairwise patterns
        return self.patterns.get("pairwise")

    def add_pattern(
        self, model_type: str, evaluation_pattern: str, judgment_pattern: str
    ) -> None:
        """
        Add new regex patterns for a single model evaluation.

        Args:
            model_type: The model type to add patterns for
            evaluation_pattern: The regex pattern for extracting evaluation text
            judgment_pattern: The regex pattern for extracting judgment score
        """
        self.patterns[model_type] = {
            "evaluation": evaluation_pattern,
            "judgment": judgment_pattern,
        }

    def add_pairwise_pattern(
        self,
        model_type: str,
        analysis_pattern: str,
        winner_pattern: str,
        reason_pattern: str,
    ) -> None:
        """
        Add new regex patterns for pairwise comparison.

        Args:
            model_type: The model type to add patterns for
            analysis_pattern: The regex pattern for extracting analysis text
            winner_pattern: The regex pattern for extracting winner
            reason_pattern: The regex pattern for extracting reason
        """
        pattern_key = f"pairwise_{model_type.lower()}"
        self.patterns[pattern_key] = {
            "analysis": analysis_pattern,
            "winner": winner_pattern,
            "reason": reason_pattern,
        }
