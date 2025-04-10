"""
Prompts module for the evaluation pipeline.
"""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from wattelse.evaluation_pipeline.prompts.prompt_eval import PROMPTS
from wattelse.evaluation_pipeline.prompts.regex_patterns import RegexPatterns

__all__ = [
    "PROMPTS",
    "RegexPatterns",
]
