"""
Configuration module for the evaluation pipeline.
"""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from wattelse.evaluation_pipeline.config.eval_config import EvalConfig
from wattelse.evaluation_pipeline.config.server_config import ServerConfig

__all__ = [
    "EvalConfig",
    "ServerConfig",
]
