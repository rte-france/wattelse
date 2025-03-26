#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from pathlib import Path
import tomllib
from pydantic_settings import BaseSettings


class RAGOrchestratorAPIConfig(BaseSettings):
    host: str
    port: int


# Load config file
config_file = Path(__file__).parent / "default_config.toml"
with open(config_file, "rb") as f:
    CONFIG = RAGOrchestratorAPIConfig(**tomllib.load(f))
