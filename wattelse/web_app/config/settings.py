#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from pathlib import Path
from pydantic_settings import BaseSettings

from wattelse.config_utils import load_toml_config


class DjangoConfig(BaseSettings):
    host: str
    port: str


class GPTConfig(BaseSettings):
    base_url: str
    api_key: str
    default_model: str
    max_tokens: int
    max_messages: int


class GlobalConfig(BaseSettings):
    django: DjangoConfig
    gpt: GPTConfig


# Load config file
config_file = Path(__file__).parent / "default_config.toml"
CONFIG = GlobalConfig(**load_toml_config(config_file))
