#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import os
from pathlib import Path
import tomllib
from pydantic_settings import BaseSettings

DEFAULT_CONFIG_FILE = Path(__file__).parent / "default_config.toml"


class VLLMConfig(BaseSettings):
    host: str
    port: str
    port_controller: str
    port_worker: str
    model_name: str
    cuda_visible_devices: str


# Use environment variable to specify which config to load
def get_config() -> VLLMConfig:
    """
    Return the configuration for the vLLM API.
    The configuration is loaded from a toml file.
    The path to the toml file can be specified using the environment variable `VLLM_API_CONFIG_FILE`.
    If the environment variable is not set, the default configuration file is used.
    """
    config_file = os.environ.get("VLLM_API_CONFIG_FILE", None)
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    with open(config_file, "rb") as f:
        return VLLMConfig(**tomllib.load(f))
