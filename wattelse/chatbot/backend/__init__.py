#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import configparser
import os

from pathlib import Path

from wattelse.common import BASE_DATA_PATH, BASE_CACHE_PATH
from wattelse.common.config_utils import parse_literal, EnvInterpolation

# Ensures to write with +rw for both user and groups
os.umask(0o002)

# Make dirs if not exist
DATA_DIR = BASE_DATA_PATH / "chatbot"
CACHE_DIR = BASE_CACHE_PATH / "chatbot"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Retriever methods
MMR = "mmr"
SIMILARITY = "similarity"
SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"
BM25 = "bm25"
ENSEMBLE = "ensemble"

# Config for retriever and generator
config = configparser.ConfigParser(
    converters={"literal": parse_literal}, interpolation=EnvInterpolation()
)  # takes into account environment variables
config.read(Path(__file__).parent / "rag_config.cfg")

retriever_config = parse_literal(dict(config["retriever"]))

generator_config = parse_literal(dict(config["generator"]))
