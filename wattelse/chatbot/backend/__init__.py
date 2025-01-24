#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import os

from wattelse.common import BASE_DATA_PATH, BASE_CACHE_PATH

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
