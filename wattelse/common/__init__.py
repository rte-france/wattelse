#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import os
from pathlib import Path

TEXT_COLUMN = "text"
FILENAME_COLUMN = "filename"
SEED = 666

WATTELSE_BASE_DIR = os.getenv("WATTELSE_BASE_DIR", None)
BASE_PATH = Path(WATTELSE_BASE_DIR) if WATTELSE_BASE_DIR else Path(__file__).parent.parent.parent

BASE_DATA_PATH = BASE_PATH / "data"
BASE_OUTPUT_PATH = BASE_PATH / "output"
BASE_CACHE_PATH = BASE_PATH / "cache"
MODEL_BASE_PATH = BASE_PATH / "models"
FEED_BASE_PATH = BASE_DATA_PATH / "bertopic" / "feeds"
BERTOPIC_LOG_PATH = BASE_PATH / "bertopic" / "log"
WATTELSE_LOG_PATH = BASE_PATH / "wattelse" / "log"

BERTOPIC_LOG_PATH.mkdir(parents=True, exist_ok=True)
WATTELSE_LOG_PATH.mkdir(parents=True, exist_ok=True)