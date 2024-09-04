#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import os
from pathlib import Path

TEXT_COLUMN = "text"
FILENAME_COLUMN = "filename"
SEED = 666

# Linux command to find the index of the GPU device currently less used than the others
BEST_CUDA_DEVICE = "\`nvidia-smi --query-gpu=index,memory.used --format=csv,nounits | tail -n +2 | sort -t',' -k2 -n  | head -n 1 | cut -d',' -f1\`"

WATTELSE_BASE_DIR = os.getenv("WATTELSE_BASE_DIR", None)
BASE_PATH = (
    Path(WATTELSE_BASE_DIR)
    if WATTELSE_BASE_DIR
    else Path(__file__).parent.parent.parent
)

BASE_DATA_PATH = BASE_PATH / "data"
BASE_OUTPUT_PATH = BASE_PATH / "output"
BASE_CACHE_PATH = BASE_PATH / "cache"
MODEL_BASE_PATH = BASE_PATH / "models"
FEED_BASE_PATH = BASE_DATA_PATH / "bertopic" / "feeds"
BERTOPIC_LOG_PATH = BASE_PATH / "logs" / "bertopic"
WATTELSE_LOG_PATH = BASE_PATH / "logs" / "wattelse"

BERTOPIC_LOG_PATH.mkdir(parents=True, exist_ok=True)
WATTELSE_LOG_PATH.mkdir(parents=True, exist_ok=True)
