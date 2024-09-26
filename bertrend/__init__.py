#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import os
from pathlib import Path

# Linux command to find the index of the GPU device currently less used than the others
BEST_CUDA_DEVICE = (
    "\`nvidia-smi --query-gpu=index,memory.used --format=csv,nounits | tail -n +2 | sort -t',' -k2 -n  "
    "| head -n 1 | cut -d',' -f1\`"
)

BERTREND_BASE_DIR = os.getenv("BERTREND_BASE_DIR", None)
BASE_PATH = (
    Path(BERTREND_BASE_DIR)
    if BERTREND_BASE_DIR
    else Path(__file__).parent.parent.parent
)

BASE_DATA_PATH = BASE_PATH / "data"
BASE_CACHE_PATH = BASE_PATH / "cache"
FEED_BASE_PATH = BASE_DATA_PATH / "bertrend" / "feeds"
BERTREND_LOG_PATH = BASE_PATH / "logs" / "bertrend"
BERTREND_LOG_PATH.mkdir(parents=True, exist_ok=True)
