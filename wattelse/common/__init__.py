#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import socket
from pathlib import Path

TEXT_COLUMN = "text"
FILENAME_COLUMN = "filename"
SEED = 666
GPU_SERVERS = ["groesplu0", "GROESSLAO01"]
GPU_DSVD = ["pf9sodsia001", "pf9sodsia002", "pf9sodsia003"]

BASE_DIR = (
    Path("/data/weak_signals")
    if socket.gethostname() in GPU_SERVERS
    else Path("/scratch/nlp")
    if socket.gethostname() in GPU_DSVD
    else Path(__file__).parent.parent.parent
)

BASE_DATA_DIR = BASE_DIR / "data"
BASE_OUTPUT_DIR = BASE_DIR / "output"
BASE_CACHE_PATH = BASE_DIR / "cache"
MODEL_BASE_PATH = BASE_DIR / "models"
FEED_BASE_DIR = BASE_DATA_DIR / "bertopic" / "feeds"
LOG_DIR = BASE_DIR / "log"

LOG_DIR.mkdir(parents=True, exist_ok=True)