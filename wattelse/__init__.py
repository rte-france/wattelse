#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import os
from pathlib import Path

WATTELSE_BASE_DIR = os.getenv("WATTELSE_BASE_DIR", None)
BASE_PATH = (
    Path(WATTELSE_BASE_DIR)
    if WATTELSE_BASE_DIR
    else Path(__file__).parent.parent.parent
)

BASE_DATA_PATH = BASE_PATH
