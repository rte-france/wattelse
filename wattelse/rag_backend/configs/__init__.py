#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from pathlib import Path

# Mapping between the configuration name as it should appear in Django (file name without extension and full path
CONFIG_NAME_TO_CONFIG_PATH = {
    path.stem: path for path in Path(__file__).parent.glob("*.toml")
}
