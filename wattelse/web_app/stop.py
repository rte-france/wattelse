#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import os

from wattelse.web_app.config.settings import CONFIG

# Stop processes associated to API port
os.system(f"kill $(lsof -t -i:{CONFIG.django.port})")
