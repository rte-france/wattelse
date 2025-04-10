#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import subprocess
from wattelse.web_app.config.settings import CONFIG

# Start Django application
command = [
    "python",
    "manage.py",
    "runserver",
    f"{CONFIG.host}:{CONFIG.port}",
    "--insecure",
]
subprocess.run(command)
