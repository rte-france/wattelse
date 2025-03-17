#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import os
import uvicorn
from wattelse.api.embedding.config.settings import get_config

# Load the configuration
CONFIG = get_config()

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG.cuda_visible_devices

# Start the FastAPI application
if __name__ == "__main__":
    uvicorn.run(
        "wattelse.api.embedding.main:app",
        host=CONFIG.host,
        port=CONFIG.port,
        workers=CONFIG.number_workers,
        ssl_keyfile="../key.pem",
        ssl_certfile="../cert.pem",
    )
