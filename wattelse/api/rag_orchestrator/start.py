#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import uvicorn
from wattelse.api.rag_orchestrator.config.settings import CONFIG

# Start the FastAPI application
if __name__ == "__main__":
    uvicorn.run(
        "wattelse.api.rag_orchestrator.main:app",
        host=CONFIG.host,
        port=CONFIG.port,
        reload=True,
        ssl_keyfile="../key.pem",
        ssl_certfile="../cert.pem",
    )
