#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from fastapi import FastAPI
from wattelse.api.rag_orchestrator.routers import (
    sessions,
    info,
    rag_query,
    documents,
    authentication,
)

app = FastAPI()

app.include_router(info.router, tags=["Info"])
app.include_router(sessions.router, tags=["Sessions"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(rag_query.router, tags=["Queries"])
app.include_router(authentication.router, tags=["Authentication"])
