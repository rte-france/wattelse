#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from fastapi import FastAPI
from wattelse.api.embedding.routers import info, embeddings

app = FastAPI()

app.include_router(info.router, tags=["Info"])
app.include_router(embeddings.router, tags=["Embedding"])
